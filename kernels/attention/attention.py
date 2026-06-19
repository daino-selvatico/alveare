import argparse
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.config import cxx_header_path

@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def attention_npu(
    Q: In,
    KVCache: In,
    O: Out,
    *,
    seq_len: CompileTime[int],
    head_dim: CompileTime[int],
):
    # Local buffers inside L1
    q_ty = np.ndarray[(4 * head_dim,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(seq_len * head_dim * 2,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(4 * head_dim,), np.dtype[bfloat16]]

    kernel_flags = [f"-DDIM_H={head_dim}", f"-DMAX_SEQ_LEN={seq_len}"]

    attn_kernel = ExternalFunction(
        "attention",
        source_file=str(Path(__file__).parent / "attention.cc"),
        arg_types=[q_ty, kv_ty, o_ty],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    of_q = ObjectFifo(q_ty, name="of_q")
    of_kv = ObjectFifo(kv_ty, name="of_kv")
    of_o = ObjectFifo(o_ty, name="of_o")

    def core_fn(of_q, of_kv, of_o, attn_k):
        for _ in range_(8):
            elem_q = of_q.acquire(1)
            elem_kv = of_kv.acquire(1)
            elem_o = of_o.acquire(1)
            
            attn_k(elem_q, elem_kv, elem_o)
            
            of_q.release(1)
            of_kv.release(1)
            of_o.release(1)

    worker = Worker(
        core_fn,
        fn_args=[
            of_q.cons(),
            of_kv.cons(),
            of_o.prod(),
            attn_kernel,
        ],
        stack_size=0xF00,
    )

    # DRAM shapes
    Q_ty = np.ndarray[(8, 4 * head_dim), np.dtype[bfloat16]]
    KVCache_ty = np.ndarray[(8, seq_len * head_dim * 2), np.dtype[bfloat16]]
    O_ty = np.ndarray[(8, 4 * head_dim), np.dtype[bfloat16]]

    q_tap = TensorTiler2D.group_tiler((8, 4 * head_dim), (1, 4 * head_dim), (8, 1), prune_step=False)[0]
    kv_tap = TensorTiler2D.group_tiler((8, seq_len * head_dim * 2), (1, seq_len * head_dim * 2), (8, 1), prune_step=False)[0]
    o_tap = TensorTiler2D.group_tiler((8, 4 * head_dim), (1, 4 * head_dim), (8, 1), prune_step=False)[0]

    rt = Runtime()
    with rt.sequence(Q_ty, KVCache_ty, O_ty) as (q_in, kv_in, o_out):
        rt.start(worker)
        rt.fill(of_q.prod(), q_in, q_tap)
        rt.fill(of_kv.prod(), kv_in, kv_tap)
        rt.drain(of_o.cons(), o_out, o_tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()

def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Attention")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-S", "--seq_len", type=int, default=32)
    p.add_argument("-H", "--head_dim", type=int, default=64)
    add_benchmark_args(p)
    return p

def _run_and_verify(opts):
    print(f"Verifying Attention shape: seq_len={opts.seq_len}, head_dim={opts.head_dim}")
    
    # Generate random test data
    rng = np.random.default_rng(42)
    q_np = rng.uniform(-1.0, 1.0, size=(8, 4, opts.head_dim)).astype(np.float32)
    k_np = rng.uniform(-1.0, 1.0, size=(8, opts.seq_len, opts.head_dim)).astype(np.float32)
    v_np = rng.uniform(-1.0, 1.0, size=(8, opts.seq_len, opts.head_dim)).astype(np.float32)
    
    # Pack key and value into KV cache shape (8, seq_len, 2 * head_dim)
    kv_cache_np = np.zeros((8, opts.seq_len, opts.head_dim * 2), dtype=np.float32)
    kv_cache_np[:, :, :opts.head_dim] = k_np
    kv_cache_np[:, :, opts.head_dim:] = v_np
    
    # Compute CPU reference using our formula
    expected = np.zeros((8, 4, opts.head_dim), dtype=np.float32)
    scale = 1.0 / np.sqrt(opts.head_dim)
    for g in range(8):
        for q in range(4):
            q_head = q_np[g, q]
            scores = np.zeros(opts.seq_len, dtype=np.float32)
            max_score = -1e9
            for t in range(opts.seq_len):
                score = np.dot(q_head, k_np[g, t]) * scale
                scores[t] = score
                if score > max_score:
                    max_score = score
            # Softmax
            exp_scores = np.exp(scores - max_score)
            probs = exp_scores / np.sum(exp_scores)
            
            # Aggregate
            out = np.zeros(opts.head_dim, dtype=np.float32)
            for t in range(opts.seq_len):
                out += probs[t] * v_np[g, t]
            expected[g, q] = out
            
    # Create NPU tensors
    q_t = iron.tensor(q_np.astype(bfloat16).reshape(8, -1), dtype=bfloat16, device="npu")
    kv_t = iron.tensor(kv_cache_np.astype(bfloat16).reshape(8, -1), dtype=bfloat16, device="npu")
    o_t = iron.zeros((8, 4 * opts.head_dim), dtype=bfloat16, device="npu")
    
    run_iters(
        attention_npu,
        q_t,
        kv_t,
        o_t,
        seq_len=opts.seq_len,
        head_dim=opts.head_dim,
        warmup=opts.warmup,
        iters=opts.iters,
    )
    
    actual = o_t.numpy().reshape(8, 4, opts.head_dim)
    
    rtol = 0.05
    atol = 0.05
    close = np.allclose(actual, expected.astype(bfloat16), rtol=rtol, atol=atol)
    if not close:
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        print(f"Max abs diff: {np.max(diff)}")
        assert close, "NPU output does not match CPU reference!"
        
    print("PASS!")

def _compile_kwargs(opts):
    return dict(seq_len=opts.seq_len, head_dim=opts.head_dim)

def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        attention_npu,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )

if __name__ == "__main__":
    main()
