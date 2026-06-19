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
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.config import cxx_header_path

@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def rope_npu(
    X: In,
    CosSin: In,
    Y: Out,
    *,
    K: CompileTime[int],
    head_dim: CompileTime[int],
):
    # Local buffers inside L1
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    cs_ty = np.ndarray[(head_dim * 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(K,), np.dtype[bfloat16]]

    kernel_flags = [f"-DDIM_K={K}", f"-DDIM_H={head_dim}"]

    rope_kernel = ExternalFunction(
        "rope",
        source_file=str(Path(__file__).parent / "rope.cc"),
        arg_types=[x_ty, cs_ty, y_ty],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    of_x = ObjectFifo(x_ty, name="of_x")
    of_cs = ObjectFifo(cs_ty, name="of_cs")
    of_y = ObjectFifo(y_ty, name="of_y")

    def core_fn(of_x, of_cs, of_y, rope_k):
        elem_x = of_x.acquire(1)
        elem_cs = of_cs.acquire(1)
        elem_y = of_y.acquire(1)
        
        rope_k(elem_x, elem_cs, elem_y)
        
        of_x.release(1)
        of_cs.release(1)
        of_y.release(1)

    worker = Worker(
        core_fn,
        fn_args=[
            of_x.cons(),
            of_cs.cons(),
            of_y.prod(),
            rope_kernel,
        ],
        stack_size=0x800,
    )

    # DRAM shapes
    X_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    CosSin_ty = np.ndarray[(head_dim * 2,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(K,), np.dtype[bfloat16]]

    x_tap = TensorTiler2D.group_tiler((1, K), (1, K), (1, 1), prune_step=False)[0]
    cs_tap = TensorTiler2D.group_tiler((1, head_dim * 2), (1, head_dim * 2), (1, 1), prune_step=False)[0]
    y_tap = TensorTiler2D.group_tiler((1, K), (1, K), (1, 1), prune_step=False)[0]

    rt = Runtime()
    with rt.sequence(X_ty, CosSin_ty, Y_ty) as (x_in, cs_in, y_out):
        rt.start(worker)
        rt.fill(of_x.prod(), x_in, x_tap)
        rt.fill(of_cs.prod(), cs_in, cs_tap)
        rt.drain(of_y.cons(), y_out, y_tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()

def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE RoPE")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-H", type=int, default=64)
    add_benchmark_args(p)
    return p

def _run_and_verify(opts):
    print(f"Verifying RoPE shape: K={opts.K}, head_dim={opts.H}")
    
    # Generate random data
    rng = np.random.default_rng(42)
    x_np = rng.uniform(-1.0, 1.0, size=(opts.K,)).astype(np.float32)
    cos_np = rng.uniform(-1.0, 1.0, size=(opts.H,)).astype(np.float32)
    sin_np = rng.uniform(-1.0, 1.0, size=(opts.H,)).astype(np.float32)
    
    # Pack cos and sin
    cos_sin_np = np.concatenate([cos_np, sin_np])
    
    # Compute CPU reference
    expected = np.zeros_like(x_np)
    num_heads = opts.K // opts.H
    half = opts.H // 2
    for h in range(num_heads):
        head = x_np[h * opts.H : (h + 1) * opts.H]
        x1 = head[:half]
        x2 = head[half:]
        expected[h * opts.H : h * opts.H + half] = x1 * cos_np[:half] - x2 * sin_np[:half]
        expected[h * opts.H + half : (h + 1) * opts.H] = x2 * cos_np[:half] + x1 * sin_np[:half]
    
    # Create NPU tensors
    x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
    cos_sin_t = iron.tensor(cos_sin_np.astype(bfloat16), dtype=bfloat16, device="npu")
    y_t = iron.zeros(opts.K, dtype=bfloat16, device="npu")
    
    run_iters(
        rope_npu,
        x_t,
        cos_sin_t,
        y_t,
        K=opts.K,
        head_dim=opts.H,
        warmup=opts.warmup,
        iters=opts.iters,
    )
    
    actual = y_t.numpy()
    
    rtol = 0.01
    atol = 0.02
    close = np.allclose(actual, expected.astype(bfloat16), rtol=rtol, atol=atol)
    if not close:
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        print(f"Max abs diff: {np.max(diff)}")
        assert close, "NPU output does not match CPU reference!"
        
    print("PASS!")

def _compile_kwargs(opts):
    return dict(K=opts.K, head_dim=opts.H)

def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        rope_npu,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )

if __name__ == "__main__":
    main()
