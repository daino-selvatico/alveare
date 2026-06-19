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

# Import reference and converter
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from tools.ref.gemv_q import gemv_q_combined as ref_gemv_q
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined

@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def gemv_q_npu(
    W_combined: In,
    X: In,
    Y: Out,
    *,
    N: CompileTime[int],
    K: CompileTime[int],
    m: CompileTime[int],
    k_tile: CompileTime[int],
):
    # Dimensions inside core
    w_ty = np.ndarray[(m * (k_tile // 32) * 20,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(k_tile,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(m,), np.dtype[bfloat16]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k_tile}"]

    # Custom AIE Kernels
    gemv_kernel = ExternalFunction(
        "gemv_q",
        source_file=str(Path(__file__).parent / "gemv_q.cc"),
        arg_types=[w_ty, x_ty, y_ty],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    zero_kernel = ExternalFunction(
        "zero_kernel_bf16",
        source_file=str(Path(__file__).parent / "zero_kernel.cc"),
        arg_types=[y_ty],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    # Object FIFOs in local AIE memory
    of_w = ObjectFifo(w_ty, name="of_w")
    of_x = ObjectFifo(x_ty, name="of_x")
    of_y = ObjectFifo(y_ty, name="of_y")

    # Core function execution flow
    def core_fn(of_w, of_x, of_y, zero_k, gemv_k):
        for _ in range_(N // m):
            elem_y = of_y.acquire(1)
            zero_k(elem_y)
            for _ in range_(K // k_tile):
                elem_w = of_w.acquire(1)
                elem_x = of_x.acquire(1)
                gemv_k(elem_w, elem_x, elem_y)
                of_w.release(1)
                of_x.release(1)
            of_y.release(1)

    # Instantiate worker on AIE core
    worker = Worker(
        core_fn,
        fn_args=[
            of_w.cons(),
            of_x.cons(),
            of_y.prod(),
            zero_kernel,
            gemv_kernel,
        ],
        stack_size=0xF00,
    )

    # DRAM/Host buffers shapes
    W_ty = np.ndarray[(N, (K // 32) * 20), np.dtype[np.uint8]]
    X_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(N,), np.dtype[bfloat16]]

    # Combined weights tiler (streams all tiles row-by-row sequentially)
    w_tap = TensorTiler2D.group_tiler(
        (N, (K // 32) * 20), (m, (k_tile // 32) * 20), (N // m, K // k_tile), prune_step=False
    )[0]
    
    # Activation vector tiling (shape 1xK, tiled to 1xk_tile, repeated N/m times)
    x_tap = TensorTiler2D.group_tiler(
        (1, K), (1, k_tile), (1, K // k_tile), pattern_repeat=N // m, prune_step=False
    )[0]
    
    # Output vector tiling (shape 1xN, tiled to 1xm)
    y_tap = TensorTiler2D.group_tiler(
        (1, N), (1, m), (1, N // m), prune_step=False
    )[0]

    rt = Runtime()
    with rt.sequence(W_ty, X_ty, Y_ty) as (w_in, x_in, y_out):
        rt.start(worker)
        rt.fill(of_x.prod(), x_in, x_tap)
        rt.fill(of_w.prod(), w_in, w_tap)
        rt.drain(of_y.cons(), y_out, y_tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()

def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Quantized GEMV")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-N", type=int, default=256)
    p.add_argument("-K", type=int, default=256)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=256)
    add_benchmark_args(p)
    return p

def _run_and_verify(opts):
    print(f"Verifying shape: N={opts.N}, K={opts.K} with tiling: m={opts.m}, k={opts.k}")
    
    # Generate random FP32 weights and activation vector
    rng = np.random.default_rng(1726250518)
    W_fp32 = rng.uniform(-1.0, 1.0, size=(opts.N, opts.K)).astype(np.float32)
    x_np = rng.uniform(-1.0, 1.0, size=(opts.K,)).astype(np.float32)
    
    # Quantize and pack using our Q4_0 and combined routines
    w_q4_np, scales_np = quantize_to_q4_0(W_fp32)
    w_combined_np = pack_to_combined(w_q4_np, scales_np)
    
    # Create NPU tensors
    w_combined_t = iron.tensor(w_combined_np.reshape(-1), dtype=np.uint8, device="npu")
    x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
    y_t = iron.zeros(opts.N, dtype=bfloat16, device="npu")
    
    bench = run_iters(
        gemv_q_npu,
        w_combined_t,
        x_t,
        y_t,
        N=opts.N,
        K=opts.K,
        m=opts.m,
        k_tile=opts.k,
        warmup=opts.warmup,
        iters=opts.iters,
    )
    
    # Compute CPU reference
    expected = ref_gemv_q(w_combined_np, x_np.astype(bfloat16))
    actual = y_t.numpy()
    
    # Compare results
    rtol = 0.05
    atol = 1.0
    
    close = np.allclose(actual, expected, rtol=rtol, atol=atol)
    if not close:
        # Print differences
        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        print(f"Max absolute difference: {max_diff}")
        print(f"Actual (first 10): {actual[:10]}")
        print(f"Expected (first 10): {expected[:10]}")
        assert close, "NPU output does not match NumPy CPU reference!"
        
    print("PASS!")
    if bench:
        print(bench)

def _compile_kwargs(opts):
    return dict(
        N=opts.N,
        K=opts.K,
        m=opts.m,
        k_tile=opts.k,
    )

def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gemv_q_npu,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )

if __name__ == "__main__":
    main()
