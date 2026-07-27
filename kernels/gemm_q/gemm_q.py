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

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined

@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def gemm_q_npu(
    W_combined: In,
    X: In,
    Y: Out,
    *,
    B: CompileTime[int],
    N: CompileTime[int],
    K: CompileTime[int],
    m: CompileTime[int],
    k_tile: CompileTime[int],
):
    if N % (8 * m) == 0:
        n_cores = 8
    elif N % (4 * m) == 0:
        n_cores = 4
    elif N % (2 * m) == 0:
        n_cores = 2
    else:
        n_cores = 1

    N_div_n_cores = N // n_cores

    w_ty = np.ndarray[(m * (k_tile // 32) * 20,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(B * k_tile,), np.dtype[bfloat16]]
    # y accumulates in fp32 across the K//k_tile kernel calls (the fp32 mmul
    # accumulator must persist between calls; full-K per call would not fit L1).
    y_ty = np.ndarray[(B * m,), np.dtype[np.float32]]

    kernel_flags = [f"-DDIM_M={m}", f"-DDIM_K={k_tile}", f"-DDIM_B={B}", "-O3"]

    gemm_kernel = ExternalFunction(
        "gemm_q",
        source_file=str(Path(__file__).parent / "gemm_q.cc"),
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

    of_x = ObjectFifo(x_ty, name="of_x")
    
    memW_fifos = []
    outY_fifos = []
    workers = []

    def core_fn(of_w, of_x, of_y, zero_k, gemm_k):
        for _ in range_(N_div_n_cores // m):
            elem_y = of_y.acquire(1)
            zero_k(elem_y)
            for _ in range_(K // k_tile):
                elem_w = of_w.acquire(1)
                elem_x = of_x.acquire(1)
                gemm_k(elem_w, elem_x, elem_y)
                of_w.release(1)
                of_x.release(1)
            of_y.release(1)

    for i in range(n_cores):
        w_fifo = ObjectFifo(w_ty, name=f"of_w_{i}")
        y_fifo = ObjectFifo(y_ty, name=f"of_y_{i}")
        memW_fifos.append(w_fifo)
        outY_fifos.append(y_fifo)
        
        workers.append(
            Worker(
                core_fn,
                fn_args=[
                    w_fifo.cons(),
                    of_x.cons(),
                    y_fifo.prod(),
                    zero_kernel,
                    gemm_kernel,
                ],
                stack_size=0xF00,
            )
        )

    W_ty = np.ndarray[(N, (K // 32) * 20), np.dtype[np.uint8]]
    X_ty = np.ndarray[(B, K), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(B, N), np.dtype[np.float32]]

    w_taps = TensorTiler2D.group_tiler(
        (N, (K // 32) * 20),
        (m, (k_tile // 32) * 20),
        (N_div_n_cores // m, K // k_tile),
        prune_step=False
    )
    
    x_tap = TensorTiler2D.group_tiler(
        (B, K), (B, k_tile), (1, K // k_tile), pattern_repeat=N_div_n_cores // m, prune_step=False
    )[0]
    
    y_taps = TensorTiler2D.group_tiler(
        (B, N), (B, m), (1, N_div_n_cores // m), prune_step=False
    )

    rt = Runtime()
    with rt.sequence(W_ty, X_ty, Y_ty) as (w_in, x_in, y_out):
        rt.start(*workers)
        
        tg = rt.task_group()
        rt.fill(of_x.prod(), x_in, x_tap, task_group=tg)
        for i in range(n_cores):
            rt.fill(memW_fifos[i].prod(), w_in, w_taps[i], task_group=tg)
        
        for i in range(n_cores):
            rt.drain(outY_fifos[i].cons(), y_out, y_taps[i], wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()

def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Quantized GEMM")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-B", type=int, default=16)
    p.add_argument("-N", type=int, default=1024)
    p.add_argument("-K", type=int, default=1024)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=256)
    add_benchmark_args(p)
    return p

def _run_and_verify(opts):
    print(f"Verifying shape: B={opts.B}, N={opts.N}, K={opts.K} with tiling: m={opts.m}, k={opts.k}")
    
    rng = np.random.default_rng(1726250518)
    W_fp32 = rng.uniform(-1.0, 1.0, size=(opts.N, opts.K)).astype(np.float32)
    X_np = rng.uniform(-1.0, 1.0, size=(opts.B, opts.K)).astype(np.float32)
    
    from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined
    w_q4_np, scales_np = quantize_to_q4_0(W_fp32)
    w_combined_np = pack_to_combined(w_q4_np, scales_np)
    
    w_combined_t = iron.tensor(w_combined_np.reshape(-1), dtype=np.uint8, device="npu")
    X_t = iron.tensor(X_np.astype(bfloat16).reshape(-1), dtype=bfloat16, device="npu")
    Y_t = iron.zeros(opts.B * opts.N, dtype=np.float32, device="npu")
    
    bench = run_iters(
        gemm_q_npu,
        w_combined_t,
        X_t,
        Y_t,
        B=opts.B,
        N=opts.N,
        K=opts.K,
        m=opts.m,
        k_tile=opts.k,
        warmup=opts.warmup,
        iters=opts.iters,
    )
    
    from tools.ref.gemv_q import dequantize_combined
    w_fp32_deq = dequantize_combined(w_combined_np)

    expected = (X_np @ w_fp32_deq.T).astype(bfloat16)
    actual = Y_t.numpy().reshape(opts.B, opts.N)
    
    rtol = 0.05
    atol = 1.0
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    print(f"CHECK max_diff_vs_cpu_ref={float(np.max(diff)):.4f} (expect ~2-4 if correct; huge = broken)")
    print(f"  actual[0,:6]={actual[0,:6]}  expected[0,:6]={expected[0,:6]}")
    if bench:
        print(bench)

def _compile_kwargs(opts):
    return dict(
        B=opts.B,
        N=opts.N,
        K=opts.K,
        m=opts.m,
        k_tile=opts.k,
    )

def _resolve_full_device(opts):
    """Resolve ``--dev`` to the max-column variant (n_cols=None), else
    ``from_name`` defaults to n_cols=1 and placement fails with "no available
    compute tiles". Mirrors gemv_q.py."""
    from aie.iron.device import from_name
    return from_name(opts.dev, n_cols=None)

def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gemm_q_npu,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=_resolve_full_device,
    )

if __name__ == "__main__":
    main()
