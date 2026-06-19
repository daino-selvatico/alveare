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
def rmsnorm_npu(
    X: In,
    W: In,
    Y: Out,
    *,
    K: CompileTime[int],
):
    # Local buffers inside AIE L1 memory
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(K,), np.dtype[np.float32]]
    y_ty = np.ndarray[(K,), np.dtype[bfloat16]]

    kernel_flags = [f"-DDIM_K={K}"]

    rmsnorm_kernel = ExternalFunction(
        "rmsnorm",
        source_file=str(Path(__file__).parent / "rmsnorm.cc"),
        arg_types=[x_ty, w_ty, y_ty],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    of_x = ObjectFifo(x_ty, name="of_x")
    of_w = ObjectFifo(w_ty, name="of_w")
    of_y = ObjectFifo(y_ty, name="of_y")

    def core_fn(of_x, of_w, of_y, norm_k):
        # We only process 1 tile of size K
        elem_x = of_x.acquire(1)
        elem_w = of_w.acquire(1)
        elem_y = of_y.acquire(1)
        
        norm_k(elem_x, elem_w, elem_y)
        
        of_x.release(1)
        of_w.release(1)
        of_y.release(1)

    worker = Worker(
        core_fn,
        fn_args=[
            of_x.cons(),
            of_w.cons(),
            of_y.prod(),
            rmsnorm_kernel,
        ],
        stack_size=0x800,
    )

    # DRAM shapes
    X_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    W_ty = np.ndarray[(K,), np.dtype[np.float32]]
    Y_ty = np.ndarray[(K,), np.dtype[bfloat16]]

    # 1D tiles formatted as 2D groups for the tiler
    x_tap = TensorTiler2D.group_tiler((1, K), (1, K), (1, 1), prune_step=False)[0]
    w_tap = TensorTiler2D.group_tiler((1, K), (1, K), (1, 1), prune_step=False)[0]
    y_tap = TensorTiler2D.group_tiler((1, K), (1, K), (1, 1), prune_step=False)[0]

    rt = Runtime()
    with rt.sequence(X_ty, W_ty, Y_ty) as (x_in, w_in, y_out):
        rt.start(worker)
        rt.fill(of_x.prod(), x_in, x_tap)
        rt.fill(of_w.prod(), w_in, w_tap)
        rt.drain(of_y.cons(), y_out, y_tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()

def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE RMSNorm")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-K", type=int, default=2048)
    add_benchmark_args(p)
    return p

def _run_and_verify(opts):
    print(f"Verifying RMSNorm shape: K={opts.K}")
    
    # Generate random test data
    rng = np.random.default_rng(42)
    x_np = rng.uniform(-1.0, 1.0, size=(opts.K,)).astype(np.float32)
    w_np = rng.uniform(0.5, 1.5, size=(opts.K,)).astype(np.float32)
    
    # Compute CPU reference
    variance = np.mean(x_np**2)
    inv_std = 1.0 / np.sqrt(variance + 1e-5)
    expected = (x_np * inv_std * w_np).astype(bfloat16)
    
    # Create NPU tensors
    x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
    w_t = iron.tensor(w_np, dtype=np.float32, device="npu")
    y_t = iron.zeros(opts.K, dtype=bfloat16, device="npu")
    
    run_iters(
        rmsnorm_npu,
        x_t,
        w_t,
        y_t,
        K=opts.K,
        warmup=opts.warmup,
        iters=opts.iters,
    )
    
    actual = y_t.numpy()
    
    # Compare
    rtol = 1e-3
    atol = 1e-3
    close = np.allclose(actual, expected, rtol=rtol, atol=atol)
    if not close:
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        print(f"Max abs diff: {np.max(diff)}")
        print(f"Actual (first 10): {actual[:10]}")
        print(f"Expected (first 10): {expected[:10]}")
        assert close, "NPU output does not match CPU reference!"
        
    print("PASS!")

def _compile_kwargs(opts):
    return dict(K=opts.K)

def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        rmsnorm_npu,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
    )

if __name__ == "__main__":
    main()
