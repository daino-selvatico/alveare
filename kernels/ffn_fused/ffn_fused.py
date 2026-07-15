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
def ffn_fused_npu(
    W_fused: In,
    X: In,
    Y_partial: Out,
    *,
    H: CompileTime[int],
    I: CompileTime[int],
    m_I: CompileTime[int],
    k_tile: CompileTime[int],
    activation: CompileTime[str],
):
    # Enforce m_H = k_tile for perfectly aligned homogenous weight tile sizing (zero padding)
    m_H = k_tile

    # Dynamically select n_cores based on intermediate size I and m_I
    if I % (8 * m_I) == 0:
        n_cores = 8
    elif I % (4 * m_I) == 0:
        n_cores = 4
    elif I % (2 * m_I) == 0:
        n_cores = 2
    else:
        n_cores = 1

    I_div_n_cores = I // n_cores
    num_blocks_I = I_div_n_cores // m_I

    # Both gate/up tile and down tile are of size (m_I * (k_tile // 32) * 20) bytes.
    tile_size = m_I * (k_tile // 32) * 20
    w_ty = np.ndarray[(tile_size,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(H,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(m_H,), np.dtype[bfloat16]]

    kernel_flags = [f"-DDIM_M={m_I}", f"-DDIM_K={k_tile}", f"-DDIM_H={H}", "-O3"]
    if activation == "silu":
        kernel_flags.append("-DACTIVATION_SILU")

    # Declare C++ kernels pointing to separate sources to prevent linker duplicate symbols
    init_kernel = ExternalFunction(
        "ffn_init",
        source_file=str(Path(__file__).parent / "ffn_init.cc"),
        arg_types=[],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    init_gate_up_kernel = ExternalFunction(
        "ffn_init_gate_up",
        source_file=str(Path(__file__).parent / "ffn_init_gate_up.cc"),
        arg_types=[],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    compute_gate_up_kernel = ExternalFunction(
        "ffn_compute_gate_up",
        source_file=str(Path(__file__).parent / "ffn_compute_gate_up.cc"),
        arg_types=[w_ty, w_ty, x_ty, np.int32],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    compute_activation_kernel = ExternalFunction(
        "ffn_compute_activation",
        source_file=str(Path(__file__).parent / "ffn_compute_activation.cc"),
        arg_types=[],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    accumulate_down_kernel = ExternalFunction(
        "ffn_accumulate_down",
        source_file=str(Path(__file__).parent / "ffn_accumulate_down.cc"),
        arg_types=[w_ty, np.int32],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    finalize_kernel = ExternalFunction(
        "ffn_finalize",
        source_file=str(Path(__file__).parent / "ffn_finalize.cc"),
        arg_types=[y_ty, np.int32],
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    # Input activation FIFO (depth 1, size H, broadcasted to all cores)
    of_x = ObjectFifo(x_ty, name="of_x")

    memW_fifos = []
    outY_fifos = []
    workers = []

    # AIE Core logic
    def core_fn(of_w, of_x, of_y, init_k, init_gu_k, comp_k, act_k, down_k, fin_k):
        # 1. Initialize static y_accum to zero
        init_k()

        # 2. Acquire activation vector x once and hold it
        elem_x = of_x.acquire(1)

        # Loop over intermediate dimension slice
        for _ in range_(num_blocks_I):
            # Reset gate/up accumulators
            init_gu_k()

            # Compute gate and up projections
            for h_blk in range_(H // k_tile):
                # Acquire 2 weight tiles contiguously (1 for gate, 1 for up)
                elem_w_both = of_w.acquire(2)
                elem_w_gate = elem_w_both[0]
                elem_w_up = elem_w_both[1]

                comp_k(elem_w_gate, elem_w_up, elem_x, h_blk * k_tile)

                of_w.release(2)

            # Compute activation
            act_k()

            # Multiply by W_down and accumulate into y_accum
            for h_blk_down in range_(H // m_H):
                # Acquire 1 tile for W_down
                elem_w_down = of_w.acquire(1)

                down_k(elem_w_down, h_blk_down * m_H)

                of_w.release(1)

        # Release x
        of_x.release(1)

        # 3. Finalize: write static y_accum to output FIFO
        for h_blk_out in range_(H // m_H):
            elem_y = of_y.acquire(1)
            fin_k(elem_y, h_blk_out * m_H)
            of_y.release(1)

    # Configure per-core ObjectFifos and Workers
    for i in range(n_cores):
        w_fifo = ObjectFifo(w_ty, depth=4, name=f"of_w_{i}")
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
                    init_kernel,
                    init_gate_up_kernel,
                    compute_gate_up_kernel,
                    compute_activation_kernel,
                    accumulate_down_kernel,
                    finalize_kernel,
                ],
                stack_size=0xF00,
            )
        )

    # Compute weight sequence size per core
    tiles_per_block = 3 * (H // k_tile)
    total_tiles_per_core = num_blocks_I * tiles_per_block
    size_per_core_bytes = total_tiles_per_core * tile_size
    
    # DRAM/Host buffers shapes
    W_fused_ty = np.ndarray[(n_cores, size_per_core_bytes), np.dtype[np.uint8]]
    X_ty = np.ndarray[(H,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(n_cores, H), np.dtype[bfloat16]]

    # Setup DRAM-to-L1 weight tiling sequence (linear stream of tiles per core)
    w_taps = TensorTiler2D.group_tiler(
        (n_cores, size_per_core_bytes),
        (1, tile_size),
        (1, total_tiles_per_core),
        prune_step=False
    )

    # Activation broadcast tiler
    x_tap = TensorTiler2D.group_tiler((1, H), (1, H), (1, 1), prune_step=False)[0]

    # Output drain tilers (one row of Y_partial per core)
    y_taps = TensorTiler2D.group_tiler(
        (n_cores, H),
        (1, m_H),
        (1, H // m_H),
        prune_step=False
    )

    rt = Runtime()
    with rt.sequence(W_fused_ty, X_ty, Y_ty) as (w_fused_in, x_in, y_out):
        rt.start(*workers)

        tg = rt.task_group()
        # Fill activation vector once (broadcasted to all worker cores)
        rt.fill(of_x.prod(), x_in, x_tap, task_group=tg)

        # Fill fused weight FIFO for each core
        for i in range(n_cores):
            rt.fill(memW_fifos[i].prod(), w_fused_in, w_taps[i], task_group=tg)

        # Drain output partial vectors from each core
        for i in range(n_cores):
            rt.drain(outY_fifos[i].cons(), y_out, y_taps[i], wait=True, task_group=tg)

        rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()

def pack_ffn_fused_weights(w_gate, w_up, w_down, H, I, m_I, k_tile):
    """
    Packs the weights of W_gate, W_up, and W_down into a single contiguous 
    fused weight matrix of shape (n_cores, size_per_core_bytes) matching the 
    traversal order of the fused FFN kernel.
    """
    m_H = k_tile
    
    if I % (8 * m_I) == 0:
        n_cores = 8
    elif I % (4 * m_I) == 0:
        n_cores = 4
    elif I % (2 * m_I) == 0:
        n_cores = 2
    else:
        n_cores = 1

    I_div_n_cores = I // n_cores
    num_blocks_I = I_div_n_cores // m_I
    chunks_per_gate_up = k_tile // 32

    core_buffers = []

    for c in range(n_cores):
        start_I = c * I_div_n_cores
        end_I = (c + 1) * I_div_n_cores
        
        w_gate_slice = w_gate[start_I:end_I]
        w_up_slice = w_up[start_I:end_I]
        
        # W_down slice is along columns (inputs to down projection)
        start_block = start_I // 32
        end_block = end_I // 32
        w_down_slice = w_down[:, start_block * 20 : end_block * 20]
        
        core_bytes = []
        
        for b_I in range(num_blocks_I):
            row_start = b_I * m_I
            row_end = (b_I + 1) * m_I
            
            # Interleave Gate and Up tiles to match of_w.acquire(2) in core loop
            for h_blk in range(H // k_tile):
                # 1. Gate tile
                col_start_bytes = h_blk * chunks_per_gate_up * 20
                col_end_bytes = (h_blk + 1) * chunks_per_gate_up * 20
                tile_gate = w_gate_slice[row_start:row_end, col_start_bytes:col_end_bytes]
                core_bytes.append(tile_gate.tobytes())
                
                # 2. Up tile
                tile_up = w_up_slice[row_start:row_end, col_start_bytes:col_end_bytes]
                core_bytes.append(tile_up.tobytes())
                
            # 3. Down tiles
            col_start_bytes = b_I * (m_I // 32) * 20
            col_end_bytes = (b_I + 1) * (m_I // 32) * 20
            for h_blk_down in range(H // m_H):
                row_start_down = h_blk_down * m_H
                row_end_down = (h_blk_down + 1) * m_H
                tile = w_down_slice[row_start_down:row_end_down, col_start_bytes:col_end_bytes]
                core_bytes.append(tile.tobytes())
                
        core_buf = np.frombuffer(b"".join(core_bytes), dtype=np.uint8)
        core_buffers.append(core_buf)
        
    return np.stack(core_buffers)

def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Fused FFN")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-H", "--hidden_size", type=int, default=1024)
    p.add_argument("-I", "--intermediate_size", type=int, default=2048)
    p.add_argument("-m_I", type=int, default=32)
    p.add_argument("-k", "--k_tile", type=int, default=256)
    p.add_argument("--act", type=str, default="gelu", choices=["gelu", "silu"])
    add_benchmark_args(p)
    return p

def _run_and_verify(opts):
    print(f"Verifying shape: H={opts.hidden_size}, I={opts.intermediate_size} with tiling: m_I={opts.m_I}, k={opts.k_tile}, act={opts.act}")
    
    rng = np.random.default_rng(1726250518)
    
    # Generate random FP32 weights and input vector
    W_gate_fp32 = rng.uniform(-1.0, 1.0, size=(opts.intermediate_size, opts.hidden_size)).astype(np.float32)
    W_up_fp32 = rng.uniform(-1.0, 1.0, size=(opts.intermediate_size, opts.hidden_size)).astype(np.float32)
    W_down_fp32 = rng.uniform(-1.0, 1.0, size=(opts.hidden_size, opts.intermediate_size)).astype(np.float32)
    x_np = rng.uniform(-1.0, 1.0, size=(opts.hidden_size,)).astype(np.float32)
    
    # Quantize and pack weights
    w_gate_q4, scales_gate = quantize_to_q4_0(W_gate_fp32)
    w_gate_combined = pack_to_combined(w_gate_q4, scales_gate)
    
    w_up_q4, scales_up = quantize_to_q4_0(W_up_fp32)
    w_up_combined = pack_to_combined(w_up_q4, scales_up)
    
    w_down_q4, scales_down = quantize_to_q4_0(W_down_fp32)
    w_down_combined = pack_to_combined(w_down_q4, scales_down)
    
    # Pack weights using FFN fused packing routine
    w_fused_combined = pack_ffn_fused_weights(
        w_gate_combined, w_up_combined, w_down_combined, 
        opts.hidden_size, opts.intermediate_size, opts.m_I, opts.k_tile
    )
    
    # Create NPU tensors
    w_fused_t = iron.tensor(w_fused_combined.reshape(-1), dtype=np.uint8, device="npu")
    x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
    
    # Select n_cores
    if opts.intermediate_size % (8 * opts.m_I) == 0:
        n_cores = 8
    elif opts.intermediate_size % (4 * opts.m_I) == 0:
        n_cores = 4
    elif opts.intermediate_size % (2 * opts.m_I) == 0:
        n_cores = 2
    else:
        n_cores = 1
        
    y_partial_t = iron.zeros(n_cores * opts.hidden_size, dtype=bfloat16, device="npu")
    
    bench = run_iters(
        ffn_fused_npu,
        w_fused_t,
        x_t,
        y_partial_t,
        H=opts.hidden_size,
        I=opts.intermediate_size,
        m_I=opts.m_I,
        k_tile=opts.k_tile,
        activation=opts.act,
        warmup=opts.warmup,
        iters=opts.iters,
    )
    
    # Compute CPU reference using fast activation approximation
    from tools.ref.gemv_q import dequantize_combined
    w_gate_deq = dequantize_combined(w_gate_combined)
    w_up_deq = dequantize_combined(w_up_combined)
    w_down_deq = dequantize_combined(w_down_combined)
    
    gate_expected = (w_gate_deq @ x_np).astype(bfloat16).astype(np.float32)
    up_expected = (w_up_deq @ x_np).astype(bfloat16).astype(np.float32)
    
    # Fast math approximations matching NPU
    def exp_approx(z):
        val = z * 1.442695040888963
        ix = np.floor(val).astype(np.int32)
        fx = val - ix
        pow2_ix = (2.0 ** ix).astype(np.float32)
        pow2_fx = 1.0 + 0.6931471805599453 * fx + 0.2401598148889220 * fx * fx
        return pow2_ix * pow2_fx

    def tanh_approx(y):
        y_clipped = np.clip(y, -9.0, 9.0)
        z = 2.0 * y_clipped
        exp_z = exp_approx(z)
        return (exp_z - 1.0) / (exp_z + 1.0)

    def gelu_approx(x):
        y = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + tanh_approx(y))

    def silu_approx(x):
        sig = 1.0 / (1.0 + exp_approx(-x))
        # Clip range similar to C++
        sig = np.where(x < -9.0, 0.0, sig)
        sig = np.where(x > 9.0, 1.0, sig)
        return x * sig
        
    if opts.act == "silu":
        act_val = silu_approx(gate_expected).astype(bfloat16).astype(np.float32)
    else:
        act_val = gelu_approx(gate_expected).astype(bfloat16).astype(np.float32)
        
    act_expected = (act_val * up_expected).astype(bfloat16).astype(np.float32)
    expected = (w_down_deq @ act_expected).astype(bfloat16)
    
    # Sum AIE core partial outputs
    y_partial_np = y_partial_t.numpy().reshape(n_cores, opts.hidden_size)
    actual = np.sum(y_partial_np, axis=0).astype(bfloat16)
    
    rtol = 0.08
    atol = 1500.0
    close = np.allclose(actual, expected, rtol=rtol, atol=atol)
    if not close:
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        max_diff = np.max(diff)
        print(f"Max absolute difference: {max_diff}")
        print(f"Actual (first 10):\n{actual[:10]}")
        print(f"Expected (first 10):\n{expected[:10]}")
        assert close, "NPU output does not match CPU reference!"
    else:
        print("✓ Verification PASSED!")

def main():
    opts = _make_argparser().parse_args()
    
    def compile_kwargs(opts):
        return {
            "H": opts.hidden_size,
            "I": opts.intermediate_size,
            "m_I": opts.m_I,
            "k_tile": opts.k_tile,
            "activation": opts.act,
        }
        
    run_design_cli(
        ffn_fused_npu,
        opts,
        compile_kwargs=compile_kwargs,
        run_and_verify=_run_and_verify,
    )

if __name__ == "__main__":
    main()
