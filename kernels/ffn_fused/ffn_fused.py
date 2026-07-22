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

    # Dynamically select n_cores based on intermediate size I and m_I. npu2 has
    # 32 compute tiles (8 cols x 4 rows); use 16 (2 rows/column) via a per-column
    # memtile funnel when I splits evenly, else the original one-core-per-column.
    if I % (16 * m_I) == 0:
        n_cores = 16
    elif I % (8 * m_I) == 0:
        n_cores = 8
    elif I % (4 * m_I) == 0:
        n_cores = 4
    elif I % (2 * m_I) == 0:
        n_cores = 2
    else:
        n_cores = 1

    I_div_n_cores = I // n_cores
    num_blocks_I = I_div_n_cores // m_I

    # gate/up/GELU are computed once for the whole intermediate slice and stored
    # in act_all (DIM_IC = I/n_cores bf16). The down projection then runs in
    # N_PASSES passes over the H output, so the fp32 output accumulator only needs
    # y_accum[H/N_PASSES] and — together with act_all — fits the core .bss (a full
    # fp32 y_accum[H] overflows). N_PASSES=4 keeps y_accum[H/4] fp32 within budget.
    N_PASSES = 4
    H_out = H // N_PASSES
    down_tiles_per_pass = H_out // m_H  # down output-row tiles handled per pass

    # Both gate/up tile and down tile are of size (m_I * (k_tile // 32) * 20) bytes.
    tile_size = m_I * (k_tile // 32) * 20
    w_ty = np.ndarray[(tile_size,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(H,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(m_H,), np.dtype[bfloat16]]

    kernel_flags = [f"-DDIM_M={m_I}", f"-DDIM_K={k_tile}", f"-DDIM_H={H}",
                    f"-DDIM_HOUT={H_out}", f"-DDIM_IC={I_div_n_cores}", "-O3"]
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
        arg_types=[np.int32],  # ic_offset into act_all
        compile_flags=kernel_flags,
        include_dirs=[cxx_header_path()],
    )

    accumulate_down_kernel = ExternalFunction(
        "ffn_accumulate_down",
        source_file=str(Path(__file__).parent / "ffn_accumulate_down.cc"),
        arg_types=[w_ty, np.int32, np.int32],  # w_down, h_offset, ic_offset
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

    # Weight-stream length per core: phase 1 streams gate + up once per I-block
    # (2*(H//k_tile)); phase 2 streams the down tiles (num_blocks_I * H//m_H).
    phase1_tiles = num_blocks_I * 2 * (H // k_tile)
    phase2_tiles = num_blocks_I * (H // m_H)
    total_tiles_per_core = phase1_tiles + phase2_tiles
    size_per_core_bytes = total_tiles_per_core * tile_size

    memW_fifos = []
    outY_fifos = []
    workers = []

    # AIE Core logic
    def core_fn(of_w, of_x, of_y, init_k, init_gu_k, comp_k, act_k, down_k, fin_k):
        # Acquire the input activation vector x once and hold it for phase 1.
        elem_x = of_x.acquire(1)

        # --- Phase 1: gate/up/GELU for the whole intermediate slice, computed
        # ONCE and stored in act_all (indexed by ic_offset = b_I * m_I). ---
        for b_I in range_(num_blocks_I):
            init_gu_k()
            for h_blk in range_(H // k_tile):
                # Acquire 2 weight tiles contiguously (1 for gate, 1 for up)
                elem_w_both = of_w.acquire(2)
                comp_k(elem_w_both[0], elem_w_both[1], elem_x, h_blk * k_tile)
                of_w.release(2)
            act_k(b_I * m_I)

        of_x.release(1)

        # --- Phase 2: down projection in N_PASSES passes over the H output,
        # reusing the stored act_all (no gate/up recompute). y_accum is indexed
        # relative to the pass's H-slice; the host streams the matching down rows. ---
        for _p in range_(N_PASSES):
            init_k()  # zero y_accum[H_out] for this pass
            for b_I in range_(num_blocks_I):
                for h_blk_down in range_(down_tiles_per_pass):
                    elem_w_down = of_w.acquire(1)
                    down_k(elem_w_down, h_blk_down * m_H, b_I * m_I)
                    of_w.release(1)
            for h_blk_out in range_(down_tiles_per_pass):
                elem_y = of_y.acquire(1)
                fin_k(elem_y, h_blk_out * m_H)
                of_y.release(1)

    # --- 16-core memtile dataflow (npu2: 2 rows x 8 cols) -----------------------
    # I is split across 16 cores; each produces a full-H partial that the host
    # sums (order-independent). The weights are packed INTERLEAVED per column:
    # for a column's two cores the tiles alternate in DRAM
    # (col block = [c0.t0, c1.t0, c0.t1, c1.t1, ...]), so the per-column weight
    # fill is a single CONTIGUOUS stream (one trivial BD) instead of the strided
    # 2-row read that exhausted the memtile DMA descriptors. The column memtile
    # splits each (2*tile) object to its 2 cores; x is broadcast; y partials are
    # joined back. See pack_ffn_fused_weights (n_cores==16 branch) for the layout.
    if n_cores == 16:
        n_aie_rows, n_aie_cols = 2, 8
        col_bytes = n_aie_rows * size_per_core_bytes
        w_l2_ty = np.ndarray[(n_aie_rows * tile_size,), np.dtype[np.uint8]]
        y_l2_ty = np.ndarray[(n_aie_rows * m_H,), np.dtype[bfloat16]]

        W_fused_ty = np.ndarray[(n_aie_cols, col_bytes), np.dtype[np.uint8]]
        X_ty = np.ndarray[(H,), np.dtype[bfloat16]]
        Y_ty = np.ndarray[(n_cores, H), np.dtype[bfloat16]]

        W_l2l1 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        Y_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        X_l2l1 = [None] * n_aie_cols
        W_l3l2 = [None] * n_aie_cols
        X_l3l2 = [None] * n_aie_cols
        Y_l2l3 = [None] * n_aie_cols

        for col in range(n_aie_cols):
            W_l3l2[col] = ObjectFifo(w_l2_ty, name=f"W_L3L2_{col}", depth=2)
            wsub = W_l3l2[col].cons().split(
                [tile_size * r for r in range(n_aie_rows)],
                obj_types=[w_ty] * n_aie_rows,
                names=[f"W_L2L1_{col}_{r}" for r in range(n_aie_rows)],
                depths=[4] * n_aie_rows,
            )
            for r in range(n_aie_rows):
                W_l2l1[r][col] = wsub[r]

            X_l3l2[col] = ObjectFifo(x_ty, name=f"X_L3L2_{col}", depth=2)
            X_l2l1[col] = X_l3l2[col].cons().forward(obj_type=x_ty, name=f"X_L2L1_{col}")

            Y_l2l3[col] = ObjectFifo(y_l2_ty, name=f"Y_L2L3_{col}", depth=2)
            ysub = Y_l2l3[col].prod().join(
                [m_H * r for r in range(n_aie_rows)],
                obj_types=[y_ty] * n_aie_rows,
                names=[f"Y_L1L2_{col}_{r}" for r in range(n_aie_rows)],
                depths=[2] * n_aie_rows,
            )
            for r in range(n_aie_rows):
                Y_l1l2[r][col] = ysub[r]

        workers = Worker.grid(
            n_aie_rows,
            n_aie_cols,
            lambda row, col: Worker(
                core_fn,
                [
                    W_l2l1[row][col].cons(),
                    X_l2l1[col].cons(),
                    Y_l1l2[row][col].prod(),
                    init_kernel,
                    init_gate_up_kernel,
                    compute_gate_up_kernel,
                    compute_activation_kernel,
                    accumulate_down_kernel,
                    finalize_kernel,
                ],
                stack_size=0xF00,
            ),
        )

        # Per-column weight fill: contiguous stream of (2*tile) objects.
        w_col_taps = TensorTiler2D.group_tiler(
            (n_aie_cols, col_bytes), (1, n_aie_rows * tile_size),
            (1, total_tiles_per_core), prune_step=False,
        )
        x_tap = TensorTiler2D.group_tiler((1, H), (1, H), (1, 1), prune_step=False)[0]
        # Per-column output drain: (2, m_H) blocks into adjacent rows [2col, 2col+1].
        y_col_taps = TensorTiler2D.group_tiler(
            (n_cores, H), (n_aie_rows, m_H), (1, H // m_H), prune_step=False,
        )

        rt = Runtime()
        with rt.sequence(W_fused_ty, X_ty, Y_ty) as (w_fused_in, x_in, y_out):
            rt.start(*[w for row in workers for w in row])
            tg = rt.task_group()
            for col in range(n_aie_cols):
                rt.fill(X_l3l2[col].prod(), x_in, x_tap, task_group=tg)
                rt.fill(W_l3l2[col].prod(), w_fused_in, w_col_taps[col], task_group=tg)
                rt.drain(Y_l2l3[col].cons(), y_out, y_col_taps[col], wait=True, task_group=tg)
            rt.finish_task_group(tg)
        return Program(iron.get_current_device(), rt).resolve_program()

    # --- Fallback: original one-core-per-column dataflow ------------------------
    of_x = ObjectFifo(x_ty, name="of_x")
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

    W_fused_ty = np.ndarray[(n_cores, size_per_core_bytes), np.dtype[np.uint8]]
    X_ty = np.ndarray[(H,), np.dtype[bfloat16]]
    Y_ty = np.ndarray[(n_cores, H), np.dtype[bfloat16]]

    w_taps = TensorTiler2D.group_tiler(
        (n_cores, size_per_core_bytes),
        (1, tile_size),
        (1, total_tiles_per_core),
        prune_step=False
    )
    x_tap = TensorTiler2D.group_tiler((1, H), (1, H), (1, 1), prune_step=False)[0]
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
        rt.fill(of_x.prod(), x_in, x_tap, task_group=tg)
        for i in range(n_cores):
            rt.fill(memW_fifos[i].prod(), w_fused_in, w_taps[i], task_group=tg)
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

        n_passes = 4
        down_tiles_per_pass = (H // m_H) // n_passes

        # Phase 1: gate + up tiles for every I-block (streamed once).
        for b_I in range(num_blocks_I):
            row_start = b_I * m_I
            row_end = (b_I + 1) * m_I
            for h_blk in range(H // k_tile):
                col_start_bytes = h_blk * chunks_per_gate_up * 20
                col_end_bytes = (h_blk + 1) * chunks_per_gate_up * 20
                # Interleave Gate and Up to match of_w.acquire(2) in the core loop
                core_bytes.append(w_gate_slice[row_start:row_end, col_start_bytes:col_end_bytes].tobytes())
                core_bytes.append(w_up_slice[row_start:row_end, col_start_bytes:col_end_bytes].tobytes())

        # Phase 2: down tiles, per H-output pass, per I-block (matches core loop).
        for p in range(n_passes):
            for b_I in range(num_blocks_I):
                col_start_bytes = b_I * (m_I // 32) * 20
                col_end_bytes = (b_I + 1) * (m_I // 32) * 20
                for h_blk_down in range(p * down_tiles_per_pass, (p + 1) * down_tiles_per_pass):
                    row_start_down = h_blk_down * m_H
                    row_end_down = (h_blk_down + 1) * m_H
                    tile = w_down_slice[row_start_down:row_end_down, col_start_bytes:col_end_bytes]
                    core_bytes.append(tile.tobytes())

        core_buf = np.frombuffer(b"".join(core_bytes), dtype=np.uint8)
        core_buffers.append(core_buf)
        
    return np.stack(core_buffers)

def _resolve_full_device(opts):
    """Resolve ``--dev`` to the max-column variant for its family.

    ``aie.iron.device.from_name()`` defaults to ``n_cols=1`` (the
    single-column variant) when called with just a family name — which is
    exactly what ``run_design_cli``'s internal dispatch does when no
    ``device=`` override is supplied. That silently caps this design to a
    single column's worth of CoreTiles even when the attached NPU exposes
    the full 8-column part, and placement then fails with "no available
    compute tiles for placement" for this 8-core design. Force the
    unrestricted device explicitly so placement always sees every physical
    tile.
    """
    from aie.iron.device import from_name

    return from_name(opts.dev, n_cols=None)

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
        device=_resolve_full_device,
    )

if __name__ == "__main__":
    main()
