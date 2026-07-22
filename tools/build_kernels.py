#!/usr/bin/env python3
"""
Phase P1 — AOT kernel harvester for the C++ runtime.

Enumerates every distinct matmul shape a model actually invokes, compiles each
once via the existing IRON designs (no MLIR duplication), and emits a manifest
the C++ runtime loads at startup.

Discovery: dump_p0.py proved the AOT API —
    gemv_q_npu.specialize(N=, K=, m=, k_tile=).compile(xclbin_path=, inst_path=)
We reuse it here for every shape, and the analogous call for gemm_q_npu.

Shapes are read from the *actual* packed weight files on disk (not hardcoded),
so this stays correct across models. A packed weight of logical shape (N, K) is
stored on disk as (N, K/32 * 20) uint8 (Q4_0, 20 bytes/block of 32).

Run on a machine with the mlir-aie / IRON toolchain and the NPU present:
    python tools/build_kernels.py --weights-dir quantized_weights --out kernels/build

Output:
    kernels/build/<name>_<N>x<K>[_bB].xclbin
    kernels/build/<name>_<N>x<K>[_bB].insts
    kernels/build/manifest.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from kernels.gemv_q.gemv_q import gemv_q_npu
from kernels.gemm_q.gemm_q import gemm_q_npu

from kernels.ffn_fused.ffn_fused import ffn_fused_npu

M = 32          # kernel row tile (fixed in the IRON designs)
K_TILE = 256    # kernel K tile (fixed in the IRON designs)

# The standard GEMV/GEMM projections. FFN is handled separately as a fused block.
PROJECTIONS = ["attn_q", "attn_k", "attn_v", "attn_output"]


def n_cores_for(N: int) -> int:
    """Mirror the core-count heuristic in gemv_q.py / gemm_q.py."""
    if N % (8 * M) == 0:
        return 8
    if N % (4 * M) == 0:
        return 4
    if N % (2 * M) == 0:
        return 2
    return 1


def packed_shape_to_logical(path: Path) -> tuple[int, int]:
    """(N, K/32*20) uint8 on disk -> logical (N, K)."""
    # Read only the .npy header; never load the payload. Use the versioned
    # public readers (the private _read_array_header was removed in numpy 2.x,
    # which the alveare-aie toolchain env ships).
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"{path}: unsupported .npy version {version}")
    N, packed = shape
    assert packed % 20 == 0, f"{path}: packed dim {packed} not a multiple of 20"
    K = (packed // 20) * 32
    return int(N), int(K)


def enumerate_shapes(weights_dir: Path, num_layers: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    gemv_shapes: set[tuple[int, int]] = set()
    ffn_shapes: set[tuple[int, int]] = set()
    
    for l in range(num_layers):
        for proj in PROJECTIONS:
            p = weights_dir / f"blk.{l}.{proj}.weight_packed.npy"
            if p.exists():
                gemv_shapes.add(packed_shape_to_logical(p))

        # gemma4 fuses Q/K/V into one gemv at runtime, so the concatenated
        # output shape (N_q + N_k + N_v, K) also needs a kernel. gemma marker:
        # per-head q-norm. Sliding layers ((l+1)%6 != 0) use q++k++v; global
        # layers reuse k for v, so q++k.
        pq = weights_dir / f"blk.{l}.attn_q.weight_packed.npy"
        pk = weights_dir / f"blk.{l}.attn_k.weight_packed.npy"
        pv = weights_dir / f"blk.{l}.attn_v.weight_packed.npy"
        p_qnorm = weights_dir / f"blk.{l}.attn_q_norm.weight.npy"
        if p_qnorm.exists() and pq.exists() and pk.exists():
            nq, K = packed_shape_to_logical(pq)
            nk, _ = packed_shape_to_logical(pk)
            is_sliding = (l + 1) % 6 != 0
            if is_sliding and pv.exists():
                nv, _ = packed_shape_to_logical(pv)
                gemv_shapes.add((nq + nk + nv, K))
            else:
                gemv_shapes.add((nq + nk, K))

        # Handle FFN shapes for fusion. We read the gate projection to get (I, H).
        p_gate = weights_dir / f"blk.{l}.ffn_gate.weight_packed.npy"
        if p_gate.exists():
            I, H = packed_shape_to_logical(p_gate)
            ffn_shapes.add((H, I))
            
    lm = weights_dir / "lm_head_packed.npy"
    if lm.exists():
        # LM head is chunked to MAX_N=16384 along N in run_gemv_npu
        _, K = packed_shape_to_logical(lm)
        gemv_shapes.add((16384, K))
        
    return gemv_shapes, ffn_shapes


def compile_gemv(N: int, K: int, out: Path) -> dict:
    name = f"gemv_{N}x{K}"
    xclbin = out / f"{name}.xclbin"
    insts = out / f"{name}.insts"
    gemv_q_npu.specialize(N=N, K=K, m=M, k_tile=K_TILE).compile(
        xclbin_path=str(xclbin), inst_path=str(insts))
    return {"kind": "gemv", "N": N, "K": K, "m": M, "k_tile": K_TILE,
            "n_cores": n_cores_for(N),
            "xclbin": xclbin.name, "insts": insts.name}


def compile_gemm(B: int, N: int, K: int, out: Path) -> dict:
    name = f"gemm_{N}x{K}_b{B}"
    xclbin = out / f"{name}.xclbin"
    insts = out / f"{name}.insts"
    gemm_q_npu.specialize(B=B, N=N, K=K, m=M, k_tile=K_TILE).compile(
        xclbin_path=str(xclbin), inst_path=str(insts))
    return {"kind": "gemm", "B": B, "N": N, "K": K, "m": M, "k_tile": K_TILE,
            "n_cores": n_cores_for(N),
            "xclbin": xclbin.name, "insts": insts.name}


def compile_ffn_fused(H: int, I: int, activation: str, out: Path) -> dict:
    name = f"ffn_fused_{H}x{I}_{activation}"
    xclbin = out / f"{name}.xclbin"
    insts = out / f"{name}.insts"
    ffn_fused_npu.specialize(H=H, I=I, m_I=M, k_tile=K_TILE, activation=activation).compile(
        xclbin_path=str(xclbin), inst_path=str(insts))
    # n_cores logic matches ffn_fused.py
    if I % (32 * M) == 0:
        n_cores = 32
    elif I % (16 * M) == 0:
        n_cores = 16
    elif I % (8 * M) == 0:
        n_cores = 8
    elif I % (4 * M) == 0:
        n_cores = 4
    elif I % (2 * M) == 0:
        n_cores = 2
    else:
        n_cores = 1
        
    return {"kind": "ffn_fused", "H": H, "I": I, "m_I": M, "k_tile": K_TILE, "activation": activation,
            "n_cores": n_cores,
            "xclbin": xclbin.name, "insts": insts.name}


def main():
    ap = argparse.ArgumentParser(description="AOT-harvest NPU kernels for the C++ runtime")
    ap.add_argument("--weights-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "kernels" / "build")
    ap.add_argument("--max-batch", type=int, default=16, help="prefill GEMM batch B")
    ap.add_argument("--no-gemm", action="store_true", help="skip prefill GEMM shapes")
    args = ap.parse_args()

    # Initialize the IRON NPU device context (required before compilation)
    import aie.iron as iron
    _ = iron.tensor([0], device="npu")

    cfg = json.loads((args.weights_dir / "config.json").read_text())
    num_layers = cfg.get("num_hidden_layers", 48)
    # Default to gelu, though llama uses silu (hidden_act = "silu")
    activation = cfg.get("hidden_act", "gelu") 
    
    args.out.mkdir(parents=True, exist_ok=True)

    gemv_shapes, ffn_shapes = enumerate_shapes(args.weights_dir, num_layers)
    gemv_shapes = sorted(gemv_shapes)
    ffn_shapes = sorted(ffn_shapes)
    
    total_shapes = len(gemv_shapes) + len(ffn_shapes)
    print(f"Model '{cfg.get('model_type')}' — {total_shapes} distinct matmul shapes:")
    for N, K in gemv_shapes:
        print(f"  GEMV N={N:6d} K={K:6d}  n_cores={n_cores_for(N)}")
    for H, I in ffn_shapes:
        print(f"  FFN  H={H:6d} I={I:6d}  activation={activation}")

    # Decision #2 sanity check: decode must fit the ~8-context budget resident.
    if total_shapes > 8:
        print(f"\n[!] {total_shapes} > 8 hardware contexts: the C++ registry must "
              f"bucket-pad these into <=8 resident contexts (plan decision #2) to "
              f"avoid xclbin reloads inside the decode loop.")

    entries = []
    for N, K in gemv_shapes:
        print(f"Compiling gemv {N}x{K} ...")
        entries.append(compile_gemv(N, K, args.out))
        if not args.no_gemm:
            print(f"Compiling gemm {N}x{K} b{args.max_batch} ...")
            entries.append(compile_gemm(args.max_batch, N, K, args.out))
            
    for H, I in ffn_shapes:
        print(f"Compiling ffn_fused {H}x{I} ({activation}) ...")
        entries.append(compile_ffn_fused(H, I, activation, args.out))

    manifest = {
        "model_type": cfg.get("model_type"),
        "num_hidden_layers": num_layers,
        "m": M, "k_tile": K_TILE, "max_batch": args.max_batch,
        "kernel_name": "MLIR_AIE",   # xrt::kernel entry (see meta.json ABI)
        "opcode": 3,
        "kernels": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {args.out / 'manifest.json'} with {len(entries)} kernels.")


if __name__ == "__main__":
    main()
