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

M = 32          # kernel row tile (fixed in the IRON designs)
K_TILE = 256    # kernel K tile (fixed in the IRON designs)

# The 7 projections streamed per layer, plus the tied LM head.
PROJECTIONS = ["attn_q", "attn_k", "attn_v", "attn_output",
               "ffn_gate", "ffn_up", "ffn_down"]


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
    # Read only the .npy header; never load the payload.
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, _fortran, _dtype = np.lib.format._read_array_header(f, version)
    N, packed = shape
    assert packed % 20 == 0, f"{path}: packed dim {packed} not a multiple of 20"
    K = (packed // 20) * 32
    return int(N), int(K)


def enumerate_shapes(weights_dir: Path, num_layers: int) -> set[tuple[int, int]]:
    shapes: set[tuple[int, int]] = set()
    for l in range(num_layers):
        for proj in PROJECTIONS:
            p = weights_dir / f"blk.{l}.{proj}.weight_packed.npy"
            if p.exists():
                shapes.add(packed_shape_to_logical(p))
    lm = weights_dir / "lm_head_packed.npy"
    if lm.exists():
        # LM head is chunked to MAX_N=16384 along N in run_gemv_npu; harvest the
        # chunk shape actually launched, not the full (vocab, K). Padding/bucketing
        # policy (plan decision #2) is applied by the C++ registry, not here.
        _, K = packed_shape_to_logical(lm)
        shapes.add((16384, K))
    return shapes


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


def main():
    ap = argparse.ArgumentParser(description="AOT-harvest NPU kernels for the C++ runtime")
    ap.add_argument("--weights-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "kernels" / "build")
    ap.add_argument("--max-batch", type=int, default=16, help="prefill GEMM batch B")
    ap.add_argument("--no-gemm", action="store_true", help="skip prefill GEMM shapes")
    args = ap.parse_args()

    cfg = json.loads((args.weights_dir / "config.json").read_text())
    num_layers = cfg.get("num_hidden_layers", 48)
    args.out.mkdir(parents=True, exist_ok=True)

    shapes = sorted(enumerate_shapes(args.weights_dir, num_layers))
    print(f"Model '{cfg.get('model_type')}' — {len(shapes)} distinct matmul shapes:")
    for N, K in shapes:
        print(f"  N={N:6d} K={K:6d}  n_cores={n_cores_for(N)}")

    # Decision #2 sanity check: decode must fit the ~8-context budget resident.
    if len(shapes) > 8:
        print(f"\n[!] {len(shapes)} > 8 hardware contexts: the C++ registry must "
              f"bucket-pad these into <=8 resident contexts (plan decision #2) to "
              f"avoid xclbin reloads inside the decode loop.")

    entries = []
    for N, K in shapes:
        print(f"Compiling gemv {N}x{K} ...")
        entries.append(compile_gemv(N, K, args.out))
        if not args.no_gemm:
            print(f"Compiling gemm {N}x{K} b{args.max_batch} ...")
            entries.append(compile_gemm(args.max_batch, N, K, args.out))

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
