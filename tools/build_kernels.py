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
    kernels/build/manifest.json (merges all compiled & existing kernels)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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


def ffn_n_cores_for(I: int) -> int:
    """Mirror the core-count heuristic in ffn_fused.py."""
    if I % (32 * M) == 0:
        return 32
    elif I % (16 * M) == 0:
        return 16
    elif I % (8 * M) == 0:
        return 8
    elif I % (4 * M) == 0:
        return 4
    elif I % (2 * M) == 0:
        return 2
    else:
        return 1


def kernel_key(e: dict) -> tuple:
    kind = e.get("kind")
    if kind == "gemv":
        return ("gemv", e.get("N"), e.get("K"))
    elif kind == "gemm":
        return ("gemm", e.get("B"), e.get("N"), e.get("K"))
    elif kind == "ffn_fused":
        return ("ffn_fused", e.get("H"), e.get("I"), e.get("activation"))
    return (kind, e.get("xclbin"))


def parse_xclbin_filename(xclbin_path: Path) -> dict | None:
    fname = xclbin_path.name
    insts_path = xclbin_path.with_suffix(".insts")
    if not insts_path.exists():
        return None

    m = re.match(r"^gemv_(\d+)x(\d+)\.xclbin$", fname)
    if m:
        N, K = int(m.group(1)), int(m.group(2))
        return {
            "kind": "gemv", "N": N, "K": K, "m": M, "k_tile": K_TILE,
            "n_cores": n_cores_for(N),
            "xclbin": fname, "insts": insts_path.name
        }
    m = re.match(r"^gemm_(\d+)x(\d+)_b(\d+)\.xclbin$", fname)
    if m:
        N, K, B = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return {
            "kind": "gemm", "B": B, "N": N, "K": K, "m": M, "k_tile": K_TILE,
            "n_cores": n_cores_for(N),
            "xclbin": fname, "insts": insts_path.name
        }
    m = re.match(r"^ffn_fused_(\d+)x(\d+)_([a-zA-Z0-9_]+)\.xclbin$", fname)
    if m:
        H, I, act = int(m.group(1)), int(m.group(2)), m.group(3)
        return {
            "kind": "ffn_fused", "H": H, "I": I, "m_I": M, "k_tile": K_TILE,
            "activation": act, "n_cores": ffn_n_cores_for(I),
            "xclbin": fname, "insts": insts_path.name
        }
    return None


def packed_shape_to_logical(path: Path) -> tuple[int, int]:
    """(N, K/32*20) uint8 on disk -> logical (N, K)."""
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


def enumerate_shapes(weights_dir: Path, num_layers: int, model_type: str = "") -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    gemv_shapes: set[tuple[int, int]] = set()
    ffn_shapes: set[tuple[int, int]] = set()
    
    for l in range(num_layers):
        for proj in PROJECTIONS:
            p = weights_dir / f"blk.{l}.{proj}.weight_packed.npy"
            if p.exists():
                gemv_shapes.add(packed_shape_to_logical(p))

        # gemma4 fuses Q/K/V into one gemv at runtime, so the concatenated
        # output shape (N_q + N_k + N_v, K) also needs a kernel.
        pq = weights_dir / f"blk.{l}.attn_q.weight_packed.npy"
        pk = weights_dir / f"blk.{l}.attn_k.weight_packed.npy"
        pv = weights_dir / f"blk.{l}.attn_v.weight_packed.npy"
        is_g4 = ("gemma4" in model_type or "gemma-4" in model_type or model_type == "e4b")
        if is_g4 and pq.exists():
            nq, K = packed_shape_to_logical(pq)
            if pk.exists():
                nk, _ = packed_shape_to_logical(pk)
                is_sliding = (l + 1) % 6 != 0
                if is_sliding and pv.exists():
                    nv, _ = packed_shape_to_logical(pv)
                    gemv_shapes.add((nq + nk + nv, K))
                else:
                    gemv_shapes.add((nq + nk, K))
            else:
                # Shared-KV layers (layers 24-41 in 12B) only have Q projection
                gemv_shapes.add((nq, K))

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


def compile_gemv(N: int, K: int, out: Path, force: bool = False) -> dict:
    name = f"gemv_{N}x{K}"
    xclbin = out / f"{name}.xclbin"
    insts = out / f"{name}.insts"
    entry = {"kind": "gemv", "N": N, "K": K, "m": M, "k_tile": K_TILE,
             "n_cores": n_cores_for(N),
             "xclbin": xclbin.name, "insts": insts.name}
    if not force and xclbin.exists() and insts.exists():
        print(f"Skipping compile of {name} ({xclbin.name} exists)")
        return entry

    gemv_q_npu.specialize(N=N, K=K, m=M, k_tile=K_TILE).compile(
        xclbin_path=str(xclbin), inst_path=str(insts))
    return entry


def compile_gemm(B: int, N: int, K: int, out: Path, force: bool = False) -> dict:
    name = f"gemm_{N}x{K}_b{B}"
    xclbin = out / f"{name}.xclbin"
    insts = out / f"{name}.insts"
    entry = {"kind": "gemm", "B": B, "N": N, "K": K, "m": M, "k_tile": K_TILE,
             "n_cores": n_cores_for(N),
             "xclbin": xclbin.name, "insts": insts.name}
    if not force and xclbin.exists() and insts.exists():
        print(f"Skipping compile of {name} ({xclbin.name} exists)")
        return entry

    with tempfile.TemporaryDirectory(prefix=f"gc_{N}x{K}_b{B}_") as tmpdir:
        env = os.environ.copy()
        env["NPU_CACHE_HOME"] = tmpdir
        cmd = [
            sys.executable,
            str(ROOT / "kernels" / "gemm_q" / "gemm_q.py"),
            "-N", str(N),
            "-K", str(K),
            "-B", str(B),
            "-m", "32",
            "-k", "256",
            "--xclbin-path", str(xclbin),
            "--insts-path", str(insts),
        ]
        res = subprocess.run(
            cmd,
            env=env,
            timeout=300,
            capture_output=True,
            text=True,
        )

        if res.returncode != 0 or not xclbin.exists() or not insts.exists():
            raise RuntimeError(
                f"Failed to compile GEMM for shape N={N}, K={K}, B={B}.\n"
                f"Exit code: {res.returncode}\n"
                f"Stdout:\n{res.stdout}\n"
                f"Stderr:\n{res.stderr}"
            )

    return entry


def compile_ffn_fused(H: int, I: int, activation: str, out: Path, force: bool = False) -> dict:
    name = f"ffn_fused_{H}x{I}_{activation}"
    xclbin = out / f"{name}.xclbin"
    insts = out / f"{name}.insts"
    n_cores = ffn_n_cores_for(I)
    entry = {"kind": "ffn_fused", "H": H, "I": I, "m_I": M, "k_tile": K_TILE, "activation": activation,
             "n_cores": n_cores,
             "xclbin": xclbin.name, "insts": insts.name}
    if not force and xclbin.exists() and insts.exists():
        print(f"Skipping compile of {name} ({xclbin.name} exists)")
        return entry

    # IMPORTANT: the direct xclbin_path/inst_path compile BYPASSES the jit on-disk
    # cache and, for the fused-FFN design, produces a BROKEN kernel on some shapes —
    # e.g. H=2048 came out ~65KB smaller and yielded NaN logits at runtime, while the
    # cache path (same source/flags/device) self-verifies and runs correctly. So warm
    # the cache with a no-arg compile() and copy the correct artifacts into out/.
    cached_xclbin, cached_insts = ffn_fused_npu.specialize(
        H=H, I=I, m_I=M, k_tile=K_TILE, activation=activation).compile()
    shutil.copy(cached_xclbin, xclbin)
    shutil.copy(cached_insts, insts)
    return entry


def main():
    ap = argparse.ArgumentParser(description="AOT-harvest NPU kernels for the C++ runtime")
    ap.add_argument("--weights-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "kernels" / "build")
    ap.add_argument("--max-batch", type=int, default=16, help="prefill GEMM batch B")
    ap.add_argument("--no-gemm", action="store_true", help="skip prefill GEMM shapes")
    ap.add_argument("--force", action="store_true", help="recompile xclbin even if present")
    args = ap.parse_args()

    # Initialize the IRON NPU device context for npu2
    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name("npu2", n_cols=None))

    cfg = json.loads((args.weights_dir / "config.json").read_text())
    num_layers = cfg.get("num_hidden_layers", 48)
    activation = cfg.get("hidden_act", "gelu") 
    
    args.out.mkdir(parents=True, exist_ok=True)

    model_type = cfg.get("model_type", "")
    gemv_shapes, ffn_shapes = enumerate_shapes(args.weights_dir, num_layers, model_type)
    gemv_shapes = sorted(gemv_shapes)
    ffn_shapes = sorted(ffn_shapes)
    
    total_shapes = len(gemv_shapes) + len(ffn_shapes)
    print(f"Model '{cfg.get('model_type')}' — {total_shapes} distinct matmul shapes:")
    for N, K in gemv_shapes:
        print(f"  GEMV N={N:6d} K={K:6d}  n_cores={n_cores_for(N)}")
    for H, I in ffn_shapes:
        print(f"  FFN  H={H:6d} I={I:6d}  activation={activation}")

    # 1. Collect existing manifest entries and xclbins on disk to form a superset
    kernel_map = {}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        try:
            m_old = json.loads(manifest_path.read_text())
            for e in m_old.get("kernels", []):
                xcl = args.out / e.get("xclbin", "")
                inst = args.out / e.get("insts", "")
                if xcl.exists() and inst.exists():
                    kernel_map[kernel_key(e)] = e
        except Exception as err:
            print(f"Warning: could not parse existing manifest.json: {err}")

    # 2. Scan all .xclbin files in args.out to recover any unlisted kernels
    for xclbin_file in args.out.glob("*.xclbin"):
        parsed = parse_xclbin_filename(xclbin_file)
        if parsed:
            k_key = kernel_key(parsed)
            if k_key not in kernel_map:
                kernel_map[k_key] = parsed

    # 3. Compile or reuse shapes for the current model
    for N, K in gemv_shapes:
        print(f"Checking/compiling gemv {N}x{K} ...")
        entry = compile_gemv(N, K, args.out, force=args.force)
        kernel_map[kernel_key(entry)] = entry
        if not args.no_gemm:
            print(f"Checking/compiling gemm {N}x{K} b{args.max_batch} ...")
            entry_gemm = compile_gemm(args.max_batch, N, K, args.out, force=args.force)
            kernel_map[kernel_key(entry_gemm)] = entry_gemm
            
    for H, I in ffn_shapes:
        print(f"Checking/compiling ffn_fused {H}x{I} ({activation}) ...")
        entry_ffn = compile_ffn_fused(H, I, activation, args.out, force=args.force)
        kernel_map[kernel_key(entry_ffn)] = entry_ffn

    sorted_entries = sorted(kernel_map.values(), key=lambda e: (
        e.get("kind", ""), e.get("N", e.get("H", 0)), e.get("K", e.get("I", 0)), e.get("B", 0)
    ))

    manifest = {
        "model_type": cfg.get("model_type"),
        "num_hidden_layers": num_layers,
        "m": M, "k_tile": K_TILE, "max_batch": args.max_batch,
        "kernel_name": "MLIR_AIE",   # xrt::kernel entry (see meta.json ABI)
        "opcode": 3,
        "kernels": sorted_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path} with {len(sorted_entries)} total kernels (superset).")


if __name__ == "__main__":
    main()
