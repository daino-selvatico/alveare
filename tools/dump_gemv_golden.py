#!/usr/bin/env python3
"""
Generate GEMV parity goldens for an arbitrary (N, K) shape.

Generalizes dump_p0.py: writes packed weights, the bf16 activation vector, and
the CPU dequant reference output for one shape, so the C++ NPU-registry smoke
test can run the *already-harvested* xclbin (kernels/build/manifest.json) and
compare against a known-good result. Unlike dump_p0.py this does NOT recompile a
kernel -- it only produces .npy goldens, so it needs no NPU/toolchain, just numpy.

    python tools/dump_gemv_golden.py --N 2048 --K 4096 --out kernels/build/golden

Emits (in --out):
    gemv_<N>x<K>_W.npy         uint8  (N, K/32*20)  packed Q4_0 weights
    gemv_<N>x<K>_x.npy         bf16   (K,)          activation vector
    gemv_<N>x<K>_expected.npy  bf16   (N,)          CPU dequant reference y = W@x

With --npu (requires the mlir-aie/IRON toolchain + NPU), also runs the shape on
the NPU via pyxrt/IRON and emits the true device output, which the C++ registry
must reproduce byte-for-byte (same xclbin, same instruction stream, same device):
    gemv_<N>x<K>_npu.npy       bf16   (N,)          Python/pyxrt NPU output
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from tools.ref.gemv_q import gemv_q_combined as ref_gemv_q
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined


def run_on_npu(w_combined, x_bf16, N, K, m, k_tile):
    """Run the same gemv on the NPU through IRON and return the bf16 output."""
    import aie.iron as iron
    from kernels.gemv_q.gemv_q import gemv_q_npu

    w_t = iron.tensor(w_combined.reshape(-1), dtype=np.uint8, device="npu")
    x_t = iron.tensor(x_bf16, dtype=bfloat16, device="npu")
    y_t = iron.zeros(N, dtype=bfloat16, device="npu")
    gemv_q_npu(w_t, x_t, y_t, N=N, K=K, m=m, k_tile=k_tile)
    # y_t.numpy() is a view over the device-mapped buffer; element-copy it into
    # freshly allocated host memory so np.save's tofile() works (a plain
    # ascontiguousarray is a no-op on the already-contiguous device view).
    y_host = np.empty(N, dtype=bfloat16)
    y_host[:] = y_t.numpy()
    return y_host


def main():
    ap = argparse.ArgumentParser(description="Dump GEMV parity goldens for one shape")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--k-tile", type=int, default=256)
    ap.add_argument("--npu", action="store_true",
                    help="also run on the NPU and emit gemv_<N>x<K>_npu.npy")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    W_fp32 = rng.uniform(-1.0, 1.0, size=(args.N, args.K)).astype(np.float32)
    x_fp32 = rng.uniform(-1.0, 1.0, size=(args.K,)).astype(np.float32)
    x_bf16 = x_fp32.astype(bfloat16)

    w_q4, scales = quantize_to_q4_0(W_fp32)
    w_combined = pack_to_combined(w_q4, scales)
    expected = ref_gemv_q(w_combined, x_bf16)

    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / f"gemv_{args.N}x{args.K}"
    np.save(f"{base}_W.npy", w_combined)
    np.save(f"{base}_x.npy", x_bf16)
    np.save(f"{base}_expected.npy", expected)
    print(f"wrote goldens for gemv {args.N}x{args.K} to {args.out} "
          f"(W {w_combined.shape} {w_combined.dtype})")

    if args.npu:
        y_npu = run_on_npu(w_combined, x_bf16, args.N, args.K, args.m, args.k_tile)
        np.save(f"{base}_npu.npy", y_npu)
        max_ref = float(np.max(np.abs(y_npu.astype(np.float32)
                                      - expected.astype(np.float32))))
        print(f"wrote NPU golden {base}_npu.npy "
              f"(max |npu - cpu_ref| = {max_ref})")


if __name__ == "__main__":
    main()
