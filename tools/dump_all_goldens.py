#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from tools.ref.gemv_q import gemv_q_combined as ref_gemv_q
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined

def run_npu_gemv(w_combined, x_bf16, N, K, m, k_tile):
    import aie.iron as iron
    from kernels.gemv_q.gemv_q import gemv_q_npu
    w_t = iron.tensor(w_combined.reshape(-1), dtype=np.uint8, device="npu")
    x_t = iron.tensor(x_bf16, dtype=bfloat16, device="npu")
    y_t = iron.zeros(N, dtype=bfloat16, device="npu")
    gemv_q_npu(w_t, x_t, y_t, N=N, K=K, m=m, k_tile=k_tile)
    y_host = np.empty(N, dtype=bfloat16)
    y_host[:] = y_t.numpy()
    return y_host

def run_npu_ffn_fused(w_combined, x_bf16, H, I, m_I, k_tile, activation):
    import aie.iron as iron
    from kernels.ffn_fused.ffn_fused import ffn_fused_npu
    if I % (8 * m_I) == 0: n_cores = 8
    elif I % (4 * m_I) == 0: n_cores = 4
    elif I % (2 * m_I) == 0: n_cores = 2
    else: n_cores = 1

    w_t = iron.tensor(w_combined.reshape(-1), dtype=np.uint8, device="npu")
    x_t = iron.tensor(x_bf16, dtype=bfloat16, device="npu")
    y_partial_t = iron.zeros(n_cores * H, dtype=bfloat16, device="npu")
    ffn_fused_npu(w_t, x_t, y_partial_t, H=H, I=I, m_I=m_I, k_tile=k_tile, activation=activation)
    y_partial_np = y_partial_t.numpy().reshape(n_cores, H)
    actual = np.sum(y_partial_np, axis=0).astype(bfloat16)

    y_host = np.empty(H, dtype=bfloat16)
    y_host[:] = actual
    return y_host

def main():
    manifest_path = ROOT / "kernels" / "build" / "manifest.json"
    out_dir = ROOT / "kernels" / "build" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    rng = np.random.default_rng(42)
    
    for spec in manifest["kernels"]:
        kind = spec["kind"]
        if kind == "gemm":
            continue # Skip prefill for now, focus on decode/FFN
            
        if kind == "gemv":
            N, K = spec["N"], spec["K"]
            m, k_tile = spec["m"], spec["k_tile"]
            base = out_dir / f"gemv_{N}x{K}"
            
            W_fp32 = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)
            x_fp32 = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)
            x_bf16 = x_fp32.astype(bfloat16)
            
            w_q4, scales = quantize_to_q4_0(W_fp32)
            w_combined = pack_to_combined(w_q4, scales)
            expected = ref_gemv_q(w_combined, x_bf16)
            
            y_npu = run_npu_gemv(w_combined, x_bf16, N, K, m, k_tile)
            
            np.save(f"{base}_W.npy", w_combined)
            np.save(f"{base}_x.npy", x_bf16)
            np.save(f"{base}_expected.npy", expected)
            np.save(f"{base}_npu.npy", y_npu)
            print(f"Dumped gemv {N}x{K}")
            
        elif kind == "ffn_fused":
            H, I = spec["H"], spec["I"]
            m_I, k_tile = spec["m_I"], spec["k_tile"]
            act = spec["activation"]
            base = out_dir / f"ffn_fused_{H}x{I}_{act}"
            
            # W_fused expects concatenated quantized gate/up/down
            # Each is (I, H). So total is (3*I, H).
            # Wait, pack_ffn_fused_weights handles this. Let's just generate it directly.
            # actually we can just generate random uint8 for w_combined for testing!
            # The bit-exact test doesn't care if it's mathematically a perfect FFN,
            # just that the C++ registry executes the same op as Python.
            
            w_size = ((3 * I) * (H // 32 * 20))
            w_combined = rng.integers(0, 256, size=(w_size,), dtype=np.uint8)
            x_fp32 = rng.uniform(-1.0, 1.0, size=(H,)).astype(np.float32)
            x_bf16 = x_fp32.astype(bfloat16)
            
            y_npu = run_npu_ffn_fused(w_combined, x_bf16, H, I, m_I, k_tile, act)
            
            np.save(f"{base}_W.npy", w_combined)
            np.save(f"{base}_x.npy", x_bf16)
            np.save(f"{base}_npu.npy", y_npu)
            print(f"Dumped ffn_fused {H}x{I}_{act}")

if __name__ == "__main__":
    main()
