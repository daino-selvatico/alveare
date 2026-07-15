import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron

sys.path.append(str(Path(__file__).resolve().parent))
from tools.ref.gemv_q import gemv_q_combined as ref_gemv_q
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined
from kernels.gemv_q.gemv_q import gemv_q_npu

def main():
    N, K, m, k_tile = 256, 256, 32, 256
    
    rng = np.random.default_rng(42)
    W_fp32 = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)
    x_np = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)
    
    w_q4, scales = quantize_to_q4_0(W_fp32)
    w_combined = pack_to_combined(w_q4, scales)
    expected = ref_gemv_q(w_combined, x_np.astype(bfloat16))
    
    out_dir = Path("runtime/cpp/build")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "W.npy", w_combined)
    np.save(out_dir / "x.npy", x_np.astype(bfloat16))
    np.save(out_dir / "expected.npy", expected)
    
    w_t = iron.tensor(w_combined.reshape(-1), dtype=np.uint8, device="npu")
    x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
    y_t = iron.zeros(N, dtype=bfloat16, device="npu")
    
    gemv_q_npu.specialize(N=N, K=K, m=m, k_tile=k_tile).compile(
        xclbin_path=str(out_dir / "gemv_256_256.xclbin"),
        inst_path=str(out_dir / "gemv_256_256.insts")
    )
    print("Done dumping P0 golden files.")

if __name__ == "__main__":
    main()
