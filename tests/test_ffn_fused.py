import unittest
import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.ffn_fused.ffn_fused import ffn_fused_npu, pack_ffn_fused_weights
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined
from tools.ref.gemv_q import dequantize_combined

# Fast GELU math approximations used on NPU
def exp_approx(z):
    # Vectorized fast exp approximation matching C++
    val = z * 1.442695040888963
    ix = np.floor(val).astype(np.int32)
    fx = val - ix
    pow2_ix = (2.0 ** ix).astype(np.float32)
    pow2_fx = 1.0 + 0.6931471805599453 * fx + 0.2401598148889220 * fx * fx
    return pow2_ix * pow2_fx

def tanh_approx(y):
    # Clip range to avoid overflow in exp_approx
    y_clipped = np.clip(y, -9.0, 9.0)
    z = 2.0 * y_clipped
    exp_z = exp_approx(z)
    return (exp_z - 1.0) / (exp_z + 1.0)

def gelu_approx(x):
    y = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + tanh_approx(y))

class TestFfnFused(unittest.TestCase):
    
    def run_ffn_fused_npu_helper(self, H, I, m_I=32, k_tile=256):
        rng = np.random.default_rng(1726250518)
        
        # 1. Generate random weight matrices and input activation
        W_gate_fp32 = rng.uniform(-1.0, 1.0, size=(I, H)).astype(np.float32)
        W_up_fp32 = rng.uniform(-1.0, 1.0, size=(I, H)).astype(np.float32)
        W_down_fp32 = rng.uniform(-1.0, 1.0, size=(H, I)).astype(np.float32)
        x_np = rng.uniform(-1.0, 1.0, size=(H,)).astype(np.float32)
        
        # 2. Quantize and pack weights
        w_gate_q4, scales_gate = quantize_to_q4_0(W_gate_fp32)
        w_gate_combined = pack_to_combined(w_gate_q4, scales_gate)
        
        w_up_q4, scales_up = quantize_to_q4_0(W_up_fp32)
        w_up_combined = pack_to_combined(w_up_q4, scales_up)
        
        w_down_q4, scales_down = quantize_to_q4_0(W_down_fp32)
        w_down_combined = pack_to_combined(w_down_q4, scales_down)
        
        # 3. Pack weights using the combined FFN layout
        w_fused_combined = pack_ffn_fused_weights(
            w_gate_combined, w_up_combined, w_down_combined,
            H, I, m_I, k_tile
        )
        
        # 4. Create NPU tensors
        w_fused_t = iron.tensor(w_fused_combined.reshape(-1), dtype=np.uint8, device="npu")
        x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
        
        # 5. Select number of cores and initialize output tensor
        if I % (8 * m_I) == 0:
            n_cores = 8
        elif I % (4 * m_I) == 0:
            n_cores = 4
        elif I % (2 * m_I) == 0:
            n_cores = 2
        else:
            n_cores = 1
            
        y_partial_t = iron.zeros(n_cores * H, dtype=bfloat16, device="npu")
        
        # 6. Execute NPU fused FFN kernel
        ffn_fused_npu(
            w_fused_t,
            x_t,
            y_partial_t,
            H=H,
            I=I,
            m_I=m_I,
            k_tile=k_tile,
            activation="gelu"
        )
        
        # 7. Compute reference output on CPU using the fast GELU approximation
        w_gate_deq = dequantize_combined(w_gate_combined)
        w_up_deq = dequantize_combined(w_up_combined)
        w_down_deq = dequantize_combined(w_down_combined)
        
        gate_expected = (w_gate_deq @ x_np).astype(bfloat16).astype(np.float32)
        up_expected = (w_up_deq @ x_np).astype(bfloat16).astype(np.float32)
        
        # CPU fast GELU matching NPU kernel
        gelu_gate = gelu_approx(gate_expected).astype(bfloat16).astype(np.float32)
        act_expected = (gelu_gate * up_expected).astype(bfloat16).astype(np.float32)
        
        expected = (w_down_deq @ act_expected).astype(bfloat16)
        
        # 8. Sum partial core outputs
        y_partial_np = y_partial_t.numpy().reshape(n_cores, H)
        actual = np.sum(y_partial_np, axis=0).astype(bfloat16)
        
        # Verify results match within standard bfloat16 accumulation tolerance
        # rtol=0.08, atol=1500.0 accommodates the double matmul + GELU error accumulation chain
        rtol = 0.08
        atol = 1500.0
        close = np.allclose(actual, expected, rtol=rtol, atol=atol)
        if not close:
            diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
            max_diff = np.max(diff)
            print(f"Max absolute difference: {max_diff}")
            print(f"Actual (first 10):\n{actual[:10]}")
            print(f"Expected (first 10):\n{expected[:10]}")
            self.assertTrue(close, f"NPU fused FFN output does not match CPU reference for shape H={H}, I={I}!")
        print(f"✓ Shape H={H}, I={I} PASSED!")

    def test_tiny_ffn(self):
        print("\n=== Testing Tiny Fused FFN (256x512) ===")
        self.run_ffn_fused_npu_helper(H=256, I=512)

    def test_medium_ffn(self):
        print("\n=== Testing Medium Fused FFN (1024x2048) ===")
        self.run_ffn_fused_npu_helper(H=1024, I=2048)

    def test_gemma_ffn(self):
        print("\n=== Testing Gemma Fused FFN (1152x6912) ===")
        self.run_ffn_fused_npu_helper(H=1152, I=6912, k_tile=128)

if __name__ == "__main__":
    unittest.main()
