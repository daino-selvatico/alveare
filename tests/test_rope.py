import unittest
import sys
import os
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.rope.rope import rope_npu

class TestRoPE(unittest.TestCase):
    
    def test_random_rope(self):
        """Test RoPE on random data (Query-like shape)."""
        print("\n--- Testing Random RoPE (K=2048, H=64) ---")
        K = 2048
        H = 64
        rng = np.random.default_rng(42)
        x_np = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)
        cos_np = rng.uniform(-1.0, 1.0, size=(H,)).astype(np.float32)
        sin_np = rng.uniform(-1.0, 1.0, size=(H,)).astype(np.float32)
        
        # Pack cos and sin
        cos_sin_np = np.concatenate([cos_np, sin_np])
        
        # CPU reference
        expected = np.zeros_like(x_np)
        num_heads = K // H
        half = H // 2
        for h in range(num_heads):
            head = x_np[h * H : (h + 1) * H]
            x1 = head[:half]
            x2 = head[half:]
            expected[h * H : h * H + half] = x1 * cos_np[:half] - x2 * sin_np[:half]
            expected[h * H + half : (h + 1) * H] = x2 * cos_np[:half] + x1 * sin_np[:half]
            
        # NPU execution
        x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
        cos_sin_t = iron.tensor(cos_sin_np.astype(bfloat16), dtype=bfloat16, device="npu")
        y_t = iron.zeros(K, dtype=bfloat16, device="npu")
        
        rope_npu(x_t, cos_sin_t, y_t, K=K, head_dim=H)
        
        actual = y_t.numpy()
        close = np.allclose(actual, expected.astype(bfloat16), rtol=0.01, atol=0.02)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS")

    def test_real_query_rope(self):
        """Test RoPE on Llama 3.2 Query activations."""
        print("\n--- Testing Real Query RoPE (K=2048, H=64) ---")
        data_dir = Path(__file__).resolve().parents[1] / "tools" / "ref" / "data"
        
        x_np = np.load(data_dir / "q_val.npy")
        cos_np = np.load(data_dir / "cos_val.npy")
        sin_np = np.load(data_dir / "sin_val.npy")
        expected = np.load(data_dir / "q_rope.npy").reshape(-1).astype(bfloat16)
        
        K = x_np.shape[0] # 2048
        H = cos_np.shape[0] # 64
        
        # Pack cos and sin
        cos_sin_np = np.concatenate([cos_np, sin_np])
        
        # NPU execution
        x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
        cos_sin_t = iron.tensor(cos_sin_np.astype(bfloat16), dtype=bfloat16, device="npu")
        y_t = iron.zeros(K, dtype=bfloat16, device="npu")
        
        rope_npu(x_t, cos_sin_t, y_t, K=K, head_dim=H)
        
        actual = y_t.numpy()
        close = np.allclose(actual, expected, rtol=0.01, atol=0.02)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS")

    def test_real_key_rope(self):
        """Test RoPE on Llama 3.2 Key activations."""
        print("\n--- Testing Real Key RoPE (K=512, H=64) ---")
        data_dir = Path(__file__).resolve().parents[1] / "tools" / "ref" / "data"
        
        x_np = np.load(data_dir / "k_val.npy")
        cos_np = np.load(data_dir / "cos_val.npy")
        sin_np = np.load(data_dir / "sin_val.npy")
        expected = np.load(data_dir / "k_rope.npy").reshape(-1).astype(bfloat16)
        
        K = x_np.shape[0] # 512
        H = cos_np.shape[0] # 64
        
        # Pack cos and sin
        cos_sin_np = np.concatenate([cos_np, sin_np])
        
        # NPU execution
        x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
        cos_sin_t = iron.tensor(cos_sin_np.astype(bfloat16), dtype=bfloat16, device="npu")
        y_t = iron.zeros(K, dtype=bfloat16, device="npu")
        
        rope_npu(x_t, cos_sin_t, y_t, K=K, head_dim=H)
        
        actual = y_t.numpy()
        close = np.allclose(actual, expected, rtol=0.01, atol=0.02)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS")

if __name__ == "__main__":
    unittest.main()
