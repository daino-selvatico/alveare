import unittest
import sys
import os
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.attention.attention import attention_npu

class TestAttention(unittest.TestCase):
    
    def test_random_attention(self):
        """Test attention on random data (shape 8 heads, seq_len=32, head_dim=64)."""
        print("\n--- Testing Random Attention (S=32, H=64) ---")
        seq_len = 32
        head_dim = 64
        rng = np.random.default_rng(42)
        
        q_np = rng.uniform(-1.0, 1.0, size=(8, 4, head_dim)).astype(np.float32)
        k_np = rng.uniform(-1.0, 1.0, size=(8, seq_len, head_dim)).astype(np.float32)
        v_np = rng.uniform(-1.0, 1.0, size=(8, seq_len, head_dim)).astype(np.float32)
        
        # Pack Key and Value caches
        kv_cache_np = np.zeros((8, seq_len, head_dim * 2), dtype=np.float32)
        kv_cache_np[:, :, :head_dim] = k_np
        kv_cache_np[:, :, head_dim:] = v_np
        
        # Compute CPU reference
        expected = np.zeros((8, 4, head_dim), dtype=np.float32)
        scale = 1.0 / np.sqrt(head_dim)
        for g in range(8):
            for q in range(4):
                q_head = q_np[g, q]
                scores = np.zeros(seq_len, dtype=np.float32)
                max_score = -1e9
                for t in range(seq_len):
                    score = np.dot(q_head, k_np[g, t]) * scale
                    scores[t] = score
                    if score > max_score:
                        max_score = score
                exp_scores = np.exp(scores - max_score)
                probs = exp_scores / np.sum(exp_scores)
                
                out = np.zeros(head_dim, dtype=np.float32)
                for t in range(seq_len):
                    out += probs[t] * v_np[g, t]
                expected[g, q] = out
                
        # NPU execution
        q_t = iron.tensor(q_np.astype(bfloat16).reshape(8, -1), dtype=bfloat16, device="npu")
        kv_t = iron.tensor(kv_cache_np.astype(bfloat16).reshape(8, -1), dtype=bfloat16, device="npu")
        o_t = iron.zeros((8, 4 * head_dim), dtype=bfloat16, device="npu")
        
        attention_npu(q_t, kv_t, o_t, seq_len=seq_len, head_dim=head_dim)
        
        actual = o_t.numpy().reshape(8, 4, head_dim)
        
        rtol = 0.05
        atol = 0.05
        close = np.allclose(actual, expected.astype(bfloat16), rtol=rtol, atol=atol)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS")

    def test_real_attention(self):
        """Test attention using real activations from Llama 3.2."""
        print("\n--- Testing Real Llama 3.2 Attention (S=32, H=64) ---")
        data_dir = Path(__file__).resolve().parents[1] / "tools" / "ref" / "data"
        
        q_np = np.load(data_dir / "q_rope.npy") # shape (8, 4, 64)
        k_np = np.load(data_dir / "k_cache.npy") # shape (8, 32, 64)
        v_np = np.load(data_dir / "v_cache.npy") # shape (8, 32, 64)
        expected = np.load(data_dir / "attn_out.npy").astype(bfloat16) # shape (8, 4, 64)
        
        seq_len = k_np.shape[1] # 32
        head_dim = k_np.shape[2] # 64
        
        self.assertEqual(seq_len, 32)
        self.assertEqual(head_dim, 64)
        
        # Pack Key and Value caches
        kv_cache_np = np.zeros((8, seq_len, head_dim * 2), dtype=np.float32)
        kv_cache_np[:, :, :head_dim] = k_np
        kv_cache_np[:, :, head_dim:] = v_np
        
        # NPU execution
        q_t = iron.tensor(q_np.astype(bfloat16).reshape(8, -1), dtype=bfloat16, device="npu")
        kv_t = iron.tensor(kv_cache_np.astype(bfloat16).reshape(8, -1), dtype=bfloat16, device="npu")
        o_t = iron.zeros((8, 4 * head_dim), dtype=bfloat16, device="npu")
        
        attention_npu(q_t, kv_t, o_t, seq_len=seq_len, head_dim=head_dim)
        
        actual = o_t.numpy().reshape(8, 4, head_dim)
        
        # We allow a loose tolerance of 0.05 because the custom AIE exp function
        # uses polynomial approximation instead of high-precision math.
        rtol = 0.05
        atol = 0.05
        close = np.allclose(actual, expected, rtol=rtol, atol=atol)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS")

if __name__ == "__main__":
    unittest.main()
