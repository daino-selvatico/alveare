import unittest
import sys
import os
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.rmsnorm.rmsnorm import rmsnorm_npu

class TestRMSNorm(unittest.TestCase):
    
    def test_random_rmsnorm(self):
        """Test RMSNorm on random data."""
        print("\n--- Testing Random RMSNorm (K=2048) ---")
        K = 2048
        rng = np.random.default_rng(42)
        x_np = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)
        w_np = rng.uniform(0.5, 1.5, size=(K,)).astype(np.float32)
        
        # CPU reference
        variance = np.mean(x_np**2)
        inv_std = 1.0 / np.sqrt(variance + 1e-5)
        expected = (x_np * inv_std * w_np).astype(bfloat16)
        
        # NPU execution
        x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
        w_t = iron.tensor(w_np, dtype=np.float32, device="npu")
        y_t = iron.zeros(K, dtype=bfloat16, device="npu")
        
        rmsnorm_npu(x_t, w_t, y_t, K=K)
        
        actual = y_t.numpy()
        close = np.allclose(actual, expected, rtol=0.01, atol=0.02)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS")

    def test_real_rmsnorm(self):
        """Test RMSNorm using real activations and weights from Llama 3.2."""
        print("\n--- Testing Real Llama 3.2 RMSNorm (K=2048) ---")
        data_dir = Path(__file__).resolve().parents[1] / "tools" / "ref" / "data"
        
        x_np = np.load(data_dir / "input_hidden_states.npy")
        w_np = np.load(data_dir / "input_norm_weights.npy")
        expected = np.load(data_dir / "x_norm.npy").astype(bfloat16)
        
        K = x_np.shape[0]
        self.assertEqual(K, 2048)
        
        # NPU execution
        x_t = iron.tensor(x_np.astype(bfloat16), dtype=bfloat16, device="npu")
        w_t = iron.tensor(w_np, dtype=np.float32, device="npu")
        y_t = iron.zeros(K, dtype=bfloat16, device="npu")
        
        rmsnorm_npu(x_t, w_t, y_t, K=K)
        
        actual = y_t.numpy()
        
        # Note: tolerance is extremely tight because the implementation on AIE
        # accumulates in fp32, matching the PyTorch implementation exactly.
        close = np.allclose(actual, expected, rtol=0.01, atol=0.02)
        if not close:
            print(f"Max abs diff: {np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))}")
        self.assertTrue(close)
        print("PASS")

if __name__ == "__main__":
    unittest.main()
