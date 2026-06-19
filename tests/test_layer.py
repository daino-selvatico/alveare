import unittest
import sys
import os
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from runtime.py.layer import run_llama_layer, run_rmsnorm_npu, run_gemv_q_unified, run_rope_npu, run_attention_npu

class TestLlamaLayer(unittest.TestCase):
    
    def test_full_layer(self):
        """
        Verify the complete decoder layer against Hugging Face reference.
        Also inspects intermediate outputs to locate drift.
        """
        print("\n=== Testing Complete Decoder Layer ===")
        data_dir = Path(__file__).resolve().parents[0] / "tools" / "ref" / "data"
        if not data_dir.exists():
            data_dir = Path(__file__).resolve().parents[1] / "tools" / "ref" / "data"
            
        self.assertTrue(data_dir.exists(), f"Reference data dir {data_dir} does not exist!")
        
        # Load weights
        weights = {
            "attn_norm": np.load(data_dir / "input_norm_weights.npy"),
            "w_q": np.load(data_dir / "w_q.npy"),
            "w_k": np.load(data_dir / "w_k.npy"),
            "w_v": np.load(data_dir / "w_v.npy"),
            "w_o": np.load(data_dir / "w_o.npy"),
            "ffn_norm": np.load(data_dir / "ffn_norm_weights.npy"),
            "w_gate": np.load(data_dir / "w_gate.npy"),
            "w_up": np.load(data_dir / "w_up.npy"),
            "w_down": np.load(data_dir / "w_down.npy"),
            "cos": np.load(data_dir / "cos_val.npy").reshape(1, 64),  # we saved a single pos, but we replicate it for the test
            "sin": np.load(data_dir / "sin_val.npy").reshape(1, 64),
        }
        
        # Replicate cos/sin values for all positions to simplify indexing
        weights["cos"] = np.repeat(weights["cos"], 32, axis=0)
        weights["sin"] = np.repeat(weights["sin"], 32, axis=0)
        
        # Load inputs
        x_np = np.load(data_dir / "input_hidden_states.npy")
        pos = 31  # 32nd token (0-indexed)
        
        # Load reference caches
        k_cache_ref = np.load(data_dir / "k_cache.npy") # (8, 32, 64)
        v_cache_ref = np.load(data_dir / "v_cache.npy") # (8, 32, 64)
        
        # Setup host KV Cache (initialize past 31 tokens with reference cache values)
        k_cache = np.zeros((8, 32, 64), dtype=np.float32).astype(bfloat16)
        v_cache = np.zeros((8, 32, 64), dtype=np.float32).astype(bfloat16)
        k_cache[:, :pos, :] = k_cache_ref[:, :pos, :].astype(bfloat16)
        v_cache[:, :pos, :] = v_cache_ref[:, :pos, :].astype(bfloat16)
        
        # Run NPU layer execution
        x_bf16 = x_np.astype(bfloat16)
        y_npu = run_llama_layer(x_bf16, pos, k_cache, v_cache, weights)
        
        # Load intermediate references to report error metrics
        x_norm_ref = np.load(data_dir / "x_norm.npy").astype(bfloat16)
        q_val_ref = np.load(data_dir / "q_val.npy").astype(bfloat16)
        q_rope_ref = np.load(data_dir / "q_rope.npy").reshape(-1).astype(bfloat16)
        attn_out_ref = np.load(data_dir / "attn_out.npy").reshape(-1).astype(bfloat16)
        attn_proj_ref = np.load(data_dir / "attn_proj.npy").astype(bfloat16)
        x_post_attn_ref = np.load(data_dir / "x_post_attn.npy").astype(bfloat16)
        x_norm2_ref = np.load(data_dir / "x_norm2.npy").astype(bfloat16)
        gate_ref = np.load(data_dir / "gate.npy").astype(bfloat16)
        up_ref = np.load(data_dir / "up.npy").astype(bfloat16)
        down_ref = np.load(data_dir / "down.npy").astype(bfloat16)
        expected_output = np.load(data_dir / "output_hidden_states.npy").astype(bfloat16)
        
        # Trace sub-op by sub-op correctness (with custom NPU calls)
        # Note: we re-run intermediate steps manually to record their precise relative/absolute errors.
        x_norm_actual = run_rmsnorm_npu(x_bf16, weights["attn_norm"])
        q_actual = run_gemv_q_unified(weights["w_q"], x_norm_ref) # use ref inputs to measure local drift
        q_rope_actual = run_rope_npu(q_val_ref, weights["cos"][pos], weights["sin"][pos])
        
        # Cache check
        k_rope_ref = np.load(data_dir / "k_rope.npy").reshape(-1).astype(bfloat16)
        v_ref = np.load(data_dir / "v_val.npy").reshape(-1).astype(bfloat16)
        k_cache_test = k_cache_ref.copy().astype(bfloat16)
        v_cache_test = v_cache_ref.copy().astype(bfloat16)
        attn_out_actual = run_attention_npu(q_rope_ref, k_cache_test, v_cache_test, pos)
        
        attn_proj_actual = run_gemv_q_unified(weights["w_o"], attn_out_ref)
        gate_actual = run_gemv_q_unified(weights["w_gate"], x_norm2_ref)
        up_actual = run_gemv_q_unified(weights["w_up"], x_norm2_ref)
        
        silu_out_ref = np.load(data_dir / "silu_out.npy").astype(bfloat16)
        down_actual = run_gemv_q_unified(weights["w_down"], silu_out_ref)
        
        def print_stats(name, actual, expected):
            act_fp32 = actual.astype(np.float32)
            exp_fp32 = expected.astype(np.float32)
            abs_diff = np.abs(act_fp32 - exp_fp32)
            max_abs = np.max(abs_diff)
            mean_abs = np.mean(abs_diff)
            
            # Relative error = ||actual - expected||_2 / ||expected||_2
            rel_err = np.linalg.norm(act_fp32 - exp_fp32) / (np.linalg.norm(exp_fp32) + 1e-9)
            
            print(f"Sub-op: {name:<20} | Max Abs Error: {max_abs:.5f} | Mean Abs: {mean_abs:.5f} | Rel Error: {rel_err:.5f}")
            return rel_err, max_abs
            
        print("\n--- Sub-op Error Tracking (NPU vs HF Reference) ---")
        print_stats("RMSNorm 1", x_norm_actual, x_norm_ref)
        print_stats("Query Proj (GEMV)", q_actual, q_val_ref)
        print_stats("Query RoPE", q_rope_actual, q_rope_ref)
        print_stats("Attention (NPU)", attn_out_actual, attn_out_ref)
        print_stats("Attn Proj (GEMV)", attn_proj_actual, attn_proj_ref)
        print_stats("Gate Proj (GEMV)", gate_actual, gate_ref)
        print_stats("Up Proj (GEMV)", up_actual, up_ref)
        print_stats("Down Proj (GEMV)", down_actual, down_ref)
        
        print("\n--- End-to-End Layer Error Tracking ---")
        rel_err, max_abs = print_stats("Full Layer Output", y_npu, expected_output)
        
        # We enforce a documented tolerance:
        # Full-layer relative error must be < 0.05
        # Full-layer absolute error must be < 2.0 (compounding GEMV errors across the layer)
        self.assertLess(rel_err, 0.05, "Full layer relative error exceeds tolerance (0.05)!")
        self.assertLess(max_abs, 2.0, "Full layer absolute error exceeds tolerance (2.0)!")
        print("PASS: Full layer output within documented tolerance.")

if __name__ == "__main__":
    unittest.main()
