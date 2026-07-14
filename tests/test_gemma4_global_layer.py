import unittest
import sys
import os
from pathlib import Path
from ml_dtypes import bfloat16
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from runtime.py.model import LlamaNPUModel

class TestGemma4GlobalLayer(unittest.TestCase):
    
    def test_global_layer(self):
        """
        Verify the complete Gemma-4 global decoder layer (Layer 5) against CPU reference.
        Also inspects intermediate outputs to locate drift.
        """
        print("\n=== Testing Complete Gemma-4 Global Decoder Layer (Layer 5) ===")
        ref_dir = Path(__file__).resolve().parents[0] / "tools" / "ref" / "data_gemma4_layer_5"
        if not ref_dir.exists():
            ref_dir = Path(__file__).resolve().parents[1] / "tools" / "ref" / "data_gemma4_layer_5"
            
        self.assertTrue(ref_dir.exists(), f"Reference data dir {ref_dir} does not exist!")
        
        weights_dir = Path(__file__).resolve().parents[1] / "quantized_weights_gemma4"
        self.assertTrue(weights_dir.exists(), f"Weights dir {weights_dir} does not exist!")
        
        # Load model using the generalized runtime
        model = LlamaNPUModel(weights_dir)
        
        # Load inputs
        x_bf16 = np.load(ref_dir / "input_hidden_states.npy").astype(bfloat16)
        pos = 31
        
        # Initialize KV caches up to pos
        k_cache_ref = np.load(ref_dir / "k_cache.npy") # (1, 32, 512)
        v_cache_ref = np.load(ref_dir / "v_cache.npy") # (1, 32, 512)
            
        model.reset_caches()
        model.k_caches[5][:, :pos, :] = k_cache_ref[:, :pos, :].astype(bfloat16)
        model.v_caches[5][:, :pos, :] = v_cache_ref[:, :pos, :].astype(bfloat16)
        
        # Run Gemma layer on NPU
        print("Running Gemma-4 global layer (Layer 5) on NPU...")
        y_npu = model.run_layer(x_bf16, pos, l=5, use_npu=True)
        print("Gemma-4 global layer run completed.")
        
        # Load intermediate references to report error metrics
        x_norm_ref = np.load(ref_dir / "x_norm.npy").astype(bfloat16)
        q_val_ref = np.load(ref_dir / "q_val.npy").astype(bfloat16)
        q_normed_ref = np.load(ref_dir / "q_normed.npy").astype(bfloat16)
        q_rope_ref = np.load(ref_dir / "q_rope.npy").astype(bfloat16)
        attn_out_ref = np.load(ref_dir / "attn_out.npy").astype(bfloat16)
        attn_proj_ref = np.load(ref_dir / "attn_proj.npy").astype(bfloat16)
        x_post_attn_ref = np.load(ref_dir / "x_post_attn.npy").astype(bfloat16)
        x_norm2_ref = np.load(ref_dir / "x_norm2.npy").astype(bfloat16)
        gate_ref = np.load(ref_dir / "gate.npy").astype(bfloat16)
        up_ref = np.load(ref_dir / "up.npy").astype(bfloat16)
        down_ref = np.load(ref_dir / "down.npy").astype(bfloat16)
        expected_output = np.load(ref_dir / "output_hidden_states.npy").astype(bfloat16)
        
        # Trace sub-op by sub-op correctness using ref inputs to avoid compounding error
        # 1. Input RMSNorm
        x_norm_actual = model.run_rmsnorm_cpu(x_bf16, model.layer_attn_norms[5])
        # 2. Query Projection (NPU)
        q_actual = model.run_gemv_npu(model.layer_weights[5]["attn_q"], x_norm_ref)[:8192]
        # 3. QK-Norm
        q_normed_actual = np.zeros_like(q_val_ref)
        for h in range(16):
            q_h = q_val_ref[h * 512 : (h + 1) * 512]
            q_normed_actual[h * 512 : (h + 1) * 512] = model.run_rmsnorm_cpu(q_h, model.layer_q_norms[5])
        # 4. RoPE
        q_rope_actual = model.run_rope_cpu_gemma(q_normed_ref, pos, base_freq=1000000.0)
        # 5. Attention (Host CPU)
        model.k_caches[5][:, :pos+1, :] = k_cache_ref.astype(bfloat16)
        model.v_caches[5][:, :pos+1, :] = v_cache_ref.astype(bfloat16)
        attn_out_actual = model.run_attention_host(q_rope_ref, pos, l=5)
        # 6. Output Projection (NPU)
        attn_proj_actual = model.run_gemv_npu(model.layer_weights[5]["attn_output"], attn_out_ref)[:3840]
        # 7. Gate & Up projections (NPU)
        gate_actual = model.run_gemv_npu(model.layer_weights[5]["ffn_gate"], x_norm2_ref)[:15360]
        up_actual = model.run_gemv_npu(model.layer_weights[5]["ffn_up"], x_norm2_ref)[:15360]
        # 8. Down projection (NPU)
        geglu_out_ref = np.load(ref_dir / "geglu_out.npy").astype(bfloat16)
        down_actual = model.run_gemv_npu(model.layer_weights[5]["ffn_down"], geglu_out_ref)[:3840]
        
        def print_stats(name, actual, expected):
            act_fp32 = actual.astype(np.float32).reshape(-1)
            exp_fp32 = expected.astype(np.float32).reshape(-1)
            abs_diff = np.abs(act_fp32 - exp_fp32)
            max_abs = np.max(abs_diff)
            mean_abs = np.mean(abs_diff)
            rel_err = np.linalg.norm(act_fp32 - exp_fp32) / (np.linalg.norm(exp_fp32) + 1e-9)
            print(f"Sub-op: {name:<20} | Max Abs Error: {max_abs:.5f} | Mean Abs: {mean_abs:.5f} | Rel Error: {rel_err:.5f}")
            return rel_err, max_abs
            
        print("\n--- Sub-op Error Tracking (NPU/Host vs CPU Reference) ---")
        print_stats("RMSNorm 1", x_norm_actual, x_norm_ref)
        print_stats("Query Proj (GEMV)", q_actual, q_val_ref)
        print_stats("QK-Norm Query", q_normed_actual, q_normed_ref)
        print_stats("Query RoPE", q_rope_actual, q_rope_ref)
        print_stats("Attention (Host)", attn_out_actual, attn_out_ref)
        print_stats("Attn Proj (GEMV)", attn_proj_actual, attn_proj_ref)
        print_stats("Gate Proj (GEMV)", gate_actual, gate_ref)
        print_stats("Up Proj (GEMV)", up_actual, up_ref)
        print_stats("Down Proj (GEMV)", down_actual, down_ref)
        
        print("\n--- End-to-End Layer Error Tracking ---")
        rel_err, max_abs = print_stats("Full Layer Output", y_npu, expected_output)
        
        # Enforce tolerance of < 25% relative error (expected for Q4_0 block quantization)
        self.assertLess(rel_err, 0.25, "Full layer relative error exceeds tolerance (0.25)!")
        self.assertLess(max_abs, 15.0, "Full layer absolute error exceeds tolerance (15.0)!")
        print("PASS: Full layer output within documented tolerance.")

if __name__ == "__main__":
    unittest.main()
