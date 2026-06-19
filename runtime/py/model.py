import os
import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from kernels.gemv_q.gemv_q import gemv_q_npu
from kernels.rmsnorm.rmsnorm import rmsnorm_npu
from kernels.rope.rope import rope_npu

class LlamaNPUModel:
    def __init__(self, weights_dir: Path, max_seq_len: int = 2048):
        self.weights_dir = Path(weights_dir)
        self.max_seq_len = max_seq_len
        
        print("Loading non-transformer weights...")
        self.token_embd = np.load(self.weights_dir / "token_embd.npy")
        self.output_norm = np.load(self.weights_dir / "output_norm.weight.npy")
        
        print("Loading layer norms...")
        self.layer_attn_norms = [np.load(self.weights_dir / f"blk.{l}.attn_norm.weight.npy") for l in range(16)]
        self.layer_ffn_norms = [np.load(self.weights_dir / f"blk.{l}.ffn_norm.weight.npy") for l in range(16)]
        
        print("Mmap-mapping layer matmul weights...")
        self.layer_weights = []
        for l in range(16):
            layer_w = {}
            for proj in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]:
                path = self.weights_dir / f"blk.{l}.{proj}.weight_packed.npy"
                layer_w[proj] = np.load(path, mmap_mode='r')
            self.layer_weights.append(layer_w)
            
        print("Mmap-mapping LM head...")
        self.lm_head = np.load(self.weights_dir / "lm_head_packed.npy", mmap_mode='r')
        
        print("Pre-dequantizing layer weights for fast CPU prefill...")
        from tools.ref.gemv_q import dequantize_combined
        self.layer_weights_dequant = []
        for l in range(16):
            layer_w_dequant = {}
            for proj in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]:
                layer_w_dequant[proj] = dequantize_combined(self.layer_weights[l][proj]).astype(bfloat16)
            self.layer_weights_dequant.append(layer_w_dequant)
            
        print("Pre-dequantizing LM head...")
        self.lm_head_dequant = dequantize_combined(self.lm_head).astype(bfloat16)
        
        print("Precomputing RoPE cos/sin tables...")
        self.cos_sin_table = self.precompute_cos_sin_table()
        
        print("Initializing KV caches...")
        self.k_caches = [np.zeros((8, self.max_seq_len, 64), dtype=bfloat16) for _ in range(16)]
        self.v_caches = [np.zeros((8, self.max_seq_len, 64), dtype=bfloat16) for _ in range(16)]
        
        print("Allocating resident NPU tensors for zero-copy execution...")
        # GEMV tensors
        # Unified shape (2048, 2048) in Q4_0 packed format is (2048, 1280) bytes
        self.w_gemv_t = iron.tensor(np.zeros((2048, 1280), dtype=np.uint8).reshape(-1), dtype=np.uint8, device="npu")
        self.x_gemv_t = iron.tensor(np.zeros(2048, dtype=bfloat16), dtype=bfloat16, device="npu")
        self.y_gemv_t = iron.zeros(2048, dtype=bfloat16, device="npu")
        
        # RMSNorm tensors
        self.x_rmsnorm_t = iron.tensor(np.zeros(2048, dtype=bfloat16), dtype=bfloat16, device="npu")
        self.w_rmsnorm_t = iron.tensor(np.zeros(2048, dtype=np.float32), dtype=np.float32, device="npu")
        self.y_rmsnorm_t = iron.zeros(2048, dtype=bfloat16, device="npu")
        
        # RoPE tensors
        self.x_rope_q_t = iron.tensor(np.zeros(2048, dtype=bfloat16), dtype=bfloat16, device="npu")
        self.y_rope_q_t = iron.zeros(2048, dtype=bfloat16, device="npu")
        self.x_rope_k_t = iron.tensor(np.zeros(512, dtype=bfloat16), dtype=bfloat16, device="npu")
        self.y_rope_k_t = iron.zeros(512, dtype=bfloat16, device="npu")
        self.cos_sin_rope_t = iron.tensor(np.zeros(128, dtype=bfloat16), dtype=bfloat16, device="npu")
        
        print("NPU initialization complete.")
        
    def reset_caches(self):
        """Resets the KV caches to zero."""
        for l in range(16):
            self.k_caches[l].fill(0)
            self.v_caches[l].fill(0)

    def precompute_cos_sin_table(self):
        dim = 64
        base = 500000.0
        factor = 32.0
        low_freq_factor = 1.0
        high_freq_factor = 4.0
        old_context_len = 8192
        
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * np.pi / inv_freq
        
        inv_freq_llama = inv_freq.copy()
        inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq_llama)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = np.clip(smooth_factor, 0.0, 1.0)
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
        is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
        inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        
        cos_sin_table = np.zeros((self.max_seq_len, 128), dtype=np.float32)
        
        for pos in range(self.max_seq_len):
            freqs = pos * inv_freq_llama
            cos_val = np.cos(freqs)
            sin_val = np.sin(freqs)
            # Duplicate to match HF format
            cos_dup = np.concatenate([cos_val, cos_val])
            sin_dup = np.concatenate([sin_val, sin_val])
            cos_sin_table[pos] = np.concatenate([cos_dup, sin_dup])
            
        return cos_sin_table.astype(bfloat16)

    def run_gemv_npu(self, W_combined, x_bf16):
        # W_combined shape: (N, K_blocks * 20)
        # x_bf16 shape: (K,)
        # Note: both N and K are multiples of 2048 because of pre-padding.
        N = W_combined.shape[0]
        K_blocks = W_combined.shape[1] // 20
        K = K_blocks * 32
        
        target_N = 2048
        target_K = 2048
        
        y_sum = np.zeros(N, dtype=np.float32)
        
        # Loop over column chunks
        for start_col in range(0, K, target_K):
            end_col = min(start_col + target_K, K)
            x_chunk = x_bf16[start_col:end_col]
            
            # Column slice of combined weights
            b_start = start_col // 32
            b_end = end_col // 32
            W_col_slice = W_combined[:, b_start * 20 : b_end * 20]
            
            # Loop over row chunks
            for start_row in range(0, N, target_N):
                end_row = min(start_row + target_N, N)
                W_chunk = W_col_slice[start_row:end_row]
                
                # Copy to resident NPU tensors
                self.w_gemv_t.numpy()[:] = W_chunk.reshape(-1)
                self.w_gemv_t._sync_to_device()
                
                self.x_gemv_t.numpy()[:] = x_chunk
                self.x_gemv_t._sync_to_device()
                
                # Execute JIT NPU kernel (warmup compilation happens once, then hits cache)
                gemv_q_npu(self.w_gemv_t, self.x_gemv_t, self.y_gemv_t, N=2048, K=2048, m=32, k_tile=256)
                
                # Fetch result safely
                res = np.array(self.y_gemv_t.numpy()).astype(np.float32)
                y_sum[start_row:end_row] += res[:(end_row - start_row)]
                
        return y_sum.astype(bfloat16)

    def run_rmsnorm_npu_resident(self, x_bf16, w_fp32):
        K = x_bf16.shape[0]
        self.x_rmsnorm_t.numpy()[:K] = x_bf16
        self.x_rmsnorm_t._sync_to_device()
        
        self.w_rmsnorm_t.numpy()[:K] = w_fp32
        self.w_rmsnorm_t._sync_to_device()
        
        rmsnorm_npu(self.x_rmsnorm_t, self.w_rmsnorm_t, self.y_rmsnorm_t, K=2048)
        
        res = np.array(self.y_rmsnorm_t.numpy())
        return res[:K].astype(bfloat16)

    def run_rope_npu_resident(self, x_bf16, pos, is_key=False):
        K = x_bf16.shape[0]
        cos_sin = self.cos_sin_table[pos]
        
        if K == 2048:
            x_t = self.x_rope_q_t
            y_t = self.y_rope_q_t
        else:
            x_t = self.x_rope_k_t
            y_t = self.y_rope_k_t
            
        x_t.numpy()[:K] = x_bf16
        x_t._sync_to_device()
        
        self.cos_sin_rope_t.numpy()[:] = cos_sin
        self.cos_sin_rope_t._sync_to_device()
        
        rope_npu(x_t, self.cos_sin_rope_t, y_t, K=K, head_dim=64)
        
        res = np.array(y_t.numpy())
        return res[:K].astype(bfloat16)

    def run_rmsnorm_cpu(self, x_bf16, w_fp32):
        variance = np.mean(x_bf16.astype(np.float32) ** 2)
        return (x_bf16.astype(np.float32) * (1.0 / np.sqrt(variance + 1e-5)) * w_fp32).astype(bfloat16)

    def run_rope_cpu(self, x_bf16, pos):
        K = x_bf16.shape[0]
        cos_sin = self.cos_sin_table[pos]
        cos = cos_sin[:32].astype(np.float32)
        sin = cos_sin[64:96].astype(np.float32)
        
        num_heads = K // 64
        x_fp32 = x_bf16.astype(np.float32).reshape(num_heads, 64)
        y_fp32 = np.zeros_like(x_fp32)
        
        for h in range(num_heads):
            x1 = x_fp32[h, :32]
            x2 = x_fp32[h, 32:]
            y_fp32[h, :32] = x1 * cos - x2 * sin
            y_fp32[h, 32:] = x2 * cos + x1 * sin
            
        return y_fp32.reshape(-1).astype(bfloat16)

    def run_attention_host(self, q_rope, pos, l):
        # q_rope shape: (2048,) -> reshape to (8, 4, 64)
        q = q_rope.astype(np.float32).reshape(8, 4, 64)
        
        seq_len = pos + 1
        
        # K, V slice shape: (8, seq_len, 64)
        k = self.k_caches[l][:, :seq_len, :].astype(np.float32)
        v = self.v_caches[l][:, :seq_len, :].astype(np.float32)
        
        scale = 1.0 / np.sqrt(64.0)
        # q: (8, 4, 64), k: (8, seq_len, 64) -> transpose to (8, 64, seq_len)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale # (8, 4, seq_len)
        
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        output = np.matmul(probs, v) # (8, 4, 64)
        return output.reshape(-1).astype(bfloat16)

    def run_layer(self, x_bf16, pos, l, use_npu: bool = True):
        # 1. Input RMSNorm (on CPU to save NPU context)
        if use_npu:
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.layer_attn_norms[l])
            # 2. QKV Projections (quantized weight GEMVs)
            q = self.run_gemv_npu(self.layer_weights[l]["attn_q"], x_norm)
            k = self.run_gemv_npu(self.layer_weights[l]["attn_k"], x_norm)[:512]
            v = self.run_gemv_npu(self.layer_weights[l]["attn_v"], x_norm)[:512]
            # 3. Apply RoPE (Query and Key) on CPU
            q_rope = self.run_rope_cpu(q, pos)
            k_rope = self.run_rope_cpu(k, pos)
        else:
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.layer_attn_norms[l])
            # Direct CPU matrix-vector multiply on pre-dequantized weights
            q = (self.layer_weights_dequant[l]["attn_q"].astype(np.float32) @ x_norm.astype(np.float32)).astype(bfloat16)
            k = (self.layer_weights_dequant[l]["attn_k"].astype(np.float32) @ x_norm.astype(np.float32)).astype(bfloat16)[:512]
            v = (self.layer_weights_dequant[l]["attn_v"].astype(np.float32) @ x_norm.astype(np.float32)).astype(bfloat16)[:512]
            q_rope = self.run_rope_cpu(q, pos)
            k_rope = self.run_rope_cpu(k, pos)
        
        # 4. Insert K and V into host KV cache
        self.k_caches[l][:, pos, :] = k_rope.reshape(8, 64)
        self.v_caches[l][:, pos, :] = v.reshape(8, 64)
        
        # 5. Attention on Host
        attn_out = self.run_attention_host(q_rope, pos, l)
        
        # 6. Attention Output Projection
        if use_npu:
            attn_proj = self.run_gemv_npu(self.layer_weights[l]["attn_output"], attn_out)
        else:
            attn_proj = (self.layer_weights_dequant[l]["attn_output"].astype(np.float32) @ attn_out.astype(np.float32)).astype(bfloat16)
        
        # 7. First Residual Connection
        x_post_attn = (x_bf16.astype(np.float32) + attn_proj.astype(np.float32)).astype(bfloat16)
        
        # 8. Post-attention RMSNorm (on CPU to save NPU context)
        if use_npu:
            x_norm2 = self.run_rmsnorm_cpu(x_post_attn, self.layer_ffn_norms[l])
            # 9. MLP Projections (Gate & Up)
            gate = self.run_gemv_npu(self.layer_weights[l]["ffn_gate"], x_norm2)
            up = self.run_gemv_npu(self.layer_weights[l]["ffn_up"], x_norm2)
        else:
            x_norm2 = self.run_rmsnorm_cpu(x_post_attn, self.layer_ffn_norms[l])
            gate = (self.layer_weights_dequant[l]["ffn_gate"].astype(np.float32) @ x_norm2.astype(np.float32)).astype(bfloat16)
            up = (self.layer_weights_dequant[l]["ffn_up"].astype(np.float32) @ x_norm2.astype(np.float32)).astype(bfloat16)
        
        # 10. MLP Activation (SwiGLU)
        gate_fp32 = gate.astype(np.float32)
        up_fp32 = up.astype(np.float32)
        silu_out = (gate_fp32 * (1.0 / (1.0 + np.exp(-gate_fp32)))) * up_fp32
        silu_out_bf16 = silu_out.astype(bfloat16)
        
        # 11. MLP Down Projection
        if use_npu:
            down = self.run_gemv_npu(self.layer_weights[l]["ffn_down"], silu_out_bf16)
        else:
            down = (self.layer_weights_dequant[l]["ffn_down"].astype(np.float32) @ silu_out_bf16.astype(np.float32)).astype(bfloat16)
        
        # 12. Second Residual Connection
        y_final = (x_post_attn.astype(np.float32) + down.astype(np.float32)).astype(bfloat16)
        
        # return y_final
        return y_final
 
    def forward(self, token_id: int, pos: int, return_logits: bool = True, use_npu: bool = True) -> np.ndarray:
        # 1. Embedding lookup
        x_bf16 = self.token_embd[token_id].astype(bfloat16)
        
        # 2. Run layers
        for l in range(16):
            x_bf16 = self.run_layer(x_bf16, pos, l, use_npu=use_npu)
            
        if not return_logits:
            return None
            
        # 3. Final norm (on CPU to save NPU context)
        if use_npu:
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.output_norm)
            # 4. LM Head (chunked GEMV)
            logits = self.run_gemv_npu(self.lm_head, x_norm)
        else:
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.output_norm)
            logits = (self.lm_head_dequant.astype(np.float32) @ x_norm.astype(np.float32)).astype(bfloat16)
        
        # Slice to vocabulary size
        return logits[:128256].astype(np.float32)
