import os
import sys
import json
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from kernels.gemv_q.gemv_q import gemv_q_npu
from kernels.gemm_q.gemm_q import gemm_q_npu
from kernels.rmsnorm.rmsnorm import rmsnorm_npu
from kernels.rope.rope import rope_npu

class LazyLayerWeights:
    def __init__(self, weights_dir: Path, layer_idx: int):
        self.weights_dir = weights_dir
        self.layer_idx = layer_idx
        self._cache = {}
        
    def __getitem__(self, key):
        if key not in self._cache:
            path = self.weights_dir / f"blk.{self.layer_idx}.{key}.weight_packed.npy"
            self._cache[key] = np.load(path, mmap_mode='r')
        return self._cache[key]
        
    def clear(self):
        self._cache.clear()

class LlamaNPUModel:
    def __init__(self, weights_dir: Path, max_seq_len: int = 2048):
        self.weights_dir = Path(weights_dir)
        self.max_seq_len = max_seq_len
        
        # Load config if present
        config_path = self.weights_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
            self.model_type = self.config.get("model_type", "llama")
        else:
            self.config = {}
            self.model_type = "llama"
            
        print(f"Initializing model of type: {self.model_type}")
        
        if self.model_type == "gemma3":
            self.hidden_size = self.config.get("hidden_size", 1152)
            self.intermediate_size = self.config.get("intermediate_size", 6912)
            self.num_attention_heads = self.config.get("num_attention_heads", 4)
            self.num_key_value_heads = self.config.get("num_key_value_heads", 1)
            self.head_dim = self.config.get("head_dim", 256)
            self.num_hidden_layers = self.config.get("num_hidden_layers", 26)
            self.vocab_size = self.config.get("vocab_size", 262144)
        elif self.model_type == "gemma4":
            self.hidden_size = self.config.get("hidden_size", 3840)
            self.intermediate_size = self.config.get("intermediate_size", 15360)
            self.num_attention_heads = self.config.get("num_attention_heads", 16)
            self.num_key_value_heads = self.config.get("num_key_value_heads", 8)
            self.head_dim = self.config.get("head_dim", 256)
            self.num_hidden_layers = self.config.get("num_hidden_layers", 48)
            self.vocab_size = self.config.get("vocab_size", 262144)
        else:
            # Llama-3.2 defaults
            self.hidden_size = 2048
            self.intermediate_size = 8192
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            self.head_dim = 64
            self.num_hidden_layers = 16
            self.vocab_size = 128256
            
        print("Loading non-transformer weights...")
        self.token_embd = np.load(self.weights_dir / "token_embd.npy")
        self.output_norm = np.load(self.weights_dir / "output_norm.weight.npy")
        
        print("Loading layer norms...")
        self.layer_attn_norms = [np.load(self.weights_dir / f"blk.{l}.attn_norm.weight.npy") for l in range(self.num_hidden_layers)]
        self.layer_ffn_norms = [np.load(self.weights_dir / f"blk.{l}.ffn_norm.weight.npy") for l in range(self.num_hidden_layers)]
        
        if self.model_type in ["gemma3", "gemma4"]:
            self.layer_post_attn_norms = [np.load(self.weights_dir / f"blk.{l}.post_attention_norm.weight.npy") for l in range(self.num_hidden_layers)]
            self.layer_post_ffw_norms = [np.load(self.weights_dir / f"blk.{l}.post_ffw_norm.weight.npy") for l in range(self.num_hidden_layers)]
            self.layer_q_norms = [np.load(self.weights_dir / f"blk.{l}.attn_q_norm.weight.npy") for l in range(self.num_hidden_layers)]
            self.layer_k_norms = [np.load(self.weights_dir / f"blk.{l}.attn_k_norm.weight.npy") for l in range(self.num_hidden_layers)]
            
        if self.model_type == "gemma4":
            self.layer_output_scales = [np.load(self.weights_dir / f"blk.{l}.layer_output_scale.weight.npy")[0] for l in range(self.num_hidden_layers)]
            
        if self.model_type == "gemma4":
            print("Initializing lazy layer weights for Gemma-4...")
            self.layer_weights = [LazyLayerWeights(self.weights_dir, l) for l in range(self.num_hidden_layers)]
        else:
            print("Mmap-mapping layer matmul weights...")
            self.layer_weights = []
            for l in range(self.num_hidden_layers):
                layer_w = {}
                for proj in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]:
                    path = self.weights_dir / f"blk.{l}.{proj}.weight_packed.npy"
                    layer_w[proj] = np.load(path, mmap_mode='r')
                self.layer_weights.append(layer_w)
            
        print("Mmap-mapping LM head...")
        self.lm_head = np.load(self.weights_dir / "lm_head_packed.npy", mmap_mode='r')
        
        if self.model_type != "gemma4":
            print("Pre-dequantizing layer weights for fast CPU prefill...")
            from tools.ref.gemv_q import dequantize_combined
            self.layer_weights_dequant = []
            for l in range(self.num_hidden_layers):
                layer_w_dequant = {}
                for proj in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]:
                    layer_w_dequant[proj] = dequantize_combined(self.layer_weights[l][proj]).astype(bfloat16)
                self.layer_weights_dequant.append(layer_w_dequant)
                
            print("Pre-dequantizing LM head...")
            self.lm_head_dequant = dequantize_combined(self.lm_head).astype(bfloat16)
        else:
            self.layer_weights_dequant = None
            self.lm_head_dequant = None
        
        if self.model_type == "gemma3":
            print("Precomputing Gemma RoPE tables...")
            self.cos_sin_table_sliding = self.precompute_cos_sin_table_gemma(base=10000.0)
            self.cos_sin_table_full = self.precompute_cos_sin_table_gemma(base=1000000.0)
        elif self.model_type == "gemma4":
            print("Precomputing Gemma-4 RoPE tables...")
            self.cos_sin_table_sliding = self.precompute_cos_sin_table_gemma(base=10000.0, dim=256)
            self.cos_sin_table_full = self.precompute_cos_sin_table_gemma(base=1000000.0, dim=512)
        else:
            print("Precomputing Llama RoPE cos/sin tables...")
            self.cos_sin_table = self.precompute_cos_sin_table()
            
        print("Initializing KV caches...")
        if self.model_type == "gemma4":
            self.k_caches = []
            self.v_caches = []
            for l in range(self.num_hidden_layers):
                is_sliding = (l + 1) % 6 != 0
                kv_heads = 8 if is_sliding else 1
                h_dim = 256 if is_sliding else 512
                self.k_caches.append(np.zeros((kv_heads, self.max_seq_len, h_dim), dtype=bfloat16))
                self.v_caches.append(np.zeros((kv_heads, self.max_seq_len, h_dim), dtype=bfloat16))
        else:
            self.k_caches = [np.zeros((self.num_key_value_heads, self.max_seq_len, self.head_dim), dtype=bfloat16) for _ in range(self.num_hidden_layers)]
            self.v_caches = [np.zeros((self.num_key_value_heads, self.max_seq_len, self.head_dim), dtype=bfloat16) for _ in range(self.num_hidden_layers)]
        
        print("Allocating resident NPU tensors for zero-copy execution...")
        # GEMV tensors
        # Unified shape (2048, 2048) in Q4_0 packed format is (2048, 1280) bytes
        self.w_gemv_t = iron.tensor(np.zeros((2048, 1280), dtype=np.uint8).reshape(-1), dtype=np.uint8, device="npu")
        self.x_gemv_t = iron.tensor(np.zeros(2048, dtype=bfloat16), dtype=bfloat16, device="npu")
        self.y_gemv_t = iron.zeros(2048, dtype=bfloat16, device="npu")
        
        # GEMM batched prefill tensors (max batch size = 16)
        self.MAX_BATCH_SIZE = 16
        self.x_gemm_t = iron.tensor(np.zeros(self.MAX_BATCH_SIZE * 2048, dtype=bfloat16), dtype=bfloat16, device="npu")
        self.y_gemm_t = iron.zeros(self.MAX_BATCH_SIZE * 2048, dtype=bfloat16, device="npu")
        
        # RMSNorm tensors (reused only for Llama path or direct testing)
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
        for l in range(self.num_hidden_layers):
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
            cos_dup = np.concatenate([cos_val, cos_val])
            sin_dup = np.concatenate([sin_val, sin_val])
            cos_sin_table[pos] = np.concatenate([cos_dup, sin_dup])
            
        return cos_sin_table.astype(bfloat16)

    def precompute_cos_sin_table_gemma(self, base, dim=None):
        if dim is None:
            dim = self.head_dim # 256
        if self.model_type == "gemma4" and dim == 512:
            # Proportional RoPE for global layers (partial_rotary_factor = 0.25)
            rope_angles = int(0.25 * dim // 2) # 64
            inv_freq_rotated = 1.0 / (base ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / dim))
            inv_freq = np.zeros(dim // 2, dtype=np.float32)
            inv_freq[:rope_angles] = inv_freq_rotated
        else:
            inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
            
        cos_sin_table = np.zeros((self.max_seq_len, dim), dtype=np.float32)
        for pos in range(self.max_seq_len):
            freqs = pos * inv_freq
            cos_val = np.cos(freqs)
            sin_val = np.sin(freqs)
            cos_sin_table[pos] = np.concatenate([cos_val, sin_val])
        return cos_sin_table.astype(bfloat16)

    def run_gemv_npu(self, W_combined, x_bf16):
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

            # Pad activation chunk if it's smaller than 2048
            if x_chunk.shape[0] < target_K:
                x_input = np.zeros(target_K, dtype=bfloat16)
                x_input[:x_chunk.shape[0]] = x_chunk
            else:
                x_input = x_chunk

            self.x_gemv_t.numpy()[:] = x_input
            self.x_gemv_t._sync_to_device()

            b_start = start_col // 32
            b_end = end_col // 32
            W_col_slice = W_combined[:, b_start * 20 : b_end * 20]

            for start_row in range(0, N, target_N):
                end_row = min(start_row + target_N, N)
                W_chunk = W_col_slice[start_row:end_row]

                # Copy to resident NPU tensors
                self.w_gemv_t.numpy()[:] = W_chunk.reshape(-1)
                self.w_gemv_t._sync_to_device()

                # Execute JIT NPU kernel
                gemv_q_npu(self.w_gemv_t, self.x_gemv_t, self.y_gemv_t, N=2048, K=2048, m=32, k_tile=256)

                # Fetch result
                res = np.array(self.y_gemv_t.numpy()).astype(np.float32)
                y_sum[start_row:end_row] += res[:(end_row-start_row)]

        return y_sum.astype(bfloat16)

    def run_gemm_npu(self, W_combined, x_bf16):
        B = x_bf16.shape[0]
        N = W_combined.shape[0]
        K_blocks = W_combined.shape[1] // 20
        K = K_blocks * 32

        target_N = 2048
        target_K = 2048

        y_out = np.zeros((B, N), dtype=np.float32)

        # Loop over batch chunks
        for start_b in range(0, B, self.MAX_BATCH_SIZE):
            end_b = min(start_b + self.MAX_BATCH_SIZE, B)
            batch_chunk = x_bf16[start_b:end_b]
            
            y_sum = np.zeros((self.MAX_BATCH_SIZE, N), dtype=np.float32)

            # Loop over column chunks
            for start_col in range(0, K, target_K):
                end_col = min(start_col + target_K, K)
                x_chunk = batch_chunk[:, start_col:end_col]

                # Pad activation chunk if it's smaller than target
                if x_chunk.shape[1] < target_K or x_chunk.shape[0] < self.MAX_BATCH_SIZE:
                    x_input = np.zeros((self.MAX_BATCH_SIZE, target_K), dtype=bfloat16)
                    x_input[:x_chunk.shape[0], :x_chunk.shape[1]] = x_chunk
                else:
                    x_input = x_chunk

                self.x_gemm_t.numpy()[:] = x_input.reshape(-1)
                self.x_gemm_t._sync_to_device()

                b_start = start_col // 32
                b_end = end_col // 32
                W_col_slice = W_combined[:, b_start * 20 : b_end * 20]

                for start_row in range(0, N, target_N):
                    end_row = min(start_row + target_N, N)
                    W_chunk = W_col_slice[start_row:end_row]

                    # Copy to resident NPU tensors
                    self.w_gemv_t.numpy()[:] = W_chunk.reshape(-1)
                    self.w_gemv_t._sync_to_device()

                    # Execute JIT NPU kernel
                    gemm_q_npu(self.w_gemv_t, self.x_gemm_t, self.y_gemm_t, B=self.MAX_BATCH_SIZE, N=2048, K=2048, m=32, k_tile=256)

                    # Fetch result
                    res = np.array(self.y_gemm_t.numpy()).astype(np.float32).reshape(self.MAX_BATCH_SIZE, 2048)
                    y_sum[:, start_row:end_row] += res[:, :(end_row-start_row)]

            y_out[start_b:end_b] = y_sum[:(end_b-start_b)]

        return y_out.astype(bfloat16)

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
        if self.model_type in ["gemma3", "gemma4"]:
            eps = 1e-6
            variance = np.mean(x_bf16.astype(np.float32) ** 2)
            normed = x_bf16.astype(np.float32) * (1.0 / np.sqrt(variance + eps))
            if w_fp32 is not None:
                normed = normed * w_fp32
            return normed.astype(bfloat16)
        else:
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

    def run_rope_cpu_gemma(self, x_bf16, pos, base_freq):
        x_flat = x_bf16.reshape(-1)
        K = x_flat.shape[0]
        if self.model_type == "gemma3":
            dim = 256
        else:
            dim = 256 if base_freq == 10000.0 else 512
        num_heads = K // dim
        
        if base_freq == 10000.0:
            cos_sin = self.cos_sin_table_sliding[pos]
        else:
            cos_sin = self.cos_sin_table_full[pos]
            
        cos = cos_sin[:dim//2].astype(np.float32)
        sin = cos_sin[dim//2:].astype(np.float32)
        
        x_fp32 = x_flat.astype(np.float32).reshape(num_heads, dim)
        y_fp32 = np.zeros_like(x_fp32)
        
        for h in range(num_heads):
            x1 = x_fp32[h, :dim//2]
            x2 = x_fp32[h, dim//2:]
            y_fp32[h, :dim//2] = x1 * cos - x2 * sin
            y_fp32[h, dim//2:] = x2 * cos + x1 * sin
            
        return y_fp32.reshape(-1).astype(bfloat16)

    def run_attention_host(self, q_rope, pos, l):
        if self.model_type == "gemma4":
            is_sliding = (l + 1) % 6 != 0
            num_heads = 16
            num_kv_heads = 8 if is_sliding else 1
            dim = 256 if is_sliding else 512
            scale = 1.0
            window_size = 1024
        else:
            num_heads = self.num_attention_heads
            num_kv_heads = self.num_key_value_heads
            dim = self.head_dim
            scale = 1.0 / np.sqrt(dim)
            window_size = 512
            
        q = q_rope.astype(np.float32).reshape(num_heads, dim)
        seq_len = pos + 1
        
        # Sliding-window slicing for sliding layers
        is_sliding_layer = (self.model_type == "gemma3" and (l + 1) % 6 != 0) or (self.model_type == "gemma4" and (l + 1) % 6 != 0)
        if is_sliding_layer:
            W = min(seq_len, window_size)
            k = self.k_caches[l][:, pos + 1 - W : pos + 1, :].astype(np.float32)
            v = self.v_caches[l][:, pos + 1 - W : pos + 1, :].astype(np.float32)
        else:
            k = self.k_caches[l][:, :seq_len, :].astype(np.float32)
            v = self.v_caches[l][:, :seq_len, :].astype(np.float32)
            W = seq_len
            
        # Grouped query attention repetition
        group_ratio = num_heads // num_kv_heads
        if group_ratio > 1:
            k = np.repeat(k, group_ratio, axis=0)
            v = np.repeat(v, group_ratio, axis=0)
            
        # Compute attention scores
        scores = np.zeros((num_heads, W), dtype=np.float32)
        for h in range(num_heads):
            scores[h] = np.dot(k[h], q[h]) * scale
            
        # Softmax along last dimension
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Aggregation
        output = np.zeros((num_heads, dim), dtype=np.float32)
        for h in range(num_heads):
            output[h] = np.dot(probs[h], v[h])
            
        return output.reshape(-1).astype(bfloat16)

    def run_layer(self, x_bf16, pos, l, use_npu: bool = True):
        if self.model_type == "gemma3":
            # 1. Input RMSNorm
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.layer_attn_norms[l])
            
            # 2. QKV Projections (quantized weight GEMVs)
            if use_npu:
                q = self.run_gemv_npu(self.layer_weights[l]["attn_q"], x_norm)[:1024]
                k = self.run_gemv_npu(self.layer_weights[l]["attn_k"], x_norm)[:256]
                v = self.run_gemv_npu(self.layer_weights[l]["attn_v"], x_norm)[:256]
            else:
                x_norm_pad = np.zeros(2048, dtype=np.float32)
                x_norm_pad[:1152] = x_norm.astype(np.float32)
                q = (self.layer_weights_dequant[l]["attn_q"].astype(np.float32) @ x_norm_pad).astype(bfloat16)[:1024]
                k = (self.layer_weights_dequant[l]["attn_k"].astype(np.float32) @ x_norm_pad).astype(bfloat16)[:256]
                v = (self.layer_weights_dequant[l]["attn_v"].astype(np.float32) @ x_norm_pad).astype(bfloat16)[:256]
                
            # 3. QK-Norm
            q_normed = np.zeros_like(q)
            for h in range(4):
                q_h = q[h * 256 : (h + 1) * 256]
                q_normed[h * 256 : (h + 1) * 256] = self.run_rmsnorm_cpu(q_h, self.layer_q_norms[l])
            k_normed = self.run_rmsnorm_cpu(k, self.layer_k_norms[l])
            
            # 4. RoPE
            is_sliding = (l + 1) % 6 != 0
            base_freq = 10000.0 if is_sliding else 1000000.0
            q_rope = self.run_rope_cpu_gemma(q_normed, pos, base_freq)
            k_rope = self.run_rope_cpu_gemma(k_normed, pos, base_freq)
            
            # 5. Insert K and V into KV Cache
            self.k_caches[l][:, pos, :] = k_rope.reshape(1, 256)
            self.v_caches[l][:, pos, :] = v.reshape(1, 256)
            
            # 6. Attention
            attn_out = self.run_attention_host(q_rope, pos, l)
            
            # 7. Attention Output Projection
            if use_npu:
                attn_proj = self.run_gemv_npu(self.layer_weights[l]["attn_output"], attn_out)[:1152]
            else:
                attn_out_pad = np.zeros(2048, dtype=np.float32)
                attn_out_pad[:1024] = attn_out.astype(np.float32)
                attn_proj = (self.layer_weights_dequant[l]["attn_output"].astype(np.float32) @ attn_out_pad).astype(bfloat16)[:1152]
                
            # 8. Post-attention norm & residual add
            attn_proj_normed = self.run_rmsnorm_cpu(attn_proj, self.layer_post_attn_norms[l])
            x_post_attn = (x_bf16.astype(np.float32) + attn_proj_normed.astype(np.float32)).astype(bfloat16)
            
            # 9. Pre-FFN norm
            x_norm2 = self.run_rmsnorm_cpu(x_post_attn, self.layer_ffn_norms[l])
            
            # 10. MLP Gate & Up
            if use_npu:
                gate = self.run_gemv_npu(self.layer_weights[l]["ffn_gate"], x_norm2)[:6912]
                up = self.run_gemv_npu(self.layer_weights[l]["ffn_up"], x_norm2)[:6912]
            else:
                x_norm2_pad = np.zeros(2048, dtype=np.float32)
                x_norm2_pad[:1152] = x_norm2.astype(np.float32)
                gate = (self.layer_weights_dequant[l]["ffn_gate"].astype(np.float32) @ x_norm2_pad).astype(bfloat16)[:6912]
                up = (self.layer_weights_dequant[l]["ffn_up"].astype(np.float32) @ x_norm2_pad).astype(bfloat16)[:6912]
                
            # 11. GeGLU activation (gelu_pytorch_tanh)
            gate_fp32 = gate.astype(np.float32)
            up_fp32 = up.astype(np.float32)
            gelu_out = 0.5 * gate_fp32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate_fp32 + 0.044715 * (gate_fp32 ** 3))))
            geglu_out = (gelu_out * up_fp32).astype(bfloat16)
            
            # 12. MLP Down Projection
            if use_npu:
                down = self.run_gemv_npu(self.layer_weights[l]["ffn_down"], geglu_out)[:1152]
            else:
                geglu_out_pad = np.zeros(8192, dtype=np.float32)
                geglu_out_pad[:6912] = geglu_out.astype(np.float32)
                down = (self.layer_weights_dequant[l]["ffn_down"].astype(np.float32) @ geglu_out_pad).astype(bfloat16)[:1152]
                
            # 13. Post-FFN norm & second residual add
            down_normed = self.run_rmsnorm_cpu(down, self.layer_post_ffw_norms[l])
            y_final = (x_post_attn.astype(np.float32) + down_normed.astype(np.float32)).astype(bfloat16)
            
            return y_final
        elif self.model_type == "gemma4":
            # 1. Input RMSNorm
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.layer_attn_norms[l])
            
            # 2. QKV Projections (quantized weight GEMVs)
            is_sliding = (l + 1) % 6 != 0
            q_size = 4096 if is_sliding else 8192
            k_size = 2048 if is_sliding else 512
            v_size = 2048 if is_sliding else 512
            h_dim = 256 if is_sliding else 512
            n_heads = 16
            n_kv_heads = 8 if is_sliding else 1
            
            if use_npu:
                q = self.run_gemv_npu(self.layer_weights[l]["attn_q"], x_norm)[:q_size]
                k = self.run_gemv_npu(self.layer_weights[l]["attn_k"], x_norm)[:k_size]
                if is_sliding:
                    v = self.run_gemv_npu(self.layer_weights[l]["attn_v"], x_norm)[:v_size]
                else:
                    v = k
            else:
                # Dynamic on-the-fly dequantization for CPU path
                from tools.ref.gemv_q import dequantize_combined
                layer_w_dequant = {}
                for proj in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]:
                    if proj == "attn_v" and not is_sliding:
                        continue
                    layer_w_dequant[proj] = dequantize_combined(self.layer_weights[l][proj]).astype(bfloat16)

                x_norm_pad = np.zeros(4096, dtype=np.float32)
                x_norm_pad[:3840] = x_norm.astype(np.float32)
                q = (layer_w_dequant["attn_q"].astype(np.float32) @ x_norm_pad).astype(bfloat16)[:q_size]
                k = (layer_w_dequant["attn_k"].astype(np.float32) @ x_norm_pad).astype(bfloat16)[:k_size]
                if is_sliding:
                    v = (layer_w_dequant["attn_v"].astype(np.float32) @ x_norm_pad).astype(bfloat16)[:v_size]
                else:
                    v = k
                
            # 3. QK-Norm & V-Norm
            q_normed = np.zeros_like(q)
            for h in range(n_heads):
                q_h = q[h * h_dim : (h + 1) * h_dim]
                q_normed[h * h_dim : (h + 1) * h_dim] = self.run_rmsnorm_cpu(q_h, self.layer_q_norms[l])
                
            k_normed = np.zeros_like(k)
            for h in range(n_kv_heads):
                k_h = k[h * h_dim : (h + 1) * h_dim]
                k_normed[h * h_dim : (h + 1) * h_dim] = self.run_rmsnorm_cpu(k_h, self.layer_k_norms[l])
                
            v_normed = np.zeros_like(v)
            for h in range(n_kv_heads):
                v_h = v[h * h_dim : (h + 1) * h_dim]
                v_normed[h * h_dim : (h + 1) * h_dim] = self.run_rmsnorm_cpu(v_h, None)
                
            # 4. RoPE
            base_freq = 10000.0 if is_sliding else 1000000.0
            q_rope = self.run_rope_cpu_gemma(q_normed, pos, base_freq)
            k_rope = self.run_rope_cpu_gemma(k_normed, pos, base_freq)
            
            # 5. Insert K and V into KV Cache
            self.k_caches[l][:, pos, :] = k_rope.reshape(n_kv_heads, h_dim)
            self.v_caches[l][:, pos, :] = v_normed.reshape(n_kv_heads, h_dim)
            
            # 6. Attention
            attn_out = self.run_attention_host(q_rope, pos, l)
            
            # 7. Attention Output Projection
            attn_out_size = 4096 if is_sliding else 8192
            if use_npu:
                attn_proj = self.run_gemv_npu(self.layer_weights[l]["attn_output"], attn_out)[:3840]
            else:
                attn_out_pad = np.zeros(attn_out_size, dtype=np.float32)
                attn_out_pad[:len(attn_out)] = attn_out.astype(np.float32)
                attn_proj = (layer_w_dequant["attn_output"].astype(np.float32) @ attn_out_pad).astype(bfloat16)[:3840]
                
            # 8. Post-attention norm & residual add
            attn_proj_normed = self.run_rmsnorm_cpu(attn_proj, self.layer_post_attn_norms[l])
            x_post_attn = (x_bf16.astype(np.float32) + attn_proj_normed.astype(np.float32)).astype(bfloat16)
            
            # 9. Pre-FFN norm
            x_norm2 = self.run_rmsnorm_cpu(x_post_attn, self.layer_ffn_norms[l])
            
            # 10. MLP Gate & Up
            if use_npu:
                gate = self.run_gemv_npu(self.layer_weights[l]["ffn_gate"], x_norm2)[:15360]
                up = self.run_gemv_npu(self.layer_weights[l]["ffn_up"], x_norm2)[:15360]
            else:
                x_norm2_pad = np.zeros(4096, dtype=np.float32)
                x_norm2_pad[:3840] = x_norm2.astype(np.float32)
                gate = (layer_w_dequant["ffn_gate"].astype(np.float32) @ x_norm2_pad).astype(bfloat16)[:15360]
                up = (layer_w_dequant["ffn_up"].astype(np.float32) @ x_norm2_pad).astype(bfloat16)[:15360]
                
            # 11. GeGLU activation (gelu_pytorch_tanh)
            gate_fp32 = gate.astype(np.float32)
            up_fp32 = up.astype(np.float32)
            gelu_out = 0.5 * gate_fp32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate_fp32 + 0.044715 * (gate_fp32 ** 3))))
            geglu_out = (gelu_out * up_fp32).astype(bfloat16)
            
            # 12. MLP Down Projection
            if use_npu:
                down = self.run_gemv_npu(self.layer_weights[l]["ffn_down"], geglu_out)[:3840]
            else:
                geglu_out_pad = np.zeros(16384, dtype=np.float32)
                geglu_out_pad[:15360] = geglu_out.astype(np.float32)
                down = (layer_w_dequant["ffn_down"].astype(np.float32) @ geglu_out_pad).astype(bfloat16)[:3840]
                
            # 13. Post-FFN norm & second residual add
            down_normed = self.run_rmsnorm_cpu(down, self.layer_post_ffw_norms[l])
            y_final = (x_post_attn.astype(np.float32) + down_normed.astype(np.float32)).astype(bfloat16)
            
            # 14. Layer Output Scale
            y_final = (y_final.astype(np.float32) * self.layer_output_scales[l]).astype(bfloat16)
            
            # Clear lazy weight cache to reclaim memory immediately
            self.layer_weights[l].clear()
            
            return y_final
        else:
            # Llama-3.2 layer execution
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.layer_attn_norms[l])
            
            if use_npu:
                q = self.run_gemv_npu(self.layer_weights[l]["attn_q"], x_norm)
                k = self.run_gemv_npu(self.layer_weights[l]["attn_k"], x_norm)[:512]
                v = self.run_gemv_npu(self.layer_weights[l]["attn_v"], x_norm)[:512]
                q_rope = self.run_rope_cpu(q, pos)
                k_rope = self.run_rope_cpu(k, pos)
            else:
                q = (self.layer_weights_dequant[l]["attn_q"].astype(np.float32) @ x_norm.astype(np.float32)).astype(bfloat16)
                k = (self.layer_weights_dequant[l]["attn_k"].astype(np.float32) @ x_norm.astype(np.float32)).astype(bfloat16)[:512]
                v = (self.layer_weights_dequant[l]["attn_v"].astype(np.float32) @ x_norm.astype(np.float32)).astype(bfloat16)[:512]
                q_rope = self.run_rope_cpu(q, pos)
                k_rope = self.run_rope_cpu(k, pos)
                
            self.k_caches[l][:, pos, :] = k_rope.reshape(8, 64)
            self.v_caches[l][:, pos, :] = v.reshape(8, 64)
            
            attn_out = self.run_attention_host(q_rope, pos, l)
            
            if use_npu:
                attn_proj = self.run_gemv_npu(self.layer_weights[l]["attn_output"], attn_out)
            else:
                attn_proj = (self.layer_weights_dequant[l]["attn_output"].astype(np.float32) @ attn_out.astype(np.float32)).astype(bfloat16)
                
            x_post_attn = (x_bf16.astype(np.float32) + attn_proj.astype(np.float32)).astype(bfloat16)
            
            x_norm2 = self.run_rmsnorm_cpu(x_post_attn, self.layer_ffn_norms[l])
            
            if use_npu:
                gate = self.run_gemv_npu(self.layer_weights[l]["ffn_gate"], x_norm2)
                up = self.run_gemv_npu(self.layer_weights[l]["ffn_up"], x_norm2)
            else:
                gate = (self.layer_weights_dequant[l]["ffn_gate"].astype(np.float32) @ x_norm2.astype(np.float32)).astype(bfloat16)
                up = (self.layer_weights_dequant[l]["ffn_up"].astype(np.float32) @ x_norm2.astype(np.float32)).astype(bfloat16)
                
            gate_fp32 = gate.astype(np.float32)
            up_fp32 = up.astype(np.float32)
            silu_out = (gate_fp32 * (1.0 / (1.0 + np.exp(-gate_fp32)))) * up_fp32
            silu_out_bf16 = silu_out.astype(bfloat16)
            
            if use_npu:
                down = self.run_gemv_npu(self.layer_weights[l]["ffn_down"], silu_out_bf16)
            else:
                down = (self.layer_weights_dequant[l]["ffn_down"].astype(np.float32) @ silu_out_bf16.astype(np.float32)).astype(bfloat16)
                
            y_final = (x_post_attn.astype(np.float32) + down.astype(np.float32)).astype(bfloat16)
            
            return y_final

    def forward(self, token_id: int, pos: int, return_logits: bool = True, use_npu: bool = True) -> np.ndarray:
        if self.model_type == "gemma3":
            # 1. Embedding lookup scaled by sqrt(hidden_size)
            x_bf16 = (self.token_embd[token_id].astype(np.float32) * np.sqrt(self.hidden_size)).astype(bfloat16)
            
            # 2. Run layers
            for l in range(self.num_hidden_layers):
                x_bf16 = self.run_layer(x_bf16, pos, l, use_npu=use_npu)
                
            if not return_logits:
                return None
                
            # 3. Final norm
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.output_norm)
            
            # 4. LM Head (tied embedding quantized GEMV)
            if use_npu:
                logits = self.run_gemv_npu(self.lm_head, x_norm)
            else:
                x_norm_pad = np.zeros(2048, dtype=np.float32)
                x_norm_pad[:1152] = x_norm.astype(np.float32)
                logits = (self.lm_head_dequant.astype(np.float32) @ x_norm_pad).astype(bfloat16)
                
            return logits[:self.vocab_size].astype(np.float32)
        elif self.model_type == "gemma4":
            # 1. Embedding lookup scaled by sqrt(hidden_size)
            x_bf16 = (self.token_embd[token_id].astype(np.float32) * np.sqrt(self.hidden_size)).astype(bfloat16)
            
            # 2. Run layers
            for l in range(self.num_hidden_layers):
                x_bf16 = self.run_layer(x_bf16, pos, l, use_npu=use_npu)
                
            if not return_logits:
                return None
                
            # 3. Final norm
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.output_norm)
            
            # 4. LM Head (tied embedding quantized GEMV)
            if use_npu:
                logits = self.run_gemv_npu(self.lm_head, x_norm)
            else:
                from tools.ref.gemv_q import dequantize_combined
                lm_head_dequant = dequantize_combined(self.lm_head).astype(bfloat16)
                x_norm_pad = np.zeros(4096, dtype=np.float32)
                x_norm_pad[:3840] = x_norm.astype(np.float32)
                logits = (lm_head_dequant.astype(np.float32) @ x_norm_pad).astype(bfloat16)
                
            logits_unmasked = logits[:self.vocab_size].astype(np.float32)
            logits_softcapped = 30.0 * np.tanh(logits_unmasked / 30.0)
            return logits_softcapped
        else:
            # Llama-3.2
            x_bf16 = self.token_embd[token_id].astype(bfloat16)
            
            for l in range(16):
                x_bf16 = self.run_layer(x_bf16, pos, l, use_npu=use_npu)
                
            if not return_logits:
                return None
                
            x_norm = self.run_rmsnorm_cpu(x_bf16, self.output_norm)
            
            if use_npu:
                logits = self.run_gemv_npu(self.lm_head, x_norm)
            else:
                logits = (self.lm_head_dequant.astype(np.float32) @ x_norm.astype(np.float32)).astype(bfloat16)
                
            return logits[:128256].astype(np.float32)

    def run_rmsnorm_batch_cpu(self, x_bf16, w_fp32):
        x_fp32 = x_bf16.astype(np.float32)
        if self.model_type in ["gemma3", "gemma4"]:
            eps = 1e-6
            variance = np.mean(x_fp32 ** 2, axis=-1, keepdims=True)
            normed = x_fp32 * (1.0 / np.sqrt(variance + eps))
            if w_fp32 is not None:
                normed = normed * w_fp32
            return normed.astype(bfloat16)
        else:
            variance = np.mean(x_fp32 ** 2, axis=-1, keepdims=True)
            return (x_fp32 * (1.0 / np.sqrt(variance + 1e-5)) * w_fp32).astype(bfloat16)

    def run_rope_batch_cpu_gemma(self, x_bf16, pos_start, base_freq):
        B, K = x_bf16.shape
        if self.model_type == "gemma3":
            dim = 256
        else:
            dim = 256 if base_freq == 10000.0 else 512
        num_heads = K // dim
        
        pos_array = np.arange(pos_start, pos_start + B)
        if base_freq == 10000.0:
            cos_sin = self.cos_sin_table_sliding[pos_array]
        else:
            cos_sin = self.cos_sin_table_full[pos_array]
            
        cos = cos_sin[:, :dim//2].astype(np.float32)
        sin = cos_sin[:, dim//2:].astype(np.float32)
        
        x_fp32 = x_bf16.astype(np.float32).reshape(B, num_heads, dim)
        y_fp32 = np.zeros_like(x_fp32)
        
        for h in range(num_heads):
            x1 = x_fp32[:, h, :dim//2]
            x2 = x_fp32[:, h, dim//2:]
            y_fp32[:, h, :dim//2] = x1 * cos - x2 * sin
            y_fp32[:, h, dim//2:] = x2 * cos + x1 * sin
            
        return y_fp32.reshape(B, K).astype(bfloat16)

    def run_attention_batch_host(self, q_rope, pos_start, l):
        B = q_rope.shape[0]
        if self.model_type == "gemma4":
            is_sliding = (l + 1) % 6 != 0
            num_heads = 16
            num_kv_heads = 8 if is_sliding else 1
            dim = 256 if is_sliding else 512
            scale = 1.0
            window_size = 1024
        else:
            raise NotImplementedError()
            
        q = q_rope.astype(np.float32).reshape(B, num_heads, dim)
        output = np.zeros((B, num_heads, dim), dtype=np.float32)
        
        for b in range(B):
            pos = pos_start + b
            seq_len = pos + 1
            W = min(seq_len, window_size) if is_sliding else seq_len
            
            k = self.k_caches[l][:, pos + 1 - W : pos + 1, :].astype(np.float32)
            v = self.v_caches[l][:, pos + 1 - W : pos + 1, :].astype(np.float32)
            
            group_ratio = num_heads // num_kv_heads
            if group_ratio > 1:
                k = np.repeat(k, group_ratio, axis=0)
                v = np.repeat(v, group_ratio, axis=0)
                
            scores = np.zeros((num_heads, W), dtype=np.float32)
            for h in range(num_heads):
                scores[h] = np.dot(k[h], q[b, h]) * scale
                
            max_scores = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            
            for h in range(num_heads):
                output[b, h] = np.dot(probs[h], v[h])
                
        return output.reshape(B, -1).astype(bfloat16)

    def run_layer_batch(self, x_bf16, pos_start, l, use_npu=True):
        B = x_bf16.shape[0]
        if self.model_type == "gemma4":
            x_norm = self.run_rmsnorm_batch_cpu(x_bf16, self.layer_attn_norms[l])
            
            is_sliding = (l + 1) % 6 != 0
            q_size = 4096 if is_sliding else 8192
            k_size = 2048 if is_sliding else 512
            v_size = 2048 if is_sliding else 512
            h_dim = 256 if is_sliding else 512
            n_heads = 16
            n_kv_heads = 8 if is_sliding else 1
            
            if not use_npu:
                from tools.ref.gemv_q import dequantize_combined
                layer_w_dequant = {}
                for proj in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]:
                    if proj == "attn_v" and not is_sliding: continue
                    layer_w_dequant[proj] = dequantize_combined(self.layer_weights[l][proj]).astype(bfloat16)

            x_norm_pad = np.zeros((B, 4096), dtype=np.float32)
            x_norm_pad[:, :3840] = x_norm.astype(np.float32)
            
            if use_npu:
                q = self.run_gemm_npu(self.layer_weights[l]["attn_q"], x_norm_pad)[:, :q_size]
                k = self.run_gemm_npu(self.layer_weights[l]["attn_k"], x_norm_pad)[:, :k_size]
                if is_sliding:
                    v = self.run_gemm_npu(self.layer_weights[l]["attn_v"], x_norm_pad)[:, :v_size]
                else:
                    v = k
            else:
                q = (x_norm_pad @ layer_w_dequant["attn_q"].astype(np.float32).T).astype(bfloat16)[:, :q_size]
                k = (x_norm_pad @ layer_w_dequant["attn_k"].astype(np.float32).T).astype(bfloat16)[:, :k_size]
                if is_sliding:
                    v = (x_norm_pad @ layer_w_dequant["attn_v"].astype(np.float32).T).astype(bfloat16)[:, :v_size]
                else:
                    v = k
                
            q_normed = np.zeros_like(q)
            for h in range(n_heads):
                q_normed[:, h * h_dim : (h + 1) * h_dim] = self.run_rmsnorm_batch_cpu(q[:, h * h_dim : (h + 1) * h_dim], self.layer_q_norms[l])
            k_normed = np.zeros_like(k)
            for h in range(n_kv_heads):
                k_normed[:, h * h_dim : (h + 1) * h_dim] = self.run_rmsnorm_batch_cpu(k[:, h * h_dim : (h + 1) * h_dim], self.layer_k_norms[l])
            v_normed = np.zeros_like(v)
            for h in range(n_kv_heads):
                v_normed[:, h * h_dim : (h + 1) * h_dim] = self.run_rmsnorm_batch_cpu(v[:, h * h_dim : (h + 1) * h_dim], None)
                
            base_freq = 10000.0 if is_sliding else 1000000.0
            q_rope = self.run_rope_batch_cpu_gemma(q_normed, pos_start, base_freq)
            k_rope = self.run_rope_batch_cpu_gemma(k_normed, pos_start, base_freq)
            
            k_rope_reshaped = k_rope.reshape(B, n_kv_heads, h_dim)
            v_normed_reshaped = v_normed.reshape(B, n_kv_heads, h_dim)
            
            for b in range(B):
                self.k_caches[l][:, pos_start + b, :] = k_rope_reshaped[b]
                self.v_caches[l][:, pos_start + b, :] = v_normed_reshaped[b]
            
            attn_out = self.run_attention_batch_host(q_rope, pos_start, l)
            
            attn_out_size = 4096 if is_sliding else 8192
            attn_out_pad = np.zeros((B, attn_out_size), dtype=np.float32)
            attn_out_pad[:, :attn_out.shape[1]] = attn_out.astype(np.float32)
            if use_npu:
                attn_proj = self.run_gemm_npu(self.layer_weights[l]["attn_output"], attn_out_pad)[:, :3840]
            else:
                attn_proj = (attn_out_pad @ layer_w_dequant["attn_output"].astype(np.float32).T).astype(bfloat16)[:, :3840]
                
            attn_proj_normed = self.run_rmsnorm_batch_cpu(attn_proj, self.layer_post_attn_norms[l])
            x_post_attn = (x_bf16.astype(np.float32) + attn_proj_normed.astype(np.float32)).astype(bfloat16)
            
            x_norm2 = self.run_rmsnorm_batch_cpu(x_post_attn, self.layer_ffn_norms[l])
            
            x_norm2_pad = np.zeros((B, 4096), dtype=np.float32)
            x_norm2_pad[:, :3840] = x_norm2.astype(np.float32)
            if use_npu:
                gate = self.run_gemm_npu(self.layer_weights[l]["ffn_gate"], x_norm2_pad)[:, :15360]
                up = self.run_gemm_npu(self.layer_weights[l]["ffn_up"], x_norm2_pad)[:, :15360]
            else:
                gate = (x_norm2_pad @ layer_w_dequant["ffn_gate"].astype(np.float32).T).astype(bfloat16)[:, :15360]
                up = (x_norm2_pad @ layer_w_dequant["ffn_up"].astype(np.float32).T).astype(bfloat16)[:, :15360]
                
            gate_fp32 = gate.astype(np.float32)
            up_fp32 = up.astype(np.float32)
            gelu_out = 0.5 * gate_fp32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate_fp32 + 0.044715 * (gate_fp32 ** 3))))
            geglu_out = (gelu_out * up_fp32).astype(bfloat16)
            
            geglu_out_pad = np.zeros((B, 16384), dtype=np.float32)
            geglu_out_pad[:, :15360] = geglu_out.astype(np.float32)
            if use_npu:
                down = self.run_gemm_npu(self.layer_weights[l]["ffn_down"], geglu_out_pad)[:, :3840]
            else:
                down = (geglu_out_pad @ layer_w_dequant["ffn_down"].astype(np.float32).T).astype(bfloat16)[:, :3840]
                
            down_normed = self.run_rmsnorm_batch_cpu(down, self.layer_post_ffw_norms[l])
            y_final = (x_post_attn.astype(np.float32) + down_normed.astype(np.float32)).astype(bfloat16)
            
            y_final = (y_final.astype(np.float32) * self.layer_output_scales[l]).astype(bfloat16)
            self.layer_weights[l].clear()
            return y_final
        else:
            raise NotImplementedError()

    def forward_batch(self, token_ids: list[int], pos_start: int, use_npu: bool = True) -> np.ndarray:
        if self.model_type == "gemma4":
            x_bf16 = (self.token_embd[token_ids].astype(np.float32) * np.sqrt(self.hidden_size)).astype(bfloat16)
            for l in range(self.num_hidden_layers):
                x_bf16 = self.run_layer_batch(x_bf16, pos_start, l, use_npu=use_npu)
            return None
        else:
            raise NotImplementedError()
