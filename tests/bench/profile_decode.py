import sys
import time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from runtime.py.model import LlamaNPUModel
from runtime.py.tokenizer_glue import TokenizerGlue
from kernels.gemv_q.gemv_q import gemv_q_npu

class ProfiledModel(LlamaNPUModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_profiling()
        
    def reset_profiling(self):
        self.timings = {
            "gemv_calls": 0,
            "gemv_raw_npu_ms": 0.0,
            "gemv_pyxrt_overhead_ms": 0.0,
            "gemv_sync_copy_ms": 0.0,
            "gemv_total_ms": 0.0,
            
            "lm_head_gemv_total_ms": 0.0,
            
            "rmsnorm_ms": 0.0,
            "rope_ms": 0.0,
            "attention_ms": 0.0,
            "swiglu_ms": 0.0,
            "other_cpu_ms": 0.0,
            "total_forward_ms": 0.0
        }
        self.is_profiling = False
        self.in_lm_head = False

    def run_gemv_npu(self, W_combined, x_bf16):
        if not self.is_profiling:
            return super().run_gemv_npu(W_combined, x_bf16)
            
        N = W_combined.shape[0]
        K_blocks = W_combined.shape[1] // 20
        K = K_blocks * 32
        
        target_N = 2048
        target_K = 2048
        
        y_sum = np.zeros(N, dtype=np.float32)
        
        t_gemv_start = time.perf_counter()
        
        # Loop over column chunks
        for start_col in range(0, K, target_K):
            end_col = min(start_col + target_K, K)
            
            t0 = time.perf_counter()
            x_chunk = x_bf16[start_col:end_col]
            
            # Column slice of combined weights
            b_start = start_col // 32
            b_end = end_col // 32
            W_col_slice = W_combined[:, b_start * 20 : b_end * 20]
            t1 = time.perf_counter()
            self.timings["other_cpu_ms"] += (t1 - t0) * 1000.0
            
            # Loop over row chunks
            for start_row in range(0, N, target_N):
                end_row = min(start_row + target_N, N)
                W_chunk = W_col_slice[start_row:end_row]
                
                # Copy to resident NPU tensors (sync/copy time)
                t_sync_start = time.perf_counter()
                self.w_gemv_t.numpy()[:] = W_chunk.reshape(-1)
                self.w_gemv_t._sync_to_device()
                
                self.x_gemv_t.numpy()[:] = x_chunk
                self.x_gemv_t._sync_to_device()
                t_sync_end = time.perf_counter()
                self.timings["gemv_sync_copy_ms"] += (t_sync_end - t_sync_start) * 1000.0
                
                # Execute JIT NPU kernel
                t_kernel_start = time.perf_counter()
                ret = gemv_q_npu(self.w_gemv_t, self.x_gemv_t, self.y_gemv_t, N=2048, K=2048, m=32, k_tile=256)
                t_kernel_end = time.perf_counter()
                
                # Extract NPU time vs PyXRT overhead
                e2e_ms = (t_kernel_end - t_kernel_start) * 1000.0
                npu_ns = getattr(ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret, "npu_time", None)
                if npu_ns is not None:
                    npu_ms = npu_ns / 1_000_000.0
                else:
                    npu_ms = 0.0 # fallback
                
                self.timings["gemv_raw_npu_ms"] += npu_ms
                self.timings["gemv_pyxrt_overhead_ms"] += (e2e_ms - npu_ms)
                self.timings["gemv_calls"] += 1
                
                # Fetch result safely (sync/copy time)
                t_fetch_start = time.perf_counter()
                res = np.array(self.y_gemv_t.numpy()).astype(np.float32)
                t_fetch_end = time.perf_counter()
                self.timings["gemv_sync_copy_ms"] += (t_fetch_end - t_fetch_start) * 1000.0
                
                t0 = time.perf_counter()
                y_sum[start_row:end_row] += res[:(end_row - start_row)]
                t1 = time.perf_counter()
                self.timings["other_cpu_ms"] += (t1 - t0) * 1000.0
                
        t_gemv_end = time.perf_counter()
        gemv_dur_ms = (t_gemv_end - t_gemv_start) * 1000.0
        self.timings["gemv_total_ms"] += gemv_dur_ms
        if self.in_lm_head:
            self.timings["lm_head_gemv_total_ms"] += gemv_dur_ms
            
        return y_sum.astype(bfloat16)

    def run_rmsnorm_cpu(self, x_bf16, w_fp32):
        if not self.is_profiling:
            return super().run_rmsnorm_cpu(x_bf16, w_fp32)
        t0 = time.perf_counter()
        res = super().run_rmsnorm_cpu(x_bf16, w_fp32)
        t1 = time.perf_counter()
        self.timings["rmsnorm_ms"] += (t1 - t0) * 1000.0
        return res

    def run_rope_cpu(self, x_bf16, pos):
        if not self.is_profiling:
            return super().run_rope_cpu(x_bf16, pos)
        t0 = time.perf_counter()
        res = super().run_rope_cpu(x_bf16, pos)
        t1 = time.perf_counter()
        self.timings["rope_ms"] += (t1 - t0) * 1000.0
        return res

    def run_attention_host(self, q_rope, pos, l):
        if not self.is_profiling:
            return super().run_attention_host(q_rope, pos, l)
        t0 = time.perf_counter()
        res = super().run_attention_host(q_rope, pos, l)
        t1 = time.perf_counter()
        self.timings["attention_ms"] += (t1 - t0) * 1000.0
        return res

    def run_layer(self, x_bf16, pos, l, use_npu=True):
        if not self.is_profiling:
            return super().run_layer(x_bf16, pos, l, use_npu=use_npu)
            
        # 1. Input RMSNorm
        x_norm = self.run_rmsnorm_cpu(x_bf16, self.layer_attn_norms[l])
        
        # 2. QKV Projections
        q = self.run_gemv_npu(self.layer_weights[l]["attn_q"], x_norm)
        k = self.run_gemv_npu(self.layer_weights[l]["attn_k"], x_norm)[:512]
        v = self.run_gemv_npu(self.layer_weights[l]["attn_v"], x_norm)[:512]
        
        # 3. Apply RoPE
        q_rope = self.run_rope_cpu(q, pos)
        k_rope = self.run_rope_cpu(k, pos)
        
        # 4. KV cache insertion
        t0 = time.perf_counter()
        self.k_caches[l][:, pos, :] = k_rope.reshape(8, 64)
        self.v_caches[l][:, pos, :] = v.reshape(8, 64)
        t1 = time.perf_counter()
        self.timings["other_cpu_ms"] += (t1 - t0) * 1000.0
        
        # 5. Attention
        attn_out = self.run_attention_host(q_rope, pos, l)
        
        # 6. Attention Output Projection
        attn_proj = self.run_gemv_npu(self.layer_weights[l]["attn_output"], attn_out)
        
        # 7. First Residual Connection
        t0 = time.perf_counter()
        x_post_attn = (x_bf16.astype(np.float32) + attn_proj.astype(np.float32)).astype(bfloat16)
        t1 = time.perf_counter()
        self.timings["other_cpu_ms"] += (t1 - t0) * 1000.0
        
        # 8. Post-attention RMSNorm
        x_norm2 = self.run_rmsnorm_cpu(x_post_attn, self.layer_ffn_norms[l])
        
        # 9. MLP Projections (Gate & Up)
        gate = self.run_gemv_npu(self.layer_weights[l]["ffn_gate"], x_norm2)
        up = self.run_gemv_npu(self.layer_weights[l]["ffn_up"], x_norm2)
        
        # 10. SwiGLU Activation
        t_swi_start = time.perf_counter()
        gate_fp32 = gate.astype(np.float32)
        up_fp32 = up.astype(np.float32)
        silu_out = (gate_fp32 * (1.0 / (1.0 + np.exp(-gate_fp32)))) * up_fp32
        silu_out_bf16 = silu_out.astype(bfloat16)
        t_swi_end = time.perf_counter()
        self.timings["swiglu_ms"] += (t_swi_end - t_swi_start) * 1000.0
        
        # 11. MLP Down Projection
        down = self.run_gemv_npu(self.layer_weights[l]["ffn_down"], silu_out_bf16)
        
        # 12. Second Residual Connection
        t0 = time.perf_counter()
        y_final = (x_post_attn.astype(np.float32) + down.astype(np.float32)).astype(bfloat16)
        t1 = time.perf_counter()
        self.timings["other_cpu_ms"] += (t1 - t0) * 1000.0
        
        return y_final

    def forward(self, token_id: int, pos: int, return_logits: bool = True, use_npu: bool = True) -> np.ndarray:
        if not self.is_profiling:
            return super().forward(token_id, pos, return_logits, use_npu)
            
        t_fwd_start = time.perf_counter()
        
        # 1. Embedding lookup
        t0 = time.perf_counter()
        x_bf16 = self.token_embd[token_id].astype(bfloat16)
        t1 = time.perf_counter()
        self.timings["other_cpu_ms"] += (t1 - t0) * 1000.0
        
        # 2. Run layers
        for l in range(16):
            x_bf16 = self.run_layer(x_bf16, pos, l, use_npu=use_npu)
            
        if not return_logits:
            return None
            
        # 3. Final norm
        x_norm = self.run_rmsnorm_cpu(x_bf16, self.output_norm)
        
        # 4. LM Head (chunked GEMV)
        self.in_lm_head = True
        logits = self.run_gemv_npu(self.lm_head, x_norm)
        self.in_lm_head = False
        
        # Slice to vocabulary size
        t0 = time.perf_counter()
        res = logits[:128256].astype(np.float32)
        t1 = time.perf_counter()
        self.timings["other_cpu_ms"] += (t1 - t0) * 1000.0
        
        t_fwd_end = time.perf_counter()
        self.timings["total_forward_ms"] = (t_fwd_end - t_fwd_start) * 1000.0
        return res

def main():
    weights_dir = Path(__file__).resolve().parents[2] / "quantized_weights"
    print("Initializing profiled model...")
    model = ProfiledModel(weights_dir)
    tokenizer = TokenizerGlue()
    
    prompt_messages = [{"role": "user", "content": "capital of France is"}]
    input_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
    
    print("Running prefill on CPU...")
    model.reset_caches()
    for pos in range(len(input_ids) - 1):
        model.forward(input_ids[pos], pos, return_logits=False, use_npu=False)
        
    current_token_id = input_ids[-1]
    pos = len(input_ids) - 1
    
    print("Running decode warmup step 0 (NPU)...")
    logits = model.forward(current_token_id, pos, use_npu=True)
    next_token_id = int(np.argmax(logits))
    current_token_id = next_token_id
    pos += 1
    
    print("Running decode warmup step 1 (NPU)...")
    logits = model.forward(current_token_id, pos, use_npu=True)
    next_token_id = int(np.argmax(logits))
    current_token_id = next_token_id
    pos += 1
    
    print("Running profiled decode step 2 (NPU)...")
    model.reset_profiling()
    model.is_profiling = True
    
    logits = model.forward(current_token_id, pos, use_npu=True)
    
    model.is_profiling = False
    
    # Print and save profile
    t = model.timings
    total_measured_ms = t["total_forward_ms"]
    
    gemv_total = t["gemv_total_ms"]
    raw_npu = t["gemv_raw_npu_ms"]
    pyxrt_overhead = t["gemv_pyxrt_overhead_ms"]
    sync_copy = t["gemv_sync_copy_ms"]
    lm_head = t["lm_head_gemv_total_ms"]
    
    cpu_ops = t["rmsnorm_ms"] + t["rope_ms"] + t["attention_ms"] + t["swiglu_ms"]
    other_cpu = t["other_cpu_ms"]
    
    # Calculate sum of all reported to see consistency
    reported_sum = raw_npu + pyxrt_overhead + sync_copy + cpu_ops + other_cpu
    
    output = []
    output.append("=== ALVEARE DECODE STEP PROFILE ===")
    output.append(f"Total step latency: {total_measured_ms:.2f} ms")
    output.append(f"GEMV calls count: {t['gemv_calls']}")
    output.append("")
    output.append(f"- raw NPU GEMV compute (sum over all calls): {raw_npu:.2f} ms ({raw_npu / total_measured_ms * 100:.1f}%)")
    output.append(f"- per-call host/PyXRT overhead x calls: {pyxrt_overhead:.2f} ms ({pyxrt_overhead / total_measured_ms * 100:.1f}%)")
    output.append(f"- host<->device tensor sync/copy time: {sync_copy:.2f} ms ({sync_copy / total_measured_ms * 100:.1f}%)")
    output.append(f"- CPU light ops total: {cpu_ops:.2f} ms ({cpu_ops / total_measured_ms * 100:.1f}%)")
    output.append(f"  * rmsnorm: {t['rmsnorm_ms']:.2f} ms")
    output.append(f"  * rope: {t['rope_ms']:.2f} ms")
    output.append(f"  * attention/softmax: {t['attention_ms']:.2f} ms")
    output.append(f"  * swiglu: {t['swiglu_ms']:.2f} ms")
    output.append(f"- LM head GEMV specifically (chunked): {lm_head:.2f} ms ({lm_head / total_measured_ms * 100:.1f}%)")
    output.append(f"- Other CPU overhead (data prep/other): {other_cpu:.2f} ms ({other_cpu / total_measured_ms * 100:.1f}%)")
    
    profile_text = "\n".join(output)
    print(profile_text)
    
    # Write to target path
    profile_path = Path(__file__).resolve().parents[2] / "tests" / "bench" / "token_profile.txt"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w") as f:
        f.write(profile_text + "\n")
    print(f"\nProfile written to {profile_path}")

if __name__ == "__main__":
    main()
