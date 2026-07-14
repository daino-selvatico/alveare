import sys
import time
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from runtime.py.model import LlamaNPUModel, LazyLayerWeights
from runtime.py.tokenizer_glue import TokenizerGlue
from kernels.gemv_q.gemv_q import gemv_q_npu

class Profiler:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.timings = {
            "gemv_calls": 0,
            "gemv_raw_npu_ms": 0.0,
            "gemv_pyxrt_overhead_ms": 0.0,
            "gemv_sync_copy_ms": 0.0,
            "gemv_total_ms": 0.0,
            
            "lm_head_gemv_total_ms": 0.0,
            "weight_load_ms": 0.0,
            "gemv_padding_waste_ms": 0.0,
            
            "total_ms": 0.0
        }
        self.is_profiling = False
        self.in_lm_head = False

profiler = Profiler()

# Monkey patch model methods
orig_gemv = LlamaNPUModel.run_gemv_npu
orig_lazy_get = LazyLayerWeights.__getitem__

def patched_lazy_get(self, key):
    if not profiler.is_profiling:
        return orig_lazy_get(self, key)
    t0 = time.perf_counter()
    res = orig_lazy_get(self, key)
    t1 = time.perf_counter()
    profiler.timings["weight_load_ms"] += (t1 - t0) * 1000.0
    return res

LazyLayerWeights.__getitem__ = patched_lazy_get

def patched_gemv(self, W_combined, x_bf16):
    if not profiler.is_profiling:
        return orig_gemv(self, W_combined, x_bf16)
        
    N = W_combined.shape[0]
    K_blocks = W_combined.shape[1] // 20
    K = K_blocks * 32
    
    target_N = 2048
    target_K = K
    
    if K == 4096:
        w_t = self.w_gemv_t_k4096
        x_t = self.x_gemv_t_k4096
    elif K == 8192:
        w_t = self.w_gemv_t_k8192
        x_t = self.x_gemv_t_k8192
    elif K == 16384:
        w_t = self.w_gemv_t_k16384
        x_t = self.x_gemv_t_k16384
    else:
        raise ValueError(f"Unsupported K dimension {K}")
    
    y_sum = np.zeros(N, dtype=np.float32)
    t_gemv_start = time.perf_counter()
    
    t_cpu_start = time.perf_counter()
    if x_bf16.shape[0] < K:
        x_input = np.zeros(K, dtype=bfloat16)
        x_input[:x_bf16.shape[0]] = x_bf16
    else:
        x_input = x_bf16
        
    t_sync_start = time.perf_counter()
    x_t.numpy()[:] = x_input
    x_t._sync_to_device()
    t_sync_end = time.perf_counter()
    profiler.timings["gemv_sync_copy_ms"] += (t_sync_end - t_sync_start) * 1000.0
        
    for start_row in range(0, N, target_N):
        end_row = min(start_row + target_N, N)
        
        W_chunk = W_combined[start_row:end_row]
        
        t_sync_start = time.perf_counter()
        w_t.numpy()[:] = W_chunk.reshape(-1)
        w_t._sync_to_device()
        t_sync_end = time.perf_counter()
        profiler.timings["gemv_sync_copy_ms"] += (t_sync_end - t_sync_start) * 1000.0
        
        t_kernel_start = time.perf_counter()
        ret = gemv_q_npu(w_t, x_t, self.y_gemv_t, N=2048, K=K, m=32, k_tile=256)
        t_kernel_end = time.perf_counter()
        
        e2e_ms = (t_kernel_end - t_kernel_start) * 1000.0
        npu_ns = getattr(ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret, "npu_time", None)
        if npu_ns is not None:
            npu_ms = npu_ns / 1_000_000.0
        else:
            npu_ms = 0.0
            
        profiler.timings["gemv_raw_npu_ms"] += npu_ms
        profiler.timings["gemv_pyxrt_overhead_ms"] += (e2e_ms - npu_ms)
        profiler.timings["gemv_calls"] += 1
        
        actual_N = end_row - start_row
        actual_K = K
        waste_ratio = 1.0 - (actual_N * actual_K) / (2048 * K)
        profiler.timings["gemv_padding_waste_ms"] += npu_ms * waste_ratio
        
        t_fetch_start = time.perf_counter()
        res = np.array(self.y_gemv_t.numpy()).astype(np.float32)
        t_fetch_end = time.perf_counter()
        profiler.timings["gemv_sync_copy_ms"] += (t_fetch_end - t_fetch_start) * 1000.0
        
        y_sum[start_row:end_row] += res[:(end_row - start_row)]
        
    t_gemv_end = time.perf_counter()
    gemv_dur_ms = (t_gemv_end - t_gemv_start) * 1000.0
    profiler.timings["gemv_total_ms"] += gemv_dur_ms
    if profiler.in_lm_head:
        profiler.timings["lm_head_gemv_total_ms"] += gemv_dur_ms
        
    return y_sum.astype(bfloat16)

LlamaNPUModel.run_gemv_npu = patched_gemv

def main():
    weights_dir = Path(__file__).resolve().parents[2] / "quantized_weights_gemma4"
    print("Initializing model...")
    model = LlamaNPUModel(weights_dir)
    tokenizer = TokenizerGlue("google/gemma-4-12b-it")
    
    prompt_messages = [{"role": "user", "content": "The capital of France is"}]
    input_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
    
    print("Running prefill on NPU...")
    model.reset_caches()
    
    # Profile prefill
    profiler.reset()
    profiler.is_profiling = True
    t_start = time.perf_counter()
    for pos in range(len(input_ids) - 1):
        model.forward(input_ids[pos], pos, return_logits=False, use_npu=True)
    t_end = time.perf_counter()
    profiler.is_profiling = False
    profiler.timings["total_ms"] = (t_end - t_start) * 1000.0
    
    print_profile("PREFILL (17 tokens)", profiler.timings)
    
    current_token_id = input_ids[-1]
    pos = len(input_ids) - 1
    
    print("Running profiled decode step (NPU)...")
    profiler.reset()
    profiler.is_profiling = True
    
    t_start = time.perf_counter()
    # To catch lm_head:
    def patched_forward(self, token_id, pos, return_logits=True, use_npu=True):
        # We know we just call orig forward and set flag for lm_head
        # Actually, in Gemma4, LM head is run inside forward.
        # We can just intercept the call or wrap it.
        # The easiest is to just let forward run, but we want to know when it's in lm head.
        pass
        
    # We will just patch forward temporarily to set in_lm_head for the last gemv.
    # Actually, in model.py Gemma4 LM head runs at the end:
    #             logits = self.run_gemv_npu(self.lm_head, x_norm)
    # We can't easily hook just that, but we can hook run_gemv_npu to check if W_combined is self.lm_head
    orig_fwd = model.forward
    def forward_with_lm_check(token_id, pos, return_logits=True, use_npu=True):
        # we don't need to patch, we can check if W_combined is self.lm_head inside patched_gemv
        return orig_fwd(token_id, pos, return_logits, use_npu)
    
    # update patched_gemv to check lm_head
    global patched_gemv
    def new_patched_gemv(self, W_combined, x_bf16):
        if W_combined is self.lm_head:
            profiler.in_lm_head = True
        res = patched_gemv(self, W_combined, x_bf16)
        profiler.in_lm_head = False
        return res
    LlamaNPUModel.run_gemv_npu = new_patched_gemv
    
    logits = model.forward(current_token_id, pos, use_npu=True)
    
    t_end = time.perf_counter()
    profiler.is_profiling = False
    profiler.timings["total_ms"] = (t_end - t_start) * 1000.0
    
    print_profile("DECODE STEP 1", profiler.timings)
    next_token_id = int(np.argmax(logits))
    print(f"Predicted token: {next_token_id}")

def print_profile(name, t):
    total_measured_ms = t["total_ms"]
    gemv_total = t["gemv_total_ms"]
    raw_npu = t["gemv_raw_npu_ms"]
    pyxrt_overhead = t["gemv_pyxrt_overhead_ms"]
    sync_copy = t["gemv_sync_copy_ms"]
    lm_head = t["lm_head_gemv_total_ms"]
    weight_load = t["weight_load_ms"]
    
    other_ms = total_measured_ms - (gemv_total + weight_load)
    
    output = []
    output.append(f"=== ALVEARE {name} PROFILE ===")
    output.append(f"Total latency: {total_measured_ms:.2f} ms")
    output.append(f"GEMV calls count: {t['gemv_calls']}")
    output.append("")
    output.append(f"- weight streaming I/O (LazyLayerWeights load): {weight_load:.2f} ms ({weight_load / total_measured_ms * 100:.1f}%)")
    output.append(f"- raw NPU GEMV compute (sum over all calls): {raw_npu:.2f} ms ({raw_npu / total_measured_ms * 100:.1f}%)")
    output.append(f"  └-- padding waste (zeros matmul within raw NPU): {t['gemv_padding_waste_ms']:.2f} ms")
    output.append(f"- per-call host/PyXRT overhead x calls: {pyxrt_overhead:.2f} ms ({pyxrt_overhead / total_measured_ms * 100:.1f}%)")
    output.append(f"- host<->device tensor sync/copy time: {sync_copy:.2f} ms ({sync_copy / total_measured_ms * 100:.1f}%)")
    output.append(f"- LM head GEMV specifically: {lm_head:.2f} ms ({lm_head / total_measured_ms * 100:.1f}%)")
    output.append(f"- Other CPU overhead (data prep/other): {other_ms:.2f} ms ({other_ms / total_measured_ms * 100:.1f}%)")
    
    profile_text = "\n".join(output)
    print(profile_text)
    
    profile_path = Path(__file__).resolve().parents[2] / "tests" / "bench" / "gemma4_profile.txt"
    with open(profile_path, "a") as f:
        f.write(profile_text + "\n\n")

if __name__ == "__main__":
    main()
