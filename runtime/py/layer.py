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
from kernels.attention.attention import attention_npu
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined

def run_gemv_q_unified(W_fp32, x_bf16):
    """
    Runs GEMV on the NPU using ONLY the compiled shape N=2048, K=2048.
    Supports any input shape by padding and chunking along both N and K.
    """
    N, K = W_fp32.shape
    m = 32
    k_tile = 256
    
    target_N = 2048
    target_K = 2048
    
    y_bf16_sum = np.zeros(N, dtype=np.float32)
    
    for start_col in range(0, K, target_K):
        end_col = min(start_col + target_K, K)
        W_k_slice = W_fp32[:, start_col:end_col]
        x_k_slice = x_bf16[start_col:end_col]
        
        # Chunk along N
        y_bf16_chunk = np.zeros(N, dtype=np.float32)
        for start_row in range(0, N, target_N):
            end_row = min(start_row + target_N, N)
            N_chunk = end_row - start_row
            
            W_chunk = W_k_slice[start_row:end_row]
            
            # Pad if N_chunk < target_N
            if N_chunk < target_N:
                W_input = np.zeros((target_N, target_K), dtype=np.float32)
                W_input[:N_chunk, :] = W_chunk
            else:
                W_input = W_chunk
                
            w_q4, scales = quantize_to_q4_0(W_input)
            w_combined = pack_to_combined(w_q4, scales)
            
            w_t = iron.tensor(w_combined.reshape(-1), dtype=np.uint8, device="npu")
            x_t = iron.tensor(x_k_slice.copy().astype(bfloat16), dtype=bfloat16, device="npu")
            y_t = iron.zeros(target_N, dtype=bfloat16, device="npu")
            
            gemv_q_npu(w_t, x_t, y_t, N=target_N, K=target_K, m=m, k_tile=k_tile)
            
            res = np.array(y_t.numpy())
            y_bf16_chunk[start_row:end_row] = res[:N_chunk].astype(np.float32)
            
        y_bf16_sum += y_bf16_chunk
        
    return y_bf16_sum.astype(bfloat16)

def run_rmsnorm_npu(x_bf16, w_fp32):
    """
    Executes the RMSNorm kernel on the NPU.
    """
    K = x_bf16.shape[0]
    
    x_t = iron.tensor(x_bf16.copy(), dtype=bfloat16, device="npu")
    w_t = iron.tensor(w_fp32.copy(), dtype=np.float32, device="npu")
    y_t = iron.zeros(K, dtype=bfloat16, device="npu")
    
    rmsnorm_npu(x_t, w_t, y_t, K=K)
    
    return np.array(y_t.numpy())

def run_rope_npu(x_bf16, cos_bf16, sin_bf16):
    """
    Executes the RoPE kernel on the NPU.
    """
    K = x_bf16.shape[0]
    head_dim = cos_bf16.shape[0]
    
    cos_sin = np.concatenate([cos_bf16, sin_bf16]).astype(bfloat16)
    
    x_t = iron.tensor(x_bf16.copy().astype(bfloat16), dtype=bfloat16, device="npu")
    cos_sin_t = iron.tensor(cos_sin, dtype=bfloat16, device="npu")
    y_t = iron.zeros(K, dtype=bfloat16, device="npu")
    
    rope_npu(x_t, cos_sin_t, y_t, K=K, head_dim=head_dim)
    
    return np.array(y_t.numpy())

def run_attention_npu(q_bf16, k_cache, v_cache, pos):
    """
    Executes the Attention kernel on the NPU using sliced Key and Value caches.
    """
    # q_bf16 is shape (2048,) -> reshape to (8, 4, 64)
    q_reshaped = q_bf16.reshape(8, 4, 64)
    
    seq_len = pos + 1
    head_dim = 64
    
    # Slice K and V cache up to current seq_len: shape (8, seq_len, 64)
    k_slice = k_cache[:, :seq_len, :]
    v_slice = v_cache[:, :seq_len, :]
    
    # Interleave Key and Value slices into packed KV cache: shape (8, seq_len, 128)
    kv_cache_packed = np.zeros((8, seq_len, head_dim * 2), dtype=np.float32).astype(bfloat16)
    kv_cache_packed[:, :, :head_dim] = k_slice
    kv_cache_packed[:, :, head_dim:] = v_slice
    
    # NPU tensors
    q_t = iron.tensor(q_reshaped.reshape(8, -1).copy(), dtype=bfloat16, device="npu")
    kv_t = iron.tensor(kv_cache_packed.reshape(8, -1).copy(), dtype=bfloat16, device="npu")
    o_t = iron.zeros((8, 4 * head_dim), dtype=bfloat16, device="npu")
    
    attention_npu(q_t, kv_t, o_t, seq_len=seq_len, head_dim=head_dim)
    
    # Reshape back to flat attention vector (2048,)
    return np.array(o_t.numpy()).reshape(-1)

def run_llama_layer(
    x_bf16,
    pos,
    k_cache,
    v_cache,
    weights
):
    """
    Executes a complete decoder layer of Llama 3.2.
    
    Args:
        x_bf16: Input hidden state vector of shape (2048,) in bfloat16.
        pos: Current token position in the KV cache (0-indexed).
        k_cache: Global Key cache in host DRAM of shape (8, max_seq_len, 64) in bfloat16.
        v_cache: Global Value cache in host DRAM of shape (8, max_seq_len, 64) in bfloat16.
        weights: Dictionary containing the unquantized FP32 weights for this layer.
    """
    # 1. Input RMSNorm
    print("Layer trace: running input RMSNorm...")
    x_norm = run_rmsnorm_npu(x_bf16, weights["attn_norm"])
    print("Layer trace: input RMSNorm completed.")
    
    # 2. QKV Projections (quantized weight GEMVs)
    print("Layer trace: running Q projection...")
    q = run_gemv_q_unified(weights["w_q"], x_norm)
    print("Layer trace: running K projection...")
    k = run_gemv_q_unified(weights["w_k"], x_norm)
    print("Layer trace: running V projection...")
    v = run_gemv_q_unified(weights["w_v"], x_norm)
    print("Layer trace: QKV projections completed.")
    
    # 3. Apply RoPE (Query and Key)
    print("Layer trace: running Q RoPE...")
    q_rope = run_rope_npu(q, weights["cos"][pos], weights["sin"][pos])
    print("Layer trace: running K RoPE...")
    k_rope = run_rope_npu(k, weights["cos"][pos], weights["sin"][pos])
    print("Layer trace: RoPE completed.")
    
    # 4. Insert K and V into host KV cache
    print("Layer trace: inserting K and V into KV Cache...")
    k_cache[:, pos, :] = k_rope.reshape(8, 64)
    v_cache[:, pos, :] = v.reshape(8, 64)
    
    # 5. Attention (QKᵀ -> softmax -> ·V)
    print("Layer trace: running Attention...")
    attn_out = run_attention_npu(q_rope, k_cache, v_cache, pos)
    print("Layer trace: Attention completed.")
    
    # 6. Attention Output Projection
    print("Layer trace: running Output projection...")
    attn_proj = run_gemv_q_unified(weights["w_o"], attn_out)
    print("Layer trace: Output projection completed.")
    
    # 7. First Residual Connection (host-side)
    x_post_attn = (x_bf16.astype(np.float32) + attn_proj.astype(np.float32)).astype(bfloat16)
    
    # 8. Post-attention RMSNorm
    print("Layer trace: running post-attention RMSNorm...")
    x_norm2 = run_rmsnorm_npu(x_post_attn, weights["ffn_norm"])
    print("Layer trace: post-attention RMSNorm completed.")
    
    # 9. MLP Projections (Gate & Up)
    print("Layer trace: running MLP Gate projection...")
    gate = run_gemv_q_unified(weights["w_gate"], x_norm2)
    print("Layer trace: running MLP Up projection...")
    up = run_gemv_q_unified(weights["w_up"], x_norm2)
    print("Layer trace: MLP Gate/Up projections completed.")
    
    # 10. MLP Activation (SwiGLU, host-side in NumPy)
    gate_fp32 = gate.astype(np.float32)
    up_fp32 = up.astype(np.float32)
    silu_out = (gate_fp32 * (1.0 / (1.0 + np.exp(-gate_fp32)))) * up_fp32
    silu_out_bf16 = silu_out.astype(bfloat16)
    
    # 11. MLP Down Projection
    print("Layer trace: running MLP Down projection...")
    down = run_gemv_q_unified(weights["w_down"], silu_out_bf16)
    print("Layer trace: MLP Down projection completed.")
    
    # 12. Second Residual Connection (host-side)
    y_final = (x_post_attn.astype(np.float32) + down.astype(np.float32)).astype(bfloat16)
    
    return y_final
