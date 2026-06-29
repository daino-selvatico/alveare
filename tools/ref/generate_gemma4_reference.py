import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def gemma_rmsnorm(val, w, eps=1e-6):
    # val is numpy array
    variance = np.mean(val ** 2, axis=-1, keepdims=True)
    normed = val * (1.0 / np.sqrt(variance + eps))
    if w is not None:
        normed = normed * w
    return normed

def rotate_half(x):
    # x shape: (..., dim)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(x, cos, sin):
    # x shape: (num_heads, seq_len, dim)
    # cos, sin shape: (seq_len, dim)
    cos_unsq = np.expand_dims(cos, axis=0) # (1, seq_len, dim)
    sin_unsq = np.expand_dims(sin, axis=0) # (1, seq_len, dim)
    return (x * cos_unsq) + (rotate_half(x) * sin_unsq)

def gelu_pytorch_tanh(val):
    return 0.5 * val * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (val + 0.044715 * (val ** 3))))

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    weights_dir = Path(__file__).resolve().parents[2] / "quantized_weights_gemma4"
    out_dir = Path(__file__).resolve().parents[2] / "tools" / "ref" / "data_gemma4"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading unquantized weights from GGUF extraction...")
    # Since we saved dequantized float32 weights in quantized_weights_gemma4 as npy,
    # we can load them directly.
    # Note: token_embd is saved in float16, others are float32
    token_embd = np.load(weights_dir / "token_embd.npy")
    output_norm = np.load(weights_dir / "output_norm.weight.npy")
    
    # Layer 0 weights
    attn_norm = np.load(weights_dir / "blk.0.attn_norm.weight.npy")
    post_attn_norm = np.load(weights_dir / "blk.0.post_attention_norm.weight.npy")
    ffn_norm = np.load(weights_dir / "blk.0.ffn_norm.weight.npy")
    post_ffw_norm = np.load(weights_dir / "blk.0.post_ffw_norm.weight.npy")
    
    attn_q_norm = np.load(weights_dir / "blk.0.attn_q_norm.weight.npy")
    attn_k_norm = np.load(weights_dir / "blk.0.attn_k_norm.weight.npy")
    layer_output_scale = np.load(weights_dir / "blk.0.layer_output_scale.weight.npy")[0]
    
    # Linear projection weights (we dequantize from GGUF to FP32 in quantize_gemma4.py,
    # but let's load the dequantized weights from GGUF or dequantize them here from the GGUF)
    # Actually, in quantize_gemma4.py we did NOT save the unquantized weights for linear projections,
    # we only saved the quantized ones!
    # Wait, can we load the GGUF file and dequantize them here?
    # Yes, we can read the GGUF file directly in this script!
    print("Reading GGUF file to get unquantized projection weights...")
    sys.path.append("/home/daino/llama-mtp/llama.cpp/gguf-py")
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType
    
    reader = GGUFReader("/home/daino/llama-mtp/models/gemma-4-12b-it-UD-Q4_K_XL.gguf")
    
    def get_dequantized_weight(name):
        tensor = next(t for t in reader.tensors if t.name == name)
        qtype = GGMLQuantizationType(tensor.tensor_type)
        return dequantize(tensor.data, qtype).astype(np.float32)
        
    w_q = get_dequantized_weight("blk.0.attn_q.weight")
    w_k = get_dequantized_weight("blk.0.attn_k.weight")
    w_v = get_dequantized_weight("blk.0.attn_v.weight")
    w_o = get_dequantized_weight("blk.0.attn_output.weight")
    w_gate = get_dequantized_weight("blk.0.ffn_gate.weight")
    w_up = get_dequantized_weight("blk.0.ffn_up.weight")
    w_down = get_dequantized_weight("blk.0.ffn_down.weight")
    
    # Dimensions
    seq_len = 32
    hidden_size = 3840
    num_heads = 16
    num_kv_heads = 8
    head_dim = 256
    eps = 1e-6
    
    # Generate random input sequence (seq_len, hidden_size)
    x = np.random.randn(seq_len, hidden_size).astype(np.float32)
    
    # Run Layer 0 step-by-step
    # 1. Input Norm
    x_norm = gemma_rmsnorm(x, attn_norm, eps)
    
    # 2. Linear Projections (W shape is (out_features, in_features))
    # We project each token individually
    Q = x_norm @ w_q.T  # (seq_len, 4096)
    K = x_norm @ w_k.T  # (seq_len, 2048)
    V = x_norm @ w_v.T  # (seq_len, 2048)
    
    # Reshape Q, K, V
    Q_reshaped = Q.reshape(seq_len, num_heads, head_dim) # (seq_len, 16, 256)
    K_reshaped = K.reshape(seq_len, num_kv_heads, head_dim) # (seq_len, 8, 256)
    V_reshaped = V.reshape(seq_len, num_kv_heads, head_dim) # (seq_len, 8, 256)
    
    # 3. QK-Norm & V-Norm
    # Q and K are normalized with weights, V is normalized with unit weights (no weights)
    Q_normed = np.zeros_like(Q_reshaped)
    for h in range(num_heads):
        Q_normed[:, h, :] = gemma_rmsnorm(Q_reshaped[:, h, :], attn_q_norm, eps)
        
    K_normed = np.zeros_like(K_reshaped)
    for h in range(num_kv_heads):
        K_normed[:, h, :] = gemma_rmsnorm(K_reshaped[:, h, :], attn_k_norm, eps)
        
    V_normed = np.zeros_like(V_reshaped)
    for h in range(num_kv_heads):
        V_normed[:, h, :] = gemma_rmsnorm(V_reshaped[:, h, :], None, eps) # None means no weights
        
    # 4. RoPE
    # SWA layer 0 uses base_freq = 10000.0
    base_freq = 10000.0
    inv_freq = 1.0 / (base_freq ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    cos_sin_table = np.zeros((seq_len, head_dim), dtype=np.float32)
    for pos in range(seq_len):
        freqs = pos * inv_freq
        cos_sin_table[pos] = np.concatenate([np.cos(freqs), np.sin(freqs)])
        
    # Apply RoPE
    cos = cos_sin_table[:, :head_dim//2]
    sin = cos_sin_table[:, head_dim//2:]
    
    Q_rope = np.zeros_like(Q_normed)
    K_rope = np.zeros_like(K_normed)
    for h in range(num_heads):
        # Q_normed shape is (seq_len, 16, 256) -> transpose to (16, seq_len, 256) for apply_rotary
        # Let's just do it token by token or transpose
        q_h = Q_normed[:, h, :].T # (256, seq_len)
        # Wait, apply_rotary_pos_emb expects (num_heads, seq_len, head_dim)
        pass
    
    # Let's write apply_rotary directly for numpy:
    # x shape: (seq_len, num_heads, head_dim)
    def rotate_half_2d(val):
        half = val.shape[-1] // 2
        return np.concatenate((-val[..., half:], val[..., :half]), axis=-1)
        
    cos_expanded = np.expand_dims(cos, axis=1) # (seq_len, 1, 128)
    sin_expanded = np.expand_dims(sin, axis=1) # (seq_len, 1, 128)
    
    # We apply RoPE on the full head_dim (256), meaning we split it into half_dim (128)
    # x1 is (seq_len, num_heads, 128), x2 is (seq_len, num_heads, 128)
    Q_rope = np.zeros_like(Q_normed)
    K_rope = np.zeros_like(K_normed)
    
    for pos in range(seq_len):
        c = cos[pos]
        s = sin[pos]
        for h in range(num_heads):
            q_val = Q_normed[pos, h]
            q1 = q_val[:head_dim//2]
            q2 = q_val[head_dim//2:]
            Q_rope[pos, h, :head_dim//2] = q1 * c - q2 * s
            Q_rope[pos, h, head_dim//2:] = q2 * c + q1 * s
        for h in range(num_kv_heads):
            k_val = K_normed[pos, h]
            k1 = k_val[:head_dim//2]
            k2 = k_val[head_dim//2:]
            K_rope[pos, h, :head_dim//2] = k1 * c - k2 * s
            K_rope[pos, h, head_dim//2:] = k2 * c + k1 * s

    # 5. Attention scores
    # Gemma-4 attention scale is 1.0 (no scaling by 1/sqrt(head_dim))
    # We do Grouped-Query Attention
    # num_heads = 16, num_kv_heads = 8 -> ratio = 2
    ratio = num_heads // num_kv_heads
    K_rope_rep = np.repeat(K_rope, ratio, axis=1) # (seq_len, 16, 256)
    V_normed_rep = np.repeat(V_normed, ratio, axis=1) # (seq_len, 16, 256)
    
    # Attention scores shape: (16, seq_len, seq_len)
    attn_out = np.zeros((seq_len, num_heads, head_dim), dtype=np.float32)
    for t in range(seq_len):
        # Current token t queries all tokens up to t (causal)
        q_t = Q_rope[t] # (16, 256)
        k_past = K_rope_rep[:t+1] # (t+1, 16, 256)
        v_past = V_normed_rep[:t+1] # (t+1, 16, 256)
        
        for h in range(num_heads):
            # dot product of q_t[h] and k_past[:, h]
            scores = np.dot(k_past[:, h, :], q_t[h, :]) # shape (t+1,)
            # Softmax
            max_score = np.max(scores)
            exp_scores = np.exp(scores - max_score)
            probs = exp_scores / np.sum(exp_scores)
            
            # Weighted sum of v_past
            attn_out[t, h] = np.dot(probs, v_past[:, h, :])
            
    # Reshape back to (seq_len, hidden_size_proj) where hidden_size_proj = 16 * 256 = 4096
    attn_out_flat = attn_out.reshape(seq_len, num_heads * head_dim)
    
    # 6. Output Projection
    attn_proj = attn_out_flat @ w_o.T # (seq_len, hidden_size)
    
    # 7. Post-Attention Norm & residual add
    attn_proj_normed = gemma_rmsnorm(attn_proj, post_attn_norm, eps)
    x_post_attn = x + attn_proj_normed
    
    # 8. FFN pre-norm
    x_norm2 = gemma_rmsnorm(x_post_attn, ffn_norm, eps)
    
    # MLP projections
    gate = x_norm2 @ w_gate.T
    up = x_norm2 @ w_up.T
    
    # GeGLU activation
    geglu = gelu_pytorch_tanh(gate) * up
    down = geglu @ w_down.T
    
    # Post-FFN norm & second residual add
    down_normed = gemma_rmsnorm(down, post_ffw_norm, eps)
    y_layer = x_post_attn + down_normed
    
    # 9. Layer Output Scale
    y_final = y_layer * layer_output_scale
    
    print(f"Verify shapes: x_norm={x_norm.shape}, Q_rope={Q_rope.shape}, attn_out={attn_out.shape}, y_final={y_final.shape}")
    
    # Save the 32nd token (decode step, t = 31)
    t = 31
    np.save(out_dir / "input_hidden_states.npy", x[t])
    np.save(out_dir / "x_norm.npy", x_norm[t])
    
    # QKV Projections
    np.save(out_dir / "q_val.npy", Q[t])
    np.save(out_dir / "k_val.npy", K[t])
    np.save(out_dir / "v_val.npy", V[t])
    np.save(out_dir / "q_normed.npy", Q_normed[t].reshape(-1))
    np.save(out_dir / "k_normed.npy", K_normed[t].reshape(-1))
    np.save(out_dir / "v_normed.npy", V_normed[t].reshape(-1))
    
    # RoPE
    np.save(out_dir / "cos_val.npy", cos[t])
    np.save(out_dir / "sin_val.npy", sin[t])
    np.save(out_dir / "q_rope.npy", Q_rope[t].reshape(-1))
    np.save(out_dir / "k_rope.npy", K_rope[t].reshape(-1))
    
    # KV Cache state for sequence length 32
    # Store K_rope and V_normed up to pos 31
    np.save(out_dir / "k_cache.npy", K_rope[:32].transpose(1, 0, 2)) # (8, 32, 256)
    np.save(out_dir / "v_cache.npy", V_normed[:32].transpose(1, 0, 2)) # (8, 32, 256)
    
    # Attention Output
    np.save(out_dir / "attn_out.npy", attn_out[t].reshape(-1))
    np.save(out_dir / "attn_proj.npy", attn_proj[t])
    np.save(out_dir / "x_post_attn.npy", x_post_attn[t])
    
    # Pre-FFN Norm
    np.save(out_dir / "x_norm2.npy", x_norm2[t])
    
    # MLP projections
    np.save(out_dir / "gate.npy", gate[t])
    np.save(out_dir / "up.npy", up[t])
    np.save(out_dir / "geglu_out.npy", geglu[t])
    np.save(out_dir / "down.npy", down[t])
    
    # Output scale & final residual
    np.save(out_dir / "output_hidden_states.npy", y_final[t])
    
    print("Gemma-4 reference data saved successfully!")

if __name__ == "__main__":
    main()
