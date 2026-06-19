import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    model_id = "unsloth/Llama-3.2-1B-Instruct"
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    layer = model.model.layers[0]

    seq_len = 32
    hidden_size = 2048
    
    # Generate random input sequence (1, seq_len, hidden_size)
    x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
    
    # Run the RMSNorm before attention
    input_norm_weight = layer.input_layernorm.weight.detach()
    eps = layer.input_layernorm.variance_epsilon
    
    # Manual RMSNorm implementation to verify against PyTorch
    # RMSNorm: x * gamma / sqrt(mean(x^2) + eps)
    def rmsnorm(val, w, eps):
        variance = val.pow(2).mean(-1, keepdim=True)
        return val * torch.rsqrt(variance + eps) * w
        
    x_norm = rmsnorm(x, input_norm_weight, eps)
    
    # Self-attention linear projections
    q_proj = layer.self_attn.q_proj.weight.detach()
    k_proj = layer.self_attn.k_proj.weight.detach()
    v_proj = layer.self_attn.v_proj.weight.detach()
    
    # PyTorch linear weights are transposed: y = x @ W.T
    Q = x_norm @ q_proj.T
    K = x_norm @ k_proj.T
    V = x_norm @ v_proj.T
    
    # Reshape Q, K, V for attention
    num_heads = model.config.num_attention_heads # 32
    num_kv_heads = model.config.num_key_value_heads # 8
    head_dim = model.config.hidden_size // num_heads # 64
    
    Q_reshaped = Q.view(1, seq_len, num_heads, head_dim).transpose(1, 2) # (1, 32, seq_len, 64)
    K_reshaped = K.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2) # (1, 8, seq_len, 64)
    V_reshaped = V.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2) # (1, 8, seq_len, 64)
    
    # Apply rotary embedding (RoPE)
    # Get the cos and sin tables
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = model.model.rotary_emb(V_reshaped, position_ids) # cos, sin shape: (1, seq_len, 64)
    
    # Squeeze to (seq_len, 64)
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rotary_pos_emb(x, cos, sin):
        # x is (1, num_heads, seq_len, head_dim)
        # cos, sin are (seq_len, head_dim), unsqueezed to (1, 1, seq_len, head_dim)
        cos_unsq = cos.unsqueeze(0).unsqueeze(0)
        sin_unsq = sin.unsqueeze(0).unsqueeze(0)
        return (x * cos_unsq) + (rotate_half(x) * sin_unsq)
        
    Q_rope = apply_rotary_pos_emb(Q_reshaped, cos, sin)
    K_rope = apply_rotary_pos_emb(K_reshaped, cos, sin)
    
    # Compute attention scores: Q_rope @ K_rope.T / sqrt(head_dim)
    # Q_rope is (1, 32, seq_len, 64), K_rope is (1, 8, seq_len, 64)
    # We repeat/broadcast K_rope to 32 heads: grouping factor is 4
    K_rope_rep = torch.repeat_interleave(K_rope, num_heads // num_kv_heads, dim=1) # (1, 32, seq_len, 64)
    V_rep = torch.repeat_interleave(V_reshaped, num_heads // num_kv_heads, dim=1) # (1, 32, seq_len, 64)
    
    # Dot product along head_dim
    scores = torch.matmul(Q_rope, K_rope_rep.transpose(-1, -2)) / np.sqrt(head_dim) # (1, 32, seq_len, seq_len)
    
    # Apply causal mask
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    scores = scores + mask
    
    # Softmax
    attn_probs = torch.softmax(scores, dim=-1)
    
    # Aggregation
    attn_out = torch.matmul(attn_probs, V_rep) # (1, 32, seq_len, 64)
    
    # Reshape back to contiguous vector
    attn_out_flat = attn_out.transpose(1, 2).contiguous().view(1, seq_len, hidden_size)
    
    # Output projection
    o_proj = layer.self_attn.o_proj.weight.detach()
    attn_proj = attn_out_flat @ o_proj.T # (1, seq_len, hidden_size)
    
    # First residual add
    x_post_attn = x + attn_proj
    
    # RMSNorm before MLP
    ffn_norm_weight = layer.post_attention_layernorm.weight.detach()
    x_norm2 = rmsnorm(x_post_attn, ffn_norm_weight, eps)
    
    # MLP projections
    gate_proj = layer.mlp.gate_proj.weight.detach()
    up_proj = layer.mlp.up_proj.weight.detach()
    down_proj = layer.mlp.down_proj.weight.detach()
    
    # SwiGLU MLP
    gate = x_norm2 @ gate_proj.T
    up = x_norm2 @ up_proj.T
    
    def silu(val):
        return val * torch.sigmoid(val)
        
    silu_out = silu(gate) * up
    down = silu_out @ down_proj.T
    
    # Second residual add
    y_final = x_post_attn + down
    
    # Now verify that our step-by-step layer output matches the original layer forward output
    orig_out = layer(x, position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)))[0]
    max_err = torch.max(torch.abs(orig_out - y_final)).item()
    print(f"Verify step-by-step vs HuggingFace Layer: Max error = {max_err}")
    assert max_err < 1e-5, "Manual step-by-step computation diverges from layer forward!"
    
    # Extract tensors for the 32nd token (decode step, t = 31)
    t = 31
    
    # Output data directory
    out_dir = "/home/daino/progetti/alveare/tools/ref/data"
    os.makedirs(out_dir, exist_ok=True)
    
    # Inputs
    np.save(os.path.join(out_dir, "input_hidden_states.npy"), x[0, t].numpy()) # (2048,)
    np.save(os.path.join(out_dir, "input_norm_weights.npy"), input_norm_weight.numpy()) # (2048,)
    np.save(os.path.join(out_dir, "x_norm.npy"), x_norm[0, t].numpy()) # (2048,)
    
    # Projections at token t before RoPE
    np.save(os.path.join(out_dir, "q_val.npy"), Q[0, t].numpy()) # (2048,)
    np.save(os.path.join(out_dir, "k_val.npy"), K[0, t].numpy()) # (512,)
    np.save(os.path.join(out_dir, "v_val.npy"), V[0, t].numpy()) # (512,)
    
    # RoPE tables at position t (31)
    np.save(os.path.join(out_dir, "cos_val.npy"), cos[t].numpy()) # (64,)
    np.save(os.path.join(out_dir, "sin_val.npy"), sin[t].numpy()) # (64,)
    
    # Rotated Q, K
    # Save Q_rope as shape (8, 4, 64) to match our split-head contiguous layout
    q_rope_reshaped = Q_rope[0, :, t].numpy().reshape(8, 4, 64)
    np.save(os.path.join(out_dir, "q_rope.npy"), q_rope_reshaped) # (8, 4, 64)
    
    k_rope_reshaped = K_rope[0, :, t].numpy() # (8, 64)
    np.save(os.path.join(out_dir, "k_rope.npy"), k_rope_reshaped) # (8, 64)
    
    # KV cache including past tokens (tokens 0..t)
    # The cache shape is (num_kv_heads, seq_len, head_dim) -> (8, 32, 64)
    k_cache_raw = K_rope[0, :, :t+1].numpy() # (8, 32, 64)
    v_cache_raw = V_reshaped[0, :, :t+1].numpy() # (8, 32, 64)
    np.save(os.path.join(out_dir, "k_cache.npy"), k_cache_raw)
    np.save(os.path.join(out_dir, "v_cache.npy"), v_cache_raw)
    
    # Attention output (before O projection) at token t
    # Shape of attn_out: (1, 32, seq_len, 64). Sliced at t: (32, 64).
    # We reshape it to our contiguous format (8, 4, 64)
    attn_out_sliced = attn_out[0, :, t].numpy().reshape(8, 4, 64)
    np.save(os.path.join(out_dir, "attn_out.npy"), attn_out_sliced) # (8, 4, 64)
    
    np.save(os.path.join(out_dir, "attn_proj.npy"), attn_proj[0, t].numpy()) # (2048,)
    np.save(os.path.join(out_dir, "x_post_attn.npy"), x_post_attn[0, t].numpy()) # (2048,)
    
    # MLP norm and MLP inputs
    np.save(os.path.join(out_dir, "ffn_norm_weights.npy"), ffn_norm_weight.numpy()) # (2048,)
    np.save(os.path.join(out_dir, "x_norm2.npy"), x_norm2[0, t].numpy()) # (2048,)
    
    # MLP projections
    np.save(os.path.join(out_dir, "gate.npy"), gate[0, t].numpy()) # (8192,)
    np.save(os.path.join(out_dir, "up.npy"), up[0, t].numpy()) # (8192,)
    np.save(os.path.join(out_dir, "silu_out.npy"), silu_out[0, t].numpy()) # (8192,)
    np.save(os.path.join(out_dir, "down.npy"), down[0, t].numpy()) # (2048,)
    
    # Final output
    np.save(os.path.join(out_dir, "output_hidden_states.npy"), y_final[0, t].numpy()) # (2048,)
    
    # Layer weights
    np.save(os.path.join(out_dir, "w_q.npy"), q_proj.numpy())
    np.save(os.path.join(out_dir, "w_k.npy"), k_proj.numpy())
    np.save(os.path.join(out_dir, "w_v.npy"), v_proj.numpy())
    np.save(os.path.join(out_dir, "w_o.npy"), o_proj.numpy())
    np.save(os.path.join(out_dir, "w_gate.npy"), gate_proj.numpy())
    np.save(os.path.join(out_dir, "w_up.npy"), up_proj.numpy())
    np.save(os.path.join(out_dir, "w_down.npy"), down_proj.numpy())
    
    print("Reference tensors generated and saved to:", out_dir)

if __name__ == "__main__":
    main()
