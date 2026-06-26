import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    model_id = "unsloth/gemma-3-1b-it"
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    layer = model.model.layers[0]

    seq_len = 32
    hidden_size = 1152
    
    # Generate random input sequence (1, seq_len, hidden_size)
    x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32)
    
    eps = layer.input_layernorm.eps
    
    def gemma_rmsnorm(val, w, eps):
        variance = val.pow(2).mean(-1, keepdim=True)
        return val * torch.rsqrt(variance + eps) * (1.0 + w)
        
    x_norm = gemma_rmsnorm(x, layer.input_layernorm.weight.detach(), eps)
    
    # Self-attention linear projections
    q_proj = layer.self_attn.q_proj.weight.detach()
    k_proj = layer.self_attn.k_proj.weight.detach()
    v_proj = layer.self_attn.v_proj.weight.detach()
    
    Q = x_norm @ q_proj.T
    K = x_norm @ k_proj.T
    V = x_norm @ v_proj.T
    
    # Reshape Q, K, V for attention
    num_heads = model.config.num_attention_heads # 4
    num_kv_heads = model.config.num_key_value_heads # 1
    head_dim = model.config.head_dim # 256
    
    # QK-norm
    q_norm_weight = layer.self_attn.q_norm.weight.detach()
    k_norm_weight = layer.self_attn.k_norm.weight.detach()
    
    Q_reshaped = Q.view(1, seq_len, num_heads, head_dim)
    K_reshaped = K.view(1, seq_len, num_kv_heads, head_dim)
    
    Q_normed = gemma_rmsnorm(Q_reshaped, q_norm_weight, eps).transpose(1, 2) # (1, 4, seq_len, 256)
    K_normed = gemma_rmsnorm(K_reshaped, k_norm_weight, eps).transpose(1, 2) # (1, 1, seq_len, 256)
    V_reshaped = V.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2) # (1, 1, seq_len, 256)
    
    # Apply rotary embedding (RoPE)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    
    # In transformers, position_embeddings is passed down.
    # Let's get them from layer.self_attn or model.model.rotary_emb
    # For layer 0, it's sliding_attention, so we use the sliding_attention inv_freq
    cos_ref, sin_ref = model.model.rotary_emb(V_reshaped, position_ids, layer_type="sliding_attention")
    cos = cos_ref.squeeze(0) # shape (seq_len, 256)
    sin = sin_ref.squeeze(0)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rotary_pos_emb(x, cos, sin):
        # x is (1, num_heads, seq_len, head_dim)
        cos_unsq = cos.unsqueeze(0).unsqueeze(0)
        sin_unsq = sin.unsqueeze(0).unsqueeze(0)
        return (x * cos_unsq) + (rotate_half(x) * sin_unsq)
        
    Q_rope = apply_rotary_pos_emb(Q_normed, cos, sin)
    K_rope = apply_rotary_pos_emb(K_normed, cos, sin)
    
    # Attention scores: scaling is config.query_pre_attn_scalar ** -0.5
    scaling = model.config.query_pre_attn_scalar ** -0.5 # 256 ** -0.5 = 0.0625
    
    K_rope_rep = torch.repeat_interleave(K_rope, num_heads // num_kv_heads, dim=1) # (1, 4, seq_len, 256)
    V_rep = torch.repeat_interleave(V_reshaped, num_heads // num_kv_heads, dim=1) # (1, 4, seq_len, 256)
    
    scores = torch.matmul(Q_rope, K_rope_rep.transpose(-1, -2)) * scaling # (1, 4, seq_len, seq_len)
    
    # Apply causal mask
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    scores = scores + mask
    
    # Softmax
    attn_probs = torch.softmax(scores, dim=-1)
    
    # Aggregation
    attn_out = torch.matmul(attn_probs, V_rep) # (1, 4, seq_len, 256)
    
    # Reshape back to contiguous vector
    attn_out_flat = attn_out.transpose(1, 2).contiguous().view(1, seq_len, num_heads * head_dim)
    
    # Output projection
    o_proj = layer.self_attn.o_proj.weight.detach()
    attn_proj = attn_out_flat @ o_proj.T # (1, seq_len, hidden_size)
    
    # Post attention norm & first residual add
    x_post_attn = x + gemma_rmsnorm(attn_proj, layer.post_attention_layernorm.weight.detach(), eps)
    
    # pre_feedforward_layernorm before MLP
    x_norm2 = gemma_rmsnorm(x_post_attn, layer.pre_feedforward_layernorm.weight.detach(), eps)
    
    # MLP projections
    gate_proj = layer.mlp.gate_proj.weight.detach()
    up_proj = layer.mlp.up_proj.weight.detach()
    down_proj = layer.mlp.down_proj.weight.detach()
    
    gate = x_norm2 @ gate_proj.T
    up = x_norm2 @ up_proj.T
    
    # gelu_pytorch_tanh
    def gelu_pytorch_tanh(val):
        return 0.5 * val * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (val + 0.044715 * val.pow(3))))
        
    geglu_out = gelu_pytorch_tanh(gate) * up
    down = geglu_out @ down_proj.T
    
    # post_feedforward_layernorm & second residual add
    y_final = x_post_attn + gemma_rmsnorm(down, layer.post_feedforward_layernorm.weight.detach(), eps)
    
    # Verify vs model forward
    # Prepare inputs for layer forward
    position_embeddings = (cos_ref, sin_ref)
    orig_out = layer(x, position_embeddings=position_embeddings)[0]
    
    max_err = torch.max(torch.abs(orig_out - y_final)).item()
    print(f"Verify step-by-step vs HuggingFace Layer: Max error = {max_err}")
    assert max_err < 1e-5, "Manual step-by-step computation diverges from layer forward!"
    
    # Save the 32nd token (decode step, t = 31)
    t = 31
    out_dir = "/home/daino/progetti/alveare/tools/ref/data_gemma"
    os.makedirs(out_dir, exist_ok=True)
    
    # Inputs & Weights
    np.save(os.path.join(out_dir, "input_hidden_states.npy"), x[0, t].numpy())
    np.save(os.path.join(out_dir, "input_norm_weights.npy"), layer.input_layernorm.weight.detach().numpy())
    np.save(os.path.join(out_dir, "x_norm.npy"), x_norm[0, t].numpy())
    
    # QKV Projections
    np.save(os.path.join(out_dir, "w_q.npy"), q_proj.numpy())
    np.save(os.path.join(out_dir, "w_k.npy"), k_proj.numpy())
    np.save(os.path.join(out_dir, "w_v.npy"), v_proj.numpy())
    np.save(os.path.join(out_dir, "q_val.npy"), Q[0, t].numpy())
    np.save(os.path.join(out_dir, "k_val.npy"), K[0, t].numpy())
    np.save(os.path.join(out_dir, "v_val.npy"), V[0, t].numpy())
    
    # QK-Norm
    np.save(os.path.join(out_dir, "q_norm_weights.npy"), q_norm_weight.numpy())
    np.save(os.path.join(out_dir, "k_norm_weights.npy"), k_norm_weight.numpy())
    np.save(os.path.join(out_dir, "q_normed.npy"), Q_normed[0, :, t, :].numpy())
    np.save(os.path.join(out_dir, "k_normed.npy"), K_normed[0, :, t, :].numpy())
    
    # RoPE
    np.save(os.path.join(out_dir, "cos_val.npy"), cos[t].numpy())
    np.save(os.path.join(out_dir, "sin_val.npy"), sin[t].numpy())
    np.save(os.path.join(out_dir, "q_rope.npy"), Q_rope[0, :, t, :].numpy()) # shape (4, 256)
    np.save(os.path.join(out_dir, "k_rope.npy"), K_rope[0, :, t, :].numpy()) # shape (1, 256)
    
    # KV Cache state for sequence length 32
    # k_cache, v_cache shape: (1, seq_len, 256) squeezable to (1, 32, 256)
    np.save(os.path.join(out_dir, "k_cache.npy"), K_normed[0].numpy())
    np.save(os.path.join(out_dir, "v_cache.npy"), V_reshaped[0].numpy().transpose(1, 0, 2)) # (1, 32, 256)
    
    # Attention Output
    np.save(os.path.join(out_dir, "attn_out.npy"), attn_out[0, :, t, :].numpy()) # (4, 256)
    np.save(os.path.join(out_dir, "w_o.npy"), o_proj.numpy())
    np.save(os.path.join(out_dir, "attn_proj.npy"), attn_proj[0, t].numpy())
    
    # Post-Attn Norm & residual
    np.save(os.path.join(out_dir, "post_attn_norm_weights.npy"), layer.post_attention_layernorm.weight.detach().numpy())
    np.save(os.path.join(out_dir, "x_post_attn.npy"), x_post_attn[0, t].numpy())
    
    # Pre-FFN Norm
    np.save(os.path.join(out_dir, "ffn_norm_weights.npy"), layer.pre_feedforward_layernorm.weight.detach().numpy())
    np.save(os.path.join(out_dir, "x_norm2.npy"), x_norm2[0, t].numpy())
    
    # MLP projections
    np.save(os.path.join(out_dir, "w_gate.npy"), gate_proj.numpy())
    np.save(os.path.join(out_dir, "w_up.npy"), up_proj.numpy())
    np.save(os.path.join(out_dir, "gate.npy"), gate[0, t].numpy())
    np.save(os.path.join(out_dir, "up.npy"), up[0, t].numpy())
    
    # GeGLU & Down proj
    np.save(os.path.join(out_dir, "geglu_out.npy"), geglu_out[0, t].numpy())
    np.save(os.path.join(out_dir, "w_down.npy"), down_proj.numpy())
    np.save(os.path.join(out_dir, "down.npy"), down[0, t].numpy())
    
    # Post-FFN Norm & final residual
    np.save(os.path.join(out_dir, "post_ffw_norm_weights.npy"), layer.post_feedforward_layernorm.weight.detach().numpy())
    np.save(os.path.join(out_dir, "output_hidden_states.npy"), y_final[0, t].numpy())
    
    print("Gemma reference data saved successfully!")

if __name__ == "__main__":
    main()
