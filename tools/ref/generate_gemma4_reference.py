import os
import sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoConfig

# Add transformers models to path to import Gemma4UnifiedForConditionalGeneration
sys.path.append("/home/daino/miniconda3/envs/gemma4-ref/lib/python3.12/site-packages")
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration, apply_rotary_pos_emb

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    model_id = "google/gemma-4-12b-it"
    print(f"Loading model {model_id} in bfloat16...")
    # Load model on CPU/CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    # Text model layers are under model.model.language_model.layers
    layer = model.model.language_model.layers[0]
    config = model.config.text_config
    
    seq_len = 32
    hidden_size = config.hidden_size # 3840
    
    # Generate random input sequence (1, seq_len, hidden_size)
    x = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    eps = layer.input_layernorm.eps # 1e-6
    
    def gemma_rmsnorm(val, w, eps):
        variance = val.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = val.to(torch.float32) * torch.rsqrt(variance + eps)
        if w is not None:
            normed = normed * w.to(torch.float32)
        return normed.to(val.dtype)
        
    # Step 1: Input norm
    x_norm = gemma_rmsnorm(x, layer.input_layernorm.weight, eps)
    
    # Step 2: Projections
    q_proj = layer.self_attn.q_proj.weight.detach()
    k_proj = layer.self_attn.k_proj.weight.detach()
    v_proj = layer.self_attn.v_proj.weight.detach()
    
    Q = torch.matmul(x_norm, q_proj.T)
    K = torch.matmul(x_norm, k_proj.T)
    V = torch.matmul(x_norm, v_proj.T)
    
    # Dimensions
    num_heads = config.num_attention_heads # 16
    num_kv_heads = config.num_key_value_heads # 8
    head_dim = config.head_dim # 256
    
    # Reshape
    Q_reshaped = Q.view(1, seq_len, num_heads, head_dim)
    K_reshaped = K.view(1, seq_len, num_kv_heads, head_dim)
    V_reshaped = V.view(1, seq_len, num_kv_heads, head_dim)
    
    # Step 3: QK-Norm & V-norm
    q_norm_weight = layer.self_attn.q_norm.weight.detach()
    k_norm_weight = layer.self_attn.k_norm.weight.detach()
    Q_normed = gemma_rmsnorm(Q_reshaped, q_norm_weight, eps)
    K_normed = gemma_rmsnorm(K_reshaped, k_norm_weight, eps)
    V_normed = gemma_rmsnorm(V_reshaped, None, eps)
    
    # Step 4: RoPE
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    rotary_emb = model.model.language_model.rotary_emb
    cos_ref, sin_ref = rotary_emb(V_normed, position_ids, layer_type="sliding_attention")
    cos = cos_ref.squeeze(0) # shape (seq_len, head_dim)
    sin = sin_ref.squeeze(0)
    
    # Apply RoPE
    Q_rope = apply_rotary_pos_emb(Q_normed, cos_ref, sin_ref, unsqueeze_dim=2)
    K_rope = apply_rotary_pos_emb(K_normed, cos_ref, sin_ref, unsqueeze_dim=2)
    
    # Step 5: Attention GQA
    scaling = layer.self_attn.scaling # 1.0
    
    Q_rope_trans = Q_rope.transpose(1, 2) # (1, 16, seq_len, 256)
    K_rope_trans = K_rope.transpose(1, 2) # (1, 8, seq_len, 256)
    V_normed_trans = V_normed.transpose(1, 2) # (1, 8, seq_len, 256)
    
    ratio = num_heads // num_kv_heads # 2
    K_rope_rep = torch.repeat_interleave(K_rope_trans, ratio, dim=1) # (1, 16, seq_len, 256)
    V_normed_rep = torch.repeat_interleave(V_normed_trans, ratio, dim=1) # (1, 16, seq_len, 256)
    
    # Dot product attention scores
    scores = torch.matmul(Q_rope_trans, K_rope_rep.transpose(-1, -2)) * scaling # (1, 16, seq_len, seq_len)
    
    # Apply causal mask
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    scores = scores + mask
    
    # Softmax
    attn_probs = torch.softmax(scores.to(torch.float32), dim=-1).to(torch.bfloat16)
    
    # Weighted sum
    attn_out = torch.matmul(attn_probs, V_normed_rep) # (1, 16, seq_len, 256)
    
    # Reshape back to contiguous
    attn_out_flat = attn_out.transpose(1, 2).contiguous().view(1, seq_len, num_heads * head_dim)
    
    # Step 6: Output projection
    o_proj = layer.self_attn.o_proj.weight.detach()
    attn_proj = torch.matmul(attn_out_flat, o_proj.T)
    
    # Step 7: Post attention norm & residual add
    x_post_attn = x + gemma_rmsnorm(attn_proj, layer.post_attention_layernorm.weight, eps)
    
    # Step 8: Pre-FFN norm
    x_norm2 = gemma_rmsnorm(x_post_attn, layer.pre_feedforward_layernorm.weight, eps)
    
    # Step 9: MLP projections
    gate_proj = layer.mlp.gate_proj.weight.detach()
    up_proj = layer.mlp.up_proj.weight.detach()
    down_proj = layer.mlp.down_proj.weight.detach()
    
    gate = torch.matmul(x_norm2, gate_proj.T)
    up = torch.matmul(x_norm2, up_proj.T)
    
    # GeGLU activation
    def gelu_pytorch_tanh(val):
        import math
        precomputed_constant = math.sqrt(2 / math.pi)
        return 0.5 * val * (1 + torch.tanh(precomputed_constant * (val + 0.044715 * torch.pow(val, 3))))
        
    geglu_out = gelu_pytorch_tanh(gate.to(torch.float32)).to(torch.bfloat16) * up
    down = torch.matmul(geglu_out, down_proj.T)
    
    # Step 10: Post-FFN norm & residual add
    down_normed = gemma_rmsnorm(down, layer.post_feedforward_layernorm.weight, eps)
    y_layer = x_post_attn + down_normed
    
    # Step 11: Layer Output Scale
    y_final = y_layer * layer.layer_scalar.to(device)
    
    # Verify vs model forward
    # Prepare inputs for layer forward
    position_embeddings = (cos_ref, sin_ref)
    with torch.no_grad():
        orig_out = layer(x, position_embeddings=position_embeddings)
    
    max_err = torch.max(torch.abs(orig_out - y_final)).item()
    print(f"Verify step-by-step vs HuggingFace Layer: Max error = {max_err}")
    assert max_err < 0.5, "Manual step-by-step computation diverges from layer forward!"
    
    # Save the 32nd token (decode step, t = 31)
    t = 31
    out_dir = Path(__file__).resolve().parents[2] / "tools" / "ref" / "data_gemma4"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper to save tensor to npy
    def save_npy(filename, tensor):
        np.save(out_dir / filename, tensor.detach().cpu().to(torch.float32).numpy())
        
    # Inputs & Weights
    save_npy("input_hidden_states.npy", x[0, t])
    save_npy("input_norm_weights.npy", layer.input_layernorm.weight)
    save_npy("x_norm.npy", x_norm[0, t])
    
    # QKV Projections
    save_npy("w_q.npy", q_proj)
    save_npy("w_k.npy", k_proj)
    save_npy("w_v.npy", v_proj)
    save_npy("q_val.npy", Q[0, t])
    save_npy("k_val.npy", K[0, t])
    save_npy("v_val.npy", V[0, t])
    
    # QK-Norm
    save_npy("q_norm_weights.npy", q_norm_weight)
    save_npy("k_norm_weights.npy", k_norm_weight)
    save_npy("q_normed.npy", Q_normed[0, t].view(-1))
    save_npy("k_normed.npy", K_normed[0, t].view(-1))
    save_npy("v_normed.npy", V_normed[0, t].view(-1))
    
    # RoPE
    save_npy("cos_val.npy", cos[t])
    save_npy("sin_val.npy", sin[t])
    save_npy("q_rope.npy", Q_rope[0, t].view(-1))
    save_npy("k_rope.npy", K_rope[0, t].view(-1))
    
    # KV Cache state for sequence length 32
    # k_cache, v_cache shape: (num_kv_heads, seq_len, head_dim) -> (8, 32, 256)
    save_npy("k_cache.npy", K_rope[0].transpose(0, 1)) # (8, 32, 256)
    save_npy("v_cache.npy", V_normed[0].transpose(0, 1)) # (8, 32, 256)
    
    # Attention Output
    save_npy("attn_out.npy", attn_out[0, :, t, :].reshape(-1)) # (16 * 256)
    save_npy("w_o.npy", o_proj)
    save_npy("attn_proj.npy", attn_proj[0, t])
    
    # Post-Attn Norm & residual
    save_npy("post_attn_norm_weights.npy", layer.post_attention_layernorm.weight)
    save_npy("x_post_attn.npy", x_post_attn[0, t])
    
    # Pre-FFN Norm
    save_npy("ffn_norm_weights.npy", layer.pre_feedforward_layernorm.weight)
    save_npy("x_norm2.npy", x_norm2[0, t])
    
    # MLP projections
    save_npy("w_gate.npy", gate_proj)
    save_npy("w_up.npy", up_proj)
    save_npy("gate.npy", gate[0, t])
    save_npy("up.npy", up[0, t])
    
    # GeGLU & Down proj
    save_npy("geglu_out.npy", geglu_out[0, t])
    save_npy("w_down.npy", down_proj)
    save_npy("down.npy", down[0, t])
    
    # Post-FFN Norm & final residual
    save_npy("post_ffw_norm_weights.npy", layer.post_feedforward_layernorm.weight)
    save_npy("output_hidden_states.npy", y_final[0, t])
    
    print("Gemma-4 reference data saved successfully from HuggingFace oracle!")

if __name__ == "__main__":
    main()
