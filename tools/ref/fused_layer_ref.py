"""
NumPy reference implementation for ONE decoder layer of Gemma/Llama models.
Computes a single token forward pass reading quantized weights directly from disk
(quantized_weights_* layout) to validate fused NPU per-layer kernels.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Try importing quantization helper for self-testing dequant_q4_0
try:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "convert"))
    from gemv_q_convert import quantize_to_q4_0, pack_to_combined
    HAS_CONVERT_TOOL = True
except ImportError:
    HAS_CONVERT_TOOL = False


def dequant_q4_0(packed: np.ndarray, K: int) -> np.ndarray:
    """
    Dequantize a Q4_0 packed uint8 array of shape (N, K // 32 * 20) to float32 (N, K).

    Layout per 20-byte block (32 elements):
    - Bytes 0..15: 16 bytes of interleaved signed 4-bit quants.
      (byte j: low nibble = idx 2j, high nibble = idx 2j+1; signed: subtract 16 if >= 8).
    - Bytes 16..17: 2-byte little-endian bfloat16 scale.
    - Bytes 18..19: 2 pad bytes.
    """
    if packed.ndim == 1:
        packed = packed.reshape(1, -1)

    N = packed.shape[0]
    num_blocks = K // 32
    expected_bytes = num_blocks * 20
    assert packed.shape[1] == expected_bytes, (
        f"Expected {expected_bytes} bytes per row for K={K}, got {packed.shape[1]}"
    )

    packed_blocks = packed.reshape(N, num_blocks, 20)

    quants_raw = packed_blocks[:, :, :16]
    lo = (quants_raw & 0x0F).astype(np.int8)
    lo[lo >= 8] -= 16

    hi = ((quants_raw >> 4) & 0x0F).astype(np.int8)
    hi[hi >= 8] -= 16

    q_vals = np.empty((N, num_blocks, 32), dtype=np.float32)
    q_vals[:, :, 0::2] = lo
    q_vals[:, :, 1::2] = hi

    b0 = packed_blocks[:, :, 16].astype(np.uint32)
    b1 = packed_blocks[:, :, 17].astype(np.uint32)
    u16_scale = b0 | (b1 << 8)

    scales = (u16_scale << 16).view(np.float32)

    W_blocks = q_vals * scales[:, :, None]
    return W_blocks.reshape(N, K).astype(np.float32)


def validate_dequant_q4_0() -> bool:
    """Validate dequant_q4_0 against gemv_q_convert quantization."""
    if not HAS_CONVERT_TOOL:
        print("[dequant_q4_0 validation skipped: gemv_q_convert not found]")
        return True

    np.random.seed(42)
    N, K = 64, 128
    W_orig = np.random.randn(N, K).astype(np.float32)
    w_q4, scales = quantize_to_q4_0(W_orig)
    packed = pack_to_combined(w_q4, scales)

    W_dequant = dequant_q4_0(packed, K)

    W_blocks = W_orig.reshape(N, K // 32, 32)
    max_vals = np.max(np.abs(W_blocks), axis=2)
    max_step = np.max(max_vals / 7.0)

    max_err = np.max(np.abs(W_orig - W_dequant))
    assert max_err <= max_step + 1e-4, f"Dequant error {max_err} exceeds step size {max_step}"
    print(f"[dequant_q4_0 self-check passed: max_err={max_err:.6f} <= max_step={max_step:.6f}]")
    return True


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU activation with PyTorch/Gemma tanh approximation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def rmsnorm(x: np.ndarray, weight: np.ndarray = None, eps: float = 1e-6, is_hf_weight: bool = False) -> np.ndarray:
    """Gemma / Llama RMSNorm: x / sqrt(mean(x^2) + eps) * w."""
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(variance + eps)
    if weight is not None:
        w_eff = (1.0 + weight) if is_hf_weight else weight
        x_norm = x_norm * w_eff
    return x_norm


def apply_rope(x: np.ndarray, pos: int, base: float = 10000.0) -> np.ndarray:
    """Apply Rotary Position Embedding to per-head tensor (num_heads, head_dim)."""
    dim = x.shape[-1]
    half = dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    freqs = pos * inv_freq
    c = np.cos(freqs)
    s = np.sin(freqs)

    x1 = x[..., :half]
    x2 = x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * c - x2 * s
    out[..., half:] = x2 * c + x1 * s
    return out


def load_weight_matrix(path: str) -> tuple[np.ndarray, bool]:
    """Load a weight matrix from .npy (dequantizing if Q4_0 packed uint8). Returns (arr, is_hf_weight)."""
    if path is None or not os.path.exists(path):
        return None, False
    arr = np.load(path)
    is_hf_weight = ("_weights.npy" in path) or ("data_gemma" in path and not path.endswith("_packed.npy"))
    if arr.dtype == np.uint8:
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        row_bytes = arr.shape[1]
        K = (row_bytes // 20) * 32
        return dequant_q4_0(arr, K), is_hf_weight
    return arr.astype(np.float32), is_hf_weight


def layer_forward(weights_dir: str, layer_idx: int, x: np.ndarray, pos: int,
                  kv_cache: dict = None, config: dict = None):
    """
    Compute ONE decoder layer for a single token input x at position pos.

    Order of operations:
    input RMSNorm -> fused QKV projection -> per-head Q/K norm (and V norm on gemma4) -> RoPE ->
    KV-cache append -> causal attention (GQA; sliding window on sliding layers) -> O projection ->
    post-attention norm + residual -> pre-FFN norm -> FFN down(GELU(gate*x) * up(x)) (GELU tanh) ->
    post-FFN norm + residual -> optional PLE injection (Gemma-4-E4B) -> layer output scale.

    Returns:
        (out, intermediates_dict)
    """
    weights_dir = str(weights_dir)
    if config is None:
        cfg_path = os.path.join(weights_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

    model_type = config.get("model_type", "gemma3")
    is_gemma4 = model_type in ["gemma4", "gemma4-e4b", "e4b"]
    is_gemma3 = model_type == "gemma3"

    hidden_size = config.get("hidden_size", x.shape[-1])
    intermediate_size = config.get("intermediate_size", 6912)
    num_attention_heads = config.get("num_attention_heads", 4)
    num_kv_heads = config.get("num_key_value_heads", 1)
    head_dim = config.get("head_dim", 256)
    head_dim_global = config.get("head_dim_global", head_dim)
    max_seq_len = config.get("max_position_embeddings", config.get("max_seq_len", 2048))
    sliding_pattern_period = config.get("sliding_pattern_period", 2)
    sliding_window = config.get("sliding_window", 512)
    eps = float(config.get("rms_norm_eps", 1e-6))
    per_layer_input = config.get("per_layer_input", 0)

    is_sliding = (is_gemma3 or is_gemma4) and ((layer_idx + 1) % sliding_pattern_period != 0)

    cur_head_dim = head_dim
    if is_gemma4:
        cur_head_dim = head_dim if is_sliding else head_dim_global
        if model_type != "gemma4-e4b" and not is_sliding:
            num_kv_heads = 1

    x_1d = x.reshape(-1).astype(np.float32)
    assert x_1d.shape[0] == hidden_size, f"Expected input size {hidden_size}, got {x_1d.shape[0]}"

    def lpath(name):
        p1 = os.path.join(weights_dir, f"blk.{layer_idx}.{name}.weight_packed.npy")
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(weights_dir, f"blk.{layer_idx}.{name}.weight.npy")
        if os.path.exists(p2):
            return p2
        p3 = os.path.join(weights_dir, f"{name}.npy")
        if os.path.exists(p3):
            return p3
        p4 = os.path.join(weights_dir, f"w_{name.replace('attn_', '').replace('ffn_', '')}.npy")
        if os.path.exists(p4):
            return p4
        return None

    w_attn_norm, hf_attn_norm = load_weight_matrix(lpath("attn_norm") or os.path.join(weights_dir, "input_norm_weights.npy"))
    w_q, _ = load_weight_matrix(lpath("attn_q") or os.path.join(weights_dir, "w_q.npy"))
    w_k, _ = load_weight_matrix(lpath("attn_k") or os.path.join(weights_dir, "w_k.npy"))
    w_v, _ = load_weight_matrix(lpath("attn_v") or os.path.join(weights_dir, "w_v.npy"))
    w_q_norm, hf_q_norm = load_weight_matrix(lpath("attn_q_norm") or os.path.join(weights_dir, "q_norm_weights.npy"))
    w_k_norm, hf_k_norm = load_weight_matrix(lpath("attn_k_norm") or os.path.join(weights_dir, "k_norm_weights.npy"))
    w_o, _ = load_weight_matrix(lpath("attn_output") or os.path.join(weights_dir, "w_o.npy"))
    w_post_attn_norm, hf_post_attn_norm = load_weight_matrix(lpath("post_attention_norm") or os.path.join(weights_dir, "post_attn_norm_weights.npy"))
    w_ffn_norm, hf_ffn_norm = load_weight_matrix(lpath("ffn_norm") or os.path.join(weights_dir, "ffn_norm_weights.npy"))
    w_gate, _ = load_weight_matrix(lpath("ffn_gate") or os.path.join(weights_dir, "w_gate.npy"))
    w_up, _ = load_weight_matrix(lpath("ffn_up") or os.path.join(weights_dir, "w_up.npy"))
    w_down, _ = load_weight_matrix(lpath("ffn_down") or os.path.join(weights_dir, "w_down.npy"))
    w_post_ffw_norm, hf_post_ffw_norm = load_weight_matrix(lpath("post_ffw_norm") or os.path.join(weights_dir, "post_ffw_norm_weights.npy"))
    w_out_scale, _ = load_weight_matrix(lpath("layer_output_scale"))

    # 1. Input RMSNorm
    x_norm = rmsnorm(x_1d, w_attn_norm, eps=eps, is_hf_weight=hf_attn_norm)

    # Pad x_norm for GEMV if weight K is larger
    K_padded = w_q.shape[1] if w_q is not None else hidden_size
    x_norm_padded = np.zeros(K_padded, dtype=np.float32)
    x_norm_padded[:hidden_size] = x_norm

    # 2. QKV Projections
    N_q = num_attention_heads * cur_head_dim
    N_kv = num_kv_heads * cur_head_dim

    q_raw = w_q @ x_norm_padded
    q = q_raw[:N_q]

    k_raw = w_k @ x_norm_padded
    k = k_raw[:N_kv]

    if w_v is not None:
        v_raw = w_v @ x_norm_padded
        v = v_raw[:N_kv]
    else:
        v = k.copy()

    # 3. Per-head Q/K norm (and V norm on Gemma-4)
    q_reshaped = q.reshape(num_attention_heads, cur_head_dim)
    k_reshaped = k.reshape(num_kv_heads, cur_head_dim)
    v_reshaped = v.reshape(num_kv_heads, cur_head_dim)

    if is_gemma3 or is_gemma4:
        q_normed = rmsnorm(q_reshaped, w_q_norm, eps=eps, is_hf_weight=hf_q_norm)
        k_normed = rmsnorm(k_reshaped, w_k_norm, eps=eps, is_hf_weight=hf_k_norm)
        if is_gemma4:
            v_normed = rmsnorm(v_reshaped, None, eps=eps)
        else:
            v_normed = v_reshaped
    else:
        q_normed = q_reshaped
        k_normed = k_reshaped
        v_normed = v_reshaped

    # 4. RoPE
    base_freq = 10000.0 if is_sliding else 1000000.0
    q_rope = apply_rope(q_normed, pos, base=base_freq)
    k_rope = apply_rope(k_normed, pos, base=base_freq)

    # 5. KV-cache append
    if kv_cache is not None:
        if layer_idx not in kv_cache:
            kv_cache[layer_idx] = {
                "k": np.zeros((num_kv_heads, max_seq_len, cur_head_dim), dtype=np.float32),
                "v": np.zeros((num_kv_heads, max_seq_len, cur_head_dim), dtype=np.float32),
            }
        kv_cache[layer_idx]["k"][:, pos, :] = k_rope
        kv_cache[layer_idx]["v"][:, pos, :] = v_normed
        k_cache_layer = kv_cache[layer_idx]["k"]
        v_cache_layer = kv_cache[layer_idx]["v"]
    else:
        k_cache_layer = np.zeros((num_kv_heads, max_seq_len, cur_head_dim), dtype=np.float32)
        v_cache_layer = np.zeros((num_kv_heads, max_seq_len, cur_head_dim), dtype=np.float32)
        k_cache_layer[:, pos, :] = k_rope
        v_cache_layer[:, pos, :] = v_normed

    # 6. Causal attention
    seq_len = pos + 1
    start_pos = 0
    if is_sliding and seq_len > sliding_window:
        start_pos = seq_len - sliding_window
    W = seq_len - start_pos

    group_ratio = num_attention_heads // num_kv_heads
    attn_scale = 1.0 if is_gemma4 else (1.0 / np.sqrt(float(cur_head_dim)))
    attn_out_heads = np.zeros((num_attention_heads, cur_head_dim), dtype=np.float32)

    for h in range(num_attention_heads):
        kv_h = h // group_ratio
        qh = q_rope[h]
        keys = k_cache_layer[kv_h, start_pos:seq_len, :]
        scores = (keys @ qh) * attn_scale

        scores_max = np.max(scores)
        scores_exp = np.exp(scores - scores_max)
        probs = scores_exp / np.sum(scores_exp)

        vals = v_cache_layer[kv_h, start_pos:seq_len, :]
        attn_out_heads[h] = probs @ vals

    attn_out = attn_out_heads.reshape(-1)

    # 7. Output projection
    N_q_padded = w_o.shape[1] if w_o is not None else N_q
    attn_out_padded = np.zeros(N_q_padded, dtype=np.float32)
    attn_out_padded[:N_q] = attn_out

    attn_proj_raw = w_o @ attn_out_padded
    attn_proj = attn_proj_raw[:hidden_size]

    # 8. Post-attention norm + residual
    if is_gemma3 or is_gemma4:
        attn_proj_normed = rmsnorm(attn_proj, w_post_attn_norm, eps=eps, is_hf_weight=hf_post_attn_norm)
        x_post_attn = x_1d + attn_proj_normed
    else:
        x_post_attn = x_1d + attn_proj

    # 9. Pre-FFN norm
    x_norm2 = rmsnorm(x_post_attn, w_ffn_norm, eps=eps, is_hf_weight=hf_ffn_norm)

    # 10. FFN (gate, up, geglu, down)
    K_ffn_padded = w_gate.shape[1] if w_gate is not None else hidden_size
    x_norm2_padded = np.zeros(K_ffn_padded, dtype=np.float32)
    x_norm2_padded[:hidden_size] = x_norm2

    gate_raw = w_gate @ x_norm2_padded
    gate = gate_raw[:intermediate_size]

    up_raw = w_up @ x_norm2_padded
    up = up_raw[:intermediate_size]

    geglu = gelu_tanh(gate) * up

    I_padded = w_down.shape[1] if w_down is not None else intermediate_size
    geglu_padded = np.zeros(I_padded, dtype=np.float32)
    geglu_padded[:intermediate_size] = geglu

    down_raw = w_down @ geglu_padded
    down = down_raw[:hidden_size]

    # 11. Post-FFN norm + residual
    if is_gemma3 or is_gemma4:
        down_normed = rmsnorm(down, w_post_ffw_norm, eps=eps, is_hf_weight=hf_post_ffw_norm)
        x_post_ffn = x_post_attn + down_normed
    else:
        x_post_ffn = x_post_attn + down

    # 12. Optional PLE injection (Gemma-4-E4B)
    w_inp_gate, _ = load_weight_matrix(lpath("inp_gate"))
    w_proj, _ = load_weight_matrix(lpath("proj"))
    w_post_norm, hf_post_norm = load_weight_matrix(lpath("post_norm"))

    if per_layer_input > 0 and w_inp_gate is not None and w_proj is not None:
        h_l = np.zeros(per_layer_input, dtype=np.float32)
        g_raw = w_inp_gate @ x_post_ffn
        z = gelu_tanh(g_raw[:per_layer_input]) * h_l
        p_raw = w_proj @ z
        y_ple = rmsnorm(p_raw[:hidden_size], w_post_norm, eps=eps, is_hf_weight=hf_post_norm)
        x_post_ffn = x_post_ffn + y_ple

    # 13. Layer output scale
    oscale = 1.0
    if is_gemma4 and w_out_scale is not None and len(w_out_scale) > 0:
        oscale = float(w_out_scale[0])

    out = x_post_ffn * oscale

    intermediates = {
        "x_norm": x_norm,
        "q": q,
        "k": k,
        "v": v,
        "q_normed": q_normed.reshape(-1),
        "k_normed": k_normed.reshape(-1),
        "q_rope": q_rope.reshape(-1),
        "k_rope": k_rope.reshape(-1),
        "attn_out": attn_out,
        "attn_proj": attn_proj,
        "x_post_attn": x_post_attn,
        "x_norm2": x_norm2,
        "gate": gate,
        "up": up,
        "geglu": geglu,
        "down": down,
        "out": out,
    }

    return out, intermediates


def run_golden_check(golden_dir: str) -> bool:
    """Run golden check comparing against data_gemma trace files."""
    if not os.path.exists(golden_dir):
        print(f"Golden directory {golden_dir} does not exist.")
        return False

    x = np.load(os.path.join(golden_dir, "input_hidden_states.npy"))
    pos = 31

    cfg = {
        "model_type": "gemma3",
        "hidden_size": 1152,
        "intermediate_size": 6912,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "max_seq_len": 2048,
        "sliding_pattern_period": 2,
        "sliding_window": 512,
        "rms_norm_eps": 1e-6,
    }

    # Pre-populate KV cache if k_cache.npy and v_cache.npy exist
    kv_cache = None
    k_cache_path = os.path.join(golden_dir, "k_cache.npy")
    v_cache_path = os.path.join(golden_dir, "v_cache.npy")
    if os.path.exists(k_cache_path) and os.path.exists(v_cache_path):
        k_raw = np.load(k_cache_path)  # (1, 32, 256)
        v_raw = np.load(v_cache_path)  # (32, 1, 256) or (1, 32, 256)
        if v_raw.shape == (32, 1, 256):
            v_raw = v_raw.transpose(1, 0, 2)
        dim = 256
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        freqs = np.arange(32)[:, None] * inv_freq[None, :]
        c, s = np.cos(freqs), np.sin(freqs)
        half = dim // 2
        x1, x2 = k_raw[0, :, :half], k_raw[0, :, half:]
        k_rope_32 = np.empty_like(k_raw[0])
        k_rope_32[:, :half] = x1 * c - x2 * s
        k_rope_32[:, half:] = x2 * c + x1 * s
        kv_cache = {0: {"k": k_rope_32[None, :, :], "v": v_raw}}

    out, intermediates = layer_forward(golden_dir, 0, x, pos, kv_cache=kv_cache, config=cfg)

    golden_keys = {
        "x_norm": "x_norm.npy",
        "q_normed": "q_normed.npy",
        "k_normed": "k_normed.npy",
        "q_rope": "q_rope.npy",
        "geglu": "geglu_out.npy",
        "down": "down.npy",
    }

    print("=== Golden Trace Self-Check (Gemma-3) ===")
    all_ok = True
    for key, filename in golden_keys.items():
        filepath = os.path.join(golden_dir, filename)
        if os.path.exists(filepath):
            target = np.load(filepath).reshape(-1)
            actual = intermediates[key].reshape(-1)
            diff = np.max(np.abs(actual - target))
            print(f"Stage '{key}': max-abs-diff = {diff:.6e}")
            if diff > 1e-3:
                all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="NumPy reference for ONE decoder layer (for validating fused per-layer kernels)."
    )
    parser.add_argument("weights_dir", nargs="?", default="", help="Directory containing quantized weights")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default: 0)")
    parser.add_argument("--pos", type=int, default=0, help="Token position (default: 0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for input vector (default: 0)")
    parser.add_argument("--dump-dir", type=str, default="", help="Directory to save intermediate .npy files")
    parser.add_argument("--check-golden", action="store_true", help="Check against golden data trace in tools/ref/data_gemma/")

    args = parser.parse_args()

    # Always validate dequant_q4_0
    validate_dequant_q4_0()

    golden_dir = os.path.join(Path(__file__).resolve().parent, "data_gemma")
    if args.check_golden or (not args.weights_dir and os.path.exists(golden_dir)):
        run_golden_check(golden_dir)
        if not args.weights_dir:
            return

    if not args.weights_dir:
        print("Usage: python tools/ref/fused_layer_ref.py <weights_dir> [--layer 0] [--seed 0] [--dump-dir DIR]")
        return

    np.random.seed(args.seed)

    cfg_path = os.path.join(args.weights_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    hidden_size = cfg.get("hidden_size", 1152)
    x = np.random.randn(hidden_size).astype(np.float32)

    out, intermediates = layer_forward(args.weights_dir, args.layer, x, args.pos, config=cfg)
    print(f"Layer {args.layer} forward complete. Output shape: {out.shape}, mean: {np.mean(out):.6f}, std: {np.std(out):.6f}")

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        for key, val in intermediates.items():
            np.save(os.path.join(args.dump_dir, f"{key}.npy"), val)
        print(f"Saved {len(intermediates)} stage tensors to {args.dump_dir}")


if __name__ == "__main__":
    main()
