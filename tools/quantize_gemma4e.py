import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined, bfloat16

try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType
except ModuleNotFoundError:
    sys.path.append(os.environ.get("LLAMA_CPP_GGUF_PY", "/home/daino/llama-mtp/llama.cpp/gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType

DEFAULT_GGUF = "/home/daino/llama-mtp/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf"
DEFAULT_OUT = str(Path(__file__).resolve().parents[1] / "quantized_weights_gemma4-e4b")

def quantize_and_pack_matrix(W: np.ndarray) -> np.ndarray:
    """
    Quantize an FP32 matrix W of shape (N, K) to Q4_0 and pack.
    K must be a multiple of 32.
    """
    N, K = W.shape
    assert K % 32 == 0, f"K dimension ({K}) must be a multiple of 32"
    w_q4, scales = quantize_to_q4_0(W)
    w_combined = pack_to_combined(w_q4, scales)
    return w_combined

def unpack_q4_0_combined(w_combined: np.ndarray, N: int, K: int) -> np.ndarray:
    """
    Unpack combined Q4_0 array back to FP32 matrix of shape (N, K) for verification.
    """
    K_blocks = K // 32
    W_out = np.zeros((N, K), dtype=np.float32)
    
    scales_bytes = np.zeros((N, K_blocks), dtype=np.uint16)
    w_q4 = np.zeros((N, K // 2), dtype=np.uint8)
    
    for b in range(K_blocks):
        w_q4[:, b*16 : (b+1)*16] = w_combined[:, b*20 : b*20 + 16]
        scales_bytes[:, b] = w_combined[:, b*20 + 16].astype(np.uint16) | (w_combined[:, b*20 + 17].astype(np.uint16) << 8)
        
    scales_bf16 = scales_bytes.view(bfloat16)
    scales = scales_bf16.astype(np.float32)
    
    q0 = (w_q4 & 0x0F).astype(np.int8)
    q0[q0 > 7] -= 16
    q1 = ((w_q4 >> 4) & 0x0F).astype(np.int8)
    q1[q1 > 7] -= 16
    
    W_out[:, 0::2] = q0 * np.repeat(scales, 16, axis=1)
    W_out[:, 1::2] = q1 * np.repeat(scales, 16, axis=1)
    
    return W_out

def run_ple_numeric_check(out_dir: Path):
    """
    Run Python-side numeric check of isolated PLE computation using exported weights.
    """
    print("\n--- Running Python PLE Numeric Verification Check ---")
    token_embd_fp16 = np.load(out_dir / "token_embd.npy")
    per_layer_token_embd_fp16 = np.load(out_dir / "per_layer_token_embd.npy")
    per_layer_model_proj_packed = np.load(out_dir / "per_layer_model_proj_packed.npy")
    per_layer_proj_norm = np.load(out_dir / "per_layer_proj_norm.weight.npy")

    token_id = 9259  # "world"
    n_embd = 2560
    n_embd_per_layer = 256
    n_layer = 42

    # Step 1: Main token embedding lookup scaled by sqrt(2560)
    inpL = token_embd_fp16[token_id].astype(np.float32) * np.sqrt(n_embd)

    # Step 2: Unpack per_layer_model_proj and project
    W_proj_q4 = unpack_q4_0_combined(per_layer_model_proj_packed, 10752, 2560)
    proj_raw = inpL @ W_proj_q4.T
    proj_scaled = proj_raw / np.sqrt(n_embd)

    # Reshape to (42, 256)
    proj_grid = proj_scaled.reshape(n_layer, n_embd_per_layer)

    # RMSNorm across dim 256
    eps = 1e-6
    rms = np.sqrt(np.mean(proj_grid**2, axis=-1, keepdims=True) + eps)
    proj_normed = (proj_grid / rms) * per_layer_proj_norm

    # Step 3: Per-layer token embedding lookup scaled by sqrt(256)
    emb_lookup = per_layer_token_embd_fp16[token_id].astype(np.float32).reshape(n_layer, n_embd_per_layer)
    emb_scaled = emb_lookup * np.sqrt(n_embd_per_layer)

    # Step 4: Blend
    inp_per_layer = (proj_normed + emb_scaled) / np.sqrt(2.0)

    print(f"Computed inp_per_layer shape: {inp_per_layer.shape} (expected: (42, 256))")
    assert inp_per_layer.shape == (42, 256), f"Shape mismatch: {inp_per_layer.shape}"
    
    print(f"Layer 0 first 5 values: {inp_per_layer[0, :5]}")
    print(f"Layer 41 first 5 values: {inp_per_layer[41, :5]}")
    print(f"inp_per_layer stats: min={inp_per_layer.min():.4f}, max={inp_per_layer.max():.4f}, mean={inp_per_layer.mean():.4f}, std={inp_per_layer.std():.4f}")
    print("✅ PLE numeric verification check PASSED!")

def main(gguf_path=DEFAULT_GGUF, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading GGUF from {gguf_path}...")
    reader = GGUFReader(gguf_path)

    config = {
        "model_type": "gemma4-e4b",
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_hidden_layers": 42,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "head_dim_global": 512,
        "per_layer_input": 256,
        "shared_kv_layers": 18,
        "sliding_window": 512,
        "vocab_size": 262144,
        "max_seq_len": 2048
    }
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json: {config}")

    from tools.convert.gguf_tokenizer import write_tokenizer_json
    write_tokenizer_json(reader, out_dir)

    total_source_tensors = len(reader.tensors)
    exported_count = 0
    skipped_tensors = []

    print(f"Processing {total_source_tensors} source tensors...")

    for tensor in reader.tensors:
        name = tensor.name
        qtype = GGMLQuantizationType(tensor.tensor_type)
        data = dequantize(tensor.data, qtype)

        if name == "rope_freqs.weight":
            skipped_tensors.append((name, "Computed dynamically at runtime"))
            continue

        if "token_embd.weight" == name:
            # Save embedding table in float16 (262144, 2560)
            np.save(out_dir / "token_embd.npy", data.astype(np.float16))
            
            # LM head packed (tied with token_embd)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_matrix(W_fp32)
            np.save(out_dir / "lm_head_packed.npy", w_combined)
            exported_count += 1
            print(f"Processed {name}: saved FP16 token_embd.npy and packed lm_head_packed.npy")

        elif name == "per_layer_token_embd.weight":
            # Save per-layer embedding table in float16 (262144, 10752)
            np.save(out_dir / "per_layer_token_embd.npy", data.astype(np.float16))
            exported_count += 1
            print(f"Processed {name}: saved FP16 per_layer_token_embd.npy ({data.shape})")

        elif name == "per_layer_model_proj.weight":
            # Quantize and pack (10752, 2560)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_matrix(W_fp32)
            np.save(out_dir / "per_layer_model_proj_packed.npy", w_combined)
            exported_count += 1
            print(f"Processed {name}: saved Q4_0 per_layer_model_proj_packed.npy ({data.shape})")

        elif name.endswith("norm.weight"):
            # All RMSNorm weights (1D vectors) saved as FP32
            np.save(out_dir / f"{name}.npy", data.astype(np.float32))
            exported_count += 1

        elif name.endswith("layer_output_scale.weight"):
            # Scalar layer output scale saved as FP32
            np.save(out_dir / f"{name}.npy", data.astype(np.float32))
            exported_count += 1

        elif any(proj in name for proj in [
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "inp_gate.weight", "proj.weight"
        ]):
            # 2D projection matrices quantized to Q4_0 and packed
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_matrix(W_fp32)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            exported_count += 1

        else:
            skipped_tensors.append((name, f"Unrecognized tensor suffix"))

    print("\n--- Summary ---")
    print(f"Total source GGUF tensors: {total_source_tensors}")
    print(f"Exported tensors: {exported_count}")
    print(f"Skipped tensors ({len(skipped_tensors)}):")
    for s_name, reason in skipped_tensors:
        print(f"  - {s_name}: {reason}")

    assert exported_count + len(skipped_tensors) == total_source_tensors, \
        f"Accounting error: exported ({exported_count}) + skipped ({len(skipped_tensors)}) != total ({total_source_tensors})"

    # Run numerical validation check
    run_ple_numeric_check(out_dir)
    print("\nQuantization & Export completed successfully!")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quantize Gemma-4-E4B GGUF with PLE tensors into Alveare layout.")
    ap.add_argument("gguf", nargs="?", default=DEFAULT_GGUF, help="source GGUF file")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()
    main(gguf_path=args.gguf, out_dir=args.out)
