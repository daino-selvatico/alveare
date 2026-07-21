import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined

# GGUFReader + dequantize come from the `gguf` pip package (see requirements.txt).
# Only fall back to a local llama.cpp gguf-py checkout if the package is absent.
try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType
except ModuleNotFoundError:
    sys.path.append(os.environ.get("LLAMA_CPP_GGUF_PY", "/home/daino/llama-mtp/llama.cpp/gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType

DEFAULT_GGUF = "/home/daino/llama-mtp/models/gemma-4-12b-it-UD-Q4_K_XL.gguf"
DEFAULT_OUT = str(Path(__file__).resolve().parents[1] / "quantized_weights_gemma4")

def pad_matrix(W: np.ndarray, target_N: int, target_K: int) -> np.ndarray:
    """Pad matrix W to target_N x target_K with zeros."""
    N, K = W.shape
    if N == target_N and K == target_K:
        return W
    print(f"Padding matrix of shape {W.shape} to {(target_N, target_K)}")
    padded = np.zeros((target_N, target_K), dtype=W.dtype)
    padded[:N, :K] = W
    return padded

def quantize_and_pack_tensor(W: np.ndarray, target_N: int, target_K: int) -> np.ndarray:
    """Pad, quantize, and pack weight matrix W."""
    W_padded = pad_matrix(W, target_N, target_K)
    w_q4, scales = quantize_to_q4_0(W_padded)
    w_combined = pack_to_combined(w_q4, scales)
    return w_combined

def main(gguf_path=DEFAULT_GGUF, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading GGUF from {gguf_path}...")
    reader = GGUFReader(gguf_path)
    
    # Read config dynamically from GGUF metadata with robust fallbacks
    try:
        block_count = int(reader.fields["gemma4.block_count"].parts[0])
    except Exception:
        block_count = 48
    try:
        hidden_size = int(reader.fields["gemma4.embedding_length"].parts[0])
    except Exception:
        hidden_size = 3840
    try:
        intermediate_size = int(reader.fields["gemma4.feed_forward_length"].parts[0])
    except Exception:
        intermediate_size = 15360
    try:
        num_attention_heads = int(reader.fields["gemma4.attention.head_count"].parts[0])
    except Exception:
        num_attention_heads = 16
    try:
        head_dim = int(reader.fields["gemma4.attention.key_length_swa"].parts[0])
    except Exception:
        head_dim = 256
        
    num_key_value_heads = 8
    try:
        kv_field = reader.fields.get("gemma4.attention.head_count_kv")
        if kv_field is not None:
            for part in kv_field.parts:
                if isinstance(part, (int, np.integer)):
                    num_key_value_heads = int(part)
                    break
                elif hasattr(part, "item"):
                    num_key_value_heads = int(part.item())
                    break
    except Exception:
        pass

    config = {
        "model_type": "gemma4",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "num_hidden_layers": block_count,
        "max_seq_len": 2048,
        "vocab_size": 262144
    }
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json: {config}")

    # Emit tokenizer.json (from the GGUF's embedded tokenizer) for the native
    # C++ runtime — keeps the pipeline fully offline, no HF download needed.
    from tools.convert.gguf_tokenizer import write_tokenizer_json
    write_tokenizer_json(reader, out_dir)
    
    # Iterate through tensors and process them
    for tensor in reader.tensors:
        name = tensor.name
        
        is_needed = ("token_embd" in name or 
                     "output_norm" in name or 
                     name.startswith("blk."))
        
        if not is_needed:
            continue
            
        print(f"Dequantizing tensor {name}...")
        qtype = GGMLQuantizationType(tensor.tensor_type)
        data = dequantize(tensor.data, qtype)
        
        print(f"Processing tensor {name} with dequantized shape {data.shape}...")
        
        if "token_embd.weight" in name:
            # Save embedding table in float16
            np.save(out_dir / "token_embd.npy", data.astype(np.float16))
            print(f"Saved float16 embedding table.")
            
            # Since embeddings are tied, we reuse token_embd.weight as the LM head!
            # Shape is (262144, 3840) -> pad K to 4096
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 262144, 4096)
            np.save(out_dir / "lm_head_packed.npy", w_combined)
            print(f"Quantized and saved packed LM head (tied embedding).")
            
        elif "norm.weight" in name:
            # Norm weights are saved in float32 directly
            np.save(out_dir / f"{name}.npy", data.astype(np.float32))
            print(f"Saved float32 norm weights.")
            
        elif "layer_output_scale.weight" in name:
            # Save layer scalar (scale factor) in float32 directly
            np.save(out_dir / f"{name}.npy", data.astype(np.float32))
            print(f"Saved float32 layer output scale.")
            
        elif "attn_q.weight" in name:
            W_fp32 = data.astype(np.float32)
            # Sliding (4096, 3840) -> (4096, 4096)
            # Global (8192, 3840) -> (8192, 4096)
            w_combined = quantize_and_pack_tensor(W_fp32, W_fp32.shape[0], 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Q projection.")
            
        elif "attn_k.weight" in name:
            W_fp32 = data.astype(np.float32)
            # Sliding (2048, 3840) -> (2048, 4096)
            # Global (512, 3840) -> pad rows to 2048 -> (2048, 4096)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed K projection.")
            
        elif "attn_v.weight" in name:
            W_fp32 = data.astype(np.float32)
            # Sliding (2048, 3840) -> (2048, 4096)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed V projection.")
            
        elif "attn_output.weight" in name:
            W_fp32 = data.astype(np.float32)
            # Sliding (3840, 4096) -> (4096, 4096)
            # Global (3840, 8192) -> (4096, 8192)
            w_combined = quantize_and_pack_tensor(W_fp32, 4096, W_fp32.shape[1])
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed O projection.")
            
        elif "ffn_gate.weight" in name or "ffn_up.weight" in name:
            W_fp32 = data.astype(np.float32)
            # Shape (15360, 3840) -> (16384, 4096)
            w_combined = quantize_and_pack_tensor(W_fp32, 16384, 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Gate/Up projection.")
            
        elif "ffn_down.weight" in name:
            W_fp32 = data.astype(np.float32)
            # Shape (3840, 15360) -> (4096, 16384)
            w_combined = quantize_and_pack_tensor(W_fp32, 4096, 16384)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Down projection.")
            
        else:
            print(f"Skipping tensor {name}")
            
    print("Quantization completed successfully!")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quantize a Gemma-4-12B GGUF into Alveare's Q4_0 NPU weight layout.")
    ap.add_argument("gguf", nargs="?", default=DEFAULT_GGUF, help="source GGUF file (default: %(default)s)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output weights directory (default: %(default)s)")
    args = ap.parse_args()
    main(gguf_path=args.gguf, out_dir=args.out)
