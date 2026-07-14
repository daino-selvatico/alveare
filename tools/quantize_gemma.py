import os
import sys
import json
import numpy as np
from pathlib import Path
from gguf import GGUFReader

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined

def pad_matrix(W: np.ndarray, target_N: int, target_K: int) -> np.ndarray:
    """Pad matrix W to target_N x target_K with zeros."""
    N, K = W.shape
    if N == target_N and K == target_K:
        return W
    print(f"Padding matrix of shape {W.shape} to {(target_N, target_K)}")
    padded = np.zeros((target_N, target_K), dtype=W.dtype)
    padded[:N, :K] = W
    return padded

def unpermute(weights: np.ndarray, n_head: int) -> np.ndarray:
    """Reverse the llama.cpp RoPE permutation layout."""
    head_dim = weights.shape[0] // n_head
    return (weights.reshape(n_head, head_dim // 2, 2, weights.shape[1])
            .swapaxes(1, 2)
            .reshape(weights.shape))

def quantize_and_pack_tensor(W: np.ndarray, target_N: int, target_K: int) -> np.ndarray:
    """Pad, quantize, and pack weight matrix W."""
    W_padded = pad_matrix(W, target_N, target_K)
    w_q4, scales = quantize_to_q4_0(W_padded)
    w_combined = pack_to_combined(w_q4, scales)
    return w_combined

DEFAULT_GGUF = "/home/daino/llama-mtp/models/google_gemma-3-1b-it-bf16.gguf"
DEFAULT_OUT = str(Path(__file__).resolve().parents[1] / "quantized_weights_gemma")

def main(gguf_path=DEFAULT_GGUF, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading GGUF from {gguf_path}...")
    reader = GGUFReader(gguf_path)
    
    # Save config
    config = {
        "model_type": "gemma3",
        "hidden_size": 1152,
        "intermediate_size": 6912,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "num_hidden_layers": 26,
        "max_seq_len": 2048,
        "vocab_size": 262144
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")
    
    # Iterate through tensors and process them
    for tensor in reader.tensors:
        name = tensor.name
        # GGUF Reader might read weights as uint8 raw bytes for bf16
        # Let's view them as bfloat16 first if they are raw bytes
        from ml_dtypes import bfloat16
        if tensor.data.dtype == np.uint8 and len(tensor.data.shape) == 2 and tensor.data.shape[1] % 2 == 0:
            data = tensor.data.view(bfloat16)
        else:
            data = tensor.data
            
        print(f"Processing tensor {name} with shape {data.shape}...")
        
        if "token_embd.weight" in name:
            # Save raw embedding in float16 for host lookup
            # In GGUF, shape is (262144, 1152)
            np.save(out_dir / "token_embd.npy", data.astype(np.float16))
            print(f"Saved float16 embedding table.")
            
            # Since embeddings are tied, we reuse token_embd.weight as the LM head!
            # Shape is (262144, 1152) -> pad K to 2048
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 262144, 2048)
            np.save(out_dir / "lm_head_packed.npy", w_combined)
            print(f"Quantized and saved packed LM head (tied embedding).")
            
        elif "norm.weight" in name:
            # Norm weights are saved in float32 directly
            # For gemma, we keep them as is
            np.save(out_dir / f"{name}.npy", data.astype(np.float32))
            print(f"Saved float32 norm weights.")
            
        elif "attn_q.weight" in name:
            # Shape (1024, 1152)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Q projection.")
            
        elif "attn_k.weight" in name:
            # Shape (256, 1152)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed K projection.")
            
        elif "attn_v.weight" in name:
            # Shape (256, 1152)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed V projection.")
            
        elif "attn_output.weight" in name:
            # Shape (1152, 1024)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed O projection.")
            
        elif "ffn_gate.weight" in name or "ffn_up.weight" in name:
            # Shape (6912, 1152) -> pad N to 8192, K to 2048
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 8192, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Gate/Up projection.")
            
        elif "ffn_down.weight" in name:
            # Shape (1152, 6912) -> pad N to 2048, K to 8192
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 8192)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Down projection.")
            
        else:
            print(f"Skipping tensor {name}")
            
    print("Quantization completed successfully!")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quantize a Gemma-3-1B GGUF into Alveare's Q4_0 NPU weight layout.")
    ap.add_argument("gguf", nargs="?", default=DEFAULT_GGUF, help="source GGUF file (default: %(default)s)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output weights directory (default: %(default)s)")
    args = ap.parse_args()
    main(gguf_path=args.gguf, out_dir=args.out)
