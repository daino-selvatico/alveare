import os
import sys
import numpy as np
from pathlib import Path
from gguf import GGUFReader
from ml_dtypes import bfloat16

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

DEFAULT_GGUF = "/home/daino/llama-mtp/models/Llama-3.2-1B-Instruct-f16.gguf"
DEFAULT_OUT = str(Path(__file__).resolve().parents[1] / "quantized_weights")

def main(gguf_path=DEFAULT_GGUF, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading GGUF from {gguf_path}...")
    reader = GGUFReader(gguf_path)
    
    # We will iterate through tensors and process them
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        print(f"Processing tensor {name} with shape {data.shape}...")
        
        # Standard GGUF weight tensors have shape [K, N] in python GGML order,
        # but wait, let's verify if they need transposing.
        # In GGUFReader, tensor.data shape is standard. Let's look at the shape.
        # blk.0.attn_q.weight has shape (2048, 2048).
        # blk.0.attn_k.weight has shape (512, 2048).
        # blk.0.ffn_down.weight has shape (2048, 8192).
        # token_embd.weight has shape (128256, 2048).
        # We need N (rows) and K (cols) to be multiples of 2048.
        
        if "token_embd.weight" in name:
            # Save the raw embedding in float16 for the host-side lookup
            # We also save a quantized version for the LM head
            np.save(out_dir / "token_embd.npy", data.astype(np.float16))
            print(f"Saved float16 embedding table.")
            
            # Pad vocabulary dimension from 128256 to 129024 (63 * 2048)
            W_padded = pad_matrix(data.astype(np.float32), 129024, 2048)
            w_combined = quantize_and_pack_tensor(W_padded, 129024, 2048)
            np.save(out_dir / "lm_head_packed.npy", w_combined)
            print(f"Quantized and saved packed LM head.")
            
        elif "attn_norm.weight" in name or "ffn_norm.weight" in name or "output_norm.weight" in name:
            # Norm weights are saved in float32 directly
            np.save(out_dir / f"{name}.npy", data.astype(np.float32))
            print(f"Saved float32 norm weights.")
            
        elif "attn_q.weight" in name:
            # Shape (2048, 2048)
            W_fp32 = data.astype(np.float32)
            W_unpermuted = unpermute(W_fp32, 32)
            w_combined = quantize_and_pack_tensor(W_unpermuted, 2048, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed unpermuted Q projection.")
            
        elif "attn_k.weight" in name:
            # Shape (512, 2048) -> pad N to 2048 after unpermuting
            W_fp32 = data.astype(np.float32)
            W_unpermuted = unpermute(W_fp32, 8)
            w_combined = quantize_and_pack_tensor(W_unpermuted, 2048, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed unpermuted K projection.")
            
        elif "attn_output.weight" in name:
            # Shape (2048, 2048)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed (2048, 2048) projection.")
            
        elif "attn_v.weight" in name:
            # Shape (512, 2048) -> pad N to 2048
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed padded (2048, 2048) V projection.")
            
        elif "ffn_gate.weight" in name or "ffn_up.weight" in name:
            # Shape (8192, 2048) -> already multiple of 2048
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 8192, 2048)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed (8192, 2048) projection.")
            
        elif "ffn_down.weight" in name:
            # Shape (2048, 8192) -> already multiple of 2048
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 8192)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed (2048, 8192) projection.")
            
        else:
            print(f"Skipping tensor {name} (not a matmul or layer norm).")
            
    print("Quantization completed successfully!")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quantize a Llama-3.2-1B GGUF into Alveare's Q4_0 NPU weight layout.")
    ap.add_argument("gguf", nargs="?", default=DEFAULT_GGUF, help="source GGUF file (default: %(default)s)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output weights directory (default: %(default)s)")
    args = ap.parse_args()
    main(gguf_path=args.gguf, out_dir=args.out)
