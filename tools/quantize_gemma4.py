import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined

# Add llama.cpp gguf-py to path to import GGUFReader and dequantize
sys.path.append("/home/daino/llama-mtp/llama.cpp/gguf-py")
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType

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

def main():
    gguf_path = "/home/daino/llama-mtp/models/gemma-4-12b-it-UD-Q4_K_XL.gguf"
    out_dir = Path("/home/daino/progetti/alveare/quantized_weights_gemma4")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading GGUF from {gguf_path}...")
    reader = GGUFReader(gguf_path)
    
    # Save config (only 1 layer to save space, but model parameters match Gemma-4-12B)
    config = {
        "model_type": "gemma4",
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,  # for local layer
        "head_dim": 256,           # for local layer
        "num_hidden_layers": 1,    # We only bring up one layer for M6!
        "max_seq_len": 2048,
        "vocab_size": 262144
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")
    
    # Iterate through tensors and process them
    for tensor in reader.tensors:
        name = tensor.name
        
        # We only need:
        # - non-transformer weights: token_embd, output_norm
        # - layer 0 weights: blk.0.*
        is_needed = ("token_embd" in name or 
                     "output_norm" in name or 
                     name.startswith("blk.0."))
        
        if not is_needed:
            continue
            
        print(f"Dequantizing tensor {name}...")
        qtype = GGMLQuantizationType(tensor.tensor_type)
        data = dequantize(tensor.data, qtype)
        
        # NumPy dequantized shape might need to be cast / handled
        print(f"Processing tensor {name} with dequantized shape {data.shape}...")
        
        if "token_embd.weight" in name:
            # GGUF shape is usually (vocab_size, hidden_size) or (hidden_size, vocab_size)
            # In our dequantize test we saw shape (262144, 3840)
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
            # Shape (4096, 3840)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 4096, 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Q projection.")
            
        elif "attn_k.weight" in name:
            # Shape (2048, 3840)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed K projection.")
            
        elif "attn_v.weight" in name:
            # Shape (2048, 3840)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 2048, 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed V projection.")
            
        elif "attn_output.weight" in name:
            # Shape (3840, 4096)
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 4096, 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed O projection.")
            
        elif "ffn_gate.weight" in name or "ffn_up.weight" in name:
            # Shape (15360, 3840) -> pad N to 16384, K to 4096
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 16384, 4096)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Gate/Up projection.")
            
        elif "ffn_down.weight" in name:
            # Shape (3840, 15360) -> pad N to 4096, K to 16384
            W_fp32 = data.astype(np.float32)
            w_combined = quantize_and_pack_tensor(W_fp32, 4096, 16384)
            np.save(out_dir / f"{name}_packed.npy", w_combined)
            print(f"Quantized and saved packed Down projection.")
            
        else:
            print(f"Skipping tensor {name}")
            
    print("Quantization completed successfully!")

if __name__ == "__main__":
    main()
