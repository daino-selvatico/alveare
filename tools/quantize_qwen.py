import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.convert.gemv_q_convert import quantize_to_q4_0, pack_to_combined
from tools.convert.base_quantizer import BaseQuantizer

try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType
except ModuleNotFoundError:
    sys.path.append(os.environ.get("LLAMA_CPP_GGUF_PY", "/home/daino/llama-mtp/llama.cpp/gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType

DEFAULT_GGUF = "/home/daino/llama-mtp/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
DEFAULT_OUT = str(Path(__file__).resolve().parents[1] / "quantized_weights_qwen")

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

class QwenQuantizer(BaseQuantizer):
    """
    Quantizer for Qwen models including Qwen 2.5 / Qwen 3.8 GSQ-RCO with MTP.
    Extracts Q/K/V/O, FFN (gate/up/down), RMSNorm and MTP speculative heads.
    """
    def __init__(self):
        super().__init__(name="qwen")

    def quantize(self, gguf_path: str, out_dir: str):
        return main(gguf_path=gguf_path, out_dir=out_dir)

def main(gguf_path=DEFAULT_GGUF, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[quantize_qwen] Loading GGUF from {gguf_path}...")
    reader = GGUFReader(gguf_path)

    # Read config metadata dynamically with fallbacks for Qwen 27B
    arch_prefix = "qwen2"
    if "qwen3" in reader.fields:
        arch_prefix = "qwen3"
    elif "qwen" in reader.fields:
        arch_prefix = "qwen"

    block_count = 64
    hidden_size = 5120
    intermediate_size = 27648
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    vocab_size = 152064

    for k, v in reader.fields.items():
        if k.endswith(".block_count"):
            block_count = int(v.parts[0])
        elif k.endswith(".embedding_length"):
            hidden_size = int(v.parts[0])
        elif k.endswith(".feed_forward_length"):
            intermediate_size = int(v.parts[0])
        elif k.endswith(".attention.head_count"):
            num_attention_heads = int(v.parts[0])
        elif k.endswith(".attention.head_count_kv"):
            num_key_value_heads = int(v.parts[0])
        elif k.endswith(".attention.key_length"):
            head_dim = int(v.parts[0])
        elif k.endswith(".vocab_size"):
            vocab_size = int(v.parts[0])

    config = {
        "model_type": "qwen",
        "architecture": "qwen",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "num_hidden_layers": block_count,
        "max_seq_len": 4096,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-6,
        "has_mtp": any("mtp" in t.name.lower() or "spec" in t.name.lower() for t in reader.tensors)
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[quantize_qwen] Saved config.json: {config}")

    try:
        from tools.convert.gguf_tokenizer import write_tokenizer_json
        write_tokenizer_json(reader, out_dir)
    except Exception as e:
        print(f"[quantize_qwen] Tokenizer export note: {e}")

    # Process tensors
    for tensor in reader.tensors:
        name = tensor.name
        is_needed = (
            "token_embd" in name or
            "output_norm" in name or
            "output.weight" in name or
            name.startswith("blk.") or
            "mtp" in name or
            "speculative" in name
        )
        if not is_needed:
            continue

        try:
            qtype = GGMLQuantizationType(tensor.tensor_type)
            data = dequantize(tensor.data, qtype)
        except Exception:
            data = tensor.data.astype(np.float32)

        if "token_embd.weight" in name:
            np.save(out_dir / "token_embd.npy", data.astype(np.float16))
            # LM head if tied or untied
            padded_vocab = int(np.ceil(data.shape[0] / 2048.0)) * 2048
            padded_hidden = int(np.ceil(data.shape[1] / 2048.0)) * 2048
            w_combined = quantize_and_pack_tensor(data.astype(np.float32), padded_vocab, padded_hidden)
            np.save(out_dir / "lm_head_packed.npy", w_combined)
            print(f"[quantize_qwen] Saved token_embd and packed LM head.")

        elif "output.weight" in name:
            padded_vocab = int(np.ceil(data.shape[0] / 2048.0)) * 2048
            padded_hidden = int(np.ceil(data.shape[1] / 2048.0)) * 2048
            w_combined = quantize_and_pack_tensor(data.astype(np.float32), padded_vocab, padded_hidden)
            np.save(out_dir / "lm_head_packed.npy", w_combined)
            print(f"[quantize_qwen] Saved dedicated packed LM head.")

        elif "norm.weight" in name:
            np.save(out_dir / f"{name}.npy", data.astype(np.float32))

        elif "attn_q.weight" in name or "attn_k.weight" in name or "attn_v.weight" in name or "attn_output.weight" in name:
            W_fp32 = data.astype(np.float32)
            target_n = int(np.ceil(W_fp32.shape[0] / 2048.0)) * 2048
            target_k = int(np.ceil(W_fp32.shape[1] / 2048.0)) * 2048
            w_combined = quantize_and_pack_tensor(W_fp32, target_n, target_k)
            np.save(out_dir / f"{name}_packed.npy", w_combined)

        elif "ffn_gate.weight" in name or "ffn_up.weight" in name or "ffn_down.weight" in name:
            W_fp32 = data.astype(np.float32)
            target_n = int(np.ceil(W_fp32.shape[0] / 2048.0)) * 2048
            target_k = int(np.ceil(W_fp32.shape[1] / 2048.0)) * 2048
            w_combined = quantize_and_pack_tensor(W_fp32, target_n, target_k)
            np.save(out_dir / f"{name}_packed.npy", w_combined)

        elif "mtp" in name or "speculative" in name:
            # Multi-Token Prediction speculative head weights
            np.save(out_dir / f"{name}.npy", data.astype(np.float32))
            print(f"[quantize_qwen] Saved MTP speculative head tensor {name}")

    print("[quantize_qwen] Quantization and weight packing completed successfully!")
    return config

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Quantize Qwen 2.5 / 3.8 GSQ-RCO into Alveare layout.")
    ap.add_argument("gguf", nargs="?", default=DEFAULT_GGUF, help="source GGUF file")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output weights directory")
    args = ap.parse_args()
    main(gguf_path=args.gguf, out_dir=args.out)
