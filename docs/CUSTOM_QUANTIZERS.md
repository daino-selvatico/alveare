# Writing Custom Quantizers for Alveare NPU

Alveare supports custom quantization plugins for new models or experimental weight layouts. You can pass a path to any Python script (`/path/to/custom_quantizer.py`) via the **Web UI ("Aggiungi Modello" -> Mode Manuale)** or the CLI (`./alveare quantize <alias> <file.gguf> --quantizer /path/to/script.py`).

---

## 🛠️ Quantizer Plugin Requirements

A custom quantizer plugin can be implemented in one of three simple ways:

### Option A: Subclassing `BaseQuantizer` (Recommended)

Inherit from `tools.convert.base_quantizer.BaseQuantizer` and implement the `quantize(gguf_path, out_dir)` method:

```python
import json
import numpy as np
from pathlib import Path
from tools.convert.base_quantizer import BaseQuantizer
from gguf import GGUFReader, dequantize, GGMLQuantizationType

class CustomMyModelQuantizer(BaseQuantizer):
    def __init__(self):
        super().__init__(name="custom_mymodel")

    def quantize(self, gguf_path: str, out_dir: str) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        reader = GGUFReader(gguf_path)
        
        # 1. Save config.json
        config = {
            "model_type": "llama", # or gemma3, gemma4
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "num_hidden_layers": 16,
            "vocab_size": 128256
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # 2. Extract, pad, and quantize linear projection weights to Q4_0 layout
        for tensor in reader.tensors:
            name = tensor.name
            qtype = GGMLQuantizationType(tensor.tensor_type)
            data = dequantize(tensor.data, qtype).astype(np.float32)

            if "attn_q.weight" in name:
                # Use self.quantize_and_pack_tensor(data, target_N, target_K)
                w_packed = self.quantize_and_pack_tensor(data, 2048, 2048)
                np.save(out_dir / f"{name}_packed.npy", w_packed)

        return config
```

---

### Option B: Defining a `quantize(gguf_path, out_dir)` Function

Alternatively, your standalone script can export a top-level function `quantize(gguf_path, out_dir)`:

```python
def quantize(gguf_path: str, out_dir: str) -> dict:
    # Read GGUF, process weights, write config.json & packed .npy files
    ...
```

---

### Option C: Defining a `main(gguf_path, out_dir)` Function

Or a top-level `main(gguf_path, out_dir)` function:

```python
def main(gguf_path: str, out_dir: str) -> dict:
    # Read GGUF, process weights, write config.json & packed .npy files
    ...
```

---

## 📦 Required Output Files in `out_dir`

After quantization, the `out_dir` must contain:

1. **`config.json`**: Model dimensions (`model_type`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `num_hidden_layers`, `vocab_size`).
2. **`tokenizer.json`**: Tokenizer specification for the C++ runtime.
3. **`token_embd.npy`**: Embedding table in `float16`.
4. **`*.weight_packed.npy`**: Linear projections packed into Alveare's `Q4_0` NPU layout (`(N, K // 32 * 20)` uint8).
5. **`*.norm.weight.npy`**: LayerNorm / RMSNorm weights in `float32`.

---

## 🚀 Testing Your Custom Quantizer

Run your custom quantizer via CLI:

```bash
./alveare quantize mymodel ./path/to/model.gguf --quantizer /path/to/my_quantizer.py
```

Or open the **Web UI**, click **"Aggiungi Modello"**, select **Mode Manuale**, choose your GGUF file and pick **Custom** under architecture to point to your script.
