# 📦 Adding & Quantizing Models

Alveare provides flexible workflows for acquiring, quantizing, and registering large language models for execution on the AMD Ryzen™ AI (XDNA2) NPU.

---

## 🌟 Supported Model Presets

Alveare includes native Q4_0 quantizer pipelines for the following model architectures:

| Alias / ID | Model Name | Architecture Key | Default HuggingFace Source | Size (Q4_0) | Features |
|---|---|---|---|---|---|
| `gemma4` | Gemma 4 12B Instruct | `gemma4` | `unsloth/gemma-4-12b-it-GGUF` | ~9.7 GB | 48 layers, fused NPU FFN, multi-tile GEMV |
| `gemma4-e4b` | Gemma 4 E4B (Edge 4B) | `gemma4-e4b` | `unsloth/gemma-4-e4b-it-GGUF` | ~3.8 GB | Per-Layer Embedding (PLE) injection |
| `gemma3` | Gemma 3 1B Instruct | `gemma3` | `unsloth/gemma-3-1b-it-GGUF` | ~0.8 GB | 26 layers, SentencePiece SPM, NPU FFN |
| `llama` | Llama 3.2 1B Instruct | `llama` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | ~0.8 GB | Llama architecture, GQA attention |
| `whisper-base` | Whisper Base STT | `whisper` | `openai/whisper-base` | ~145 MB | High-accuracy speech-to-text, Italian & 90+ langs |

---

## 🚀 Workflow 1: 1-Click Setup via Web UI

The easiest way to install a model is using the interactive React Web UI dashboard:

1. Launch Alveare:
   ```bash
   ./alveare serve
   ```
2. Open `http://127.0.0.1:8000` in your browser.
3. Click **"Aggiungi Modello"** in the top navigation bar.
4. Select a supported preset (e.g. **Gemma 4 12B**, **Gemma 4 E4B**, or **Gemma 3 1B**), or select **Custom** to specify a custom Hugging Face GGUF repository or local `.gguf` file path.
5. Click **"Installa"**.

### What happens automatically:
- **Download**: Fetches the GGUF model file from Hugging Face via `huggingface_hub` (or direct HTTPS stream).
- **Quantization**: Converts tensor weights into Alveare's NPU-aligned `Q4_0` layout (32-element blocks with FP32 scale factors pre-tiled for AIE memory controllers).
- **Kernel Build**: Compiles or verifies the matching `.xclbin` NPU hardware kernels in `kernels/build/`.
- **Package Registration**: Emits `config.json` and `tokenizer.json` into `./quantized_weights_<alias>/`.

---

## 💻 Workflow 2: Automated Setup via CLI

You can trigger the 1-click pipeline directly from the command line using `./alveare setup` or `tools/setup_model.py`:

```bash
# Install Gemma 4 12B from default HuggingFace repository
./alveare setup gemma4 --arch gemma4

# Install Gemma 3 1B
./alveare setup gemma3 --arch gemma3

# Install Gemma 4 E4B
./alveare setup gemma4-e4b --arch gemma4-e4b

# Install & Initialize Whisper Base STT
./alveare setup whisper-base
```

### Advanced `setup_model.py` arguments:

```bash
python tools/setup_model.py <alias> \
  --arch gemma4 \
  --url https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf \
  --filename gemma-4-12b-it-Q4_K_M.gguf
```

- `--arch <arch>`: Architecture key (`gemma4`, `gemma4-e4b`, `gemma3`, `llama`, or `custom`).
- `--url <URL>`: Direct download URL or Hugging Face repository ID.
- `--filename <name>`: Specific `.gguf` filename inside the HF repository.
- `--gguf <path>`: Local path to an already downloaded GGUF file.

---

## 🛠️ Workflow 3: Manual GGUF Quantization

If you already have a local `.gguf` file on disk:

```bash
./alveare quantize g4-12b /path/to/gemma-4-12b-it-Q4_K_M.gguf
```

- **Architecture Auto-Detection**: `alveare quantize` reads `general.architecture` from the GGUF header and selects the corresponding quantizer.
- **Manual Arch Override**: Force a specific quantizer with `--arch`:
  ```bash
  ./alveare quantize g3-1b /path/to/gemma-3-1b-it.gguf --arch gemma3
  ```

---

## 🔌 Custom Quantizer Plugins

Alveare supports modular quantizer plugins. You can add support for a new model family by inheriting from `BaseQuantizer` in `tools/convert/base_quantizer.py`:

```python
from tools.convert.base_quantizer import BaseQuantizer

class CustomQuantizer(BaseQuantizer):
    def __init__(self):
        super().__init__(name="custom_model")

    def quantize(self, gguf_path: str, out_dir: str) -> dict:
        # Load GGUF, process weights, pack into Q4_0 layout, write config.json
        ...
```

For a comprehensive plugin development guide, see [`docs/CUSTOM_QUANTIZERS.md`](CUSTOM_QUANTIZERS.md).

---

## 🗑️ Managing & Deleting Models

Models are stored in package directories under the project root (`./quantized_weights_<alias>`).

- **Via Web UI**: Click the red trash icon on the model card in the dashboard.
- **Via CLI**:
  ```bash
  ./alveare delete gemma3
  ```
- **List Installed Models**:
  ```bash
  ./alveare list
  ```
