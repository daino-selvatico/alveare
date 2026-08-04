# 🐝 Alveare

**An open-source LLM inference runtime and compiler for the AMD Ryzen™ AI (XDNA2) NPU on Linux.**

Alveare runs large language models on the AMD Ryzen AI NPU with a **100% open-source stack** — including the AIE hardware kernels, the native C++ inference engine, the Python control server, and an interactive React Web UI dashboard.

> **Status: Production-ready Native C++ Runtime with Fused NPU Kernels & Web UI.** Alveare runs **Gemma-4-12B**, **Gemma-4-E4B** (with Per-Layer Embedding support), **Gemma-3-1B**, and **Llama-3.2-1B** end-to-end on the NPU. The default backend is a high-performance **native C++ server** (`alveare_runtime`) featuring fused AIE hardware kernels, memory-resident weights, C++ KV caching, and a self-contained BPE/SPM tokenizer.

---

## 🌟 Key Features

- **🚀 Native C++ Inference Server (`alveare_runtime`)**: Zero Python overhead in the generation loop. Direct XRT integration, fused FFN (gate/up + GeGLU + down in a single `.xclbin`), and multi-core GEMV kernels distributed across all 32 NPU compute tiles.
- **🖥️ Modern React Web UI Dashboard**: A sleek, real-time control center accessible at `http://127.0.0.1:8000`. Features real-time NPU telemetry, server controls, terminal log streaming, and an interactive chat playground.
- **⚡ 1-Click Automated Model Installation**: "Aggiungi Modello" modal in the Web UI allows 1-click downloads & quantization from HuggingFace for supported models (Gemma-4 12B, Gemma-4 E4B, Gemma-3 1B), as well as manual custom GGUF setup.
- **🧩 Extensible Plugin System for Custom Models**: Build and register custom quantizers by extending `BaseQuantizer` (`tools/convert/base_quantizer.py`). Bring any architecture to the NPU with custom tensor tiling and layout rules.
- **🔌 Drop-In OpenAI-Compatible API**: Fully compliant `/v1/chat/completions` and `/v1/models` HTTP endpoints. Seamlessly connects to standard client libraries (OpenAI Python SDK, LangChain, LlamaIndex, Continue, etc.).
- **🗑️ Full Model Lifecycle Management**: List, inspect, serve, setup, and delete quantized model packages via both CLI (`./alveare delete <alias>`) and Web UI.

---

## 📦 Supported Models

| Model | Architecture Key | Features | Output Size (Q4_0) |
|---|---|---|---|
| **Gemma 4 12B** | `gemma4` | 48 layers, fused FFN, multi-tile GEMV | ~9.7 GB |
| **Gemma 4 E4B** | `gemma4e` | Per-Layer Embedding (PLE) injection | ~3.8 GB |
| **Gemma 3 1B** | `gemma3` | SentencePiece SPM tokenizer, 26 layers | ~0.8 GB |
| **Llama 3.2 1B** | `llama` | Llama architecture, GQA attention | ~0.8 GB |

---

## 🚀 Quick Start

### 1. One-Command Installation

Run the automated installer to set up the Conda environment (`alveare-aie`), XRT dependencies, AIE compiler toolchain (`mlir-aie`, `llvm-aie`), and Python/C++ binaries:

```bash
./install.sh
```

### 2. Verify NPU Hardware

Run the NPU smoke test to verify Linux driver access (`/dev/accel/accel0`), XRT environment, and kernel execution:

```bash
./alveare check
```

### 3. Add and Quantize a Model

#### Option A: Via Web UI (Recommended)
Launch the server control panel and open `http://127.0.0.1:8000`:

```bash
./alveare serve
```
Click **"Aggiungi Modello"**, select a model (e.g. Gemma 4 12B), and click **"Installa"**. Alveare will automatically download the GGUF from Hugging Face, extract and quantize the weights into Alveare's NPU Q4_0 layout, and prepare the model for serving.

#### Option B: Via CLI
Download a GGUF file from HuggingFace and run `alveare quantize`:

```bash
./alveare quantize g4-12b /path/to/gemma-4-12b-it.gguf
```

### 4. Serve the Model

Start the server for your model:

```bash
./alveare serve g4-12b
```

### 5. Chat & Interact

- **Web Dashboard**: Open `http://127.0.0.1:8000` for real-time chat and system metrics.
- **Terminal CLI**: Run `./alveare chat` in another terminal window.
- **OpenAI Client / API**: Connect any OpenAI SDK to `http://127.0.0.1:8000/v1`.

---

## 🛠️ The `alveare` CLI Launcher

The `./alveare` CLI launcher automatically manages the Conda environment and NPU stack activation.

```
alveare install                               One-command full setup (env + toolchain + build).
alveare check                                 NPU smoke test and driver verification.
alveare quantize [alias] <gguf> [--arch A]    Quantize GGUF into Alveare Q4 weights.
alveare setup <alias> --arch A [--url URL]    Download and quantize a model in one step.
alveare list  (or: models)                    List all installed model packages.
alveare delete <model_id>                     Remove a quantized model package from disk.
alveare serve [model] [--host H] [--port P]   Launch the C++ runtime & Python API/Web UI server.
alveare chat [--host H] [--port P]            Interactive terminal chat client.
alveare help                                  Show help message.
```

---

## 💻 Developer & Custom Quantizer Plugins

Alveare features a modular quantizer architecture located in `tools/convert/`. Anyone can write a custom quantizer plugin to bring new GGUF models or experimental weight layouts to the NPU.

Inherit from `BaseQuantizer` in `tools/convert/base_quantizer.py`:

```python
from tools.convert.base_quantizer import BaseQuantizer
from gguf import GGUFReader, dequantize, GGMLQuantizationType
import numpy as np

class CustomModelQuantizer(BaseQuantizer):
    def __init__(self):
        super().__init__(name="custom_model")

    def quantize(self, gguf_path: str, out_dir: str) -> dict:
        # 1. Load GGUF and extract metadata
        # 2. Dequantize tensors using gguf.quants.dequantize
        # 3. Pack weights into Alveare Q4_0 layout using self.quantize_and_pack_tensor
        # 4. Save config.json, tokenizer.json, and *.npy files to out_dir
        ...
```

For full documentation and examples, see [`docs/CUSTOM_QUANTIZERS.md`](docs/CUSTOM_QUANTIZERS.md).

---

## ⚡ OpenAI API Usage Examples

Alveare's API server is a drop-in replacement for OpenAI endpoints.

### Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="gemma4",
    messages=[{"role": "user", "content": "Explain NPU acceleration in three bullet points."}],
    max_tokens=128,
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Streaming via `curl`

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Ciao! Chi sei?"}],
    "stream": true
  }'
```

---

## 🔬 Hardware & Reference Environment

Alveare is optimized for the **AMD Ryzen™ AI (XDNA2)** NPU architecture on Linux:

| Component | Specification |
|---|---|
| **SoC / APU** | AMD Ryzen AI 9 HX 470 / Ryzen AI 300 series (Gorgon Point / Strix Point) |
| **NPU Silicon** | AMD XDNA2 (32 AIE tile array), exposed at `/dev/accel/accel0` |
| **Driver / Kernel** | `amdxdna` (mainline Linux kernel driver) |
| **Firmware** | `/lib/firmware/amdnpu/` |
| **Userspace Stack** | AMD XRT `2.21.75`, `mlir-aie` `1.3.3`, `llvm-aie` (Peano) |
| **OS** | Ubuntu 24.04 / 26.04 LTS (Linux 6.x / 7.x) |

Detailed hardware architecture & compiler notes are in [`docs/hardware.md`](docs/hardware.md) and [`docs/background.md`](docs/background.md).

---

## 📁 Repository Structure

```
alveare               Top-level CLI launcher script
frontend/             React Web UI dashboard (Vite + Tailwind/CSS)
runtime/cpp/          Native C++ inference server (alveare_runtime) & XRT engine
runtime/py/           Python control server (control_server.py) & HTTP REST API
kernels/              Open AIE hardware kernel sources (IRON / MLIR-AIE)
tools/                Model quantizers, setup scripts, and conversion plugins
docs/                 Architecture documentation, setup guides, and specifications
tests/                Correctness test suites & microbenchmarks
```

---

## 📄 License

**MIT License** — See [`LICENSE`](LICENSE) for details. Alveare is 100% open-source, including all AIE hardware kernels.
