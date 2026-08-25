# 🐝 Alveare

**An open-source LLM inference runtime and compiler for the AMD Ryzen™ AI (XDNA2) NPU on Linux.**

Alveare runs large language models on the AMD Ryzen AI NPU with a **100% open-source stack** — including open AIE hardware kernels (MLIR-AIE / IRON), a high-throughput native C++ inference engine (`alveare_runtime`), a Python control server, and an interactive React Web UI dashboard.

> **Alveare 2.0:** Supports **THREE Gemma model families** end-to-end on the NPU (**Gemma-4-12B**, **Gemma-4-E4B**, **Gemma-3-1B**) and **Llama-3.2-1B**. Includes a full-featured **React Web UI Dashboard** with in-browser Hugging Face model setup, real-time streaming chat, conversation history management, multimodal file upload, generation sampling controls (`temperature`, `top_p`, `top_k`, `enable_thinking`), and a dark/light theme switcher.

---

## 🌟 Key Features

- **🚀 Native C++ Inference Server (`alveare_runtime`)**: Zero Python overhead in the execution loop. Direct XRT device management, fused FFN NPU kernels (gate/up + GeGLU + down), native KV caching, and multi-core GEMV tiled across all 32 NPU compute tiles.
- **🖥️ Modern React Web UI Dashboard**: A sleek, real-time control center at `http://127.0.0.1:8000`. Monitors NPU telemetry, streams server logs live, manages model lifecycle, and provides an interactive chat playground.
- **⚡ In-Browser & Automated 1-Click Model Setup**: "Aggiungi Modello" modal enables 1-click GGUF downloading and quantization from Hugging Face for Gemma-4 12B, Gemma-4 E4B, Gemma-3 1B, and Llama 3.2 1B.
- **💬 Rich Chat Experience**: Real-time token streaming with Markdown & code highlighting, persistent multi-turn conversation sessions (sidebar history manager), multimodal file upload (PDF/text/image/audio parsing), and dark/light theme toggle.
- **🎛️ Advanced Sampling Controls**: Full control over `temperature` (0.0 to 2.0 with greedy decoding at 0.0), `top_p` (nucleus sampling), `top_k`, `max_tokens`, `max_context_length`, system prompt customization, and `enable_thinking` toggle for reasoning models.
- **🔌 OpenAI-Compatible API**: Standard `/v1/chat/completions` and `/v1/models` REST endpoints for direct drop-in replacement with the OpenAI Python SDK, LangChain, LlamaIndex, Continue, etc.
- **🧩 Custom Quantizer Plugin Architecture**: Bring any custom architecture to the NPU by implementing `BaseQuantizer` (`tools/convert/base_quantizer.py`).

---

## 📦 Supported Models & NPU Performance

| Model | Architecture Key | Features | Output Size (Q4_0) | NPU Status & Performance (32 Cores) |
|---|---|---|---|---|
| **Gemma 4 E4B** | `gemma4-e4b` | 42 layers, Per-Layer Embedding (PLE) | ~2.4 GB | **290–305 ms/tok (3.3–3.45 tok/s native)**, bursts **4.5–17 tok/s** |
| **Gemma 4 12B** | `gemma4` | 48 layers, 32-core unified GEMV | ~6.8 GB | **696–705 ms/tok (1.43–1.94 tok/s)**, full 12.8B model |
| **Gemma 3 1B** | `gemma3` | SentencePiece SPM, 26 layers | ~0.8 GB | **7–12 tok/s**, lightweight edge model |
| **Llama 3.2 1B** | `llama` | Llama architecture, GQA attention | ~0.8 GB | **10–15 tok/s**, fast conversational model |

> **⚡ Performance Transparency**: All models execute natively across all 32 AIE2 cores with zero-bubble asynchronous pipelining, saturating the physical memory bandwidth of the AMD Ryzen AI 300 series processor.

---

## 🚀 Quick Start

### 1. One-Command Installation

```bash
./install.sh
```

### 2. Verify NPU Hardware

```bash
./alveare check
```

### 3. Launch Web UI & Server

```bash
./alveare start
```
Open `http://127.0.0.1:8000` in your browser.

### 4. Pick or Add a Model
Click **"Aggiungi Modello"** in the Web UI to download and quantize **Gemma 4 12B**, **Gemma 4 E4B**, or **Gemma 3 1B** directly from Hugging Face in one click.

Or from the command line:
```bash
./alveare setup gemma4 --arch gemma4
```

For the complete guide, see 🚀 [**Quickstart Guide**](docs/quickstart.md).

---

## 📚 Documentation & Guides

- 🚀 [**Quickstart Guide**](docs/quickstart.md): Step-by-step 5-minute setup and usage.
- 📦 [**Adding & Quantizing Models**](docs/adding_models.md): 1-click HF downloads, CLI setup, manual GGUF quantization, and quantizer plugins.
- 🎛️ [**Sampling Controls & API Reference**](docs/sampling.md): Sampling parameters (`temperature`, `top_p`, `top_k`, `seed`, `enable_thinking`, system prompt) and OpenAI API schema.
- 🛠️ [**Full Hardware & Toolchain Setup**](docs/SETUP.md): Deep dive into Linux driver permissions, XRT libraries, and AIE compiler requirements.
- 🔌 [**Custom Quantizer Plugins**](docs/CUSTOM_QUANTIZERS.md): Build custom tensor tiling and layout plugins for new model architectures.
- 🏛️ [**System Architecture**](docs/architecture.md): Overview of C++ engine, Python control server, and AIE kernel pipeline.

---

## 🛠️ The `alveare` CLI Launcher

```
alveare install                               One-command full setup (env + toolchain + build).
alveare check                                 NPU smoke test and driver verification.
alveare quantize [alias] <gguf> [--arch A]    Quantize local GGUF into Alveare Q4 weights.
alveare setup <alias> --arch A [--url URL]    Download HF GGUF and quantize in one step.
alveare list  (or: models)                    List all installed model packages.
alveare delete <model_id>                     Remove a quantized model package from disk.
alveare serve [model] [--host H] [--port P]   Launch C++ runtime & Python API/Web UI server.
alveare chat [--host H] [--port P]            Interactive terminal chat client.
alveare help                                  Show help message.
```

---

## ⚡ OpenAI API Usage Examples

Alveare's API server is a drop-in replacement for OpenAI endpoints.

### Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="gemma4",
    messages=[
        {"role": "system", "content": "Be direct and helpful."},
        {"role": "user", "content": "Explain NPU acceleration in three bullet points."}
    ],
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_tokens=128,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

### Streaming via `curl`

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-e4b",
    "messages": [{"role": "user", "content": "Ciao! Chi sei?"}],
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": true
  }'
```

---

## 🔬 Hardware & Reference Environment

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
