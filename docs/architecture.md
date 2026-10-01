# Architecture

High-level architectural overview of the Alveare Heterogeneous Neural Computing Stack (NPU • GPU • CPU).

---

## 🏛️ System Architecture

Alveare is divided into four distinct components operating across AMD Ryzen AI hardware:

```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  React Web UI Dashboard  (http://127.0.0.1:8000)                             │
    │  - Real-time telemetry, multi-device slot controls, live terminal logs      │
    │  - Full-Duplex Live Voice-to-Voice Studio with context window controls      │
    │  - 1-Click model setup modal ("Aggiungi Modello") & model deletion manager  │
    └─────────────────────────────────────┬───────────────────────────────────────┘
                                          │ HTTP / WebSocket
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Python Control Server (runtime/py/control_server.py)                       │
    │  - Multi-slot lifecycle manager (llm, stt, tts) across NPU, GPU, and CPU    │
    │  - Live Orchestrator (/ws/live) full-duplex conversational pipeline         │
    │  - Forwards /v1 OpenAI REST endpoints (chat, speech, transcriptions)        │
    └─────────────────────────────────────┬───────────────────────────────────────┘
                                          │ IPC / Direct Subprocess Routing
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Execution Engines:                                                         │
    │  1. Native C++ Engine (runtime/cpp/ -> alveare_runtime):                    │
    │     - Vulkan 1.4 GPU backend (Radeon 890M subgroup GLSL compute shaders)    │
    │     - XRT NPU backend (Ryzen AI XDNA2 32-core AIE2 .xclbin)                 │
    │     - AVX-512 / OpenMP CPU backend                                          │
    │  2. Speech-to-Text Engine (runtime/py/whisper_stt.py): Whisper on NPU/CPU   │
    │  3. Text-to-Speech Engine (runtime/py/tts_factory.py):                      │
    │     - Kokoro-82M (CPU Zen 5 / GPU CUDA, sub-500ms TTFA, StyleTTS2)          │
    │     - Audio8 TTS (AMD NPU / CPU, DualAR 10-codebook, zero-shot cloning)     │
    └─────────────────────────────────────┬───────────────────────────────────────┘
                                          │ Hardware Instructions & DMA
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Accelerated Hardware Targets:                                              │
    │  - ⚡ AMD Ryzen AI NPU (32 AIE2 Cores): Matrix GEMV Q4_0 & Fused FFN        │
    │  - 🎮 AMD Radeon 890M GPU (Vulkan 1.4 SPIR-V compute shaders)              │
    │  - 🖥️ AMD Zen 5 CPU (24 Threads AVX-512 vector pipeline)                    │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Component Responsibilities

1. **Native C++ Engine (`runtime/cpp/`)**:
   - Executes the token generation loop with zero Python overhead.
   - Manages memory-resident weight contexts in XRT buffers to avoid per-layer buffer allocation latency.
   - Implements native C++ KV caching, RMSNorm, RoPE embeddings, sliding window & global attention math, and greedy/top-k samplers.
   - Provides native tokenization using embedded model `tokenizer.json` files.

2. **Python Control Server (`runtime/py/control_server.py`)**:
   - Serves the compiled React Web UI static assets at port 8000.
   - Exposes REST API endpoints (`/api/models/setup`, `/api/models/setup/status`, `/api/models/{id}`).
   - Manages background quantization sub-processes, downloads, and model directory packages.

3. **React Web UI Dashboard (`frontend/`)**:
   - Modern glassmorphic interface for browser-based monitoring and interaction.
   - Interactive 1-click model setup modal with HuggingFace repository selection and custom plugin execution.
   - Interactive chat testing playground and live log streaming.

4. **AIE Hardware Kernels (`kernels/`)**:
   - Multi-core matrix-vector (`gemv_q`) kernels scaled across all 32 AIE tiles.
   - Fused FFN kernels (`ffn_fused`) performing gate projection, up projection, GeGLU activation, and down projection on-chip with FP32 accumulation.

---

## ⚙️ Model Specification & Quantization Layout

- Models are converted from GGUF format into Alveare's `Q4_0` NPU block quantization format (block size 32, scale per block).
- Weights are pre-tiled into memory-aligned block-column major matrices for optimal DMA streaming into AIE tile local memory.
- Quantizer plugins implement `BaseQuantizer` (`tools/convert/base_quantizer.py`) for plug-and-play architecture support.
