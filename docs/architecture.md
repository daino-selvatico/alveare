# Architecture

High-level architectural overview of the Alveare NPU LLM stack.

---

## 🏛️ System Architecture

Alveare is divided into four distinct components:

```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  React Web UI Dashboard  (http://127.0.0.1:8000)                             │
    │  - Real-time NPU telemetry, server controls, live terminal logs             │
    │  - 1-Click model setup modal ("Aggiungi Modello") & model deletion manager  │
    └─────────────────────────────────────┬───────────────────────────────────────┘
                                          │ HTTP / WebSocket
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Python Control Server (runtime/py/control_server.py)                       │
    │  - Manages background download & setup tasks, model package discovery       │
    │  - Forwards /v1 OpenAI endpoints to Native C++ Engine                       │
    └─────────────────────────────────────┬───────────────────────────────────────┘
                                          │ IPC / HTTP Proxy
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Native C++ Inference Server (runtime/cpp/ -> alveare_runtime)              │
    │  - Zero Python overhead decode loop, native C++ KV cache & samplers         │
    │  - Self-contained BPE / SentencePiece tokenizer                             │
    │  - Direct XRT memory-resident weight context management                     │
    └─────────────────────────────────────┬───────────────────────────────────────┘
                                          │ XRT Buffer Objects & Kernel Submissions
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Open AIE Hardware Kernels (kernels/ -> .xclbin)                            │
    │  - Fused FFN (gate/up + GeGLU + down in 1 xclbin), 32-core GEMV kernels     │
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
