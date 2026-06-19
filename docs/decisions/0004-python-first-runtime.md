# 0004 — Python-first runtime implementation

- Status: accepted
- Date: 2026-06-19

## Context

For Milestone M3, we need to implement a complete end-to-end Llama-3.2-1B model served over an OpenAI-compatible server.
While the ultimate goal is a high-performance C++ runtime (similar to llama.cpp or FLM), building the full graph orchestration, KV cache management, HTTP server, and tensor manipulation directly in C++ at this stage introduces a high risk of programming errors and slow iteration cycles.
Additionally, we need to ensure that the mathematical output matches references precisely and that kernel JIT-compilations are handled safely.

## Decision

We will implement the end-to-end runtime orchestrator in Python (extending `runtime/py/`).
The C++ port is deferred to a future milestone (Milestone M4) to keep the project correctness-first.

Concretely:
- We will write a Python script `model.py` to orchestrate the layer loops and stream weights.
- We will use FastAPI and Uvicorn to implement the OpenAI HTTP API endpoints (`/v1/chat/completions`).
- We will run the attention mechanism, RMSNorm, and RoPE on the host (CPU). RMSNorm and RoPE are run on CPU to conserve AMD XDNA NPU hardware contexts (avoiding system-wide context limits by only creating one context for GEMV), while running all heavy weight-bound matmuls (GEMVs) on the NPU.

## Consequences

- Faster development and debugging of the end-to-end generation loop.
- Easy integration with Hugging Face's `tokenizers` library and python web servers.
- Simplified verification: we can directly inspect NumPy arrays and compare them with the reference.
- Performance will be limited by Python interpreter overhead and host-NPU data copy overhead (to be optimized during Milestone M4).
