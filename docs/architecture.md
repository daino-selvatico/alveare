# Architecture

How Alveare is structured. This is the target design; most of it is unbuilt.

## Two halves

```
                         ┌─────────────────────────────────────────┐
   OpenAI HTTP  ───────► │  runtime/  (host, C++)                   │
   /v1/chat/...          │                                          │
                         │  server → scheduler → model graph        │
                         │     │                                    │
                         │     ├─ tokenizer, chat template          │
                         │     ├─ weight streamer (DRAM → NPU)      │
                         │     ├─ KV cache manager                  │
                         │     ├─ sampler                           │
                         │     └─ XRT: load xclbin, BOs, submit ────┼──► /dev/accel/accel0
                         └─────────────────────────────────────────┘
                                              ▲
                                              │ .xclbin
                         ┌─────────────────────────────────────────┐
                         │  kernels/  (AIE, IRON/MLIR-AIE)          │
                         │  gemv_q4 · gemm_q4 · attn · rmsnorm ·    │
                         │  rope · softmax · dequant · lm_head      │
                         └─────────────────────────────────────────┘
```

- **`runtime/`** — the "easy 70%". Conventional C++. Orchestrates everything, owns the inference loop, talks to XRT. Knows nothing about *how* a kernel is implemented, only its interface (input/output buffer shapes + xclbin name).
- **`kernels/`** — the "hard 30%". AIE designs compiled to `.xclbin`. Each kernel has a documented host-facing contract (buffer layout, dtypes, tiling parameters).

The boundary between them is a small, stable **kernel ABI**: shapes, dtypes, buffer descriptors. Keeping it narrow lets us swap a slow reference kernel for an optimized one later without touching the runtime.

## Model definition layer

Each model architecture gets a description (config + a graph of kernel calls), analogous to FLM's `modeling_*.cpp`. Initially hand-written per model. A model definition specifies:

- dimensions (hidden size, layers, heads, head_dim, intermediate size, vocab)
- attention variant (full / sliding-window, GQA grouping, any softcapping, QK-norm)
- norm type, activation, RoPE parameters, embedding tying
- quantization format of the weights
- the per-layer sequence of kernel invocations

We start with **one small dense model** to avoid drowning in architecture-specific features. Gemma-4 dense (the 12B end goal) is deliberately *not* first — its sliding/full attention alternation, QK-norm and logit softcapping are added once the simple path works.

## Weight format & streaming

- Weights are quantized (int4/int8) and pre-laid-out by a converter in `tools/` into a tiling-friendly format (our analog of FLM's Q4NX).
- The **weight streamer** prefetches the next layer's weights from DRAM into NPU-accessible buffers while the current layer computes. Hiding this latency is the core perf challenge (deferred until correctness is done — first version may stream synchronously and be slow).

## KV cache

- Per-layer K/V stored in DRAM-backed buffers, grown per token.
- Attention kernel reads the cache for the current head group. Layout chosen for contiguous DMA.

## Inference scheduler

- Decode: single-token loop, latency-oriented.
- Prefill: batched over prompt tokens, throughput-oriented (matrix-matrix kernels). May be added after decode works.

## Non-goals (for now)

- Multi-NPU / multi-device.
- Training or fine-tuning.
- Vision/audio modalities.
- Matching FLM throughput. Correctness and openness first.

## Decisions

Architecture decisions are recorded as ADRs in [`docs/decisions/`](decisions/). See:
- [`0001-record-architecture-decisions.md`](decisions/0001-record-architecture-decisions.md)
- [`0002-kernel-strategy.md`](decisions/0002-kernel-strategy.md) — hand-written kernels vs `iree-amd-aie`.
