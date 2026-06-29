# Roadmap

Honest, milestone-based plan. Each milestone has a single **definition of done** that is testable. We do not move on until the previous gate passes.

This is a multi-month project. The split:

- **~70% is conventional plumbing** (host runtime, XRT buffer management, weight streaming, KV cache, tokenizer, sampler, OpenAI server). Hard work, but not research.
- **~30% is AIE kernels** — and that 30% is where ~90% of the difficulty and value lives. "Correct but slow" kernels are achievable; "fast like FLM" is the patented, very hard part. We explicitly target *correct first, fast later*.

Performance philosophy: **a working open runtime that is slower than FLM is still a win**, because it runs models nobody can put on the NPU otherwise.

---

## M0 — Toolchain validation

**Status**: **Completed**.

---

## M1 — Quantized matrix-vector kernel

**Status**: **Completed**.

---

## M2 — One transformer layer on NPU

**Status**: **Completed**.

---

## M3 — End-to-end small model + server

**Status**: **Completed**.

---

## M4 — Scale up + optimize

**Status**: **Completed**.

---

## M5 — Gemma Bringup

**Status**: **Completed**.

**Goal**: Bring up a small Gemma-family model (Gemma-3-1B-it) end-to-end on the Alveare runtime, implementing all architecture-specific features (QK-norm, sliding window attention, GeGLU activation, tied embeddings, and layer-dependent RoPE theta).

**Done when**: The Gemma model runs end-to-end on the NPU and produces coherent text whose greedy continuation matches `llama.cpp` side-by-side.

Spec: [`docs/milestones/M5-gemma-bringup.md`](docs/milestones/M5-gemma-bringup.md)

---

## M6 — Gemma-4 Layer Bringup

**Status**: **Completed**.

**Goal**: Implement the dense Gemma-4 architecture and validate one decoder layer (Layer 0) of Gemma-4-12B against a reference, reusing the vectorized multi-core `gemv_q` kernel.

**Done when**: A unit test runs Layer 0 forward on the NPU and matches a CPU-dequantized reference layer output within a 25% relative error tolerance (expected quantization loss).

Spec: [`docs/milestones/M6-gemma4-layer.md`](docs/milestones/M6-gemma4-layer.md)

---

## M7 — Gemma-4-12B End-to-End on NPU

**Status**: **Completed**.

**Goal**: Run the full Gemma-4-12B model end-to-end on the Ryzen AI NPU and generate coherent text.

**Done when**: The model runs end-to-end on the NPU and generates coherent text whose greedy continuation matches `llama.cpp` side-by-side.

Spec: [`docs/milestones/M7-gemma4-12b.md`](docs/milestones/M7-gemma4-12b.md)

---

## Strategic shortcuts (evaluate before greenfielding)

We do **not** want to hand-write every kernel from zero if there's leverage:

1. **MLIR-AIE example kernels** — GEMM, matrix-vector, softmax, eltwise reference designs already exist. Adapt, don't reinvent.
2. **`iree-amd-aie`** — AMD's open path to compile ML graphs → AIE via IREE/MLIR. Potentially compiles a model instead of hand-writing kernels. Immature for LLMs, but the highest-leverage option if viable. Evaluated in [`docs/decisions/0002-kernel-strategy.md`](docs/decisions/0002-kernel-strategy.md).
3. **Existing community efforts** — before committing, scan for any open NPU-LLM runtime to contribute to rather than duplicate.
