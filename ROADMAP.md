# Roadmap

Honest, milestone-based plan. Each milestone has a single **definition of done** that is testable. We do not move on until the previous gate passes.

This is a multi-month project. The split:

- **~70% is conventional plumbing** (host runtime, XRT buffer management, weight streaming, KV cache, tokenizer, sampler, OpenAI server). Hard work, but not research.
- **~30% is AIE kernels** — and that 30% is where ~90% of the difficulty and value lives. "Correct but slow" kernels are achievable; "fast like FLM" is the patented, very hard part. We explicitly target *correct first, fast later*.

Performance philosophy: **a working open runtime that is slower than FLM is still a win**, because it runs models nobody can put on the NPU otherwise.

---

## M0 — Toolchain validation  ⟵ current

**Goal:** prove the full chain *kernel source → xclbin → XRT → `/dev/accel/accel0`* works on this machine, end to end, with code we control.

**Done when:** we compile a trivial AIE example (a vector add or a single small GEMM) from MLIR-AIE/IRON and run it on the NPU, reading back a correct result.

Spec: [`docs/milestones/M0-toolchain-validation.md`](docs/milestones/M0-toolchain-validation.md)

Risk: toolchain install friction (Peano/LLVM-AIE, XRT versions). Cheap to attempt; fails fast if it fails.

---

## M1 — Quantized matrix-vector kernel

**Goal:** the single most important LLM primitive — a quantized weight × bf16 activation matrix-vector multiply (the decode-time GEMV) — running on the NPU.

**Done when:** an int4/int8-weight, bf16-activation GEMV kernel produces results matching a CPU reference within tolerance, for a realistic matrix shape (e.g. 2560×2560), and we have a microbenchmark vs CPU.

Spec: [`docs/milestones/M1-quantized-matvec.md`](docs/milestones/M1-quantized-matvec.md)

---

## M2 — One transformer layer on NPU

**Goal:** a full decoder layer (attention + MLP + norms + RoPE) executing on the NPU for a tiny model.

**Done when:** for a small model (Gemma 3 270M or Llama-3.2-1B), one layer's output matches a reference (HF/llama.cpp) within tolerance, with weights streamed from DRAM.

Spec: [`docs/milestones/M2-transformer-layer.md`](docs/milestones/M2-transformer-layer.md)

Note: pick a *small, dense, well-documented* model first. Gemma-specific complications (sliding/full attention alternation, QK-norm, logit softcapping) are deferred to a later model.

---

## M3 — End-to-end small model + server

**Goal:** a complete small model generating coherent text on the NPU, exposed over an OpenAI-compatible endpoint, wired into the existing `lemonade_router`.

**Done when:** `POST /v1/chat/completions` returns sane tokens generated on the NPU for a small model, and the router can route to it.

Spec: [`docs/milestones/M3-end-to-end-small-model.md`](docs/milestones/M3-end-to-end-small-model.md)

---

## M4 — Scale up + optimize

**Goal:** bigger models (the real target: Gemma 4 dense 12B) and performance work — DMA/compute overlap, weight-stream prefetch, better tiling, prompt/KV cache.

**Done when:** open-ended. Sub-goals tracked as issues. This is the long tail where FLM's patented cleverness lives; progress here is incremental and indefinite.

Spec: [`docs/milestones/M4-scale-and-optimize.md`](docs/milestones/M4-scale-and-optimize.md)

---

## Strategic shortcuts (evaluate before greenfielding)

We do **not** want to hand-write every kernel from zero if there's leverage:

1. **MLIR-AIE example kernels** — GEMM, matrix-vector, softmax, eltwise reference designs already exist. Adapt, don't reinvent.
2. **`iree-amd-aie`** — AMD's open path to compile ML graphs → AIE via IREE/MLIR. Potentially compiles a model instead of hand-writing kernels. Immature for LLMs, but the highest-leverage option if viable. Evaluated in [`docs/decisions/0002-kernel-strategy.md`](docs/decisions/0002-kernel-strategy.md).
3. **Existing community efforts** — before committing, scan for any open NPU-LLM runtime to contribute to rather than duplicate.
