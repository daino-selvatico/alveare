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

**Status**: **Completed (fidelity gap resolved in M8).**

**Goal**: Run the full Gemma-4-12B model end-to-end on the Ryzen AI NPU and generate coherent text.

**Achieved**: full 48-layer forward on the NPU with per-layer weight streaming (~5.5 GB peak RAM), correct answer ("Paris"), ~8 s/token. **Open gap**: greedy output diverges from `llama.cpp` (same Q4 GGUF) at token 1 — the NPU skips the reasoning the model is primed to produce. Cause not yet distinguished (compounded numerical error vs an unvalidated global-layer bug). See the milestone doc; resolved in M8.

Spec: [`docs/milestones/M7-gemma4-12b.md`](docs/milestones/M7-gemma4-12b.md)

---

## M8 — Close the Gemma-4-12B fidelity gap

**Status**: **Completed.**

**Goal**: Make the NPU Gemma-4-12B match `llama.cpp` greedy, or explain the residual difference.

**Done when**: a *global*/full-attention layer is validated against the HF oracle (as M6 did for a sliding layer), the chat template is aligned exactly with `llama.cpp`, and the greedy trajectories match (or the remaining gap is attributed to quantified, benign numerical error).

---

## M9 — Gemma-4-12B decode speed (multi-core)

**Status**: **Completed.**

**Goal**: Make the 12B decode meaningfully faster without changing outputs.

**Achieved**: profiled the 12B (raw NPU GEMV compute dominant); scaled the `gemv_q` kernel from 4 to 8 AIE cores. Decode ~8 → ~5.5 s/token; prefill ~162 → ~102 s. Same-input greedy tokens still match `llama.cpp` exactly.

---

## M10 — Gemma-4-12B speed cycle 2 (streaming + sync)

**Status**: **Completed.**

**Goal**: One more profile-then-optimize pass, outputs unchanged.

**Achieved**: reduced weight-streaming allocation via `mmap` and hoisted the activation sync out of the inner chunk loop (fewer `_sync_to_device` calls). Decode ~5.5 → ~4.6 s/token (measured); prefill ~146 → ~99 s. Same-input greedy tokens still match `llama.cpp` exactly; all 5 correctness gates green.

**Note**: diminishing returns on incremental host-side tuning. Further large wins need kernel/dataflow work (padding-waste elimination, DMA/compute overlap) — deferred.

---

## M11 — Fused FFN NPU kernel

**Status**: **Completed.**

**Goal**: Collapse the FFN (gate + up + GeGLU + down) into a single AIE kernel instead of separate GEMVs + host-side activation, cutting launches and host round-trips.

**Achieved**: an open `ffn_fused` kernel that does the whole FFN on-chip across `n_cores`, with **fp32 accumulation** for the gate/up and down projections (bf16 accumulation compounded to ~13% error over 48 layers and produced garbage). The fp32 output accumulator is made to fit tile memory via an **H-output split** (process the output hidden in 2 passes so `y_accum` is fp32 at half size). Weight packing is matched byte-for-byte between the Python design, the verify, and the C++ runtime.

---

## M12 — Native C++ runtime, Gemma-4-12B coherent end-to-end

**Status**: **Completed.**

**Goal**: Remove Python from the decode loop — a single native binary driving the NPU.

**Achieved**: `runtime/cpp` decode loop + OpenAI server (cpp-httplib), native XRT kernel registry with a resident-weight context policy, a hand-ported CPU math path (RMSNorm / RoPE / sliding + global attention / sampling), the LM head tiled onto the NPU, and a self-contained byte-level BPE tokenizer that loads the model's `tokenizer.json` plus the Gemma chat template. Fixed the Gemma-4 per-layer output scale, float16 embedding decode, and embedding `sqrt(hidden)` scale. Greedy output matches the Python runtime token-for-token; Gemma-4-12B produces coherent text (`"Ciao! Sto bene, grazie mille. E tu come stai?..."`) at ~3.6 s/token.

---

## Strategic shortcuts (evaluate before greenfielding)

We do **not** want to hand-write every kernel from zero if there's leverage:

1. **MLIR-AIE example kernels** — GEMM, matrix-vector, softmax, eltwise reference designs already exist. Adapt, don't reinvent.
2. **`iree-amd-aie`** — AMD's open path to compile ML graphs → AIE via IREE/MLIR. Potentially compiles a model instead of hand-writing kernels. Immature for LLMs, but the highest-leverage option if viable. Evaluated in [`docs/decisions/0002-kernel-strategy.md`](docs/decisions/0002-kernel-strategy.md).
3. **Existing community efforts** — before committing, scan for any open NPU-LLM runtime to contribute to rather than duplicate.
