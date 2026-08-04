# Roadmap

Honest, milestone-based plan. Each milestone has a single **definition of done** that is testable. We do not move on until the previous gate passes.

The split of work across Alveare:

- **~70% is conventional plumbing** (C++ native engine, host runtime, XRT buffer management, weight streaming, KV cache, tokenizer, sampler, Python control server, Web UI).
- **~30% is AIE hardware kernels** — where the core optimization and hardware integration live. We explicitly target *correctness first, speed later*.

Our philosophy: **A working, 100% open-source runtime unlocks LLM models on the NPU for everyone.**

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

**Done when**: The Gemma model runs end-to-end on the NPU and produces coherent text whose greedy continuation matches reference outputs side-by-side.

Spec: [`docs/milestones/M5-gemma-bringup.md`](docs/milestones/M5-gemma-bringup.md)

---

## M6 — Gemma-4 Layer Bringup

**Status**: **Completed**.

**Goal**: Implement the dense Gemma-4 architecture and validate one decoder layer (Layer 0) of Gemma-4-12B against a reference, reusing the vectorized multi-core `gemv_q` kernel.

**Done when**: A unit test runs Layer 0 forward on the NPU and matches a CPU-dequantized reference layer output within expected quantization loss.

Spec: [`docs/milestones/M6-gemma4-layer.md`](docs/milestones/M6-gemma4-layer.md)

---

## M7 — Gemma-4-12B End-to-End on NPU

**Status**: **Completed.**

**Goal**: Run the full Gemma-4-12B model end-to-end on the Ryzen AI NPU and generate coherent text.

Spec: [`docs/milestones/M7-gemma4-12b.md`](docs/milestones/M7-gemma4-12b.md)

---

## M8 — Close the Gemma-4-12B fidelity gap

**Status**: **Completed.**

**Goal**: Ensure NPU Gemma-4-12B matches reference greedy generation token-for-token.

---

## M9 — Gemma-4-12B decode speed (multi-core)

**Status**: **Completed.**

**Goal**: Optimize decode performance across multi-core AIE tiles.

**Achieved**: Scaled `gemv_q` kernel from 4 to 32 AIE cores. Multi-tile distribution significantly accelerated layer execution.

---

## M10 — Gemma-4-12B speed cycle 2 (streaming + sync)

**Status**: **Completed.**

**Goal**: Reduce weight-streaming overhead and host sync calls.

---

## M11 — Fused FFN NPU kernel

**Status**: **Completed.**

**Goal**: Collapse the FFN (gate + up + GeGLU + down) into a single AIE kernel instead of separate GEMVs + host-side activation, cutting launches and host round-trips.

---

## M12 — Native C++ runtime & Full Web UI

**Status**: **Completed.**

**Goal**: Eliminate Python interpreter overhead from decode loop and provide a modern control dashboard.

**Achieved**: `runtime/cpp` native C++ engine + OpenAI server, resident-weight XRT contexts, fused FFN AIE hardware kernels, custom quantizer plugin architecture, and a modern React Web UI dashboard with 1-click model setup and real-time NPU monitoring.
