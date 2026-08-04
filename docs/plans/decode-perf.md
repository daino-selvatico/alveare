# Plan — Decode Performance & Throughput Optimization

**Goal.** Maximize autoregressive decode throughput on the AMD XDNA2 NPU for Gemma-4, Gemma-3, and Llama models.

---

## 📊 Where Alveare's Per-Token Time Goes (Profile Breakdown)

Measured decode step breakdown:

1. **Host CPU Execution**: Attention, RMSNorm, RoPE, PLE, and KV updates.
2. **NPU GEMV Dispatches**: QKV + O projections (~2 GEMVs × layers + LM head tiles).
3. **FFN Compute**: Fused-FFN AIE kernel calls (`ffn_fused`).
4. **LM Head Projection**: Vocabulary projection matrix-vector computation.

Memory bandwidth streaming is the primary physical constraint for LLM decode on unified memory architectures.

---

## 🚀 Performance Levers (Priority Order)

### Tier 1 — Dispatch Minimization & Host Overhead Reduction
1. **Batch LM Head Tiles**: Collapse multiple LM head sub-tiles into single contiguous GEMV / GEMM dispatches.
2. **Fused Layer Projections**: Combine QKV and O projections into single resident weight contexts.
3. **NPU RMSNorm & RoPE Offloading**: Move RMSNorm and Rotary Embeddings onto AIE tiles to prevent host round-trips.

### Tier 2 — On-NPU Attention Kernels
4. Implement multi-head attention (QKᵀ softmax V, GQA, sliding window) directly as AIE tile kernels with KV caches mapped in NPU-accessible memory.

### Tier 3 — Fused On-Chip Pipeline
5. A persistent AIE dataflow pipeline running all decoder layers directly on-chip with asynchronous DRAM weight prefetching.

---

## 🔬 Optimization Principles

- **Data-Driven Profiling**: Always profile before and after every optimization (`ALVEARE_PROFILE_DECODE`).
- **Bit-Exact Fidelity**: Verify that greedy generation outputs match reference baseline implementations token-for-token.
- **Tiling Efficiency**: Balance tile memory allocations with DMA transfer sizes.
