# M4 — Scale up + optimize

**Blocked by:** M3. Open-ended; tracked as issues, not a single gate.

## Goal

Two intertwined fronts:

### A. Bigger / more models
- Add larger dense models (Qwen3-class, then the real target: **Gemma-4 dense 12B**).
- Implement the deferred Gemma-4 features: sliding/full attention **alternation** per layer, **QK-norm**, **logit softcapping**, the specific GQA grouping, tied embeddings.
- Prefill path: matrix-matrix (`gemm_q`) kernels for processing prompts at throughput, not one token at a time.

### B. Performance (the long tail — where FLM's patented edge lives)
- **DMA/compute overlap**: prefetch layer N+1's weights while computing layer N. This is the single biggest decode-time win, since LLM decode is weight-bandwidth-bound.
- **Tiling**: spread matmuls across the AIE array efficiently; tune tile M/K/N.
- **Fusion**: dequant+matmul, norm+matmul, etc.
- **Quant format**: move from the simple Q4_0-style layout to something on-chip-friendlier if it pays off.
- **KV cache & prompt cache**: efficient layouts, reuse across turns.

## Reality check

This milestone is effectively unbounded. Matching FLM is the hard, patented part and may never fully happen — and that's acceptable per the project's performance philosophy (open + slower still unlocks models nobody can run otherwise). Progress is measured by:

- models that run at all that couldn't before, and
- tokens/sec creeping toward practical (interactive) latency.

## Success definition for the 12B end goal

Gemma-4 dense 12B generating tokens on the NPU at *any* speed = the project's headline objective met (first open NPU 12B). Interactive speed is a separate, later bar.
