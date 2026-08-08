# Plan — Fused on-NPU decode (toward FastFlowLM-level throughput)

**Status: IN PROGRESS (branch `feat/2.0-fused-decode`, started 2026-08-08).**
This doc is the persistent state across autonomous token-limited cycles — update the
Progress Log every cycle so work resumes cleanly.

## Goal
Close the ~11× decode gap vs FastFlowLM at the SAME Q4 quant. FLM: e4b ~12.6 tok/s;
Alveare: e4b ~1.1 tok/s (~907 ms/tok), 12B ~1 tok/s, gemma3 ~3.6 tok/s (~276 ms/tok).

## Honest assessment (read before diving in)
- FLM's fused-decode kernels are closed/patent-pending — the exact recipe is UNKNOWN.
  This is empirical research: measure before/after every change, don't guess.
- Prior findings (memory [[cpp-runtime-decode-ceiling]]): batch-1 decode is
  memory/**dequant**-bound. Per-core dequant ~0.3–0.5 GB/s; 32 cores → ~10–16 GB/s
  effective vs ~120 GB/s LPDDR5 peak (~8–12% util). int8≈bf16 (only ~8%), and
  dequant/mmul OVERLAP was REJECTED (~2× regression, shared AIE vector datapath).
  Host CPU work is only ~3% (~11 ms/tok on gemma3) — so removing host round-trips
  alone is LOW value; the win must come from the NPU weight-read/dequant path.
- Therefore the real lever is **effective bandwidth / dequant throughput**, not just
  fewer dispatches. The FLM gap is HOW weights are streamed+dequanted, not the quant.

## Work plan (incremental, each step measured + validated; iterate on gemma3 first — loads ~8s)
- [ ] **A. Baseline profile** — `ALVEARE_PROFILE_DECODE=1` on gemma3, e4b, 12B. Record
      ms/tok + phase split (ffn/gemv/lm_head/attn/cpu) + dispatches. Confirm bottleneck.
- [ ] **B. Dequant/bandwidth experiment** — the crux. Micro-bench the gemv/FFN kernel:
      is per-token time DMA-bound or vector-dequant-bound? Try: (b1) larger DMA tiles /
      double-buffer weights to hide DMA; (b2) a leaner Q4→bf16 dequant (LUT / vectorized
      unpack); (b3) more work per weight-read (process multiple output rows per weight
      tile). Measure effective GB/s. This decides whether higher bandwidth is reachable.
- [ ] **C. Fused layer kernel** — QKV+O(+FFN) for one layer on-chip, activations
      resident (fewer dispatches + no host round-trip). Measure (expect small if
      dispatch overhead is ~9%, but validates the fused structure).
- [ ] **D. On-NPU attention + RMSNorm + RoPE** — KV cache in NPU memory; removes the
      per-layer host attention. Big new kernel.
- [ ] **E. Multi-layer streaming** — several layers per dispatch, weights streamed near
      peak. The FLM-level step.

## Fallback levers (if the moonshot stalls — still real gains)
- **Speculative decode** (already built, `ALVEARE_SPECULATIVE`, gemma4): caps ~3 tok/s
  (batched verify is mmul-compute-bound). Could productionize / default-on where it wins.
- **Q3/Q2 quant**: ~1.3–1.5× (fewer bytes/token) — but needs new quant + kernels.

## Validation (every step, non-negotiable)
- Keep 12B/e4b/gemma3 **coherent** (greedy: sensible answers) and ideally bit-exact vs
  the current runtime. Rerun reproduces. NPU: ONE instance at a time.
- Merge to feat/rc-2.0 ONLY a step that measurably improves tok/s without regressing
  correctness. Everything stays on this branch until proven.

## Progress Log
- 2026-08-08: branch created from feat/rc-2.0 (853014a, = v2.0.0-alpha.2). Plan written.
- 2026-08-08 **step A DONE — key finding**: gemma3 decode ~276 ms/tok, NPU ~95%
  (`ffn=137 · qkv=97 · o=8 · lm_head=33`; host attn+rest ~7 ms). ANOMALY: **qkv (97 ms)
  ≫ o (8 ms) for the SAME 2048×2048 shape** → ~88 ms/tok is per-layer kernel
  **CONTEXT-SWITCH overhead** (qkv switches shape after FFN; o reuses qkv's context,
  commit 7c67e12). So dispatch/switch ≈ **30%** of NPU time here, NOT ~9%. => the
  cheapest high-value lever is **removing per-layer context switches**, not the
  dequant-bandwidth work. Reprioritized: do **step C (fused-layer kernel / shared
  contexts)** FIRST. Est. ~1.3–1.4× on gemma3 (276→~200 ms/tok). Still profile e4b/12B
  to confirm the same pattern (likely worse — more/bigger layers).
- NEXT: step C. Approach options (measure each): (c1) cheapest — reorder / make QKV and
  FFN share a single runtime-shape kernel context so switching FFN→QKV is free; (c2) a
  fused per-layer kernel doing QKV+O+FFN on-chip in one dispatch. Start on gemma3
  (loads ~8s), validate coherent + ideally bit-exact, measure, then e4b/12B.
