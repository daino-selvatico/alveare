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

## RE-ORIENTATION (2026-08-11, user steer)
gemma3 is only a **fast dev vehicle** (loads ~8s, simplest shapes). The context-switch
lever is **gemma3-biased**: ~30% of NPU time on gemma3 but only **~12% on the 12B**
(12B profile: FFN 62%, QKV 23%, O 12% — FFN weight-read/dequant DOMINATES). The 12B is
the model that matters → **always measure the win on the 12B**, not just gemma3.
### Strategy (user-approved 2026-08-11)
- **Models that matter: e4b + 12B** (the only ones used). gemma3 = fast dev vehicle only.
- **NO requantization** (Q3/Q2 dropped — quality loss unacceptable). Do everything
  INTERNALLY via the fused kernel. This is now THE lever.
- **Order: gemma3 (develop) → e4b (has an FLM reference ~12.6 tok/s to measure against)
  → 12B (same technique, no FLM reference, best-effort).** e4b is the yardstick.
- The fused kernel wins on (a) context-switches, (b) host round-trips, (c) KV/activations
  on-chip. HONEST CAVEAT: it does NOT by itself speed up the FFN weight-read+dequant
  (~62% on 12B, likely large on e4b too) — that's the same Q4 dequant on the AIE vector
  unit. So fusion gets PART of the gap; matching FLM's e4b may also need weight-
  streaming/dequant efficiency (the hard, unknown FLM secret). **Measure on e4b vs 12.6
  tok/s step by step** to see where fusion plateaus and whether the dequant must be
  attacked too.
- Speculative decode (built, gemma4) stays a COMPLEMENTARY lever (amortizes switches,
  ~3 tok/s cap) but is situational (structured text) — not the main path.
NEXT: profile the 12B decode (confirm split) + profile e4b (its split + baseline tok/s
vs FLM 12.6). Then build the fused attention-block on gemma3 (probe existing
attention/rope/rmsnorm kernels first), port to e4b, measure.

## Progress Log
- 2026-08-12 **🏆 BIGGEST WIN SO FAR — the runtime was built at -O0.**
  `CMAKE_BUILD_TYPE` was empty in `runtime/cpp/CMakeLists.txt`, so the whole C++ runtime
  compiled unoptimized. Fixed by defaulting to Release (-O3) (commit 061a1dc).
  Measured (ALVEARE_PROFILE_DECODE=1):
  - **e4b: 832 → ~620 ms/token (1.20 → 1.62 tok/s, +35%)**; the per-layer PLE injection
    collapsed **200 ms → ~27 ms (7.4×)**; host is now only ~7% of decode.
  - gemma3: ~276 → ~270 ms/tok (unchanged — it was already NPU-bound, host ~7 ms).
  - Outputs stay coherent on both (greedy "Paris").
  **Implication: every earlier CPU-side measurement/decision was taken at -O0 and may be
  wrong** — notably "OpenMP on the per-layer PLE loops is slower" ([[cpp-runtime-decode-ceiling]])
  and the host-vs-NPU split. Re-validate CPU-side conclusions before relying on them.
  e4b profile now: NPU 534 ms (ffn 264 / qkv 137 / o 133) + lm_head ~40 + host ~40.
  => The remaining e4b gap vs FLM (12.6 tok/s) is now almost ENTIRELY the NPU path:
  FFN weight-read/dequant + the 2 context switches/layer. Fusion + dequant are next.
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
- 2026-08-08 **step C analysis (grounded, decisive)**:
  - The ~2.6 ms/switch is **inherent XDNA2 hw-context reconfig**, NOT reload/eviction:
    `npu.cpp` caches one `xrt::hw_context` per kernel shape; `max_contexts=8` and gemma3
    uses only 3 shapes (gemv_2048x2048, gemv_16384x2048/lm_head, ffn_2048x8192) → no
    thrashing. Pinning won't help. Confirmed by profile (qkv 3.7 ms/layer WITH switch vs
    o 0.32 ms/layer reusing qkv's ctx).
  - gemv (8 cores) and ffn (32 cores) CANNOT spatially coexist in one xclbin → can't
    share a context by co-placement. The per-layer sequence qkv(ctxA)→**attention(HOST)**
    →o(ctxA)→ffn(ctxB)→next qkv(ctxA) has 2 switches/layer that ONLY vanish if the WHOLE
    layer runs in ONE kernel/context — which requires **attention on the NPU** (a single
    kernel can't yield to the host mid-run). So the switch win ⇒ full fused-layer kernel.
  - Moving ONLY rmsnorm/rope to NPU is counter-productive (host is ~7 ms; it'd ADD
    dispatches, not remove switches). Skip.
  - Existing `kernels/{attention,rope,rmsnorm}/*.cc/.py` sources exist (Python-runtime
    era) but are UNBUILT + UNINTEGRATED (C++ does these on host). Unknown if they match
    current gemma3/4 shapes.
  - **Potential**: eliminating both switches/layer ≈ 2×2.6 ms×26 ≈ 135 ms/tok on gemma3
    (276→~140, ~7 tok/s) — BUT requires the full fused-layer w/ on-NPU attention = big,
    high-risk to converge autonomously. Speculative decode (built, gemma4) AMORTIZES
    switches over a batch (switch cost /B) but is situational (repetitive text only).
- NEXT (cheap probe before committing): build + self-verify `kernels/attention/attention.py`,
  `rope.py`, `rmsnorm.py` to assess their state/shapes → decides if composing them into a
  fused attention-block is viable. If viable, prototype a fused attention-block
  (qkv→rope→attention→o in ONE context) on gemma3, measure. If not viable / not
  converging, prefer the honest deliverable: this characterization + a detailed impl plan
  for the user, and DON'T burn all tokens on a from-scratch on-NPU-attention kernel.
