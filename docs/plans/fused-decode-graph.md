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
- 2026-08-12 **DECISIVE: the 2.5 ms switch is a FIXED driver/context cost.**
  Extended `bench_switch.cpp` to also alternate two *gemv* contexts:
  `gemv(3072x2560) <-> gemv(2048x2048)` costs **2.574 ms per switch** — the same as
  `ffn <-> gemv` (2.62 ms), despite both being small 8-core designs. So the cost is per
  hw-context swap, INDEPENDENT of design size. Shrinking/co-placing designs won't help;
  the only lever is **fewer distinct kernel contexts per layer**.
  ALSO CORRECTED: my earlier "o -> ffn -> qkv are adjacent" claim was WRONG — `run_layer`
  does host work between them (post-attn norm + residual, pre-FFN norm, post-FFN norm,
  PLE, output scale). A single fused kernel would have to absorb those too (big).
  **NEW PLAN — "one kernel shape for the whole layer" (uses EXISTING kernel designs):**
  make every matmul in a layer run on ONE gemv shape (e.g. `(2048, 2560)`), by
  * tiling the output dim N (call the same kernel k times with per-tile weight handles):
    qkv 3072 -> 2 calls, o -> 2, gate+up 20480 -> 10;
  * chunking the input dim K for the down projection (K=10240 -> 5 chunks of 2560) and
    summing the partial results on the host (2560 floats, negligible at -O3).
  Then a layer issues N calls but ZERO context switches.
  Arithmetic (e4b): ~91.7 M MAC/layer at the measured ~13.7 GMAC/s = **6.7 ms/layer** vs
  today's 9.84 ms/layer (4.84 compute + 5.0 switch) => **~400 ms/token (~2.5 tok/s)**,
  i.e. +32% on top of the current 530 ms. Trade-off: the fused-FFN kernel is more
  efficient per MAC (21 GMAC/s vs gemv 13.7), so we give up some FFN efficiency to erase
  the switches — the measured numbers still favour it.
  RISK: this is a runtime restructuring (no new kernel code), so it is testable
  incrementally behind an env flag; validate numerics against the current path.
  NEXT: prototype behind `ALVEARE_ONESHAPE=1` on e4b — start by routing ONLY qkv+o
  through the tiled single-shape path (cheap, isolates the mechanism), measure, then move
  the FFN over.
- 2026-08-12 **MEASURED: the context switch costs 2.50 ms — and there is a tractable fix.**
  New micro-benchmark `runtime/cpp/test/bench_switch.cpp` (dummy weights, e4b shapes):
  ```
  ffn  same-context :  3.698 ms/call     gemv same-context : 0.572 ms/call
  alternating pair  :  9.263 ms          no-switch pair    : 4.270 ms
  => 2.496 ms per hw-context switch
  ```
  On e4b decode that is **42 layers x 2 switches x 2.5 ms = ~210 ms/token = 40% of the
  530 ms**. Real compute is only ffn 3.7 + qkv 0.57 + o 0.57 = 4.84 ms/layer (~203 ms) +
  lm_head/host ~80 => a switch-free decode would be **~320 ms/token (~3.1 tok/s)**.
  This also explains the earlier "GB/s improves with model size" illusion: t = 2.6 ms +
  bytes/(~13 GB/s) fits gemma3/e4b/12B — fixed overhead amortized over more work.
  **KEY INSIGHT — no on-NPU attention needed to win.** In a layer the order is
  `qkv -> [host attention] -> o -> ffn -> qkv(next layer)`. The host only sits between
  qkv and o, so **`o + ffn + qkv(next)` are ADJACENT with no host work in between**.
  One kernel doing [O + FFN + QKV-next] in a single hw_context removes BOTH switches per
  layer — a plain dataflow design (same class as the existing ffn_fused), far more
  tractable than an attention kernel.
  NEXT: design/prototype that fused [O+FFN+QKV] kernel for e4b (H=2560, I=10240,
  o:(2560<-2048), qkv:(3072<-2560)); validate numerics vs the current path, then measure.
- 2026-08-12 **WIN #2 — O reuses the fused-QKV kernel context on e4b** (commit 884fa66).
  Diagnosis: e4b's fused QKV gemv is (3072,2560) but O was (2560,2048) → different shape
  → a ~2.6 ms context switch per layer for O (O cost 3.2 ms/layer on e4b vs 0.32 on
  gemma3, where Q/K/V and O happen to share a shape). Fix: zero-pad O in both dims to
  (n_qkv, K_q) so it runs in the SAME context (zero Q4_0 blocks contribute nothing; only
  the first hidden_size outputs are read), guarded by `has_gemv`.
  Measured e4b: **O 133 → ~86 ms, decode 620 → ~530 ms/token (1.62 → 1.89 tok/s)**,
  output coherent. gemma3 unchanged (265 ms, guard doesn't fire).
  **Night total on e4b: 832 → 530 ms/token = 1.20 → 1.89 tok/s (+57%).**
  (The win was ~47 ms not ~110: the shared kernel does more MACs — 3072×2560 vs
  2560×2048 — so O didn't drop to gemma3's 0.32 ms/layer. Still net positive.)
  New e4b split @530 ms: **ffn 268 (51%)** / qkv 133 / o 86 / lm_head ~40 / host ~40.
  => NEXT BOTTLENECK IS THE FFN (weight-read + dequant) — the same wall as the FLM gap.
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
