# Plan — Decode performance toward FastFlowLM-level throughput

**Goal.** Close the decode-throughput gap with FastFlowLM (FLM). FLM (closed,
patent-pending AIE kernels) on the same XDNA2 NPU class does **Gemma-4 E4B ~12.6
tok/s decode @1k** (E2B 22.6), prefill 441-720 tok/s (Ryzen AI 7 350). Alveare's
e4b is **~1.1 tok/s (~907 ms/token)** — an ~11x gap. The gap is **host/dispatch
overhead, not hardware**: FLM proves the silicon does ~12 tok/s on E4B.

## Where Alveare's per-token time goes (measured, ALVEARE_PROFILE_DECODE)

e4b, ~907 ms/token, **142 NPU dispatches/token**:
- **host CPU ~320 ms (35%)** — attention, RMSNorm, RoPE, PLE, KV writes ALL run on
  the host, with activations round-tripping NPU↔host every layer.
- **gemv ~262 ms (29%)** — QKV + O projections: ~2 gemv × 42 layers + lm_head 16
  tiles ≈ 100 tiny dispatches; the matmuls are sub-ms, so this is mostly **dispatch
  overhead** (~2.6 ms/call for BO sync + submit + wait).
- **ffn ~272 ms (30%)** — 42 fused-FFN calls (~6.5 ms each); the most compute-bound
  part.
- **lm_head ~40 ms** — 16 tiles = 16 dispatches.

FLM's ~33-80 ms/token means it is memory-bandwidth-bound (streams weights near peak
LPDDR5), i.e. a **fused on-NPU decode** with few dispatches and KV/activations kept
on-chip — no per-layer host round-trips.

## Levers, in priority order (measure each before/after — don't guess)

**Tier 1 — cut dispatch count + host round-trips (biggest, most tractable):**
1. **Batch the lm_head** 16 tiles → 1 gemv (or a single tiled kernel call): ~15
   dispatches gone. Small but easy.
2. **Fuse QKV+O per layer** into one weight/one gemv where shapes allow (QKV already
   fused; fold O), and keep the qkv/o resident: ~1 dispatch/layer saved.
3. **Move RMSNorm + RoPE onto the NPU** (or fuse into the adjacent matmul kernels) so
   the hidden state doesn't round-trip to the host between attention and FFN. This is
   the lever against the 320 ms host block.

**Tier 2 — on-NPU attention:**
4. Implement attention (scores + softmax + V readout, GQA, sliding window, KV read)
   as an NPU kernel, with the KV cache in NPU-accessible memory. Removes the biggest
   host chunk and the per-layer Q/K/V round-trips. Significant kernel work.

**Tier 3 — the FLM-style rework (the real ~11x):**
5. A **single fused decode graph / persistent kernel** that runs all N layers on the
   NPU per token (attention + norms + FFN + PLE on-chip, weights streamed, KV
   on-chip), reducing 142 dispatches/token to a handful. This is essentially a second
   decode runtime; largest effort but the path to memory-bandwidth-bound decode.

## Approach / discipline
- **Profile first, every step** (ALVEARE_PROFILE_DECODE gives the phase breakdown;
  add a dispatch counter). The overlap/int8 dead-ends taught us to measure before
  rewriting.
- Start Tier 1 (incremental, low-risk, real dispatch savings) to get an early
  fraction of the gain and validate the dispatch-overhead hypothesis, before
  committing to Tier 2/3 kernel rewrites.
- Keep the 12B bit-exact and the e4b oracle-matching at every step (gated, validated).
- The mmul GEMM peak is NOT the decode bottleneck (int8≈bf16, overlap regressed —
  see [[cpp-runtime-mmul-kernel-opt]], [[cpp-runtime-decode-ceiling]]); dispatch +
  host overhead is. Focus there.

## Notes on FLM (reference, not copyable)
FLM is open-core: CLI/orchestration MIT, **AIE kernels are closed** (prebuilt
.xclbin, one set per model+size). So the exact fused-decode kernel recipe isn't
public — but the throughput proves the target is reachable with the same IRON +
MLIR-AIE tools Alveare uses. Benchmarks: fastflowlm.com/docs/benchmarks/gemma4_results.
