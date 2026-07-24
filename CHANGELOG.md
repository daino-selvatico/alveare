# Changelog

All notable changes to Alveare are documented here. This project targets the
AMD Ryzen AI (XDNA2) NPU on Linux.

## [Unreleased]

### Added
- **Batched mmul GEMM prefill (~4x faster prefill).** A new systolic `aie::mmul`
  Q4_0 GEMM kernel (`kernels/gemm_q/gemm_q.cc`, ~56 GMAC/s vs ~5 for the
  element-wise gemm) makes batched prefill correct and ~4x faster (18-token prompt:
  ~40s -> ~10.3s), behind `ALVEARE_BATCH_PREFILL`. Foundation for speculative
  decoding. Build the gemm kernels with `tools/build_gemm_mmul.sh` (the AOT
  `.compile()` path is unreliable for these; see docs/kernel-roofline.md).

## [1.4.0] — 2026-07-23

_Decode on Gemma-4-12B goes from ~2.6 s/token to **~1 s/token** — a **2.6×**
speedup — by using all 32 NPU compute tiles and eliminating kernel context-switch
overhead. All changes are bit-exact (greedy tokens identical, an identical re-send
reproduces the output). Session progression: 2596 → 1561 → 1219 → 1109 → **1006
ms/token** (`benchmarks/README.md`)._

### Changed
- **~1 token/s on Gemma-4-12B.** The output projection (`w_o`) is zero-padded in
  its output dim for gemma4 sliding layers so it reuses the **same** `(8192, 4096)`
  kernel shape as the fused `w_qkv` — the two projections run back-to-back with no
  kernel context switch between them (~2.6 ms saved/layer). Combined with fused
  Q/K/V, decode reaches **~1029 ms/token (~0.97 tok/s; early/short-context tokens
  <1000 ms, >1 tok/s)** — down from ~1123. Bit-exact (greedy tokens identical,
  rerun reproduces output). Only sliding layers share (global O has a different K).
- **Fused Q/K/V projection: fewer NPU launches, ~10% faster decode.** The three
  attention input projections share the same input, so their weights are
  concatenated into one resident weight and run as a **single gemv** — 160 NPU
  launches/token instead of 248. The win is avoiding kernel context switches: a
  micro-benchmark shows switching gemv shapes costs **~2.6 ms/call** (vs ~0.9 ms
  for a same-shape call), so each removed launch removes a switch. Bit-exact
  (greedy tokens identical, rerun matches). Decode **~1236 → ~1123 ms/token**
  (~0.89 tok/s). This is the floor for per-shape kernels (3 shapes/layer =
  QKV, O, FFN → 3 switches/layer); crossing 1 tok/s needs a shared-context
  (runtime-shape) kernel to remove the remaining switches.
- **Decode ~2× faster again: all 32 compute tiles.** The GEMV and fused-FFN
  kernels now run across **32 cores** (4 rows × 8 columns) instead of 16. Two
  things unlocked the 4th-row routing: broadcasting the activation through a
  single global ObjectFifo (keeping it off the per-column memtiles, whose DMA
  channels are the bottleneck at 4 rows) and the `basic-sequential` DMA
  allocation scheme; the FFN weight interleave was generalized to 4 cores/column.
  Bit-exact (greedy tokens identical, rerun reproduces output):
  - GEMV 16384×4096: ~13.6 → **~3.35 ms (~4× vs the original 8-core)**
  - fused FFN: ~900 → **~570 ms/token**
  - lm_head: ~110 → **~59 ms**
  - **decode: ~1560 → ~1236 ms/token** (~2580 → ~1236 overall, **2.09×**; ~0.81 tok/s)
  - 32 is the physical tile ceiling; the attention GEMVs are now host-dispatch-
    bound (fewer/fused launches is the next lever, not more cores).

## [1.3.0] — 2026-07-22

### Changed
- **Decode ~40% faster: 16-core NPU kernels.** npu2 (Strix) has 32 compute tiles
  (8 cols × 4 rows) but the GEMV and fused-FFN kernels used only 8 (one per
  column). Both now split their work across **16 cores** (2 rows/column) via a
  per-column **memtile funnel** (ObjectFifo split/join, activation broadcast), so
  the shim does one weight-in + one output DMA per column instead of one per core
  (which capped the old design at 8). The FFN weights are packed **interleaved per
  column** so each column's fill is contiguous (avoids exhausting the memtile DMA
  descriptors). Bit-exact (greedy tokens identical, rerun reproduces output):
  - fused FFN: ~1650 → ~900 ms/token (~1.84×)
  - lm_head GEMV: ~213 → ~110 ms (2×)
  - **decode: ~2580 → ~1560 ms/token (~40% faster)**
  - 32 cores (4 rows) place but fail routing; 16 is the sweet spot.

## [1.2.0] — 2026-07-21

### Added
- **KV-cache reuse across requests** — the decode loop no longer resets the KV
  cache every request. It reuses the longest common prefix between the new prompt
  and the previously cached token sequence and prefills only the new tokens.
  Output is bit-identical (validated: a rerun reproduces the prior response
  exactly). `generate()` is serialized with a mutex, since it mutates the single
  shared cache.
- **Full multi-turn reuse** — the Gemma chat template now replays a completed
  assistant turn with the same generation-prompt suffix the model saw when
  producing it (`<|channel>thought<channel|>`), so the history tokens match what
  is already cached and the **entire conversation prefix is reused** each turn —
  only the newest user turn is prefilled. End-to-end over the HTTP server: turn 2
  reused 33/58 tokens (all of turn 1 incl. its reply), prefilling only the 24 new
  ones; an identical re-send skips prefill entirely (40s → 0.00s).

### Docs
- `docs/kernel-roofline.md`: documented the core-count ceiling — npu2 has 32
  compute tiles but the kernels use only 8 (one/column); using more fails shim-DMA
  placement, and the fix is a per-column memtile split/join dataflow (future work).

## [1.1.0] — 2026-07-21

### Added
- `quantize` now emits a `tokenizer.json` for **Gemma** models, reconstructed from
  the GGUF's embedded tokenizer (`tools/convert/gguf_tokenizer.py`) — fully offline,
  bit-exact vs the upstream HuggingFace tokenizer. The native C++ runtime works
  out-of-the-box with no manual tokenizer copy.
- **NPU benchmark suite** (`alveare bench` → `tests/bench/run_bench.py`): times every
  distinct kernel shape (ms + GMAC/s) and an end-to-end prefill/decode, then writes a
  timestamped Markdown report under `benchmarks/` and prepends a row to the trend
  table (`benchmarks/README.md`) so perf changes and regressions are tracked.
- Batched GEMM prefill infrastructure — `NpuRegistry::run_gemm` / `run_gemm_streamed`
  and `Model::run_layer_batch`, gated behind `ALVEARE_BATCH_PREFILL` — plus the
  `ALVEARE_SELFTEST` in-process generation hook (fixed prompt → stdout, no server).
- `docs/kernel-roofline.md` — analysis of the ~5 GMAC/s kernel ceiling.

### Changed
- **Fused FFN: ~27% faster decode.** The kernel now computes gate/up/GELU once and
  stores the whole activation vector (`act_all`), then runs the down projection in
  N=4 H-output passes reusing it — instead of recomputing gate/up per pass. FFN
  drops from ~2610 → ~1660 ms/token; decode ~3.6 → ~2.6 s/token, prefill ~28%
  faster too. Output unchanged (verify PASSES, greedy tokens identical).
- Added lightweight NPU profiling (`NpuRegistry::npu_seconds/ffn_seconds/npu_calls`)
  and a per-token decode breakdown in the server log (ffn / gemv / lm_head / cpu).

### Notes
- Batched prefill is correct but **not faster** than the per-token fused path (the
  NPU is compute-bound, so a `gemm(B=16)` costs the same as 16 `gemv`); the default
  stays per-token. Two kernel micro-optimizations (hoisting the Q4_0 dequant, and one
  `reduce_add` per row instead of per K-block) were measured to give **no** speedup —
  `gemv` is DMA/dataflow-bound and `gemm` compute-bound on the element-wise mul/mac
  path. See `docs/kernel-roofline.md`.

## [1.0.0] — 2026-07-20

First tagged release. Gemma-4-12B runs coherently end-to-end on the NPU through a
native C++ runtime, greedy-matching the Python runtime and `llama.cpp` token-for-token.

### Added
- **Native C++ runtime** (`runtime/cpp`) — the default `alveare serve` path, no
  Python in the decode loop:
  - Native XRT kernel registry with a resident-weight, bounded-context policy.
  - Decode loop + OpenAI-compatible HTTP server (cpp-httplib): `/v1/models`,
    `/v1/chat/completions`, non-streaming JSON and streaming SSE.
  - Hand-ported CPU math: RMSNorm, Llama/Gemma RoPE, sliding + global attention,
    greedy sampling; LM head tiled onto the NPU as quantized GEMVs.
  - Self-contained byte-level **BPE tokenizer** (`GemmaTokenizer`) that parses a
    HuggingFace `tokenizer.json` (space→▁ normalizer, rank BPE, byte fallback,
    atomic special tokens) plus the Gemma chat template. Bit-exact vs HF.
- **Fused FFN AIE kernel** (`kernels/ffn_fused`) — gate + up + GeGLU + down in one
  xclbin across `n_cores`, with fp32 accumulation and an H-output 2-pass split so
  the fp32 accumulator fits tile memory.

### Fixed
- Gemma-4 **per-layer output scale** was missing — layer activations grew ~19× and
  the LM-head logits saturated the soft-cap, collapsing greedy decoding.
- FFN fused kernel accumulated gate/up and the down projection in **bf16**, which
  compounded to ~13% error over 48 layers and produced gibberish; now fp32.
- `token_embd` is IEEE **float16**, was being decoded as bfloat16; added the
  missing `sqrt(hidden_size)` embedding scale for Gemma.
- LM head segfault: the packed Q4_0 head was read as dense bf16 and indexed out of
  bounds right after prefill.
- Tokenizer toolchain: the AIE design CLI resolved devices at `n_cols=1`
  (single-column), so multi-core kernels failed placement — now forced to the full
  column device.

### Known limitations
- Decode ~3.6 s/token on Gemma-4-12B (correctness first; the fused-FFN H-split
  recomputes gate/up per pass — caching it is a planned follow-up).
- The native C++ runtime needs a `tokenizer.json` in the weights directory;
  `quantize` does not emit one yet (copy it from the source model, or use `--legacy`).
- NPU-only, Linux-only, XDNA2. Experimental — expect rough edges.
