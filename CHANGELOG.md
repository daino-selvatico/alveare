# Changelog

All notable changes to Alveare are documented here. This project targets the
AMD Ryzen AI (XDNA2) NPU on Linux.

## [Unreleased]

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
