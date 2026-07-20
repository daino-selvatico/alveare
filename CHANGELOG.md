# Changelog

All notable changes to Alveare are documented here. This project targets the
AMD Ryzen AI (XDNA2) NPU on Linux.

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
