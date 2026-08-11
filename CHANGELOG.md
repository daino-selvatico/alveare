# Changelog

All notable changes to Alveare are documented here. This project targets the
AMD Ryzen AI (XDNA2) NPU on Linux.

## [Unreleased]

## [2.0.0-alpha.3] — 2026-08-11

### Fixed (UI)
- **Chat tok/s now mirrors the server-measured rate** (same value as the Models &
  Benchmarks view). It used to fall back to a cumulative client estimate, which ramps up
  from a low value instead of showing the real decode rate; the fallback is now a sliding
  window and is labelled approximate.
- **Image/audio upload is gated as "Coming soon"** in both the menu and drag-and-drop, and
  rejected by the backend: the runtime is text-only (no vision/audio encoder), so those
  files never reached the model. Document upload (text extraction) still works.
- **`index.html` is served with `no-cache`**, so a rebuilt frontend is actually picked up
  instead of the browser silently keeping the previous bundle.

### Added
- **`ALVEARE_ONESHAPE` (opt-in): ~13% faster decode on Gemma-4-E4B.** An NPU
  hardware-context switch costs a **fixed ~2.5 ms** (measured with the new
  `bench_switch` micro-benchmark: `gemv↔gemv` 2.57 ms, `ffn↔gemv` 2.62 ms — independent
  of design size), and decode alternates kernel shapes every layer, so E4B was spending
  **~210 ms/token (40%)** on switches alone. With the flag, every matmul of a layer runs
  on ONE kernel shape — `gate++up` as tiles, `GELU(gate)*up` on the host, the `down`
  projection split along its input dim with the partials summed on the host, and QKV/O
  zero-padded onto the same shape — so those layers never switch context.
  **E4B: ~530 → ~470 ms/token (1.89 → 2.13 tok/s).** Default OFF; enable with
  `ALVEARE_ONESHAPE=1`.
  The path is self-gating: it only engages when the shared shape stays within 1.25x the
  padded hidden size, because the `down` tiles use only `hidden` of those rows. Measured
  counter-examples that the gate now excludes: E4B's global layers (shape 6144, 2.4x)
  cost 514 vs 468 ms/token, and the 12B (8192, 2.0x) 1161 vs 1010 — so the 12B and
  Gemma-3 are simply left on the fused-FFN path and are unchanged.
- **`bench_switch`** micro-benchmark (`runtime/cpp/test/`) measuring NPU context-switch
  and per-call overhead, plus a decode profiler split for the PLE cost.

### Changed
- **Gemma-4-E4B decode ~15% faster: the O projection now reuses the fused-QKV kernel
  context.** E4B's fused QKV gemv is `(3072, 2560)` while O was `(2560, 2048)` — a
  different shape, so every layer paid a ~2.6 ms NPU context switch for O. O is now
  zero-padded in both dims to the QKV kernel's `(n_qkv, K_q)` and runs in the same
  context (padded Q4_0 blocks have scale 0 and contribute nothing; only the first
  `hidden_size` outputs are read). Measured: **O 133 → ~86 ms, decode ~620 → ~530
  ms/token (1.62 → 1.89 tok/s)**. Gemma-3 and the 12B are unchanged (the path is guarded
  by `has_gemv`), all three still decode coherently.
  Together with the `-O3` build fix below, **E4B went 832 → 530 ms/token (1.20 → 1.89
  tok/s, +57%)** with no quality cost.

### Fixed
- **The native runtime was being built at `-O0` — now defaults to Release (`-O3`).**
  `runtime/cpp/CMakeLists.txt` never set `CMAKE_BUILD_TYPE`, so CMake passed no
  optimization flags and every host-side Q4 dot product was left uninlined and
  unvectorized. Measured with `ALVEARE_PROFILE_DECODE=1`:
  - **Gemma-4-E4B: ~832 → ~620 ms/token (1.20 → 1.62 tok/s, +35%)** — its per-layer PLE
    injection collapsed from **200 ms to ~27 ms/token (7.4×)** and the host share of
    decode fell from 31% to ~7%.
  - Gemma-3 (~270 ms/tok) and the 12B (~1.0 s/tok) are unchanged — both were already
    NPU-bound — and all three still decode coherently (greedy "Paris"). No quality cost.
- The decode profiler now reports the PLE cost separately from `cpu_rest`.

## [2.0.0-alpha.2] — 2026-08-08

_Progress since alpha.1, toward the 2.0 release. Gemma-3 now runs on the fast NPU
FFN (the CPU fallback is gone), decoding gained **configurable sampling**, and the
web UI grew into a real chat app: streaming, conversation history, file upload, and a
generation-settings panel._

### Added
- **Sampling (temperature / top-k / top-p / seed).** The decoder is no longer
  greedy-only: `temperature`, `top_p`, `top_k`, and `seed` are read from the
  OpenAI-style request body. `temperature<=0` keeps the exact greedy argmax (the
  12B/e4b/gemma3 defaults stay deterministic/bit-exact); `temperature>0` does
  temperature scaling → optional top-k → nucleus (top-p) → multinomial sampling. The
  RNG is seeded once per request, so a fixed `seed` reproduces the whole generation
  while `seed=0` is nondeterministic.
- **Real-time streaming chat + Markdown** in the web UI (SSE token-by-token, code
  blocks with copy, auto-scroll, Stop).
- **Conversation history & multi-turn** — persistent conversations (localStorage),
  sidebar with new/rename/delete, and full-history requests that exploit the runtime's
  KV-cache prefix reuse.
- **Multimodal file upload** — images/audio/documents with drag-and-drop, attachment
  chips, and inline players/preview.
- **Generation-settings panel** — system prompt (with presets), temperature/top-p/
  top-k sliders, max-tokens/context, thinking toggle, plus a dark/light theme; settings
  persist per-conversation.
- **Model-management UX** — live model-load progress and measured tok/s surfaced from
  the server logs, with readable error banners on load failure.
- **First-run onboarding, error resilience & polish** — onboarding wizard for the
  no-model case, React error boundary, loading/empty states, and a logs viewer.
- **Internationalization (IT/EN)** with a language switcher (persisted, browser-default).
- **Accessibility pass** — aria labels/roles, focus trap in modals, visible focus,
  contrast, and `prefers-reduced-motion`.
- **Export / import conversations** (JSON) with validation.
- **Keyboard shortcuts** (Enter/Shift+Enter, Esc to stop, Ctrl/Cmd+K new chat) + help overlay.
- **Models / benchmarks view** — per-model size/arch/status and live tok/s.
- **Frontend test suite** (Vitest + Testing Library) for core utils and components.
- **2.0 documentation** — refreshed README plus quickstart, adding-models, and sampling
  guides.

### Fixed
- **Gemma-3 back on the NPU FFN.** The H=2048 fused-FFN kernel was NaN-ing at runtime
  because `build_kernels.py`'s direct-path `.compile(xclbin_path, inst_path)` bypasses
  the jit cache and emits a broken (~65 KB smaller) xclbin. It now warms the cache with
  a no-arg `.compile()` and copies the correct artifacts, so Gemma-3 drops the CPU
  fallback and decodes ~1.4× faster on the NPU (~276 ms/token). 12B/e4b unaffected.

## [2.0.0-alpha.1] — 2026-08-05

_First 2.0 pre-release. Alveare now serves **three Gemma families** end-to-end on
the NPU — the 12B (gemma4), **Gemma-4-E4B** (new: Per-Layer Embeddings + KV-sharing),
and **Gemma-3-1B** (now coherent) — behind a **web UI** with in-browser model
management, plus one-command Hugging Face setup. This is an alpha: the 2.0 final
still targets the fused-decode performance work, and Gemma-3's FFN currently runs on
the CPU fallback (correct, slower) pending an NPU-kernel build fix._

### Added
- **Gemma-4-E4B serving** — the E4B variant (gemma4 + Per-Layer Embeddings, 42
  layers, KV-cache sharing) runs end-to-end on the NPU, validated vs a llama.cpp
  oracle.
- **Web UI** — a React/Vite frontend + FastAPI control server that discovers
  quantized-weight dirs, launches/stops models, and adds new models in-browser
  (add-model flow, max-tokens / max-context controls, thinking toggle).
- **Hugging Face auto-download + custom quantizers** — `tools/setup_model.py`
  fetches a model from HF and quantizes it in one command; `base_quantizer.py`
  provides a pluggable quantizer base (see docs/CUSTOM_QUANTIZERS.md).

### Fixed
- **Gemma-3-1B now produces coherent output on the NPU.** Three bugs: the GGUF's
  SentencePiece tokenizer shipped no BPE merges (byte-fallback word-salad) — now
  reconstructed from the SPM scores; the fused-FFN weight pack used a mismatched
  `k_tile` (128 vs the kernel's 256); and the chat prompt used Gemma-4's turn/channel
  tokens. Gemma-3's FFN is routed to the (correct) CPU fallback until its NPU kernel
  build is fixed. 12B and E4B are unaffected.
- **Thinking toggle** — enabling "thinking" no longer suppressed it (the empty
  thought-channel block was appended on the wrong branch).

## [1.5.0] — 2026-07-27

_Faster prefill and a characterized decode ceiling. A new systolic `aie::mmul`
Q4_0 GEMM makes **batched prefill ~4x faster** (18-token: ~40s -> ~10.3s) and was
tuned **1.57x** (56 -> 88 GMAC/s), all bit-exact. Opt-in **speculative decode**
(prompt-lookup drafter + batched B=8 verify) gives a clear win on repetitive/
structured output (~473 ms/tok on a fully-accepted 8-token burst). Gated profilers
established that decode is ~97% NPU dispatch and memory/dequant-bound: the
architectural ceiling on 12B Q4 is **~3 tok/s** — bf16 and int8 MAC throughput are
equal (~8% apart) and dequant/mmul overlap regresses, so breaking it needs a smaller
model or lower-bit quant, not kernel work. Finally, `./alveare` now auto-activates
its environment, so one command just works from any shell._

### Added
- **Batched mmul GEMM prefill (~4x faster prefill).** A new systolic `aie::mmul`
  Q4_0 GEMM kernel (`kernels/gemm_q/gemm_q.cc`) makes batched prefill correct and
  ~4x faster (18-token prompt: ~40s -> ~10.3s), behind `ALVEARE_BATCH_PREFILL`.
  Build the gemm kernels with `tools/build_gemm_mmul.sh` (the AOT `.compile()` path
  is unreliable for these; see docs/kernel-roofline.md).
- **1.57x faster mmul GEMM kernel (56 -> 88 GMAC/s), bit-exact.** Three safe steps:
  hoist the weight transpose out of the batch loop, run 4 concurrent batch
  accumulators to hide systolic latency, and fuse the Q4_0 dequant. Prefill for a
  39-token prompt drops 16.6s -> 12.4s. (dequant/mmul overlap was tried and rejected
  — ~2x regression on the shared vector datapath; int8 mmul benchmarked at only ~8%
  over bf16, so 88 GMAC/s is effectively the kernel ceiling.)
- **Speculative decode (opt-in, `ALVEARE_SPECULATIVE`, gemma4).** A prompt-lookup
  n-gram drafter proposes up to 7 tokens, verified in ONE batched B=8 forward;
  accept the matching prefix + one correction, fall back to the normal single-token
  decode when no draft is found (never slower than the default path). Situational: a
  clear win on repetitive/structured output (measured ~473 ms/tok vs ~910 for a
  fully-accepted 8-token burst), roughly neutral-to-negative on novel prose (a
  rejected draft pays the full ~4s verify). Deterministic; rerun reproduces output.
- **Gated decode profilers (`ALVEARE_PROFILE_DECODE`, zero cost when off).** Break
  down per-token decode and the batched verify by phase. These established that
  decode is ~97% NPU dispatch and memory/dequant-bound, pinning the architectural
  ceiling at ~3 tok/s on 12B Q4 (see docs/kernel-roofline.md).

### Changed
- **`./alveare` auto-activates its environment.** Commands that touch the NPU
  (`serve`, `check`, `build-kernels`, `quantize`, `bench`, ...) now detect and
  activate the `alveare-aie` conda env + source the mlir-aie NPU stack, then
  re-exec — so `./alveare serve gemma4` works from any shell (e.g. conda `base`)
  with no manual `conda activate` / `env_setup.sh`. If conda or the env is missing
  it points at `./alveare install`. The native C++ server no longer wrongly
  requires the Python `fastapi`/`uvicorn`/`pyxrt` deps (those gate `--legacy` only).

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
