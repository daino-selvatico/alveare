# C++ Runtime — Implementation Plan

Status: **draft, awaiting approval**
Date: 2026-07-15
Branch: intended `feature/cpp-runtime` (⚠️ working tree is currently on `main` — see Open Questions #0)
Supersedes the "deferred to M4" note in `docs/decisions/0004-python-first-runtime.md`.

## 1. Goal

Rewrite the inference loop and the OpenAI-compatible server in pure C++, calling XDNA2
through **native XRT C++** (`xrt::device`, `xrt::kernel`, `xrt::run`, `xrt::bo`) instead of
`pyxrt` + `aie.iron`. The single motivation is to kill per-op FFI / interpreter overhead: today
each token drives 336 matmul launches through Python, and the host↔NPU round-trip is dominated
by Python glue, not by the NPU.

**Non-goal for this branch:** changing the kernels themselves. We keep the existing hand-written
AIE `.cc` kernels and their host ABI byte-for-byte. This is a *runtime* port, not a *kernel* port.

## 2. What I found in the current runtime (the contract to preserve)

I read `runtime/py/{model.py,server.py,layer.py,sampler.py,tokenizer_glue.py}` and
`kernels/{gemv_q,gemm_q}/*.py`. The important, non-obvious findings that shape this plan:

**The NPU surface is only two kernels.** Despite `kernels/` containing rmsnorm / rope / attention
designs, the *actual* decode/prefill paths in `model.py` do **not** use them. In `run_layer` and
`run_layer_batch`, everything except the matmuls runs on the host in NumPy:

- **NPU:** `gemv_q` (decode, one row-vector at a time) and `gemm_q` (prefill, batch B=16).
- **CPU (host):** embedding lookup + `sqrt(hidden)` scale, RMSNorm (`run_rmsnorm_cpu`), QK/V-norm,
  RoPE (`run_rope_cpu_gemma`), attention + softmax (`run_attention_host`, sliding-window aware),
  GeGLU/SiLU, both residual adds, per-layer output scale, final norm, logit soft-cap, and sampling.

So the C++ port must faithfully reimplement a fair amount of **CPU math**, and only *two* NPU entry
points. The user brief mentions "RoPE and RMSNorm in C++" — correct, but note those are **CPU**
functions here, not NPU offloads.

**Quantized weight format (`*.weight_packed.npy`, uint8).** Q4_0, block = 32 elements → **20 bytes**:
16 bytes of packed 4-bit nibbles + 2 bytes bf16 block scale + 2 bytes padding. A weight of logical
shape `(N, K)` is stored `(N, K/32*20)` uint8. `tools/convert/gemv_q_convert.py` is the reference for
pack/unpack; the C++ side never needs to *quantize* (weights are pre-packed on disk) but does need to
read this layout into a device buffer verbatim.

**Resident weights.** For Gemma-4, `LazyLayerWeights` uploads all 7 projections of every layer to the
NPU once (`iron.tensor(..., device="npu")`) and never frees them (`clear()` is a deliberate no-op).
48 layers × 7 = 336 device-resident weight tensors, ~5.5 GB. Decode then does zero-copy: only the
activation vector is streamed per matmul.

**Resident scratch buffers.** `model.py` pre-allocates oversized device tensors and reuses them:
`w_gemv_t` `(MAX_N, MAX_K/32*20)` uint8, `x_gemv_t` `(MAX_K)` bf16, `y_gemv_t` `(MAX_N)` bf16, and the
GEMM equivalents at `MAX_BATCH_SIZE=16`. `MAX_N = MAX_K = 16384`. Each call writes a sub-slice of the
mapped host buffer, syncs to device, runs, syncs the output back.

**How `iron.jit` actually reaches the NPU (the crux — see §4).** `@iron.jit` compiles the IRON design
to an `.xclbin` **plus an instruction stream**, specialized on the compile-time constants
(`N, K, m=32, k_tile=256`, and `B` for gemm; `n_cores` is derived from `N`). At run time pyxrt loads
the xclbin into a hardware context, uploads the instruction buffer to a BO, and each kernel invocation
is effectively `xrt::run(kernel)(opcode, instr_bo, n_instr, argW_bo, argX_bo, argY_bo)`. **The xclbin
is shape-specialized**, so every distinct `(N, K)` (decode) and `(B, N, K)` (prefill) is a *different*
xclbin/context.

**The 8-context limit + `err=-22` eviction.** XDNA2 allows only ~8 concurrent hardware contexts. With
per-shape xclbins the model needs more distinct shapes than that (7 projection shapes × sliding/global
variants + lm_head + the gemm variants), so loading them all at once overflows. The Python code catches
the XRT `-22` ("out of contexts") error and evicts. This policy must be reproduced in C++.

**Server.** FastAPI, single `asyncio.Lock` serializing all generation, `/v1/models` and
`/v1/chat/completions` (streaming SSE + non-streaming). Prefill for Gemma-4 uses `forward_batch`
(NPU GEMM, chunks of 16); decode uses `forward` (NPU GEMV). Tokenization uses HuggingFace
`transformers` `AutoTokenizer` + the model's **Jinja chat template** (see Open Questions #1 — this is
the biggest non-compute gap).

## 3. Proposed layout: `runtime/cpp/`

```
runtime/cpp/
  CMakeLists.txt
  include/alveare/
    bf16.h            // bfloat16 <-> float, round-to-nearest-even (match ml_dtypes)
    npy.h             // minimal .npy v1 reader (header parse + typed span over mmap)
    config.h          // model config struct, parsed from config.json
    tensor.h          // host tensor views (row spans, bf16/fp32/uint8)
    npu.h             // XRT wrapper: device, xclbin registry, kernel/run cache, BOs
    weights.h         // resident weight store (per-layer projections, norms, embed, lm_head)
    model.h           // LlamaNpuModel: forward(), forward_batch(), run_layer()
    sampler.h
    tokenizer.h       // see Open Questions #1
    server.h
  src/
    bf16.cpp npy.cpp config.cpp npu.cpp weights.cpp
    model.cpp         // the layer loop + all CPU math (rmsnorm/rope/attn/geglu/...)
    sampler.cpp server.cpp main.cpp
  third_party/        // httplib.h, json.hpp (vendored, header-only) — or via CMake FetchContent
  test/
    parity_test.cpp   // compares C++ tensors against Python golden dumps
```

Header-only deps: **cpp-httplib** (HTTP + SSE) and **nlohmann/json**. Both vendored as single headers
or pulled with `FetchContent`.

## 4. The kernel/XCLBIN strategy (most important decision)

The C++ runtime cannot JIT. We need the `.xclbin` + instruction buffers **ahead of time (AOT)**. Two
viable models, and I want a decision (Open Questions #2):

**Option A — per-shape xclbins + LRU context eviction (mirrors today's code).**
Enumerate every `(N, K)` decode shape and every `(B, N, K)` prefill shape the target model uses
(finite: q/k/v/o/gate/up/down for sliding & global layers, + lm_head), AOT-compile one xclbin each,
and in C++ keep an LRU registry of *loaded* hardware contexts. On XRT `-22`, evict the least-recently
used context and retry. Pro: no wasted MACs, matches current numerics exactly. Con: reintroduces the
context-thrashing management and its latency cost.

**Option B — single MAX-sized xclbin + pad every op to (16384, 16384) / (16, 16384, 16384).**
Compile *one* gemv xclbin and *one* gemm xclbin at the MAX shape (N=16384 ⇒ n_cores=8). Every
projection pads up to that shape. Pro: exactly 2 contexts, the `-22` eviction logic disappears entirely,
much simpler C++. Con: large wasted compute (e.g. K padded 3840→16384 is ~4× the MACs) — likely too
slow for decode. The user brief says "we padded all shapes to MAX to avoid thrashing," which describes
Option B, but the code's fast path actually calls gemv with the *real* N,K (Option A). **This
contradiction needs resolving before I write the NPU layer.**

**AOT build tooling (either option).** Add `tools/build_kernels.py` that drives the existing
`gemv_q_npu` / `gemm_q_npu` `iron.jit` once per required shape and harvests the compiled `.xclbin` +
instruction `.bin` from the JIT cache into `kernels/build/<name>_<shape>.{xclbin,insts.bin}`. This
reuses the proven kernel sources unchanged and produces a manifest (`kernels/build/manifest.json`:
shape → files, arg order, opcode, n_instr) that the C++ `npu.cpp` loads at startup. No hand-written
MLIR duplication.

**XRT invocation shape (C++).** Per loaded shape: `xrt::xclbin` → register on `xrt::device` →
`xrt::hw_context` → `xrt::kernel(ctx, "MLIR_AIE")`; upload instructions to a `xrt::bo` (cacheable,
group id from `kernel.group_id(1)`); weight/activation/output BOs bound to their arg groups. A run is
`xrt::run r(kernel); r.set_arg(0, opcode); r.set_arg(1, instr_bo); r.set_arg(2, n_instr);
r.set_arg(3, wbo); r.set_arg(4, xbo); r.set_arg(5, ybo); r.start(); r.wait();`. Exact arg indices come
from the manifest (validated against a pyxrt trace during bring-up).

## 5. Component plan (CPU side)

- **bf16.h** — storage `uint16_t`; `to_f32` (bit expand) and `from_f32` (round-to-nearest-even, matching
  `ml_dtypes.bfloat16` so parity holds). All hidden-state math is done in `float`, stored back as bf16
  exactly where NumPy does `.astype(bfloat16)`, to preserve token-for-token equality with `llama.cpp`.
- **npy.h** — parse the `\x93NUMPY` header (dtype, shape, fortran flag), mmap the file, return a typed
  span. Handles uint8 (packed weights), fp32 (norms), and bf16/uint16 (embeddings, scales).
- **weights.h/.cpp** — mirror `LazyLayerWeights`: on load, upload each `*.weight_packed.npy` into a
  resident weight BO (device-resident, synced once). Keep norms / embeddings / lm_head / RoPE tables in
  host RAM. Handle Gemma-4 sliding vs global layer geometry (`(l+1)%6`), the tied lm_head, and per-layer
  output scales.
- **model.cpp** — `run_layer` / `run_layer_batch` reimplemented in C++: RMSNorm (eps 1e-6 Gemma / 1e-5
  Llama), QK/V-norm, Gemma RoPE (partial-rotary 0.25 for global dim=512, full for sliding dim=256, bases
  10000 / 1e6), sliding-window attention with the 1024/512 window, GeGLU (`gelu_pytorch_tanh`) / SiLU,
  residuals, output scale, final norm, `30*tanh(logits/30)` soft-cap. RoPE cos/sin tables precomputed at
  startup exactly as in `precompute_cos_sin_table_gemma`.
- **sampler.cpp** — port `sampler.py`: temperature, top-k, top-p, greedy at T=0. Same RNG semantics
  (seedable) so tests are deterministic.

## 6. Server (`server.cpp`)

cpp-httplib server, one global `std::mutex` replacing the `asyncio.Lock` (generation is already fully
serialized today, so no concurrency regression). Endpoints:

- `GET /v1/models` — static model descriptor.
- `POST /v1/chat/completions` — non-streaming JSON and streaming SSE (`text/event-stream`,
  `data: {chunk}\n\n` … `data: [DONE]\n\n`), byte-compatible with the current Python responses.

Generation flow unchanged: reset KV caches → prefill (`forward_batch`, B=16 for Gemma-4; per-token CPU
prefill for the small models) → decode loop (`forward` + `sample`) until EOS/EOT or `max_tokens`.

## 7. Build system

`CMakeLists.txt`, C++17. `find_package(XRT)` (or explicit `libxrt_coreutil` / `libxrt_core` +
`$XILINX_XRT/include`). Link `xrt_coreutil xrt_core pthread`. httplib/json header-only. A `parity_test`
target. Output binary `alveare-server`, configured via the same env vars as today
(`ALVEARE_WEIGHTS_DIR`, `ALVEARE_HOST`, `ALVEARE_PORT`).

## 8. Verification strategy

Parity-first, matching the project's "correctness came first" ethos:

1. **Golden dumps** — a small Python harness runs the existing runtime and dumps per-layer bf16 tensors
   (x_norm, q/k/v, attn_out, gate/up, layer output) and final logits for a fixed prompt/seed.
2. **`parity_test.cpp`** — runs the C++ path on the same input and asserts bitwise/near-bitwise equality
   against the dumps (bf16 exact where possible; documented tolerance where fp32 reduction order differs).
3. **End-to-end** — greedy (T=0) C++ output must match the Python runtime **and** `llama.cpp` token-for-token
   on the standard Gemma-4-12B check, before any perf comparison.
4. **Perf** — record decode s/token and TTFT vs the ~4.6 s/token Python baseline to quantify the FFI win.
5. A **subagent verification pass** on the numerics-critical files (bf16 rounding, RoPE indexing,
   sliding-window slicing) before merge.

## 9. Phasing

- **P0 — DONE.** CMake, `bf16`, `.npy` loader, XRT device open, and one hard-coded gemv (N=256,K=256)
  run end-to-end natively (`runtime/cpp/src/main.cpp`, opcode 3, kernel entry `MLIR_AIE`; ABI in
  `meta.json`). Result `max_diff = 0.0625`, within bf16 tolerance. **Caveat:** `expected.npy` is the
  **CPU dequant reference** (`ref_gemv_q`), so P0 proves *NPU-called-from-C++ is numerically correct*,
  **not** bitwise NPU-vs-Python parity (that comparison should read back the Python `pyxrt` NPU output
  and expect `0.0` — add as a follow-up check). Key API discovered:
  `gemv_q_npu.specialize(N,K,m,k_tile).compile(xclbin_path=, inst_path=)` — this is the AOT hook P1 uses.
- **P1 — in progress.** `tools/build_kernels.py` written: enumerates distinct matmul shapes from the
  on-disk packed weights (not hardcoded), compiles each gemv + gemm via the `.specialize().compile()`
  API, and emits `kernels/build/manifest.json` (shape → xclbin/insts, n_cores, ABI). Must run on the
  NPU/toolchain host (not this sandbox). Next: the C++ NPU registry that loads the manifest and applies
  the ≤8-context resident policy (§4 / decision #2 — expect ~10–11 Gemma-4 shapes, so bucketing is needed).
- **P2** — full Gemma-4 `run_layer` (decode) parity, layer-by-layer against golden dumps.
- **P3** — prefill (`forward_batch` / gemm_q) parity.
- **P4** — HTTP server (both modes) + tokenizer decision resolved (§ Open Questions #1).
- **P5** — end-to-end token-for-token parity, then perf measurement + verification pass.

## 10. Decisions (recommended — perf is the north star)

Primary goal: cut decode latency (~3 s/token today) and TTFT. The single biggest lever is removing
per-op Python/FFI + host↔device sync overhead — that is exactly what this port does — followed by
reducing kernel launches and keeping the decode working set resident. Recommendations:

0. **Branch** — DONE: created `feature/cpp-runtime` as a linked worktree at `.worktrees/cpp-runtime`;
   `main` left clean.

1. **Tokenizer → embed `tokenizers-cpp` + hand-port the chat template.** Tokenization runs once per
   request, so it has **zero impact on decode perf** — choose for a single self-contained binary, not
   speed. `tokenizers-cpp` reads the same `tokenizer.json`; the per-model chat templates are short and
   port to a small C++ function. (A Python sidecar is available as a throwaway bootstrap in P0–P3 if it
   speeds up bring-up, but the end state is no Python at runtime.)

2. **Context policy → keep the whole decode working set resident; bucket-pad only if it exceeds 8.**
   The perf killer is reloading an xclbin *inside the token loop*. In steady-state decode the same
   projection shapes repeat every token, so:
   - If the distinct decode GEMV shapes fit in ≤8 contexts, load them all once and **never evict during
     decode** (Option A, all-resident). Eviction/`-22` handling is kept only for the rare shapes
     (prefill GEMM, lm_head).
   - If they exceed 8, **bucket a few shapes into a shared padded xclbin** so the total is ≤8 and all
     stay resident. Pad only enough to merge (modest wasted MACs), never blanket-pad to 16384² (Option B
     is rejected — ~4× wasted compute).
   - Measure the distinct shape count first (`tools/build_kernels.py` will enumerate it) and pick per
     the above. This makes "no context thrashing in the hot loop" a hard design invariant.
   - **Highest-leverage follow-up (own branch, not this port): fuse projections.** Q/K/V share `x_norm`
     → one concatenated matmul; gate+up → one matmul. That turns 7 launches/layer into ~4, cutting both
     launch overhead and distinct contexts. This is a kernel+weights change (you already have
     `feature/npu-ffn-fusion`), so it stays out of the pure-port branch but is the next big win after it.

3. **Bring-up order → Llama-3.2-1B first, then straight to Gemma-4-12B.** The 1B is the fastest parity
   loop (16 layers, no QK-norm / sliding window / soft-cap) to validate the whole C++ + XRT + parity
   harness. Then jump to the 12B (the actual perf target); Gemma-3-1B only if a Gemma-specific bug needs
   isolating.

4. **Deps → vendor single headers in `third_party/`.** `httplib.h` + `json.hpp` committed to the repo.
   No build-time network (the project already supports air-gapped/offline use).

5. **CPU math → clean scalar port first, then SIMD/threading guided by profiling.** Correctness/parity
   first; write the host math over contiguous `float` buffers so it vectorizes later. Expected first
   hotspot as context grows is host attention over the KV cache — thread it across heads in the second
   pass. Also minimize bf16↔fp32 churn (keep fp32 accumulators, convert only at the documented `.astype`
   boundaries) to preserve token-for-token parity.

### Cross-cutting perf checklist (baked into the port)

- Weights are resident BOs, synced once; per matmul only the small activation vector is written/synced.
- Submit independent projections (Q/K/V) back-to-back without intermediate host round-trips; wait once.
- Reuse pinned activation/output BOs (no per-call allocation), and avoid redundant `sync` on unchanged buffers.
- Decode steady state must issue **zero** xclbin (re)loads — enforced by decision #2.
```
