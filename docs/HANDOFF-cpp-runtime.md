# Handoff — Alveare C++ runtime port (resume prompt)

Paste the block below into Claude Code running at `/home/daino/progetti/alveare`.

---

We're porting **Alveare** — an open-source LLM inference runtime for the AMD Ryzen AI **XDNA2** NPU
(Linux) — from Python to **C++**, to kill per-op FFI/interpreter overhead (today ~3 s/token; 336 matmul
launches per token driven through Python). This is a **runtime** port only: the hand-written AIE `.cc`
kernels and their host ABI stay unchanged. Repo: `/home/daino/progetti/alveare`. Work lives on branch
`feature/cpp-runtime`. **Performance is the primary goal.**

## First: fix the git state (a previous session created the worktree from a sandbox with wrong paths)

The files for this work currently sit **uncommitted** in `/home/daino/progetti/alveare/.worktrees/cpp-runtime/`
(a plain folder). Branch `feature/cpp-runtime` exists but is at old HEAD `bd04301` and does NOT yet
contain them. `main` is clean. Recover cleanly:

```bash
cd /home/daino/progetti/alveare
git worktree prune
mv .worktrees/cpp-runtime /tmp/alv-cpp-work
git worktree add .worktrees/cpp-runtime feature/cpp-runtime
rsync -a --exclude '.git' --exclude 'runtime/cpp/build/' /tmp/alv-cpp-work/ .worktrees/cpp-runtime/
cd .worktrees/cpp-runtime && git add -A && git commit -m "P0 native XRT gemv + P1 kernel harvester + C++ runtime plan"
```

(If you prefer a sibling worktree like `../alveare-cpp`, that's fine too — just get these files committed
onto `feature/cpp-runtime` and keep `main` clean.)

## What's already done

- **`docs/cpp-runtime-plan.md`** — full implementation plan + decisions. Read it first.
- **P0 DONE** — `runtime/cpp/` (CMake, `include/alveare/{bf16,npy,config}.h`, `src/{main,bf16,npy,config}.cpp`)
  runs one hard-coded quantized GEMV (N=256,K=256) on the NPU via **native XRT C++**, no Python.
  ABI: kernel entry `"MLIR_AIE"`, `opcode=3`, args `(opcode, instr_bo, ninstr, boW, boX, boY)`; instr BO
  `XCL_BO_FLAGS_CACHEABLE` on `group_id(1)`, data BOs `XRT_BO_FLAGS_HOST_ONLY` on `group_id(3/4/5)`
  (see `meta.json`). Result `max_diff=0.0625` (bf16 tolerance).
  **Caveat to fix:** `dump_p0.py` writes `expected.npy = ref_gemv_q(...)` — the **CPU dequant reference**,
  not the Python/pyxrt NPU output. So P0 proves *NPU-from-C++ is numerically correct*, NOT bitwise
  parity vs Python. Add a proper NPU-vs-NPU check (read back Python `pyxrt` output, expect `0.0`).
- **AOT API discovered:** `gemv_q_npu.specialize(N=,K=,m=,k_tile=).compile(xclbin_path=, inst_path=)`
  (from `dump_p0.py`). Same for `gemm_q_npu` with `B=`.
- **P1 in progress** — `tools/build_kernels.py` written: reads distinct matmul shapes from the on-disk
  packed weights (not hardcoded), compiles each gemv + gemm via `.specialize().compile()`, emits
  `kernels/build/manifest.json` (shape → xclbin/insts, `n_cores`, ABI). **Run it to get the real shapes:**
  ```bash
  python tools/build_kernels.py --weights-dir <path/to/gemma4_weights> --out kernels/build
  ```
  It prints the distinct shapes and warns if >8 (Gemma-4 is expected ~10–11 → bucketing needed).

## Key facts about the existing Python runtime (so you don't re-derive)

- **The NPU surface is only two kernels.** In `runtime/py/model.py`, `run_layer`/`run_layer_batch` offload
  ONLY the matmuls: `gemv_q` (decode, per-token) and `gemm_q` (prefill, batch B=16). Everything else runs
  on CPU/NumPy: embedding + `sqrt(hidden)` scale, RMSNorm, QK/V-norm, RoPE, attention+softmax (sliding-window
  aware), GeGLU/SiLU, both residuals, per-layer output scale, final norm, `30*tanh(logits/30)` soft-cap,
  sampling. The rmsnorm/rope/attention AIE kernels exist but are NOT used in the decode path. So the C++
  port must reimplement a fair amount of **CPU math** and only **two** NPU entry points.
- **Quant format:** Q4_0, block=32 → **20 bytes** (16 packed nibbles + 2-byte bf16 scale + 2 pad). Packed
  weight `(N,K)` stored on disk as `(N, K/32*20)` uint8. Ref: `tools/convert/gemv_q_convert.py`.
- **Gemma-4-12B config:** hidden 3840, intermediate 15360, head_dim 256, 16 heads, kv_heads 8 (sliding) /
  1 (global), 48 layers, vocab 262144. Layer is **sliding** unless `(l+1)%6==0` (then global). Sliding
  window 1024. RoPE base 10000 (sliding, dim 256) / 1e6 (global, partial-rotary 0.25 → dim 512). Weights
  are streamed per-layer resident on NPU (~5.5 GB). Greedy output matches `llama.cpp` token-for-token.
- **Resident weights:** Python uses `iron.tensor(..., device="npu")` uploaded once. In C++: one `xrt::bo`
  per weight, memcpy from `.npy`, sync-to-device once, keep alive.
- **8 hardware contexts max on XDNA2.** Each distinct xclbin = one context. Python catches XRT `err=-22`
  ("out of contexts") and evicts. See decision below.

## Decisions locked in the plan (perf-first)

1. **Tokenizer:** embed `tokenizers-cpp` + hand-port the chat template (runs once/request → zero decode
   perf impact; single self-contained binary, no Python at runtime).
2. **Context policy (the perf-critical one):** keep the whole decode working set **resident**; if distinct
   decode shapes exceed 8, **bucket-pad** a few into shared xclbins so total ≤8 and all stay loaded —
   the decode loop must issue **zero** xclbin reloads. `-22` eviction kept only for rare shapes (prefill
   GEMM, lm_head). Reject the single-16384² padded xclbin (≈4× wasted MACs). Highest-leverage follow-up
   (separate branch — `feature/npu-ffn-fusion` exists): **fuse Q/K/V into one matmul and gate/up into one**
   (7 launches/layer → ~4): fewer launches AND fewer contexts. Out of scope for the pure port.
3. **Bring-up order:** Llama-3.2-1B first (16 layers, no QK-norm / sliding / soft-cap → fastest parity
   loop), then straight to Gemma-4-12B (the real perf target).
4. **Deps:** vendor `httplib.h` + `nlohmann/json.hpp` single headers in `runtime/cpp/third_party/` (project
   supports air-gapped builds — no `FetchContent` network).
5. **CPU math:** clean scalar port first for token-for-token parity, then SIMD/OpenMP guided by profiling
   (first hotspot is host attention over the growing KV cache → thread across heads). Match `ml_dtypes`
   bf16 round-to-nearest-even and convert to bf16 only at the exact `.astype(bfloat16)` boundaries.

## Server (to port later)

FastAPI today; port to **cpp-httplib** + one global `std::mutex` (generation is already fully serialized).
Endpoints `/v1/models` and `/v1/chat/completions` (non-streaming JSON + streaming SSE, byte-compatible).
Flow: reset KV caches → prefill (`forward_batch`, B=16 for Gemma-4) → decode loop (`forward` + `sample`).

## Next steps

1. Fix git + commit (above).
2. Run `tools/build_kernels.py` → get real shape count → confirm bucketing need.
3. Write the C++ **NPU registry** `runtime/cpp/include/alveare/npu.h` + `src/npu.cpp`: load
   `manifest.json`, register xclbins as `xrt::hw_context`, cache `xrt::kernel`/`xrt::run` + instr BOs,
   implement the ≤8-context resident policy (+ `-22` eviction for the rare pool).
4. Add the true NPU-vs-Python parity check to `dump_p0.py` / a `parity_test`.
5. Then P2: full Gemma-4 (or Llama-1B first) `run_layer` decode parity against golden per-layer dumps.

## Verification (project is correctness-first)

Golden per-layer bf16 dumps from the Python runtime → `parity_test.cpp` asserts near-bitwise equality →
end-to-end greedy (T=0) must match Python **and** `llama.cpp` token-for-token before any perf claim →
record decode s/token + TTFT vs the Python baseline. Note: you (Claude Code) are on the real machine, so
the `mlir-aie`/IRON toolchain and the NPU device are actually available — you can build and run for real.

Start by reading `docs/cpp-runtime-plan.md`, then fix git and run the harvester.
