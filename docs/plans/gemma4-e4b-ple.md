# Plan — Serve Gemma-4-E4B (Per-Layer Embeddings) on Alveare

**Goal.** Serve `gemma-4-E4B-it` alongside the existing 12B, under alias
`gemma4-e4b`, on the XDNA2 NPU. E4B is smaller (→ likely faster decode, the
real-time lever) but adds the **Per-Layer Embeddings (PLE)** component the 12B
does not use — so this is a real architecture feature, not just a size change.

**How this is run.** Broken into self-contained tasks executed in **separate
focused sessions**. This repo is the source of truth; each task lists its files,
steps, and **acceptance criteria** so the orchestrator can verify the returned
result before unblocking the next. Do the tasks in dependency order (below).

**Status (2026-07-27).** Investigation done; GGUF downloaded; plan written. No
code yet.

---

## Key facts (already established — do not re-derive)

**Source GGUF:** `/home/daino/llama-mtp/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf`
(5.13 GB, arch=`gemma4`, name "Gemma-4-E4B-It"). Repo:
`unsloth/gemma-4-E4B-it-GGUF`.

**Reference oracle:** llama.cpp build at `~/.unsloth/llama.cpp/build/bin/`
(`llama-server` present; `llama-cli` may need building). The GGUF uses llama.cpp
tensor names (`blk.N.inp_gate` etc.), so llama.cpp supports E4B — it is the
validation oracle for correctness (Alveare must match its greedy tokens / logits).

**E4B config vs 12B (the working baseline):**

| field | 12B | **E4B** |
|---|---|---|
| `hidden_size` (embedding_length) | 3840 | **2560** |
| `intermediate_size` (feed_forward_length) | 15360 | **10240** |
| `num_hidden_layers` (block_count) | 48 | **42** |
| `num_attention_heads` (head_count) | 16 | **8** |
| `num_key_value_heads` (head_count_kv) | 1* | **2** |
| head_dim swa / global (key_length_swa / key_length) | 256 / 512 | **256 / 512** (same) |
| sliding_window | 1024? | **512** |
| **`embedding_length_per_layer_input`** | **0 (PLE off)** | **256 (PLE on)** |
| rope freq_base / swa | 1e6 / 1e4 | 1e6 / 1e4 (same) |

\* The 12B GGUF reports `head_count_kv=1` (global MQA); the runtime hard-codes
sliding kv-heads = 8. E4B reports `head_count_kv=2` — **the sliding vs global
kv-head split for E4B must be confirmed in Task 0** (the GGUF field alone is
ambiguous, as it is for the 12B).

**Extra E4B tensors the 12B does NOT have (the PLE component):**

| tensor | shape | role |
|---|---|---|
| `per_layer_token_embd.weight` | [10752, 262144] | per-layer token embedding table (vocab × 42·256); ~2.8B params — the "effective-4B" trick |
| `per_layer_model_proj.weight` | [2560, 10752] | projects the main embedding → 42·256 per-layer inputs |
| `per_layer_proj_norm.weight` | [256] | norm on the per-layer projection |
| `blk.N.inp_gate.weight` | [2560, 256] | per-layer: gate the per-layer embedding into the layer |
| `blk.N.proj.weight` | [256, 2560] | per-layer: project the per-layer embedding |
| `blk.N.post_norm.weight` | [2560] | per-layer: extra norm (12B has attn_norm/post_attention_norm/ffn_norm/post_ffw_norm; E4B adds this) |

**Runtime today (`runtime/cpp/src/model.cpp`):** the `gemma4` path **hard-codes
the 12B geometry** at **21 sites** (`grep -c 'model_type == "gemma4"'`): e.g.
`N_q = is_sliding ? 4096 : 8192`, `n_q_heads = 16`, `n_kv_heads = is_sliding ? 8 : 1`,
`h_dim = is_sliding ? 256 : 512`, `K_padded = 4096`, `I = 16384`, `I_real = 15360`,
`is_sliding = (layer+1) % 6 != 0`. The config loader
(`runtime/cpp/src/config.cpp`) **already reads** hidden/intermediate/heads/kv/
head_dim/layers from `config.json` for gemma4 — the values exist, the forward just
ignores them.

**Quantizer today (`tools/quantize_gemma4.py`):** exports `token_embd` + the
standard per-block tensors; **does not** handle any PLE tensor (it would silently
drop them). `config.json` it writes already includes heads/kv/head_dim/hidden/
intermediate/layers read from the GGUF.

---

## Architecture notes — PLE (CONFIRM exact math in Task 0)

Best current understanding (Gemma-3n-style PLE; **to be verified** against the
llama.cpp `build_gemma3n`/gemma4 graph and/or HF `transformers` modeling code):

1. **Main path** unchanged: `token_embd` lookup → scale by √hidden → the usual
   decoder stack (attn + FFN), with E4B sizes.
2. **Per-layer inputs**, computed once per token:
   - Look up `per_layer_token_embd[token]` → a [42, 256] tensor (one 256-vec per
     layer), likely scaled (√256?).
   - Project the main embedding through `per_layer_model_proj` (2560→10752),
     reshape to [42, 256], normalize with `per_layer_proj_norm` (RMSNorm).
   - Combine the two (add? average? — CONFIRM) → `per_layer_input[42, 256]`.
3. **Per decoder layer L:** inject `per_layer_input[L]` into the layer's output:
   roughly `proj`(256→2560) of the per-layer input, gated by `inp_gate`
   (2560→256 activation?), normalized by `post_norm`, added to the hidden state at
   a specific point (after FFN? — CONFIRM order and the exact gate/activation).

**The exact equations (scales, norm placement, gate activation, injection point,
residual structure) are the single biggest correctness risk and MUST come from a
reference in Task 0 — do not guess them into the runtime.**

---

## Tasks

Dependency order: **0 → 1 → 2 → 3 → 4**. Task 2.1 (parameterization) may run in
parallel with 0/1. Every downstream correctness check validates against the Task-0
oracle dumps.

### M0 — Reference & spec (de-risk) — BLOCKS everything

- **0.1 — Oracle up.** Get llama.cpp running the E4B GGUF and emit reference
  outputs. Build `llama-cli` in `~/.unsloth/llama.cpp` if missing. Run a few fixed
  greedy prompts (temp 0), capture: generated token ids, and per-token top-k
  logits if obtainable (`--logits`/server `logprobs`). Save under
  `docs/plans/e4b/ref/` (prompt → tokens/logits).
  **Accept:** deterministic reference token sequences for ≥3 prompts committed.

- **0.2 — PLE spec.** From llama.cpp `src/llama-model.cpp` (the gemma4/gemma3n
  build-graph) and/or HF `transformers` Gemma3n code, write the EXACT PLE math:
  every tensor's role, all scales, norm types/placement, the gate activation, the
  per-layer injection point, and the sliding/global kv-head split + sliding
  pattern for E4B. Output: `docs/plans/e4b/PLE-spec.md` with equations.
  **Accept:** a spec precise enough to implement without further guessing; the
  sliding pattern + per-geometry kv-head counts for E4B are pinned.

### M1 — Quantizer

- **1.1 — Export PLE tensors.** Extend `tools/quantize_gemma4.py` (or a
  `quantize_gemma4e.py` that shares code) to also emit `per_layer_token_embd`,
  `per_layer_model_proj`, `per_layer_proj_norm`, and per-block `inp_gate`, `proj`,
  `post_norm`, in Alveare's layout. Decide quant per tensor (projections → Q4_0
  like the rest; `per_layer_token_embd` is huge — evaluate Q4_0 vs fp16 for size
  vs quality; document the choice). Write `config.json` with `per_layer_input`,
  the E4B sizes, and the confirmed sliding pattern.
  **Files:** `tools/quantize_gemma4*.py`. **Output:** `quantized_weights_gemma4-e4b/`.
  **Accept:** all 720 source tensors accounted for (mapped or intentionally
  skipped); shapes match; a **Python-side numeric check** of the isolated PLE
  computation vs the Task-0 reference passes (before any NPU/runtime work).

### M2 — Runtime forward (CPU-first, correctness before speed)

- **2.1 — Parameterize the gemma4 path.** Replace the 21 hard-coded 12B literals
  in `model.cpp` (`run_layer`, `run_layer_batch`, `run_attention_host`, rope, kv
  init) with values derived from `config_` (heads, kv-heads, head_dim swa/global,
  hidden, intermediate, layers, sliding pattern, paddings). **Must not regress the
  12B.** **Accept:** 12B `ALVEARE_SELFTEST` still bit-exact (`rerun matches: YES`,
  identical tokens to pre-change); no literal `4096/8192/2048/16384/15360/16/6`
  gemma4 geometry left in the forward.

- **2.2 — Implement PLE forward.** Per `PLE-spec.md`: load the PLE tensors,
  compute per-layer inputs once per token, inject per layer. Keep it on CPU/host
  first where feasible (the PLE projections are small per token). **Accept:**
  Alveare's **logits/greedy tokens match the Task-0 llama.cpp reference** on the
  fixed prompts (allow small bf16 tolerance; greedy tokens should match). This is
  the correctness gate — do not proceed to kernels until it passes.

### M3 — NPU kernels for E4B shapes

- **3.1 — Build E4B kernels.** Enumerate E4B decode/prefill GEMV/GEMM/FFN shapes
  (hidden 2560→padded, intermediate 10240, N from 8 heads×head_dim, etc.); build
  the `gemv_q` / `ffn_fused` / `gemm_q` xclbins and regenerate the manifest
  (`tools/build_kernels.py`, `tools/build_gemm_mmul.sh`). Can take hours.
  **Accept:** manifest resolves every E4B shape; per-kernel CHECK max_diff sane.

- **3.2 — PLE projections placement.** Decide NPU vs CPU for `per_layer_model_proj`
  (2560×10752), `proj` (256×2560), `inp_gate` (2560×256). Small per-token ones can
  stay on CPU; if on NPU, add their shapes to M3.1. **Accept:** full forward runs
  on the NPU path with M2.2 correctness preserved.

### M4 — Integrate, validate, serve

- **4.1 — End-to-end.** `./alveare serve gemma4-e4b` (alias resolves), run
  `ALVEARE_SELFTEST`; confirm coherent output, **greedy-token match vs llama.cpp**,
  and rerun reproducibility (`rerun matches: YES`). Speculative decode should still
  work (gemma4 gate). **Accept:** all three hold.

- **4.2 — Benchmark + docs.** Measure decode tok/s vs the 12B (`ALVEARE_PROFILE_DECODE`),
  update `CHANGELOG.md` / `docs/kernel-roofline.md` / README model list; keep the
  alias. **Accept:** documented tok/s; if the E4B decode ceiling differs from the
  12B's ~3 tok/s, record why.

---

## Orchestration

- **Critical path:** 0.1 → 0.2 → (1.1 ∥ 2.1) → 2.2 → 3.1 → 3.2 → 4.1 → 4.2.
- **Gate discipline:** the Task-0 oracle dumps are the single source of truth for
  correctness; M1 checks against them numerically, M2.2 checks logits, M4 checks
  end-to-end tokens. Nothing ships until M2.2 matches the oracle.
- **Do-not-regress:** M2.1 must keep the 12B bit-exact — run the 12B selftest as a
  guard after any `model.cpp` change.
- **Each session brief should include:** this doc's Key-facts + the target task's
  Accept criteria, and must return the artifact + evidence it meets them.

## Risks

1. **PLE math wrong → subtly bad output.** Mitigation: Task 0 spec from a real
   reference + M2.2 logit match; never guess equations into code.
2. **12B regression from parameterization.** Mitigation: M2.1 guard selftest.
3. **`per_layer_token_embd` size** (~2.8B params). Mitigation: quant choice in
   M1.1, measured.
4. **Kernel build time** (M3, hours). Mitigation: schedule as its own session.
5. **Sliding pattern / kv-head split for E4B unknown from the GGUF field alone.**
   Mitigation: pin it in Task 0.2 from the reference graph.
