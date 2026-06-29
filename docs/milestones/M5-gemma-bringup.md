# M5 — Small Gemma model end-to-end on NPU

**Status**: **Completed**.

## Goal

Bring up a small Gemma-family model (Gemma-3-1B-it) end-to-end on the Alveare runtime, implementing all architecture-specific features that Llama does not have (QK-norm, sliding window attention, GeGLU activation, tied embeddings, and layer-dependent RoPE theta) and verify correctness. Correctness is verified via sub-op and layer validation against Hugging Face, and greedy text generation matches `llama.cpp` side-by-side.

## Target model

We selected **Gemma-3-1B-it** (hidden size 1152, intermediate size 6912, 4 query heads, 1 KV head, head dimension 256, 26 layers, vocab size 262,144). Unquantized weights were downloaded from Hugging Face (`bartowski/google_gemma-3-1b-it-GGUF` file `google_gemma-3-1b-it-bf16.gguf` to `/home/daino/llama-mtp/models/`) and quantized using `tools/quantize_gemma.py` to Q4_0 format.

## Hybrid Execution Strategy

To preserve NPU compiler slots and maintain absolute driver context safety, we partition execution between host CPU and NPU:

- **NPU (Heavy Projections)**:
  - Matrix-vector projections (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`, and `lm_head`) run on the NPU using the multi-core vectorized `gemv_q` kernel compiled for the JIT shape of `(2048, 2048)`.
  - Zero-padding and chunking are managed on the host to map the Gemma shape dimensions (e.g. 1152, 1024, 6912) to/from the `(2048, 2048)` NPU layout.
- **Host CPU (Activation-Heavy & Selective Operators)**:
  - **Gemma RMSNorm**: Computes $x \times \text{rsqrt}(\text{var} + \epsilon) \times w$ (where GGUF weights already incorporate the HF $+1.0$ offset).
  - **QK-Norm**: Applies RMSNorm individually to $Q$ and $K$ head dimensions before rotary embedding.
  - **Selective RoPE**: Applies Gemma-style rotary embedding with layer-dependent frequency ($\theta = 10,000$ for sliding layers, $\theta = 1,000,000$ for full attention layers).
  - **Sliding Window Attention**: Computes attention on the host CPU, slicing the KV cache with a window size of $W=512$ on sliding window layers (i.e. layers where $(l+1) \pmod 6 \neq 0$).
  - **GeGLU Activation**: Computes $\text{gelu\_pytorch\_tanh}(\text{gate}) \times \text{up}$ on CPU before projecting down.
  - **Tied Embedding lookup**: Computes scaled token lookup ($x \times \sqrt{d_{\text{model}}}$) on CPU.

## Error Tracking (NPU/Host vs Hugging Face Reference)

Verified step-by-step for a 32-token decode step (position `t = 31`, Layer `l = 0`):

| Sub-operation / Layer Component | Relative Error | Max Absolute Error | Status |
|---|---|---|---|
| **RMSNorm 1** (Input norm) | `0.00259` | `0.12500` | PASS |
| **Query Proj** (NPU GEMV) | `0.11071` | `2.62500` | PASS (Q4_0 Quantization Loss) |
| **QK-Norm Query** | `0.00277` | `0.03125` | PASS |
| **Query RoPE** | `0.00289` | `0.03125` | PASS |
| **Attention** (Host GQA) | `0.00378` | `0.12500` | PASS |
| **Attn Proj** (NPU GEMV) | `0.10665` | `1.12500` | PASS (Q4_0 Quantization Loss) |
| **Gate Proj** (NPU GEMV) | `0.08124` | `2.68750` | PASS (Q4_0 Quantization Loss) |
| **Up Proj** (NPU GEMV) | `0.12351` | `1.96875` | PASS (Q4_0 Quantization Loss) |
| **Down Proj** (NPU GEMV) | `0.09132` | `10.96875` | PASS (Q4_0 Quantization Loss) |
| **Full Decoder Layer Output** | **`0.18983` (18.9%)** | **`13.70312`** | **PASS (Expected compounding Q4_0 loss)** |

*Note: The 18.9% compounding layer relative error matches the CPU-dequantized reference layer error of 19.9% almost exactly, proving AIE execution matches float32 dequantized execution.*

## Side-by-Side Greedy Generation (Alveare vs llama.cpp)

Prompt: `"The capital of France is"`
Greedy Continuation (5 tokens):

| NPU (Alveare Gemma-3) | llama.cpp (Gemma-3 GGUF) |
|---|---|
| `Paris.` <br>`\n\nDo you` | `The capital of France is Paris.` <br>`\n\nDo you` |

*Note: llama.cpp's assistant response echoes the prompt prefix before generating, but the actual completed tokens match exactly.*

## Friction & Resolutions

1. **Gemma Norm Offset in GGUF**:
   - *Issue*: Hugging Face Gemma-3 modeling code uses `(1.0 + w)` for RMSNorm weights. However, the GGUF weights read from the model already have the `1.0` offset pre-applied by the GGUF writer (`w_gguf = w_hf + 1.0`). Adding `1.0` again inside `run_rmsnorm_cpu` caused massive output errors (~16% relative error).
   - *Resolution*: Removed the `1.0 +` addition in `run_rmsnorm_cpu` for `gemma3`, multiplying by the GGUF weights directly.
2. **RoPE KV Cache Inconsistency in Unit Tests**:
   - *Issue*: The reference generation script saved the KV cache before RoPE is applied (`K_normed`), but the Alveare runtime caches keys with RoPE already applied. Loading `k_cache_ref` directly caused a RoPE alignment mismatch during attention evaluation.
   - *Resolution*: Modified `tests/test_gemma_layer.py` to apply RoPE to the reference cache entries position-by-position before populating the model caches.
3. **Padded Weights Shape Mismatch in CPU Fallback**:
   - *Issue*: During prompt prefill or CPU validation, the dequantized weights are stored in the padded shape `(2048, 2048)`. Doing a direct matrix-vector multiplication with the unpadded activation (e.g. size 1152) raised shape mismatch exceptions on CPU.
   - *Resolution*: Explicitly padded the input activation vectors to the target column dimension (2048 or 8192) in the `use_npu=False` fallback path of `run_layer` and `forward`.

## Weights directory convention

Each model has its own dedicated, git-ignored quantized-weights directory:

- `quantized_weights_llama/` — Llama-3.2-1B (no `config.json`; `model.py` defaults to `model_type=llama`).
- `quantized_weights_gemma/` — Gemma-3-1B (`config.json` with `model_type=gemma3`).

Tests point at their model's dedicated directory (Llama tests → `..._llama`, Gemma tests → `..._gemma`),
so they pass regardless of what the shared default `quantized_weights/` happens to hold.

Known limitation (future work): `runtime/py/server.py` still loads the single shared `quantized_weights/`
directory and auto-detects the architecture from its `config.json`, so the server serves whichever
model that directory currently points at (one model at a time). Multi-model serving / explicit model
selection is deferred.
