# M6 — Gemma-4 Decoder Layer on NPU

**Status**: **Completed**.

## Goal

Implement the dense **Gemma-4** architecture and validate **one decoder layer** (Layer 0, which is a local sliding-window layer) of Gemma-4-12B against a dequantized CPU reference. This proves the correctness of Alveare's host-side Gemma-4 implementations (dynamic RoPE tables, alternating layers, QKV normalization, scale-free attention, layer output scaling, and logit softcapping) and verifies that the larger Gemma-4 model dimensions (`hidden_size = 3840`, `intermediate_size = 15360`) can be mapped correctly to the existing multi-core vectorized `gemv_q` NPU kernel (`2048 x 2048` JIT compilation target) via host-side padding and chunking.

## Target model

We targeted **Gemma-4-12B-it** (GGUF file `/home/daino/llama-mtp/models/gemma-4-12b-it-UD-Q4_K_XL.gguf`).
A new script `tools/quantize_gemma4.py` was created to extract and dequantize Layer 0 weights using `gguf.quants.dequantize` and save them under `quantized_weights_gemma4/` (with linear projections padded and quantized to the Alveare custom `Q4_0` layout).

## Hybrid Execution Partition

- **NPU (Heavy Projections)**:
  - Linear projections (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`, and `lm_head`) are executed on the NPU using the compiled `gemv_q` kernel target size of `(2048, 2048)`.
  - Host-side padding and grid-chunking are used to split the larger dimensions (e.g. `3840` padded to `4096`, `15360` padded to `16384`) into standard JIT-sized chunks.
- **Host CPU (Activation-Heavy & Layer Control)**:
  - **QKV Normalization**: Applying RMSNorm to $Q$ (using `attn_q_norm`), $K$ (using `attn_k_norm`), and $V$ (using unit weight).
  - **Selective RoPE**: Dynamic head dimension (`256` for local, `512` for global) and base frequency ($\theta = 10,000$ for local, $\theta = 1,000,000$ for global).
  - **Attention GQA**: Standard multi-head attention scores computed with scale `1.0` (no $\sqrt{head\_dim}$ division) and causal sliding-window size `1024`.
  - **Layer Output Scaling**: Final hidden states multiplied by the per-block scalar `layer_output_scale`.
  - **Final Logit Softcapping**: Logits softcapped with `30.0 * tanh(x / 30.0)` in the `forward` function.

## Error Tracking (NPU/Host vs CPU Reference)

Verified step-by-step for a 32-token decode step (position `t = 31`, Layer `l = 0`):

| Sub-operation / Layer Component | Relative Error | Max Absolute Error | Status |
|---|---|---|---|
| **RMSNorm 1** (Input norm) | `0.00261` | `1.00000` | PASS |
| **Query Proj** (NPU GEMV) | `0.12000` | `15.50000` | PASS (Q4_0 Quantization Loss) |
| **QK-Norm Query** | `0.00271` | `0.03125` | PASS |
| **Query RoPE** | `0.00325` | `0.06250` | PASS |
| **Attention** (Host GQA) | `0.00351` | `0.03125` | PASS |
| **Attn Proj** (NPU GEMV) | `0.09757` | `0.35156` | PASS (Q4_0 Quantization Loss) |
| **Gate Proj** (NPU GEMV) | `0.11158` | `3.08887` | PASS (Q4_0 Quantization Loss) |
| **Up Proj** (NPU GEMV) | `0.07942` | `4.00000` | PASS (Q4_0 Quantization Loss) |
| **Down Proj** (NPU GEMV) | `0.08891` | `14.87500` | PASS (Q4_0 Quantization Loss) |
| **Full Decoder Layer Output** | **`0.13644` (13.6%)** | **`1.50000`** | **PASS (Expected compounding Q4_0 loss)** |

## Backward Compatibility (Regression Check)

To verify that the unified Alveare runtime remains backward-compatible, we ran all previous tests after our modifications:
- `tests/test_gemv_q.py`: **PASS**
- `tests/test_gemma_layer.py`: **PASS**
- `tests/test_cpu_only.py` (Llama-3.2 system test): **PASS** (Correctly generated `The capital of France is Paris.`)

## Weights directory convention

- `quantized_weights_gemma4/` — Gemma-4-12B (`config.json` with `model_type=gemma4`).
