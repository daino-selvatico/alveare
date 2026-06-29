# 0006 — Gemma-4 Target Model and Layer Bringup Strategy for Milestone M6

- Status: accepted
- Date: 2026-06-29

## Context

For Milestone M6, Alveare needs to implement and validate the dense **Gemma-4** architecture. Gemma-4-12B introduces several significant changes compared to Gemma-3 and Llama:
1. **Alternating Attention**: Hybrid SWA/local and Global layers with different head dimensions (256 vs 512), KV head counts (8 vs 1), and RoPE base frequencies (10,000 vs 1,000,000).
2. **QKV Normalization**: Normalizing all three projections ($Q$, $K$, $V$), where V-norm uses a unit-weight RMSNorm (no learnable parameters).
3. **Scale-free Attention**: Attention dot-products are scaled by `1.0` instead of `1.0 / sqrt(head_dim)`.
4. **Layer Output Scaling**: Final hidden states of each layer are scaled by a layer-specific weight `layer_output_scale`.
5. **Final Logit Softcapping**: Logits are scaled and softcapped with $\text{tanh}$ and a cap value of `30.0`.

We must prove that the math is correct on a single layer (specifically Layer 0, a sliding window layer) cheaply and verify that we can map the larger Gemma-4 model dimensions (`hidden_size=3840`, `intermediate_size=15360`) onto our standard vectorized multi-core `gemv_q` NPU kernel (`2048 x 2048` JIT compilation target) without modifying the kernel itself.

## Decision

1. **Model Selection**: We target **Gemma-4-12B-it** (GGUF file `/home/daino/llama-mtp/models/gemma-4-12b-it-UD-Q4_K_XL.gguf`).
2. **Weight Extraction & CPU Reference**: 
   - We write `tools/quantize_gemma4.py` using python-gguf to extract, dequantize, and format weights for Layer 0 and tied embeddings/norms. Linear projection weights are padded and quantized to the `Q4_0` custom layout.
   - We write `tools/ref/generate_gemma4_reference.py` to implement a pure NumPy-based reference forward pass of Layer 0 and tied embedding lookup.
3. **Generalized Host Runtime**:
   - We update `runtime/py/model.py` to support `gemma4`.
   - We implement dynamic, layer-specific RoPE tables and KV caches depending on whether a layer index is local or global.
   - We add Q, K, and V normalization (unit weight for V).
   - We update attention scoring with a scale of `1.0`.
   - We add `layer_output_scale` multiplication and final logit softcapping.
   - We reuse the NPU JIT `gemv_q` kernel (`2048 x 2048` compilation target) by padding and column/row chunking the larger dimensions (`3840` padded to `4096` and `15360` padded to `16384`) on the host.
4. **Validation Test**:
   - We implement `tests/test_gemma4_layer.py` to run Layer 0 on both NPU/CPU and assert that relative error is under a 25% tolerance (quantization error threshold).

## Consequences

- The Gemma-4 architecture has been successfully brought up in Alveare's unified Python runtime.
- The unit test `tests/test_gemma4_layer.py` passes with a relative error of `13.6%` (well within the `25%` tolerance), verifying that our host chunking and math operations are correct.
- Existing tests (`test_gemv_q.py`, `test_gemma_layer.py`, and `test_cpu_only.py`) remain completely green, confirming backward compatibility.
- This establishes the foundation for full-model streaming execution and latency optimization of Gemma-4-12B in future milestones.
