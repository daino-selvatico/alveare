# 0005 — Target Model and Hybrid Execution Strategy for Milestone M5

- Status: accepted
- Date: 2026-06-26

## Context

For Milestone M5, Alveare needs to bring up a small Gemma-family model end-to-end and implement all architectural features that Llama does not have.
We need to select a suitable Gemma dense model that fits our memory constraints and enables testing Gemma-family specific features.
Furthermore, we must design an execution partition between host CPU and NPU to maximize compatibility and avoid hardware context slot exhaustion while reusing the vectorized multi-core `gemv_q` kernel (`kernels/gemv_q/gemv_q.cc`) as-is.

## Decision

1. **Model Selection**: We select **Gemma-3-1B-it** (unquantized GGUF downloaded from `bartowski/google_gemma-3-1b-it-GGUF` file `google_gemma-3-1b-it-bf16.gguf` to `/home/daino/llama-mtp/models/`). It is the smallest dense Gemma model, sharing all key features (QK-norm, sliding window attention, GeGLU activation, tied embeddings, and layer-dependent RoPE theta) with eventual larger targets.
2. **Hybrid Execution Strategy**:
   - **NPU**: Heavy matrix-vector projections (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`, and the tied `lm_head`) are executed on the NPU using the compiled `(2048, 2048)` JIT shape. Zero-padding and chunking are used on the host to map these projections to the `(2048, 2048)` kernel layout.
   - **Host CPU**: Lightweight, activation-heavy operators are executed on the host CPU. This includes:
     - Gemma RMSNorm ($x \times \text{rsqrt}(\text{var} + \epsilon) \times w_{\text{gguf}}$ where the GGUF weights already incorporate the HF $+1.0$ offset).
     - QK-Norm (RMSNorm applied individually to $Q$ and $K$ sub-heads).
     - Gemma RoPE with selective/layer-dependent base frequency ($\theta = 10,000$ for sliding window layers, $\theta = 1,000,000$ for full attention layers).
     - Sliding-Window Attention (KV caches sliced with window size $W=512$ on sliding window layers).
     - GeGLU activation ($\text{gelu\_pytorch\_tanh}$ applied to gate and multiplied by up).
     - Scaled embedding lookup ($x = \text{lookup} \times \sqrt{d_{\text{model}}}$).
3. **Robust test coverage**: We implement and verify:
   - A dedicated layer test (`tests/test_gemma_layer.py`) verifying each sub-op against a Hugging Face reference.
   - An end-to-end generation test (`tests/test_gemma_generation.py`) running greedy token generation on the NPU and verifying side-by-side coherence and completion correctness against a CPU `llama.cpp` server.

## Consequences

- The model runs end-to-end on the NPU with zero NPU JIT modifications.
- Exact greedy generation is verified against the `llama.cpp` server (matching `Paris.\n\nDo you` on the prompt continuation).
- Per-op logic has under 0.38% relative error, and the full layer has a 18.9% relative error (which is mathematically the expected compounding quantization loss of Q4_0 across 6 sequential projections).
