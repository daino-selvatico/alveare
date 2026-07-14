# M7 — Gemma-4-12B End-to-End on NPU

**Status**: **Completed (greedy output fidelity gap resolved in M8).**

The headline achievement is real: the full 48-layer Gemma-4-12B executes on the NPU with weight streaming and returns the correct answer. But greedy generation does **not** cleanly match `llama.cpp` (see "Side-by-Side" below) — this is an open fidelity gap, not a validated match. Verified independently by the orchestrator (regen + re-run), 2026-07-11.

## Goal

Run the full **Gemma-4-12B** model end-to-end on the AMD Ryzen AI XDNA2 NPU under Linux, generating coherent text matching `llama.cpp` side-by-side, while dynamically streaming weights to keep host memory consumption well within limits.

## What Works

- **Full Forward Pass**: Runs the entire 48-layer Gemma-4-12B network end-to-end. Handles the alternating local sliding causal window (`head_dim = 256`, `num_kv_heads = 8`, RoPE $\theta = 10,000$, window size `1024`) and global full causal attention (`head_dim = 512`, `num_kv_heads = 1`, RoPE $\theta = 1,000,000$) layer types.
- **Attention K==V Sharing**: Seamlessly supports the lack of `attn_v` weights in the global/full-attention layers by setting $V = K$ dynamically before normalization.
- **Weight Streaming**: Lazily loads quantized weight blocks per layer on-demand during execution and unloads them immediately afterward to preserve host RAM.
- **Tokenizer EOT**: Integrates the Gemma-4 specific EOT token `<turn|>` (ID 106) to correctly detect completion.

## Side-by-Side Greedy Continuation

- **Prompt**: `"The capital of France is"`
- **Greedy Outputs**:
  - **NPU (Alveare)**: `Paris`
  - **llama.cpp (Reference)**: `The user` (reasoning block: `The user is asking for the capital of France.\nThe capital of France is Paris.\nProvide the answer clearly.`, which resolves to `Paris` final answer).

**Honest reading of this divergence (do not gloss over it):** llama.cpp runs the *same* Q4 GGUF, so this is **not** weight-quantization noise. Both prompts prime a `thought` channel (the Gemma-4 chat template ends in `<|channel>thought`), yet llama.cpp reasons while the NPU jumps straight to the answer — i.e. they diverge at **token 1**. The NPU reaches the correct answer here, but "same final word" is not "matches llama.cpp". Two candidate causes, not yet distinguished:

1. **Compounded numerical error** across 48 layers (host-side ops in bf16, ~12% per-layer error at M6 layer-level) shifting the token-1 argmax from "reason" to "answer".
2. **A bug in the global / full-attention layers** (`head_dim = 512`, `num_kv_heads = 1`, `V = K`), which were **never individually validated** — M6 only validated Layer 0, a *sliding* layer. Such a bug would only surface end-to-end, which is exactly this symptom. A trivial factual prompt could pass by luck.

**M8 resolves this**: validate a global/full-attention layer against the HF oracle (as M6 did for a sliding layer) and align the chat template exactly with llama.cpp, before claiming a validated match.

## Performance & Latency

- **Prefill (NPU)**: ~7.3 seconds per token (131.43s total for 18 prompt tokens).
- **Decode (NPU)**: ~8.0 seconds per token (1st token: 8.07s, subsequent tokens: 8.02s).
- **Peak Host RAM**: ~5.5 GB (Python runtime, tokenizer, 2GB embedding table, and exactly one active layer's weights at a time).

## Hardware Partitioning (Host vs. NPU)

- **NPU**: All heavy matrix-vector multiplications (`gemv_q` for Q/K/O projections, gate/up/down projections, and LM head) are run on the NPU using the compiled `2048 x 2048` kernel target via host-side grid-chunking and padding.
- **Host (CPU)**: Embedding lookup, attention GQA score calculations, SwiGLU activation, RMSNorm, and RoPE.

## Weight Streaming Approach

We implement a custom `LazyLayerWeights` dictionary-like wrapper that lazy-loads a layer's quantized projection `.npy` files from disk/RAM page cache only when requested during execution. At the end of `run_layer`, we call `.clear()` on the dictionary, dropping the references and allowing Python's garbage collector to reclaim the memory immediately. This ensures that only **one layer**'s quantized weights (~157MB) reside in memory at any point, preventing host RAM consumption from ballooning to 7.5GB (quantized) or 24GB (dequantized).

## Deviations & Friction

- **NPU Prefill**: Prefill was originally done on the CPU for Llama-3.2-1B since its weights were pre-dequantized at startup. For Gemma-4-12B, pre-dequantizing all weights requires 24GB+ RAM, so we dequantize them on-the-fly. However, doing on-the-fly CPU dequantization for every prompt token is an extreme bottleneck (taking 15+ minutes). We resolved this by routing the prefill pass through the NPU (`use_npu=True`), which processes quantized weights directly and runs in only 130s.
