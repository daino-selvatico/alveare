# Kernel plan

The AIE kernels Alveare needs, roughly in build order. Each will get its own design note + host ABI as it's started. "Correct first, fast later" applies to every one.

## Priority order (why)

LLM decode time is dominated by **quantized matrix-vector products** (weights are quantized, activations are bf16, batch=1). So:

1. **`dequant`** — unpack int4/int8 weights to the working type on-chip. Often fused into the matmul rather than standalone, but built/validated alone first.
2. **`gemv_q`** — quantized weight × bf16 vector. *The* primitive. M1.
3. **`rmsnorm`** — cheap, needed for a full layer.
4. **`rope`** — rotary position embedding on Q/K.
5. **`attn`** — QKᵀ → softmax → ·V over the KV cache, with GQA. Includes `softmax`.
6. **`gemm_q`** — quantized matrix-matrix, for prefill (many tokens). Can come after decode works.
7. **`lm_head`** — large GEMV over vocab. Same family as `gemv_q` but huge output; may need its own tiling.

## Per-kernel contract (template)

Every kernel documents:

- **Inputs/outputs**: buffer names, shapes, dtypes, on-device layout.
- **Tiling parameters**: tile M/K/N, number of AIE cores used.
- **Quant format**: block size, scale/zero-point layout (for the quantized ones).
- **Reference**: the CPU/numpy implementation it must match, and the tolerance.
- **Bench**: shape(s) measured, NPU vs CPU.

## Quantization format

We need a weight layout that is efficient for AIE on-chip access (FLM's analog is "Q4NX"). Initial plan:

- Start with a **simple, well-understood format** (e.g. Q4_0-style: 4-bit weights, per-block fp16/bf16 scale, block size 32) to minimize converter and kernel complexity.
- Pre-arrange weights into the **block/column-major tiling** the GEMV kernel wants (mirroring the idea in FLM's converter `reshape_matrix_to_block_matrix_for_mvm`, row_block_size 32).
- Only later explore fancier formats for speed.

Converter lives in `tools/` and reads GGUF (we already have Gemma/Qwen GGUFs locally) so we reuse existing quantized weights rather than re-quantizing from scratch.

## Model-specific kernel features (deferred)

These matter for the eventual Gemma-4 12B target but are **not** in the first small model:

- Sliding-window vs full-attention **alternation** per layer.
- **QK-norm** (RMSNorm on Q and K before attention).
- **Logit softcapping** (tanh-based cap on attention logits and final logits).
- GQA with the specific head grouping.

Each becomes a kernel variant or a flag once the baseline path runs.

## Open questions

- Hand-written IRON kernels vs letting `iree-amd-aie` compile them — tracked in [`docs/decisions/0002-kernel-strategy.md`](decisions/0002-kernel-strategy.md). M1 may be done both ways to compare effort/quality.
- Best on-chip residency strategy for streamed weights — empirical, addressed in M4.
