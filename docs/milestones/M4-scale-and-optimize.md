# M4 — Scale up + optimize

**Blocked by:** M3. Open-ended; tracked as issues, not a single gate.

## Goal

Two intertwined fronts:

### A. Bigger / more models
- Add larger dense models (Qwen3-class, then the real target: **Gemma-4 dense 12B**).
- Implement the deferred Gemma-4 features: sliding/full attention **alternation** per layer, **QK-norm**, **logit softcapping**, the specific GQA grouping, tied embeddings.
- Prefill path: matrix-matrix (`gemm_q`) kernels for processing prompts at throughput, not one token at a time.

### B. Performance (the long tail — where FLM's patented edge lives)
- **DMA/compute overlap**: prefetch layer N+1's weights while computing layer N. This is the single biggest decode-time win, since LLM decode is weight-bandwidth-bound.
- **Tiling**: spread matmuls across the AIE array efficiently; tune tile M/K/N.
- **Fusion**: dequant+matmul, norm+matmul, etc.
- **Quant format**: move from the simple Q4_0-style layout to something on-chip-friendlier if it pays off.
- **KV cache & prompt cache**: efficient layouts, reuse across turns.

## Reality check

This milestone is effectively unbounded. Matching FLM is the hard, patented part and may never fully happen — and that's acceptable per the project's performance philosophy (open + slower still unlocks models nobody can run otherwise). Progress is measured by:

- models that run at all that couldn't before, and
- tokens/sec creeping toward practical (interactive) latency.

## Success definition for the 12B end goal

Gemma-4 dense 12B generating tokens on the NPU at *any* speed = the project's headline objective met (first open NPU 12B). Interactive speed is a separate, later bar.

---

## M4.1 gemv_q vectorization

**Status**: **PASSED (Done)**

### Goal
Vectorize the block-quantized `gemv_q` AIE kernel using AIE vector APIs to run on the vector units of a single AIE core, achieving a significant speedup.

### Approach
- **Vector APIs**: Loaded packed weights using `aie::load_unaligned_v<16>` to bypass 16-byte address alignment truncation.
- **Dequantization**: Unpacked packed int4 weights (int8_t) to int16_t first, then applied `<< 12 >> 12` and `<< 8 >> 12` shifts on `aie::vector<int16_t, 16>` to sign-extend `q0` and `q1` and avoid LLVM-AIE backend selector crashes.
- **Vectorized Type Conversion**: Replaced the initial slow scalar loop casting elements one-by-one from `int16_t` to `bfloat16` with the `aie::to_float<bfloat16>` vector intrinsic. This eliminated loop overhead, branch stalls, and pipeline bubbles, accelerating raw NPU execution by ~9x.
- **Interleaving**: Used `aie::filter_even` and `aie::filter_odd` to split the loaded activation vector `x` into even and odd indices to match the interleaved layout of `q0`/`q1` weights.
- **Multiply-Accumulate**: Multiplied weights and activations using FP32 dot products (`aie::mul`/`aie::mac`) and reduced the vector sum to a float scalar with `aie::reduce_add`.

### Latency and Speedup Results

| Shape | Scalar (Before) | Vectorized (Initial M4.1) | Optimized Vectorized (Our final) | Host Speedup (vs Scalar) | Est. Raw NPU | Est. NPU Speedup (vs Scalar) |
|---|---|---|---|---|---|---|
| **2048x2048** | `667.21 ms` | `203.46 ms` | `107.67 ms` | **6.20x** | `12.67 ms` | **45.16x** |
| **2560x2560** | `1075.57 ms` | `357.37 ms` | `207.60 ms` | **5.18x** | `17.60 ms` | **50.32x** |
| **End-to-End** | `~176.5 s/token` | `~35.3 s/token` | `~2.34 s/token` | **75.43x** | - | - |

- Correctness tests (`tests/test_gemv_q.py`) are fully green.
- Coherent text generation is verified (greedy continuation output `The capital of France is` is unchanged).
