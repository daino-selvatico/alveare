# M1 — Quantized matrix-vector kernel

**Status**: **PASSED (Done)**

## Goal

Implement the dominant LLM-decode primitive on the NPU: a **quantized weight × bf16 activation matrix-vector multiply** (GEMV), with on-chip (or fused) dequantization.

## Definition of Done (Achieved)

- [x] A kernel taking int4 (block-quantized, Q4_0-style) weights + a bf16 input vector produces an output vector matching a NumPy reference within tolerance.
- [x] Correctness verified for random matrices (`256x256` and `2560x2560`) and real weights loaded from a local GGUF (`640x3840`).
- [x] Latency microbenchmarked for CPU vs NPU and results written to `tests/bench/gemv_q_bench.txt`.
- [x] Host ABI documented in `kernels/gemv_q/README.md`.

## Implementation Summary

### 1. Unified Weight/Scale Layout (`W_combined`)
- **Friction**: The AIE core (CoreTile) only possesses **2 input DMA channels** and **2 output DMA channels**. An initial design with separate weight and scale buffers failed compilation as it required 3 inputs.
- **Resolution**: We combined the Q4_0 packed weights and `bfloat16` scales into a single contiguous buffer (`W_combined`) with 20-byte alignment per 32-element block:
  - 16 bytes: Packed weights (2 weights/byte)
  - 2 bytes: `bfloat16` scale
  - 2 bytes: padding (for 4-byte/16-byte alignment)
- This resolved the DMA limits and allowed streaming weights + scales through a single channel.

### 2. Tiling and Host-Side Chunking
- **Friction**: The AIE DMA buffer descriptor loop count has a strict hardware limitation of **64**. Thus, a single JIT invocation cannot process more than 64 row blocks (`N // m <= 64`). For `m=32`, this limits `N <= 2048`.
- **Resolution**: Implemented host-side chunking in Python. If `N // m > 64` (e.g. for `2560`), we chunk the matrix along the N dimension (into chunks of 2048 or fewer rows), invoke the NPU JIT sequence for each chunk, and stitch the results.

### 3. Numerical Verification
We verify correctness using:
- **Relative Tolerance (`rtol`)**: `0.05`
- **Absolute Tolerance (`atol`)**: `1.0`

Due to `bfloat16` low precision (7-bit mantissa) and tile-wise accumulation boundaries (where the accumulator is stored back to local L1 as `bfloat16` every 256 columns), slight rounding drift of up to 1.0 accumulates over large dimensions (e.g. `2560` columns) relative to the FP32 CPU reference. The NPU results match signs and magnitudes perfectly.

## Verification & Benchmark Results

### Correctness Tests
All tests passed successfully:
1. `test_dequantize_correctness`: PASS (Combined layout unpack matches Q4_0 reference).
2. `test_tiny_random` (256x256): PASS (Relative difference <= 5%, absolute difference <= 1.0).
3. `test_large_random` (2560x2560): PASS (Chunked execution matches reference).
4. `test_real_weights` (640x3840 from `mm.a.input_projection.weight` in local Gemma GGUF): PASS.

### Latency Numbers (AMD Ryzen AI 9 HX NPU vs CPU)
- **Shape 256x256**:
  - NumPy CPU: `0.59 ms`
  - NPU: `101.79 ms` (Raw hardware NPU time: `9.1 ms`, JIT/PyXRT overhead: `~92 ms`)
- **Shape 2560x2560** (with chunked execution):
  - NumPy CPU: `75.26 ms`
  - NPU: `1091.19 ms` (Raw hardware NPU time: `~720 ms`, JIT/PyXRT overhead: `~370 ms`)

*Performance Note*: The initial AIE kernel is written in scalar C++ without vector registers or pipeline pragmas. Sub-millisecond performance is targeted for M4 optimizations, while M1 successfully established the correctness baseline.
