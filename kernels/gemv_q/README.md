# Quantized GEMV Kernel (Milestone M1)

This directory contains the hand-written IRON AIE design for the quantized matrix-vector multiply (GEMV) kernel on the AMD Ryzen AI (XDNA2) NPU.

## Host ABI Contract

| Buffer | Direction | Shape | dtype | Description |
|---|---|---|---|---|
| `W_combined` | Input | `(N, (K // 32) * 20)` | `uint8` | Block-quantized weight matrix combined with scales and padding. |
| `X` | Input | `(K,)` | `bfloat16` | Input activation vector. |
| `Y` | Output | `(N,)` | `bfloat16` | Output vector. |

### Combined Weight Layout (`W_combined`)

To optimize data movement and comply with the AIE core's DMA limits (max 2 input DMA channels), the 4-bit weights and scales are packed together along the K dimension into a single combined buffer. 

For each row and block of 32 weights along the K dimension:
- **Bytes 0–15**: 16 bytes of packed 4-bit signed quantized weights (range `[-8, 7]`).
  - `byte[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4)` where `q0` is weight `2*i` and `q1` is weight `2*i+1`.
- **Bytes 16–17**: 2 bytes (`bfloat16`) representing the block scale.
- **Bytes 18–19**: 2 bytes of padding (set to `0`) for 4-byte/16-byte alignment.

This results in **20 bytes per block** of 32 elements.

### Tiling Parameters
- `m = 32` (row tile size)
- `k_tile = 256` (column tile size)

### Tiling and Dimensions Limits
- `N` must be a multiple of `m` (`32`).
- `K` must be a multiple of `k_tile` (`256`).
- **Hardware limit**: Due to the NPU's multi-dimensional DMA BD length limit (maximum size 64), the number of row blocks processed in a single JIT invocation must satisfy `N // m <= 64` (i.e. `N <= 2048`). For larger shapes (like `2560`), the host runtime must split the execution along the N dimension into multiple chunked NPU calls.

### Numerical Tolerance

- **Relative tolerance (`rtol`)**: `0.05`
- **Absolute tolerance (`atol`)**: `1.0`

*Rationale*: The AIE core accumulates the dot products in `float32` locally but stores the intermediate accumulator back to local L1 memory as `bfloat16` at the boundary of each `k_tile` (256 elements). This tile-wise casting/truncation to `bfloat16` (which has a 7-bit mantissa) accumulates rounding differences of up to 1.0 against the FP32 NumPy reference over large dimensions (e.g. `K=2560`).

### Performance and Vectorization Notes

The kernel was optimized using AIE vector APIs:
1. **Load/Unpack**: Packed `int4` weights are loaded from unaligned addresses using `aie::load_unaligned_v<16>`, preventing alignment truncation bugs. They are unpacked to `int16_t` using `.unpack()`.
2. **Dequantization**: Extracted using bitwise shifts `<<` and `>>` directly on `int16_t` vectors to avoid LLVM-AIE backend compiler crashes on un-unpacking shifted `int8_t` vectors. Unpacked integers are converted to `bfloat16` vectors and multiplied by the broadcasted scale.
3. **Deinterleaving**: Activations are loaded as 32-element vectors and deinterleaved using `aie::filter_even` and `aie::filter_odd` to align even and odd weights with the correct activation indices.
4. **Multiply-Accumulate**: Vector multiply-accumulate is computed in FP32 using `aie::mul` and `aie::mac`, then summed using `aie::reduce_add`.

**Measured Latency (Scalar vs Vectorized vs Optimized Vectorized):**
- **2048x2048**: Host time reduced from `667.21 ms` (Scalar) to `107.67 ms` (Optimized Vectorized). Est. raw NPU time reduced from `572.21 ms` to `12.67 ms` (**45.16x** raw speedup).
- **2560x2560**: Host time reduced from `1075.57 ms` (Scalar) to `207.60 ms` (Optimized Vectorized). Est. raw NPU time reduced from `885.57 ms` to `17.60 ms` (**50.32x** raw speedup).
- **End-to-End Generation**: Token generation latency on Llama-3.2-1B-Instruct went from `~176.5 s/token` (Scalar) to `~2.34 s/token` (Optimized Vectorized) (**75.43x** overall model generation speedup).

