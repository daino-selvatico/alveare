# Kernel Architecture & Roadmap

Overview of the open AIE hardware kernels compiled and used by Alveare on the AMD XDNA2 NPU.

---

## 🎯 Hardware Kernel Inventory

LLM decode latency is dominated by quantized matrix-vector multiplications (where weights are quantized int4, activations are bf16/fp32, and batch size is 1). Alveare compiles and utilizes the following AIE hardware kernels:

1. **`gemv_q`**: Vectorized quantized matrix-vector multiplication kernel. Tiles weight matrices across all 32 AIE cores on the XDNA2 array.
2. **`ffn_fused`**: Fused feed-forward network kernel performing Gate Projection, Up Projection, GeGLU activation, and Down Projection directly on-chip with FP32 accumulation.
3. **`rmsnorm`**: Optimized root-mean-square normalization kernel.
4. **`rope`**: Rotary position embedding kernel for Query/Key states.
5. **`attn`**: Multi-head attention (QKᵀ softmax V) kernel with sliding window and global attention support.

---

## 📐 Quantization Layout & Tiling

Alveare uses a dedicated `Q4_0` block-quantized format:
- **Block Size**: 32 weights per block.
- **Scale Factor**: 16-bit float/bf16 scale per block.
- **Memory Alignment**: Weights are pre-tiled into block-column major order in `tools/convert/` so that DMA streams align directly with AIE core memory banks without host-side runtime transposition.

---

## 🔬 Hardware Kernel ABI Contract

Every open kernel in `kernels/` documents:
- Buffer input/output descriptors, data types, and device layout.
- Tiling parameters ($M, K, N$) and active AIE core tile assignments.
- Tolerance thresholds compared against CPU reference implementations.
