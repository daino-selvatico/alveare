# Background: The XDNA2 NPU and the Open AMD AIE Stack

This document explains the hardware, driver, and compiler stack that Alveare builds on for AMD Ryzen AI NPU acceleration on Linux.

---

## 🏛️ The Hardware: AMD XDNA2 / AIE Architecture

AMD Ryzen AI processors (Strix Point, Gorgon Point, and Ryzen AI 300 series) feature a dedicated **NPU** based on the **AI Engine (AIE)** architecture. The NPU is distinct from standard GPUs:

- **Array of VLIW/SIMD Processor Tiles**: The NPU consists of 32 independent AIE compute tiles, each with dedicated vector units, local data memory, and a configurable stream/DMA interconnect.
- **Dataflow Compute Model**: Work is structured as dataflow graphs. Kernels execute on AIE cores while DMA engines stream activation vectors and matrix tiles through local memory hierarchy.
- **Weight Streaming from DRAM**: On-chip memory is intentionally small (a few megabytes). Large language model weights (gigabytes) are streamed layer-by-layer from DRAM during decode and prefill. Hiding streaming latency behind compute is a primary optimization objective.

---

## 🛠️ The Open Software Stack

From hardware to high-level API:

### 1. `amdxdna` — Mainline Linux Kernel Driver
Upstreamed into Linux kernel (`drivers/accel/amdxdna`). Exposes the NPU device node at `/dev/accel/accel0` and manages NPU firmware (`/lib/firmware/amdnpu/`).

### 2. XRT — AMD Xilinx Runtime
Userspace library providing low-level device control: allocating host and device memory buffer objects (BOs), loading `.xclbin` binaries, submitting execution packets, and managing multi-core synchronization.

### 3. `.xclbin` — Open Hardware Kernel Binaries
AXLF containers holding compiled AIE array configurations: core machine code, DMA stream routes, memory tile allocations, and execution metadata. Alveare builds and packages all `.xclbin` files directly from open source MLIR-AIE designs.

### 4. MLIR-AIE / IRON — Open Kernel Compiler & Frontend
Open-source compiler framework ([Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) and [amd/iron](https://github.com/amd/iron)) for programming AIE core arrays. Translates high-level Python/MLIR dataflow descriptions into `.xclbin` binaries.

### 5. Peano (`llvm-aie`) — AIE Core LLVM Compiler
Open LLVM backend compiler targeting AIE SIMD cores, compiling C++ kernel compute routines directly into AIE machine code.

---

## ⚡ Inference Execution Loop

For autoregressive decode:

1. **Embedding Lookup**: Current token ID → activation vector.
2. **Layer Execution Loop** (streaming layer weights from DRAM):
   - CPU / NPU RMSNorm
   - QKV projection (NPU multi-core GEMV) → Attention (QKᵀ · softmax · V over KV cache) → Output projection
   - RMSNorm
   - Fused FFN (fused gate/up + GeGLU + down AIE hardware kernel)
   - Residual additions
3. **Final RMSNorm + LM Head**: Quantized GEMV over vocabulary → logits → token sampling.
4. **KV Cache Update**: Append new Key/Value states; repeat for next token.

---

## 📚 References & Upstream Projects

- **MLIR-AIE**: https://github.com/Xilinx/mlir-aie
- **IRON**: https://github.com/amd/iron
- **Peano / llvm-aie**: https://github.com/Xilinx/llvm-aie
- **XDNA Driver**: Mainline Linux kernel (`drivers/accel/amdxdna`)
- **Riallto**: https://riallto.ai
