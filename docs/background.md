# Background: the XDNA2 NPU and the open AMD AIE stack

This document explains the pieces we build on, for anyone (including future us) who hasn't lived inside the Ryzen AI NPU world.

## The hardware: XDNA2 / AIE

AMD Ryzen AI laptop/mini-PC chips (Strix Point and later) include an **NPU** based on the **AI Engine (AIE)** architecture, inherited from Xilinx/Versal. The NPU is **not** a GPU:

- It is a 2D **array of small VLIW/SIMD processor tiles** ("AIE cores"), each with its own tiny local memory, connected by a configurable on-chip network and DMA engines. Think of it as a programmable systolic/dataflow fabric, not a SIMT machine.
- Compute is organized as **dataflow**: you place kernels on cores, stream data through them via DMA, and overlap movement with compute. Performance comes almost entirely from *how well you orchestrate data movement*, not from raw FLOPs.
- On-chip memory is **small** (a few MB total across memtiles). LLM weights (gigabytes) cannot live on-chip — they must be **streamed from DRAM** layer by layer. This is the central performance problem for LLM inference on the NPU, and the main thing a good runtime must hide.

This "beehive of tiny cores" is where the project name comes from.

## The software stack (all open)

From lowest to highest level:

### 1. `amdxdna` — the kernel driver
Upstreamed into mainline Linux. Exposes the NPU as an accel device: `/dev/accel/accel0`. Loads NPU firmware (`/lib/firmware/amdnpu/...`). This is what makes NPU compute possible on Linux at all (and why FLM can run on Linux while AMD's ONNX/OGA LLM path is still Windows-only).

### 2. XRT — Xilinx/AMD Runtime
Userspace library to talk to the device: create buffer objects (BOs), move data host↔device, load an `.xclbin` (the compiled NPU program), and submit/sync execution. Our host runtime links against XRT (or its AIE-specific API).

### 3. The `.xclbin` — a compiled NPU program
An AXLF container holding the configured AIE array program: core binaries, DMA/stream configuration, and metadata. **This is the unit FLM ships closed.** It is the *output* of the kernel compiler — and producing our own is the entire point of this project.

### 4. MLIR-AIE — the kernel compiler/framework
The open ([Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie)) framework for programming the AIE array. You describe: which cores run which compute, how buffers are placed, how DMAs/streams connect them. It lowers through MLIR to an `.xclbin`.

### 5. IRON — the Python frontend
[amd/iron](https://github.com/amd/iron): a higher-level Python API on top of MLIR-AIE for expressing AIE designs without writing MLIR by hand. The most ergonomic entry point. Comes with example designs (vector add, GEMM, matrix-vector, softmax).

### 6. Peano (`llvm-aie`) — the AIE backend
The LLVM backend that compiles the per-core compute kernels (C/C++ or generated code) down to AIE machine code. Alternative to the proprietary Chess compiler; being open is why we can stay fully open.

## How a model actually runs (the loop we must build)

For autoregressive decode (one token at a time):

1. Embedding lookup for the current token → activation vector.
2. For each layer (streaming its weights from DRAM):
   - RMSNorm
   - QKV projections (quantized GEMV) → attention (QK·softmax·V over the KV cache) → output projection
   - RMSNorm
   - MLP (gate/up/down GEMVs + activation)
   - residual adds
3. Final norm + LM head (big GEMV over vocab) → logits → sample next token.
4. Append new K/V to the KV cache; repeat.

Prefill (processing the prompt) is the same math but matrix-*matrix* (many tokens at once), which is more compute-bound and benefits from different kernels/tiling.

Almost all the time is in the quantized matmuls and in moving weights. That's why **M1 is a quantized GEMV** — nail the dominant primitive first.

## Why FLM is fast (what we're up against)

FLM's value is in the kernels and the orchestration: tiling that keeps the AIE array busy, DMA prefetch that hides weight streaming behind compute, fused operations, and quantization formats (their "Q4NX") laid out for efficient on-chip access. Those optimizations are patent-pending and closed. We will not match them quickly — and that's fine (see ROADMAP performance philosophy).

## References

- MLIR-AIE: https://github.com/Xilinx/mlir-aie
- IRON: https://github.com/amd/iron
- Peano / llvm-aie: https://github.com/Xilinx/llvm-aie
- XDNA driver: in mainline Linux (`drivers/accel/amdxdna`)
- Riallto (tutorial framework for Ryzen AI NPU): https://riallto.ai
- FLM (the closed-kernel reference we study for behavior): https://github.com/FastFlowLM/FastFlowLM
