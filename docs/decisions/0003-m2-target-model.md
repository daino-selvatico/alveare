# 0003 — Target Model and Kernel Strategy for Milestone M2

- Status: accepted
- Date: 2026-06-19

## Context

For Milestone M2, Alveare needs to execute one complete transformer decoder layer on the XDNA2 NPU and verify correctness against a standard reference.
We need to select a small, dense, well-documented model to avoid architectural complications (such as Gemma-4's sliding-window attention, QK-norm, or logit softcapping).
Additionally, we need a kernel execution strategy that doesn't trigger AMD XDNA2 userspace driver (`CREATE_HWCTX`) hardware context limits, which occur when too many unique compiled AIE kernel shapes are loaded in the same process.

## Decision

1. **Model Selection**: We select **Llama-3.2-1B-Instruct** as the target model. It is a standard, dense transformer model utilizing RMSNorm, Rotary Position Embeddings (RoPE) with split-half rotation, Grouped Query Attention (GQA), and SwiGLU MLP. The unquantized float16 weights are downloaded from `bartowski/Llama-3.2-1B-Instruct-GGUF` (file `Llama-3.2-1B-Instruct-f16.gguf`) and stored locally.
2. **Unified GEMV Kernel Execution**: Rather than compiling different NPU JIT programs for each matmul shape (e.g. 2048x2048, 512x2048, 2048x8192, 8192x2048), we pad and chunk all dense projections to a **single compiled shape of `(2048, 2048)`**. Specifically:
   - `w_q` and `w_o` (2048x2048): Executed directly.
   - `w_k` and `w_v` (512x2048): Padded along the rows to 2048x2048, executed, and the output is sliced to 512.
   - `w_gate` and `w_up` (8192x2048): Chunked along the rows into 4 calls of size 2048x2048.
   - `w_down` (2048x8192): Chunked along the columns into 4 calls of size 2048x2048, with the partial results accumulated on the host.
3. **Hardware Context Memory Safety**: We copy all NPU tensor buffers to host memory (`np.array(...)`) before the local tensor objects go out of scope, preventing segfaults caused by unmapping the virtual memory addresses when the tensors are garbage-collected.

## Consequences

- We run the layer with extremely low NPU context overhead (only 1 compiled shape `(2048, 2048)` for all matmuls).
- We avoid `AMDXDNA_CREATE_HWCTX` driver context limits and memory segmentation faults.
- The output matches the Hugging Face reference within a tight relative error of 0.52% (Max Absolute Error: 0.03125).
