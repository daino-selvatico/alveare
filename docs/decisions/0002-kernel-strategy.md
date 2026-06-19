# 0002 — Kernel strategy: hand-written IRON vs iree-amd-aie

- Status: decided (hand-written IRON chosen for primary kernels, iree-amd-aie deferred)
- Date: 2026-06-19

## Context

We need AIE kernels (GEMV, attention, etc.) as `.xclbin`. Two routes:
1. **Hand-written kernels** via IRON / MLIR-AIE. Maximum control and understanding.
2. **`iree-amd-aie`** — AMD's compiler path.

## Decision

**Hand-written IRON kernels** are chosen as the primary development strategy. 

During Milestone M1, we successfully implemented a block-quantized GEMV (int4 weights × bf16 activation) using custom C++ functions compiled via Peano and orchestrated with the `aie.iron` JIT.

The `iree-amd-aie` path remains deferred due to:
- Complexity of compiling custom mixed-precision block-quantized graphs (which are not natively supported by standard IREE frontends without complex lowering patterns).
- The success of the hand-written IRON approach in building a correct, hardware-constrained single-core tiled GEMV layout that interfaces with PyXRT cleanly.

We will continue using hand-written IRON/C++ kernels for subsequent milestones (M2 attention/norms) to maintain fine-grained layout control.
