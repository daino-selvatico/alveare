# 0002 — Kernel strategy: hand-written IRON vs iree-amd-aie

- Status: open (to be decided during M1)
- Date: 2026-06-19

## Context

We need AIE kernels (GEMV, attention, etc.) as `.xclbin`. Two broad routes:

1. **Hand-written kernels** via IRON / MLIR-AIE. Maximum control and understanding; maximum effort. This is (essentially) what FLM did.
2. **`iree-amd-aie`** — AMD's open compiler path: feed an ML graph (via MLIR/IREE) and let the compiler target the AIE array. Potentially compiles whole models, far less hand-written kernel code. But the LLM path is immature and may not handle our quantized decode shapes well.

## Decision

**Deferred until M1.** During M1 we will, if time permits, implement the quantized GEMV **both ways** for one shape and compare:
- developer effort,
- correctness/robustness,
- resulting performance,
- how much it generalizes to the other kernels we need.

Pick the primary route based on evidence, not assumption. The runtime/kernel ABI boundary (see `architecture.md`) is designed so this choice can change later without rewriting the host runtime.

## Consequences

- M1 may take a bit longer (two implementations of one kernel) but de-risks the single biggest strategic unknown early.
- If `iree-amd-aie` proves viable, the whole "hard 30%" shrinks dramatically — worth the experiment.
