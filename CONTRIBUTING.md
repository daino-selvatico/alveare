# Contributing

Thank you for your interest in contributing to Alveare!

## Principles

- **Open all the way down.** The primary reason this project exists is to provide a 100% open-source NPU LLM stack. Never introduce a closed binary blob or proprietary kernel as a load-bearing component.
- **Correct first, fast later.** Every kernel ships with a CPU reference implementation and a correctness verification test before any performance optimization.
- **Gate by milestone.** Work is organized into clear milestones. Do not proceed to M(n+1) work until M(n)'s definition of done passes. See `ROADMAP.md`.
- **Write down decisions.** Major architectural choices and trade-offs are recorded as Architecture Decision Records in `docs/decisions/`.

## Working Rhythm

- Each kernel lives in `kernels/<name>/` with documentation on its host ABI (buffer shapes, dtypes, layout, tolerance) and a reference test suite.
- Benchmarks are placed in `tests/` with the shape and target hardware details noted.
- Keep `docs/toolchain-setup.md` and `docs/SETUP.md` authoritative and version-pinned — toolchain drift is the #1 reproducibility risk for AIE compilation.

## Commit Hygiene

- Keep commits small and focused. Reference the task or milestone in commit messages.
- Do not commit large model weight files or compiled binary `.xclbin` build artifacts (see `.gitignore`); commit sources, quantizers, and build scripts.

## Legal Note

Alveare is built as a clean-room, open-source stack targeting the AMD XDNA2 NPU using open-source tools provided by AMD and Xilinx (`mlir-aie`, `llvm-aie`, `XRT`).
