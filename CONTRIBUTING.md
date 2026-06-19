# Contributing

Private/solo for now, but written so it can go public later.

## Principles

- **Open all the way down.** The reason this project exists is that the kernels are open. Never introduce a closed binary blob as a load-bearing component. (Studying FLM's *behavior* is fine; copying its closed kernels is not, and would defeat the purpose besides being patent-encumbered.)
- **Correct first, fast later.** Every kernel ships with a CPU reference and a correctness test before any optimization.
- **Gate by milestone.** Don't start M(n+1) work before M(n)'s definition-of-done passes. See `ROADMAP.md`.
- **Write down decisions.** Forced choices → an ADR in `docs/decisions/`.

## Working rhythm

- Each kernel lives in `kernels/<name>/` with a `README.md` documenting its host ABI (shapes, dtypes, layout, tolerance) and a reference + test.
- Benchmarks go in `tests/bench/` with the shape and the machine noted.
- Keep `docs/toolchain-setup.md` (created in M0) authoritative and version-pinned — toolchain drift is the #1 reproducibility risk here.

## Commit hygiene

- Small, focused commits. Reference the milestone (e.g. `M1: gemv_q single-tile correctness`).
- Don't commit model weights or `.xclbin` build artifacts (see `.gitignore`); commit the *sources* and build instructions.

## Legal note

This project must not incorporate FastFlowLM's proprietary kernels or any patent-encumbered code. It is a clean-room, from-the-open-stack effort.
