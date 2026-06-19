# M0 — Toolchain validation

**Status: current.** Nothing past this gate is worth starting until it passes.

## Goal

Prove the full chain works on *this* machine with code we control:

```
kernel source (IRON/MLIR-AIE) → compile → .xclbin → XRT load → run on /dev/accel/accel0 → correct result read back
```

## Definition of done

We build one of the upstream MLIR-AIE/IRON **example designs** (a vector-add or a small single-core GEMM), run it on the NPU, and the output matches the expected result. A second, slightly less trivial example (e.g. a tiled matrix-vector) running is a stretch bonus.

## Steps

1. **Pre-flight checks** (no install):
   - `ls -l /dev/accel/accel0` exists ✓ (verified)
   - `id` includes `render` group (so we can submit without root). If not, add user to `render` and re-login.
   - Note kernel + firmware versions for the record.

2. **Install the toolchain.** Options, in order of preference:
   - Prebuilt MLIR-AIE wheels / IRON via the documented install (Python venv). Pull in Peano (`llvm-aie`).
   - If wheels don't cover Strix/this distro, build from source (heavier).
   - Ensure an XRT that matches the driver is available (MLIR-AIE examples bring a runner; otherwise install XRT).
   - Record the exact versions that work in `docs/toolchain-setup.md` (created during this milestone).

3. **Run a known-good example** from MLIR-AIE (`programming_examples/`): vector scalar add or passthrough. Confirm it executes on `accel0` and verifies.

4. **Run a compute example**: a small matrix-vector or GEMM example. Confirm correctness.

5. **Write up** the exact, reproducible setup as `docs/toolchain-setup.md` and a `tools/check_npu.sh` that re-runs the pre-flight + a smoke test.

## Risks / known friction

- **Version matching** between Peano, MLIR-AIE, XRT, and the in-kernel `amdxdna` driver is the usual pain point. Pin versions once it works.
- Strix Point (XDNA2) vs older Phoenix (XDNA1) example targets — make sure examples target the right device/column count.
- Group permissions on `/dev/accel/accel0`.

## Output artifacts

- `docs/toolchain-setup.md` — reproducible install, pinned versions.
- `tools/check_npu.sh` — smoke test.
- A note in `docs/decisions/` if we hit a forced choice (e.g. build-from-source vs wheels).

## Explicitly NOT in M0

Writing any of our own kernels. M0 is purely "can we build and run *anything* on the NPU." Our first real kernel is M1.
