# M0 — Toolchain validation

**Status: done.** Fully verified on hardware.

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

## Results & Verification (2026-06-19)

### 1. Verified Toolchain Versions
- **Python**: `3.14.6` (Conda env `alveare-aie`)
- **`mlir_aie`**: `1.3.3.dev9+g8ed2e6b` (Git commit: `8ed2e6b`)
- **`llvm-aie`** (Peano): `21.0.0.2026061901+a76244b4` (Git commit: `a76244b4`)
- **Host XRT**: `2.21.75` (`libxrt-npu2` & `libxrt2`)
- **LLVM Toolchain** (on host): `21.1.8` (provides `llvm-objcopy`)

### 2. Examples Executed & Verified Correct
- **`00_memcpy`** (Vector Passthrough): Saturation benchmark compiled and ran on NPU.
  - **Performance**: NPU time: `~2193 us` (Effective bandwidth: `61.20 GB/s`).
  - **Status**: `PASS!`
- **`01_SAXPY`** (Vector Compute): $Z = a*X + Y$ ran on a single AIE core and verified against a CPU reference.
  - **Status**: `PASS!`

### 3. Key Deviations & Resolutions
- **Python 3.14**: The host system's `libxrt-npu2` package (Ubuntu 26.04) contains `pyxrt` Python bindings compiled specifically for Python 3.14. Running Python 3.12 (as originally planned) was impossible because it was binary-incompatible with the host's `pyxrt.so`. Recreating the conda environment with Python 3.14 resolved the import error.
- **Host `llvm` installation**: The kernel compilation/renaming step uses `llvm-objcopy` to rename symbols on AIE object files. Host's default GNU `objcopy` fails with AIE object files. Installing `llvm` via apt provided the necessary `llvm-objcopy` binary on the PATH.
