#!/usr/bin/env bash
#
# install.sh — one-command setup for Alveare on a fresh AMD Ryzen AI (XDNA2) Linux box.
#
# Creates the `alveare-aie` conda env (Python 3.14), installs the system XRT/LLVM
# packages, the pinned MLIR-AIE + Peano wheels, clones mlir-aie at the matching
# commit, and installs the runtime Python deps. Idempotent: safe to re-run.
#
# This mirrors the validated, version-pinned setup in docs/toolchain-setup.md.
# It uses `sudo` for the apt step and expects `conda` on PATH.
#
# (c) Copyright 2026 Alveare Authors — MIT License.

set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT"

ENV_NAME="alveare-aie"
PY_VER="3.14"
# Pinned toolchain versions (see docs/toolchain-setup.md). The mlir-aie clone is
# checked out to the commit matching the installed wheel.
MLIR_AIE_COMMIT="8ed2e6b817"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || die "conda not found on PATH. Install Miniconda/Anaconda first."

# 1. System packages: XRT (NPU runtime + pyxrt bindings) and LLVM (llvm-objcopy).
say "System packages (XRT + LLVM) via apt — needs sudo"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu llvm \
    || warn "apt install failed — install the XRT/LLVM packages manually (see docs/SETUP.md §1)."
else
  warn "apt-get not found; skipping. Install XRT + llvm for your distro manually (docs/SETUP.md §1)."
fi

# 2. Conda env.
say "Conda env '$ENV_NAME' (Python $PY_VER)"
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "env '$ENV_NAME' already exists — reusing it."
else
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

# Run the rest inside the env without needing `conda activate` in a script.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 3. AIE toolchain wheels (MLIR-AIE / IRON + Peano).
say "MLIR-AIE (IRON) + llvm-aie (Peano) wheels"
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

# 4. Matching mlir-aie clone (for utils/env_setup.sh + programming_examples).
say "mlir-aie repo clone @ $MLIR_AIE_COMMIT"
if [ -d "$ROOT/mlir-aie/.git" ]; then
  echo "mlir-aie/ already cloned — leaving as-is."
else
  git clone https://github.com/Xilinx/mlir-aie.git "$ROOT/mlir-aie"
  git -C "$ROOT/mlir-aie" checkout "$MLIR_AIE_COMMIT"
fi

# 5. Runtime + tooling Python deps.
say "Runtime Python deps (requirements.txt)"
pip install -r "$ROOT/requirements.txt"

# 6. Build the Native C++ Runtime Server.
say "Native C++ Server (CMake + Make)"
if command -v cmake >/dev/null 2>&1; then
  mkdir -p "$ROOT/runtime/cpp/build"
  (cd "$ROOT/runtime/cpp/build" && cmake .. && make -j$(nproc 2>/dev/null || echo 4))
else
  warn "cmake not found; skipping C++ build. Install cmake and build manually in runtime/cpp/build."
fi

say "Done."
cat <<EOF

Alveare is installed in the '$ENV_NAME' conda env.

Every session, before using alveare:
    conda activate $ENV_NAME
    cd mlir-aie && source utils/env_setup.sh && cd ..   # sets up the NPU stack

Then verify the NPU:
    ./alveare check

And the model workflow:
    ./alveare quantize gemma4 /path/to/gemma-4-12b-it.gguf
    ./alveare build-kernels gemma4
    ./alveare serve gemma4
    ./alveare chat                 # from another terminal

See docs/SETUP.md for details.
EOF
