#!/bin/bash
# (c) Copyright 2026 Alveare Authors
# Smoke test script for AMD Ryzen AI NPU development toolchain.

set -eo pipefail

echo "=== Running NPU Pre-flight Checks ==="

# 1. Verify device node
if [ ! -c /dev/accel/accel0 ]; then
  echo "ERROR: NPU device node /dev/accel/accel0 not found." >&2
  exit 1
fi
echo "✓ NPU device node (/dev/accel/accel0) exists."

# 2. Verify render group membership
if ! groups | grep -qw "render"; then
  echo "ERROR: Current user is not in the 'render' group." >&2
  exit 1
fi
echo "✓ User is in the 'render' group."

# 3. Verify XRT utilities
if ! command -v xrt-smi &>/dev/null; then
  echo "ERROR: xrt-smi utility not found. Please install libxrt-utils-npu." >&2
  exit 1
fi
echo "✓ xrt-smi is installed."

# 4. Verify Conda installation
CONDA_SH="/home/daino/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: Conda profile script not found at $CONDA_SH." >&2
  exit 1
fi

# 5. Verify dedicated conda environment
source "$CONDA_SH"
if ! conda env list | grep -q "alveare-aie"; then
  echo "ERROR: Conda environment 'alveare-aie' not found." >&2
  exit 1
fi
echo "✓ Conda environment 'alveare-aie' is present."

# 6. Execute smoke test
echo "=== Building and Running Smoke Test ==="
PROJECT_DIR="/home/daino/progetti/alveare"

if [ ! -d "$PROJECT_DIR/mlir-aie" ]; then
  echo "ERROR: mlir-aie repository clone not found in $PROJECT_DIR." >&2
  exit 1
fi

conda activate alveare-aie
cd "$PROJECT_DIR/mlir-aie"

# Source the MLIR-AIE environment
source utils/env_setup.sh

# Run the memcpy check and capture the output
echo "Running memcpy.py..."
set +e
OUTPUT=$(python programming_examples/getting_started/00_memcpy/memcpy.py 2>&1)
EXIT_CODE=$?
set -e

echo "$OUTPUT"

if [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: memcpy.py execution failed with exit code $EXIT_CODE" >&2
  exit 1
fi

if ! echo "$OUTPUT" | grep -q "PASS!"; then
  echo "ERROR: memcpy.py did not report 'PASS!'" >&2
  exit 1
fi

echo "✓ NPU smoke test completed successfully!"
exit 0
