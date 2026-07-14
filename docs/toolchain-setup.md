# AIE Toolchain Setup Guide (XDNA2)

This guide documents the version-pinned setup required to compile and execute AIE kernels on the AMD Ryzen AI 9 HX 470 NPU (XDNA2) on Linux.

## System Specifications
- **SoC**: AMD Ryzen AI 9 HX 470 — codename **Gorgon Point** (2026 Ryzen AI refresh; *not* Strix Point, but the same XDNA2 NPU generation)
- **NPU Device Node**: `/dev/accel/accel0`
- **NPU Driver**: `amdxdna`
- **Firmware**: `/lib/firmware/amdnpu/` (directories `1502_00`, `17f0_10`, `17f0_11` present)
- **Host OS**: Ubuntu 26.04 LTS
- **Kernel**: `7.0.0-22-generic`

---

## 1. System Package Requirements

The following Debian packages must be installed on the host system to provide the Xilinx Runtime (XRT) headers, NPU system utilities, and LLVM tools (such as `llvm-objcopy`, which is used during compilation to rename symbols for AIE cores).

```bash
sudo apt-get update
sudo apt-get install -y libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu llvm
```

### Version Information
- **XRT Runtime / Utilities**: `2.21.75`
- **LLVM Toolchain**: `21.1.8`

---

## 2. Python Environment Selection (Python 3.14)

> [!IMPORTANT]
> The system's XRT Debian packages compile and install the Python bindings (`pyxrt`) specifically for the default system Python. On Ubuntu 26.04, this corresponds to **Python 3.14** (`/usr/lib/python3/dist-packages/pyxrt.cpython-314-x86_64-linux-gnu.so`).
> 
> Therefore, we **must use Python 3.14** in the dedicated conda environment to ensure binary compatibility with the host `pyxrt` module.

### Conda Environment Creation

Create a dedicated environment named `alveare-aie` with Python 3.14:

```bash
conda create -y -n alveare-aie python=3.14
```

---

## 3. Toolchain & Wheel Installation

Activate the conda environment and install the version-pinned `mlir-aie` and `llvm-aie` (Peano) prebuilt wheels:

```bash
# Sourcing conda helper
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alveare-aie

# Install MLIR-AIE (IRON frontend)
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4

# Install llvm-aie (Peano backend compiler)
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

### Installed Pinned Versions
- **`mlir_aie`**: `1.3.3.dev9+g8ed2e6b` (Git commit: `8ed2e6b`)
- **`llvm-aie`**: `21.0.0.2026061901+a76244b4` (Git commit: `a76244b4`)
- **`numpy`**: `2.4.6`
- **`ml_dtypes`**: `0.5.4`

---

## 4. Repository Setup

Clone the upstream `mlir-aie` repository and checkout the exact commit matching the installed python wheel (`8ed2e6b817`):

```bash
# From /home/daino/progetti/alveare
git clone https://github.com/Xilinx/mlir-aie.git mlir-aie
cd mlir-aie
git checkout 8ed2e6b817
```

---

## 5. Running the Examples

### Step 1: Initialize the Environment

Before building or running any AIE kernel, you must source the environment setup script inside the `mlir-aie` repository. This detects the NPU version and configures the search paths:

```bash
source utils/env_setup.sh
```

Ensure the terminal output prints `NPU2=1` (XDNA2 is a Gen 2 NPU).

### Step 2: Run Memcpy / Vector Passthrough

This example performs a parallel memcpy using the NPUs shim DMAs to saturate bandwidth, and verifies the output on the device.

```bash
python programming_examples/getting_started/00_memcpy/memcpy.py
```

Expected output:
```text
NPU time     (avg/min/max us): 2281.6 / 2228.4 / 2330.2
End-to-end   (avg/min/max us): 2397.8 / 2304.8 / 2494.8
Effective bandwidth (NPU avg): 58.83 GB/s
PASS!
```

### Step 3: Run SAXPY (Compute Example)

This example runs a SAXPY compute kernel ($Z = a*X + Y$) on an AIE core and verifies the output correctness.

```bash
python programming_examples/getting_started/01_SAXPY/saxpy.py
```

Expected output:
```text
PASS!
```
