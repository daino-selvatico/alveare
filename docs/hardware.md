# Target Hardware & Reference Environment

The reference development and validation environment for Alveare.

> **Silicon:** Tested and validated on **Gorgon Point** (AMD Ryzen AI 9 HX 470, 2026 Ryzen AI refresh) and **Strix Point** processors. Both use the **AMD XDNA2 NPU architecture**, sharing the same driver, firmware, and toolchain targets.

---

## 💻 System Configuration

| Component | Value |
|---|---|
| **APU / SoC** | AMD Ryzen AI 9 HX 470 w/ Radeon 890M (Gorgon Point / Strix Point family) |
| **NPU** | AMD XDNA2 (32 AIE tile array), exposed at `/dev/accel/accel0` |
| **NPU Driver** | `amdxdna` (upstream kernel module) |
| **NPU Firmware** | `/lib/firmware/amdnpu/` (`1502_00`, `17f0_10`, `17f0_11`) |
| **iGPU** | Radeon 890M (RDNA 3.5) — *Alveare targets NPU only* |
| **System RAM** | 64 GB LPDDR5X (shared unified memory for weight streaming) |
| **OS** | Ubuntu 24.04 / 26.04 LTS (Linux 6.x / 7.x) |
| **Python** | 3.14 (conda env `alveare-aie`) |
| **XRT** | `2.21.75` |
| **mlir_aie** | `1.3.3.dev9+g8ed2e6b` |
| **llvm-aie / Peano** | `21.0.0.2026061901+a76244b4` |

---

## 🔌 NPU System Access

- **Device Node**: `/dev/accel/accel0` (`crw-rw---- root render`)
- **Driver**: Mainline `amdxdna` kernel module
- **Permissions**: User must belong to the `render` group to access the hardware interface without root privileges:
  ```bash
  id -nG | grep -qw render || sudo usermod -aG render "$USER"
  ```

---

## 🛠️ Software Toolchain Stack

- **MLIR-AIE / IRON**: High-level AIE core array programming and graph compilation.
- **Peano (`llvm-aie`)**: Open LLVM compiler backend for AIE vector cores.
- **XRT (`libxrt2`, `libxrt-npu2`)**: Userspace device execution runtime.
- **Native C++ Runtime (`alveare_runtime`)**: High-throughput C++ inference server linking directly against XRT.
