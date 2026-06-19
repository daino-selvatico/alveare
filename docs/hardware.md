# Target hardware

The development and first-target machine. Captured from the live system on 2026-06-19.

## SoC

- **AMD Ryzen AI 9 HX** (`/proc/cpuinfo` model name: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`)
- Architecture: Strix Point class, **XDNA2 NPU** + RDNA 3.5 iGPU (Radeon 890M)
- 64 GB system RAM (shared; the NPU streams weights from here)

## NPU access (verified present)

- Device node: **`/dev/accel/accel0`** (`crw-rw----  root render`)
- Driver: **`amdxdna`** loaded (`lsmod`: `amdxdna`, depends `amd_pmf`, `gpu_sched`)
- Firmware present: **`/lib/firmware/amdnpu/`** → `17f0_10`, `17f0_11`, `1502_00`
- Kernel: Linux 7.0 (driver is upstream, no out-of-tree module needed)

Membership note: the device is group `render` — the dev user must be in `render` to access it without root (verify with `id`).

## What's already installed (and relevant)

- **FastFlowLM** `/usr/bin/flm` v0.9.43 — the closed-kernel reference runtime. We study its *behavior* and its model configs, not its kernels.
  - FLM bundles its own XRT; system-wide `xrt-smi` was **not** found in PATH. We will likely install XRT (or AIE tools that bundle it) ourselves for the build toolchain.
- **Lemonade** `/usr/bin/lemonade` v10.2.0 — Linux build exposes only `llamacpp` + `sd-cpp` backends (no NPU/OGA on Linux). Not a path to the NPU here; useful as a model/GGUF source and as the router host.
- Custom **`~/lemonade_router/`** (router.py + rules.yaml): routes default/fast chat to the NPU (FLM E2B) and heavy tasks to GPU llama.cpp models. Alveare's eventual server is meant to slot in here as another NPU backend.

## What's NOT yet installed (M0 will add)

- MLIR-AIE / IRON
- Peano (`llvm-aie`)
- A standalone XRT + AIE runtime headers for building the host side

## GPU note (out of scope for the NPU project, but relevant context)

The machine also has a switchable eGPU (RTX 4070 Ti) and the iGPU 890M. Per the user's standing setup, the 4070 Ti is reserved for ComfyUI/diffusion and big-model GPU inference; LLM NPU work is the 890M-adjacent NPU. Alveare targets the **NPU only**.

## Reference: FLM's per-model kernel set (what a model needs)

For context on the surface area, FLM ships these `.xclbin` per language model (e.g. its Qwen3/Gemma4e dirs):

```
layer.xclbin  mm.xclbin  attn.xclbin  dequant.xclbin
( + swa.xclbin, lm_head.xclbin, vision_*.xclbin, audio_*.xclbin for some )
```

This is roughly the kernel inventory Alveare must eventually reproduce (open) for a given model. M1–M2 build the `mm`/`gemv`, `attn`, `dequant`, norm/rope equivalents.
