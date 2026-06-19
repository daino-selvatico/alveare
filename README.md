# Alveare

**An open-source LLM inference runtime for the AMD Ryzen™ AI (XDNA2) NPU on Linux.**

> Status: **M0 — toolchain bring-up.** Nothing runs yet. This is a from-scratch research/engineering project, not a usable runtime.

## Why

On Linux today, the only practical way to run LLMs on the AMD XDNA2 NPU is [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) (FLM). FLM is open-core: the CLI and orchestration are MIT, but the part that does the actual work — the **AIE NPU kernels** — ships only as prebuilt, patent-pending `.xclbin` binaries, one set per model **and per size**. That means:

- The community cannot add a new model (e.g. Gemma 4 12B): doing so requires compiling new kernels, and the kernel sources are not released.
- New-model support is gated entirely to the FLM team (and is also sold as a paid service).

Alveare's goal is a **fully open** alternative: an NPU LLM runtime where the kernels are open source too, so anyone can bring their own model. It will start slower than FLM — matching FLM's performance is the hard, patented part — but "open and slower" still unlocks models nobody can run on the NPU today.

The whole stack we need is open and documented by AMD. FLM is living proof it works on exactly this hardware; they used [IRON](https://github.com/amd/iron) + [MLIR-AIE](https://github.com/Xilinx/mlir-aie). We use the same tools — and publish the result.

## The open AMD stack we build on

| Layer | Component | Role |
|---|---|---|
| Kernel driver | `amdxdna` (upstream Linux) | Talks to the NPU device (`/dev/accel/accel0`) |
| Userspace runtime | XRT | Loads `.xclbin`, manages buffers, submits work |
| Kernel compiler | MLIR-AIE / IRON | Write & compile AIE kernels → `.xclbin` |
| Backend | Peano (`llvm-aie`) | LLVM backend for the AIE cores |

See [`docs/background.md`](docs/background.md) for what each of these actually is.

## What this is and isn't

- **Is:** an attempt to build an open NPU LLM runtime, documented openly, milestone by milestone.
- **Is not (yet):** fast, complete, or a drop-in FLM replacement. See [`ROADMAP.md`](ROADMAP.md) for the honest plan and effort estimate.

## Repository layout

```
docs/        Design, background, hardware notes, per-milestone specs, decisions (ADRs)
kernels/     AIE kernel sources (IRON/MLIR-AIE) — the hard 30%
runtime/     Host-side C++ runtime: XRT plumbing, weight streaming, KV cache, server
tools/       Weight converters, quantizers, helper scripts
tests/       Correctness + microbenchmarks (NPU vs CPU reference)
```

## Start here

1. [`docs/background.md`](docs/background.md) — the hardware and toolchain, explained.
2. [`docs/architecture.md`](docs/architecture.md) — how an NPU LLM runtime is structured.
3. [`ROADMAP.md`](ROADMAP.md) — milestones M0→M4 and what "done" means for each.
4. [`docs/milestones/M0-toolchain-validation.md`](docs/milestones/M0-toolchain-validation.md) — the current task.

## License

MIT (see [`LICENSE`](LICENSE)). The whole point is that the kernels are open too.
