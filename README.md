# Alveare

**An open-source LLM inference runtime for the AMD Ryzen™ AI (XDNA2) NPU on Linux.**

Alveare runs large language models on the AMD Ryzen AI NPU with a **fully open** stack — including the AIE kernels. It is a from-scratch alternative to the only other practical NPU runtime on Linux today, built so that anyone can bring their own model instead of waiting for closed, per-model kernel binaries.

> **Status: working, correct, and slow.** Alveare runs **Gemma-4-12B end-to-end on the NPU**, and its greedy output matches `llama.cpp` (same Q4 GGUF) **token-for-token**. It also runs Llama-3.2-1B and Gemma-3-1B. Decode is ~5.2 s/token on the 12B — correctness came first, speed is the ongoing work. See [`ROADMAP.md`](ROADMAP.md).

---

## Why

On Linux today, the only practical way to run LLMs on the AMD XDNA2 NPU is [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) (FLM). FLM is open-core: the CLI and orchestration are MIT, but the part that does the actual work — the **AIE NPU kernels** — ships only as prebuilt, patent-pending `.xclbin` binaries, one set per model **and per size**. That means:

- The community cannot add a new model: doing so requires compiling new kernels, and the kernel sources are not released.
- New-model support is gated entirely to the FLM team (and is also sold as a paid service).

Alveare's goal is a **fully open** alternative: an NPU LLM runtime where the kernels are open source too. It starts slower than FLM — matching FLM's performance is the hard, patented part — but "open and slower" still unlocks models nobody else can run on the NPU.

The whole stack we need is open and documented by AMD. FLM is living proof it works on exactly this hardware; they used [IRON](https://github.com/amd/iron) + [MLIR-AIE](https://github.com/Xilinx/mlir-aie). We use the same tools — and publish the result, kernels included.

## What works today

- **Gemma-4-12B-it** — full 48-layer forward on the NPU with per-layer weight streaming (~5.5 GB peak RAM). Greedy output matches `llama.cpp` token-for-token. ~5.2 s/token decode.
- **Gemma-3-1B-it** — end-to-end on the NPU, greedy continuation matches `llama.cpp`.
- **Llama-3.2-1B-Instruct** — end-to-end on the NPU.
- **OpenAI-compatible HTTP server** (`/v1/models`, `/v1/chat/completions`, streaming SSE).
- A hand-written, **MIT-licensed** vectorized multi-core AIE `gemv_q` kernel (quantized matrix-vector), plus rmsnorm / rope / attention kernels.

Alveare is **NPU-only and Linux-only.** It targets the XDNA2 NPU on AMD Ryzen AI (Strix Point) hardware. See [Tested on](#tested-on--reference-environment).

## The open AMD stack we build on

| Layer | Component | Role |
|---|---|---|
| Kernel driver | `amdxdna` (upstream Linux) | Talks to the NPU device (`/dev/accel/accel0`) |
| Userspace runtime | XRT | Loads `.xclbin`, manages buffers, submits work |
| Kernel compiler | MLIR-AIE / IRON | Write & compile AIE kernels → `.xclbin` |
| Backend | Peano (`llvm-aie`) | LLVM backend for the AIE cores |

See [`docs/background.md`](docs/background.md) for what each of these actually is.

---

## Quick start

Full, version-pinned instructions are in **[`docs/SETUP.md`](docs/SETUP.md)**. In brief:

1. **Set up the toolchain** — conda env `alveare-aie` (Python 3.14), the pinned `mlir_aie` + `llvm-aie` (Peano) wheels, XRT, and a matching `mlir-aie` clone. See [`docs/SETUP.md`](docs/SETUP.md).
2. **Verify the NPU** works end-to-end:
   ```bash
   ./alveare check
   ```
3. **Quantize a model** from a GGUF into Alveare's tiled Q4 layout (see [`docs/SETUP.md`](docs/SETUP.md) for pointing the script at your GGUF):
   ```bash
   python tools/quantize_gemma4.py      # -> ./quantized_weights_gemma4
   # or tools/quantize_gemma.py (Gemma-3-1B), tools/quantize_model.py (Llama-3.2-1B)
   ```
4. **Serve it** (OpenAI-compatible):
   ```bash
   ./alveare serve gemma4              # or: llama, gemma3, or a weights directory path
   ```

### The `alveare` command

```
alveare setup            Print setup guidance and run lightweight preflight checks.
alveare check            Run the NPU smoke test (wraps tools/check_npu.sh).
alveare serve <model>    Start the OpenAI-compatible server (model shorthand or weights dir).
alveare help             Show usage.
```

`serve` accepts a shorthand (`llama`, `gemma3`, `gemma4`) mapping to the default quantized-weights directory, or a path to any directory containing a `config.json`. Options: `--host` (default `127.0.0.1`), `--port` (default `8000`).

---

## Talking to the server

The server is **OpenAI-compatible**. Once `alveare serve …` is running (default `http://127.0.0.1:8000`):

**List models**
```bash
curl http://127.0.0.1:8000/v1/models
```

**Chat completion (curl)**
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-12b-it",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

Add `"stream": true` to receive Server-Sent Events (`text/event-stream`) instead of a single JSON response.

**Chat completion (Python `openai` client)**
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="gemma-4-12b-it",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=32,
    temperature=0.0,
)
print(resp.choices[0].message.content)
```

> The model name is auto-detected from the served weights' `config.json`; `GET /v1/models` reports the exact id to use. Generation is single-request serialized (one NPU) and slow on the 12B — expect several seconds per token.

---

## Tested on / reference environment

Alveare was developed and validated **entirely on one machine**. These are its real, live-captured specs — treat them as the known-good reference configuration.

| Component | Value |
|---|---|
| **APU / SoC** | AMD Ryzen AI 9 HX 470 w/ Radeon 890M (Strix Point) |
| **NPU** | XDNA2, device node `/dev/accel/accel0` (`crw-rw----+ root render`) |
| **NPU driver** | `amdxdna` (upstream, in-tree) |
| **NPU firmware** | `/lib/firmware/amdnpu/` → `1502_00`, `17f0_10`, `17f0_11` |
| **iGPU present** | Radeon 880M / 890M (RDNA 3.5) — *not used; Alveare targets the NPU only* |
| **RAM** | 64 GB system RAM (shared; the NPU streams weights from here) |
| **OS** | Ubuntu 26.04 LTS ("resolute") |
| **Kernel** | Linux `7.0.0-22-generic` (x86_64) |
| **Python** | 3.14 (conda env `alveare-aie`; must match the system `pyxrt` build) |
| **XRT** | `2.21.75` (`libxrt2`, `libxrt-npu2`, `libxrt-utils-npu`) |
| **mlir_aie** | `1.3.3.dev9+g8ed2e6b` (git `8ed2e6b`) |
| **llvm-aie / Peano** | `21.0.0.2026061901+a76244b4` (git `a76244b4`) |
| **LLVM (host)** | `21.1.8` |

More detail and provenance: [`docs/hardware.md`](docs/hardware.md) and [`docs/toolchain-setup.md`](docs/toolchain-setup.md).

---

## Repository layout

```
alveare      Top-level CLI launcher (setup / check / serve)
docs/        Design, background, hardware notes, setup guide, per-milestone specs, ADRs
kernels/     AIE kernel sources (IRON/MLIR-AIE) — the hard, open 30%
runtime/     Host-side Python runtime: XRT plumbing, weight streaming, KV cache, server
tools/       Weight quantizers/converters, reference oracles, NPU smoke test
tests/       Correctness tests + microbenchmarks (NPU vs CPU reference)
```

## How it was built

Alveare was built milestone by milestone, each with a single testable definition of done. The story, in order:

- [`docs/architecture.md`](docs/architecture.md) — how an NPU LLM runtime is structured.
- [`docs/kernels.md`](docs/kernels.md) — the AIE kernels and their host ABI.
- [`ROADMAP.md`](ROADMAP.md) — the milestones and what "done" meant for each.
- [`docs/milestones/`](docs/milestones/) — M0 (toolchain) → M9 (12B speed), including the M7/M8 story of closing the Gemma-4-12B fidelity gap.
- [`docs/decisions/`](docs/decisions/) — architecture decision records (ADRs).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The guiding principle: **open all the way down** — never introduce a closed binary blob as a load-bearing component.

## License

**MIT** — see [`LICENSE`](LICENSE). This includes the AIE kernels. Being open all the way down, kernels included, is the entire point of the project.
