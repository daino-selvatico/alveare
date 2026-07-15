# Alveare

**An open-source LLM inference runtime for the AMD Ryzen™ AI (XDNA2) NPU on Linux.**

Alveare runs large language models on the AMD Ryzen AI NPU with a **fully open** stack — including the AIE kernels. It is a from-scratch alternative to the only other practical NPU runtime on Linux today, built so that anyone can bring their own model instead of waiting for closed, per-model kernel binaries.

> **Status: working, correct, and slow.** Alveare runs **Gemma-4-12B end-to-end on the NPU**, and its greedy output matches `llama.cpp` (same Q4 GGUF) **token-for-token**. It also runs Llama-3.2-1B and Gemma-3-1B. Decode is ~4.6 s/token on the 12B — correctness came first, speed is the ongoing work. See [`ROADMAP.md`](ROADMAP.md).

---

## Why

On Linux today, the only practical way to run LLMs on the AMD XDNA2 NPU is [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) (FLM). FLM is open-core: the CLI and orchestration are MIT, but the part that does the actual work — the **AIE NPU kernels** — ships only as prebuilt, patent-pending `.xclbin` binaries, one set per model **and per size**. That means:

- The community cannot add a new model: doing so requires compiling new kernels, and the kernel sources are not released.
- New-model support is gated entirely to the FLM team (and is also sold as a paid service).

Alveare's goal is a **fully open** alternative: an NPU LLM runtime where the kernels are open source too. It starts slower than FLM — matching FLM's performance is the hard, patented part — but "open and slower" still unlocks models nobody else can run on the NPU.

The whole stack we need is open and documented by AMD. FLM is living proof it works on exactly this hardware; they used [IRON](https://github.com/amd/iron) + [MLIR-AIE](https://github.com/Xilinx/mlir-aie). We use the same tools — and publish the result, kernels included.

## What works today

- **Gemma-4-12B-it** — full 48-layer forward on the NPU with per-layer weight streaming (~5.5 GB peak RAM). Greedy output matches `llama.cpp` token-for-token. ~4.6 s/token decode.
- **Gemma-3-1B-it** — end-to-end on the NPU, greedy continuation matches `llama.cpp`.
- **Llama-3.2-1B-Instruct** — end-to-end on the NPU.
- **OpenAI-compatible HTTP server** (`/v1/models`, `/v1/chat/completions`, streaming SSE).
- A hand-written, **MIT-licensed** vectorized multi-core AIE `gemv_q` kernel (quantized matrix-vector), plus rmsnorm / rope / attention kernels.

### Current limitations

- **Decode speed:** Decode is currently ~4.6 s/token on Gemma-4-12B. Correctness came first; speed optimization is the ongoing focus.
- **Experimental state:** Expect rough edges!
- **Prompt batched prefill:** The initial prompt prefill phase is now successfully offloaded to the NPU with batched matrix multiplications, drastically improving Time-to-First-Token latency compared to the initial CPU-only implementation!

Alveare is **NPU-only and Linux-only.** It targets the **XDNA2** NPU on AMD Ryzen AI hardware. It was developed and validated on a **Gorgon Point** part (Ryzen AI 9 HX 470, the 2026 Ryzen AI refresh); XDNA2 is shared with Strix Point, so the targeting is the same across both. See [Tested on](#tested-on--reference-environment).

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

```bash
# 1. Install everything (conda env alveare-aie + toolchain + deps) — one command:
./install.sh                    # or:  ./alveare install

# 2. Every session, enter the env + NPU toolchain:
conda activate alveare-aie
cd mlir-aie && source utils/env_setup.sh && cd ..

# 3. Verify the NPU end-to-end:
./alveare check

# 4. Quantize a source GGUF into Alveare's Q4 layout (a single GGUF you downloaded).
#    The architecture is auto-detected from the GGUF; give it a short alias:
./alveare quantize g4-12b /path/to/gemma-4-12b-it.gguf     # -> quantized_weights_g4-12b
#    (omit the alias to name it after the GGUF file)

# 5. See what's installed:
./alveare list

# 6. Serve it (OpenAI-compatible HTTP server):
./alveare serve g4-12b

# 7. Talk to it from another terminal:
./alveare chat
```

### The `alveare` command

```
alveare install                        One-command full install (env + toolchain + deps).
alveare setup                          Setup guidance + lightweight preflight checks.
alveare check                          NPU smoke test (wraps tools/check_npu.sh).
alveare quantize [alias] <gguf> [--arch A]   GGUF -> Alveare Q4 weights dir.
alveare list  (or: models)             List installed models.
alveare serve <model> [--host --port]  Start the OpenAI-compatible server.
alveare chat [--host --port --model]   Minimal terminal chat vs a running server.
alveare help                           Show usage.
```

- **`quantize`** — architecture is **auto-detected** from the GGUF (`general.architecture`); override with `--arch llama|gemma3|gemma4` if needed. `[alias]` names the output dir `quantized_weights_<alias>` (defaults to the GGUF's filename). Any number of models coexist, each in its own directory — keep a 12B and a smaller one side by side. You then `serve <alias>`.
- **`list`** — shows every installed model (the `quantized_weights_*` directories), its architecture, and size.
- **`serve`** — `<model>` resolves as: built-in shorthand (`llama`/`gemma3`/`gemma4`) → an existing directory path → a generated alias (`quantized_weights_<model>`).
- **`chat`** — a tiny REPL over the OpenAI endpoint. Start a server first, then chat from another terminal.

---

## Models — where they go and in what format

**You download a single GGUF, exactly as usual. You do NOT download the many-file `.npy` directory — you generate it locally.**

Alveare cannot load a GGUF (or safetensors) directly at serve time: the NPU kernel needs weights pre-tiled into Alveare's own layout. So the workflow is always **one GGUF file → run the quantizer → a directory of `.npy` files → serve that directory**:

- **What you download:** a normal single-file **GGUF** of the model (from Hugging Face — e.g. `bartowski`/`unsloth`/`ggml-org` repos), the same file you'd use with `llama.cpp`.
- **What Alveare produces (locally, once):** running `tools/quantize_<model>.py` on that GGUF writes a directory of **Q4_0 block-quantized** tensors (blocks of 32, per-block scale) as `.npy` files pre-packed for the `gemv_q` NPU kernel, plus a `config.json`. This directory is **generated, never downloaded**, and is git-ignored (it's large model data).
- **What you serve:** that directory, by its alias. `alveare quantize g4-12b model.gguf` writes `./quantized_weights_g4-12b/`, and `alveare serve g4-12b` serves it. You can also pass a directory path directly.

**Naming and multiple models.** The output alias is yours to choose (`quantize <alias> <gguf>`) and defaults to the GGUF's filename. Every model lives in its own `quantized_weights_<alias>/` directory, so any number coexist — e.g. a Gemma-4 12B and a smaller one at the same time. `alveare list` shows them all.

**Architecture is auto-detected** from the GGUF's `general.architecture` metadata and mapped to the right quantizer:

| GGUF architecture | Quantizer used | Example |
|---|---|---|
| `llama` | `tools/quantize_model.py` | Llama-3.2-1B-Instruct |
| `gemma3` | `tools/quantize_gemma.py` | Gemma-3-1B-it |
| `gemma4` | `tools/quantize_gemma4.py` | Gemma-4-12B-it |

If auto-detection is wrong, force it with `--arch llama|gemma3|gemma4`. A GGUF of any other architecture isn't supported yet (it would need a new quantizer + runtime support). The quantized `quantized_weights_*` directories are **git-ignored and never committed** (large model data; the 12B is ~9.7 GB).

The server **auto-detects the architecture** from `config.json` in the served directory. These `quantized_weights*` directories are **git-ignored and must never be committed** — they are large model data, not source.

### From "I have a model" to "it's served"

Everything below runs **inside the `alveare-aie` conda env with the NPU toolchain sourced** (see [`docs/SETUP.md`](docs/SETUP.md)). Running from another env (e.g. conda `base`) is the #1 gotcha — the server deps and the NPU stack won't be there:

```bash
conda activate alveare-aie
pip install -r requirements.txt              # fastapi/uvicorn/... (one-time)
cd mlir-aie && source utils/env_setup.sh && cd ..   # NPU stack (pyxrt/aie)
```

1. **Download a source GGUF** — a single file, as usual. E.g. for Gemma-4-12B grab a GGUF from Hugging Face (`bartowski/…gemma-4-12b-it-GGUF`, `unsloth/…`, etc.), the same file you'd give `llama.cpp`.
2. **Quantize** it into Alveare's `.npy` layout (a few minutes; the 12B output is ~9.7 GB). The architecture is auto-detected; pick an alias:
   ```bash
   ./alveare quantize g4-12b /path/to/gemma-4-12b-it.gguf
   #   -> ./quantized_weights_g4-12b/   (config.json + *.npy)
   ```
3. **Serve** it by that alias:
   ```bash
   ./alveare serve g4-12b
   ```
4. **Chat** from another terminal:
   ```bash
   ./alveare chat
   ```

That's the whole path. Supported architectures today: **Llama, Gemma-3, Gemma-4** (validated on Llama-3.2-1B, Gemma-3-1B, Gemma-4-12B; NPU-only, Linux-only).

---

## Talking to the server (drop-in OpenAI-compatible)

**Alveare's server is OpenAI-compatible — point any OpenAI client at it, no code changes beyond the base URL.** It speaks the standard `/v1` endpoints, so the official `openai` SDK and any OpenAI-compatible tool or library (LangChain, LlamaIndex, `llm`, Continue, etc.) work by just setting `base_url` to Alveare and using any placeholder API key.

Once `alveare serve …` is running (default `http://localhost:8000`):

**Official `openai` Python SDK — the drop-in path**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="gemma-4-12b-it",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=32,
    temperature=0.0,
)
print(resp.choices[0].message.content)

# Streaming variant:
stream = client.chat.completions.create(
    model="gemma-4-12b-it",
    messages=[{"role": "user", "content": "Write a haiku about NPUs."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Any OpenAI-compatible library works the same way — e.g. in LangChain, set `ChatOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed", model="gemma-4-12b-it")`.

**Raw HTTP (curl)**
```bash
# List the served model
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-12b-it",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

Add `"stream": true` to the request body to receive Server-Sent Events (`text/event-stream`) instead of a single JSON response.

> The model name is auto-detected from the served weights' `config.json`; `GET /v1/models` reports the exact id to use. Generation is single-request serialized (one NPU) and slow on the 12B — expect several seconds per token.

---

## Tested on / reference environment

Alveare was developed and validated **entirely on one machine**. These are its real, live-captured specs — treat them as the known-good reference configuration. Note the silicon: this is a **Gorgon Point** part (AMD Ryzen AI 9 HX 470), the **newer 2026 Ryzen AI refresh** — not Strix Point. It uses the same **XDNA2** NPU generation, so the NPU, driver, firmware, and toolchain targeting are identical to Strix Point.

| Component | Value |
|---|---|
| **APU / SoC** | AMD Ryzen AI 9 HX 470 w/ Radeon 890M — **Gorgon Point** (2026 Ryzen AI refresh, *not* Strix Point; same XDNA2 NPU) |
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
