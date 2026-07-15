# Setup & Usage Guide

How to install Alveare's toolchain, verify the NPU, quantize a model, and serve it.

Alveare is **Linux-only and NPU-only**: it targets the AMD Ryzen AI **XDNA2** NPU. It was developed and validated on a **Gorgon Point** part (Ryzen AI 9 HX 470, the 2026 Ryzen AI refresh); XDNA2 is shared with Strix Point, so the same toolchain applies to both. The versions below are the exact, known-good ones this project was validated against (see [`toolchain-setup.md`](toolchain-setup.md) and the [Tested-on table in the README](../README.md#tested-on--reference-environment)). Toolchain drift is the #1 reproducibility risk here — pin these.

---

## 0. Prerequisites (hardware)

- An AMD Ryzen AI XDNA2 machine (Gorgon Point or Strix Point) with the NPU exposed at `/dev/accel/accel0`.
- The upstream `amdxdna` driver loaded and NPU firmware present under `/lib/firmware/amdnpu/`.
- Your user in the `render` group (so you can reach the NPU without root):
  ```bash
  id -nG | tr ' ' '\n' | grep -qx render && echo "in render group" || sudo usermod -aG render "$USER"
  ```
  (Log out/in after adding yourself to a group.)

Reference machine specs are in [`hardware.md`](hardware.md).

---

## 1. System packages (XRT + LLVM)

XRT provides the userspace runtime and the `pyxrt` Python bindings; `llvm` provides tools used while compiling AIE kernels (e.g. `llvm-objcopy`).

```bash
sudo apt-get update
sudo apt-get install -y libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu llvm
```

Validated versions:
- **XRT runtime / utilities**: `2.21.75`
- **LLVM (host)**: `21.1.8`

---

## 2. Python environment (Python 3.14)

> **Important:** the system XRT packages build `pyxrt` for the **default system Python**. On Ubuntu 26.04 that is **Python 3.14** (`/usr/lib/python3/dist-packages/pyxrt.cpython-314-…so`). The conda env **must** use Python 3.14 for binary compatibility with the host `pyxrt`.

```bash
conda create -y -n alveare-aie python=3.14
source ~/miniconda3/etc/profile.d/conda.sh   # adjust to your conda install
conda activate alveare-aie
```

---

## 3. AIE toolchain wheels (MLIR-AIE + Peano)

```bash
# MLIR-AIE (IRON frontend)
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4

# llvm-aie (Peano backend compiler for the AIE cores)
pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

Validated pinned versions:
- **`mlir_aie`**: `1.3.3.dev9+g8ed2e6b` (git `8ed2e6b`)
- **`llvm-aie` (Peano)**: `21.0.0.2026061901+a76244b4` (git `a76244b4`)
- **`numpy`**: `2.4.6`
- **`ml_dtypes`**: `0.5.4`

---

## 4. Clone `mlir-aie` (matching commit)

The runtime sources `mlir-aie/utils/env_setup.sh` to detect the NPU and configure search paths. Check out the commit matching the installed wheel:

```bash
# From the repo root (this repo). The clone is git-ignored.
git clone https://github.com/Xilinx/mlir-aie.git mlir-aie
cd mlir-aie
git checkout 8ed2e6b817
```

---

## 5. Runtime + tooling Python dependencies

The host runtime and tools need these in the `alveare-aie` env (in addition to the wheels above). Use the pinned list:

```bash
pip install -r requirements.txt
```

That covers `fastapi`/`uvicorn`/`pydantic` (the OpenAI-compatible server), `numpy`/`ml_dtypes`/`gguf` (weights), `transformers` (tokenizer/chat template), and `requests` (the llama.cpp verification tools). Add `torch` separately if you want to regenerate the HF reference oracles (`tools/ref/`), which is only needed for development, not for serving.

> **Serving fails with `ModuleNotFoundError: No module named 'fastapi'`?** You're running from the wrong environment (usually conda `base`). `conda activate alveare-aie` and install the requirements above. `alveare serve` now preflights this and prints exactly what's missing.

---

## 6. Activate the environment for each session

Before running anything on the NPU, activate the conda env and source the AIE environment:

```bash
conda activate alveare-aie
source mlir-aie/utils/env_setup.sh    # prints NPU2=1 on an XDNA2 NPU
```

`NPU2=1` confirms a Gen-2 NPU was detected.

---

## 7. Verify the NPU

Run the smoke test — it checks the device node, `render` membership, `xrt-smi`, the conda env, and runs a memcpy kernel on the NPU:

```bash
./alveare check
```

It should end with `PASS!` and `✓ NPU smoke test completed successfully!`. A lighter, non-NPU preflight is available via `./alveare setup`.

---

## 8. Quantize a model

**Alveare does not load GGUF or safetensors at serve time.** You download a single **GGUF** (as usual — from Hugging Face, the same file you'd give `llama.cpp`) and quantize it into Alveare's own layout: **Q4_0 block-quantized** tensors (blocks of 32, one scale per block), pre-packed/tiled for the `gemv_q` NPU kernel and written as `.npy` files, plus a `config.json`. The `.npy` directory is **generated locally, never downloaded**.

```bash
./alveare quantize g4-12b /path/to/gemma-4-12b-it.gguf
#   -> ./quantized_weights_g4-12b/   (config.json + *.npy)
```

- The **architecture is auto-detected** from the GGUF's `general.architecture` metadata and mapped to the right quantizer (same Q4_0 algorithm for all; only the per-model tensor wiring differs). Override with `--arch llama|gemma3|gemma4` if detection is wrong.
- The **alias** (`g4-12b` above) names the output dir `quantized_weights_<alias>` and is what you later `serve`. Omit it to default to the GGUF's filename.
- **Any number of models coexist**, each in its own directory. `./alveare list` shows them.

| GGUF architecture | Quantizer | Validated on | Approx. output |
|---|---|---|---|
| `llama` | `tools/quantize_model.py` | Llama-3.2-1B-Instruct | ~0.8 GB |
| `gemma3` | `tools/quantize_gemma.py` | Gemma-3-1B-it | ~0.8 GB |
| `gemma4` | `tools/quantize_gemma4.py` | Gemma-4-12B-it | ~9.7 GB |

A GGUF of any other architecture isn't supported yet (it needs a new quantizer + runtime work). The `quantized_weights*` directories are large and **git-ignored — never commit them**.

List what you've installed:

```bash
./alveare list
```

---

## 9. Serve a model

```bash
./alveare serve g4-12b                    # -> ./quantized_weights_g4-12b
./alveare serve ./quantized_weights_g4-12b --port 9000   # explicit path + port
```

`<model>` resolves as: built-in shorthand (`llama`/`gemma3`/`gemma4`) → an existing directory path → a generated alias (`quantized_weights_<model>`). Defaults: host `127.0.0.1`, port `8000` (override with `--host` / `--port` or the `ALVEARE_HOST` / `ALVEARE_PORT` env vars).

Under the hood this sets `ALVEARE_WEIGHTS_DIR` and launches `runtime/py/server.py`.

---

## 10. Talk to the server (OpenAI-compatible)

```bash
# List the served model
curl http://127.0.0.1:8000/v1/models

# Chat completion
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-12b-it",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 32,
    "temperature": 0.0
  }'
```

Python `openai` client:

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

Or just use the built-in terminal client (start the server first, then from another terminal):

```bash
./alveare chat                 # REPL over the OpenAI endpoint; streams tokens
```

Add `"stream": true` for Server-Sent Events. Requests are serialized (one NPU), and the 12B model runs at ~4.6 s/token — this runtime is **correct-but-slow** by design.

---

## 11. Offline Execution (Air-Gapped)

By default, the server downloads tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`) from Hugging Face into your `~/.cache/huggingface/hub/` directory.

To run the server in a completely offline environment without Hugging Face telemetry or online checks, use the `--offline` flag:

```bash
./alveare serve g4-12b --offline
```

This sets `HF_HUB_OFFLINE=1` under the hood. It will successfully start using the cached files in `~/.cache/`.

### 100% Air-Gapped Portability
If you wish to transfer the model to a machine that has *never* had internet access, you can bypass the `~/.cache` directory entirely:
1. Download `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` for your model.
2. Place them directly inside your generated weights folder (e.g., `./quantized_weights_g4-12b/`).
3. Run with `./alveare serve g4-12b --offline`.

Alveare's tokenizer glue will detect these files locally and avoid querying Hugging Face, making your quantized weights folder a fully self-contained portable package.

---

## Troubleshooting

- **`cannot import pyxrt`** — the conda env's Python must be 3.14 to match the system `pyxrt` build (step 2).
- **`NPU2` not printed** — you didn't source `mlir-aie/utils/env_setup.sh`, or the clone doesn't match the wheel commit (step 4).
- **Permission denied on `/dev/accel/accel0`** — add your user to the `render` group and re-login (step 0).
- **`xrt-smi` not found** — install `libxrt-utils-npu` (step 1).
