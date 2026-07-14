# Setup & Usage Guide

How to install Alveare's toolchain, verify the NPU, quantize a model, and serve it.

Alveare is **Linux-only and NPU-only**: it targets the AMD Ryzen AI (XDNA2) NPU on Strix Point hardware. The versions below are the exact, known-good ones this project was validated against (see [`toolchain-setup.md`](toolchain-setup.md) and the [Tested-on table in the README](../README.md#tested-on--reference-environment)). Toolchain drift is the #1 reproducibility risk here — pin these.

---

## 0. Prerequisites (hardware)

- An AMD Ryzen AI (Strix Point / XDNA2) machine with the NPU exposed at `/dev/accel/accel0`.
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

The host runtime and tools need these in the `alveare-aie` env (in addition to the wheels above):

```bash
pip install fastapi uvicorn pydantic numpy ml_dtypes torch transformers gguf requests
```

(`torch`/`transformers` are used for the tokenizer and the CPU reference oracles; `gguf` for reading source GGUF files during quantization.)

---

## 6. Activate the environment for each session

Before running anything on the NPU, activate the conda env and source the AIE environment:

```bash
conda activate alveare-aie
source mlir-aie/utils/env_setup.sh    # prints NPU2=1 on Strix Point
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

Alveare loads weights in a tiled Q4 layout produced from a GGUF file. Pick the script for your model and point it at your GGUF (edit the `gguf_path`/`out_dir` at the top of the script, or drop the GGUF at the expected path):

| Model | Script | Output directory |
|---|---|---|
| Llama-3.2-1B-Instruct | `tools/quantize_model.py` | `quantized_weights` |
| Gemma-3-1B-it | `tools/quantize_gemma.py` | `quantized_weights_gemma` |
| Gemma-4-12B-it | `tools/quantize_gemma4.py` | `quantized_weights_gemma4` |

```bash
python tools/quantize_gemma4.py
```

The output directory contains the tiled weights plus a `config.json` the server uses to auto-detect the architecture. Quantized weights are large and **git-ignored** — never commit them.

---

## 9. Serve a model

```bash
./alveare serve gemma4                    # shorthand -> ./quantized_weights_gemma4
./alveare serve ./quantized_weights_gemma4 --port 9000   # explicit path + port
```

Shorthands: `llama`, `gemma3`, `gemma4`. Defaults: host `127.0.0.1`, port `8000`
(override with `--host` / `--port` or the `ALVEARE_HOST` / `ALVEARE_PORT` env vars).

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

Add `"stream": true` for Server-Sent Events. Requests are serialized (one NPU), and the 12B model runs at ~5.2 s/token — this runtime is **correct-but-slow** by design.

---

## Troubleshooting

- **`cannot import pyxrt`** — the conda env's Python must be 3.14 to match the system `pyxrt` build (step 2).
- **`NPU2` not printed** — you didn't source `mlir-aie/utils/env_setup.sh`, or the clone doesn't match the wheel commit (step 4).
- **Permission denied on `/dev/accel/accel0`** — add your user to the `render` group and re-login (step 0).
- **`xrt-smi` not found** — install `libxrt-utils-npu` (step 1).
