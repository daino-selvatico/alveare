# Setup & Usage Guide

Comprehensive guide on installing Alveare's toolchain, verifying NPU hardware, setting up models, and serving inference over HTTP or Web UI.

Alveare is **Linux-only and NPU-only**: it targets the AMD Ryzen AI **XDNA2** NPU (Strix Point & Gorgon Point processors).

---

## 🛠️ 1. Automated Installation

Run the automated installer from the repository root:

```bash
./install.sh
```

This single command will:
1. Create and configure the Conda environment (`alveare-aie` with Python 3.14).
2. Install system XRT libraries (`libxrt2`, `libxrt-npu2`, `libxrt-dev`).
3. Install AIE compiler wheels (`mlir_aie`, `llvm-aie`).
4. Clone required compiler repositories (`mlir-aie` pinned commit).
5. Build the Native C++ Server runtime (`alveare_runtime`).
6. Install Web UI frontend dependencies (`frontend/`).

---

## 🔍 2. Verify NPU Hardware

Verify that your system recognizes the XDNA2 NPU and that driver permissions are set correctly:

```bash
./alveare check
```

Expected output: `✓ NPU smoke test completed successfully!`.

> **Permissions Note:** Ensure your user is in the `render` group to access `/dev/accel/accel0`:
> ```bash
> id -nG | grep -qw render || sudo usermod -aG render "$USER"
> ```
> (Log out and back in after adding yourself to the `render` group.)

---

## 📦 3. Adding Models

Alveare supports both automated 1-click model installation and manual custom quantization.

### Option A: 1-Click Automated Setup (Web UI)

1. Launch Alveare:
   ```bash
   ./alveare serve
   ```
2. Open your browser to `http://127.0.0.1:8000`.
3. Click the **"Aggiungi Modello"** button.
4. Select a supported model from the list (e.g. **Gemma 4 12B**, **Gemma 4 E4B**, **Gemma 3 1B**).
5. Click **"Installa"**. Alveare will handle downloading the GGUF from Hugging Face, dequantizing tensors, packing them into Alveare's NPU `Q4_0` layout, and building the NPU hardware kernels.

### Option B: Automated CLI Setup

You can also run the 1-click automated setup directly from the terminal:

```bash
./alveare setup gemma4 --arch gemma4
```

### Option C: Manual GGUF Quantization

If you have downloaded a local `.gguf` file:

```bash
./alveare quantize g4-12b /path/to/gemma-4-12b-it.gguf
```

- Architecture auto-detection reads `general.architecture` from the GGUF. You can override it with `--arch llama|gemma3|gemma4`.
- Output weights are stored under `./quantized_weights_<alias>/`.

---

## 🖥️ 4. Serving & Web UI Dashboard

Start the inference server and management API:

```bash
./alveare serve g4-12b
```

Once running:
- **Web UI Dashboard**: Access `http://127.0.0.1:8000` to monitor NPU performance, manage models, view live logs, or test responses in the chat playground.
- **OpenAI Endpoint**: Connect any OpenAI SDK to `http://127.0.0.1:8000/v1`.

### Command Line Options

```bash
./alveare serve g4-12b --host 0.0.0.0 --port 8000
```

---

## 💬 5. Interacting with the Server

### Terminal Chat REPL

Open a second terminal window and run:

```bash
./alveare chat
```

### Python OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="gemma4",
    messages=[{"role": "user", "content": "What is the capital of Italy?"}],
    max_tokens=64,
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 🗑️ 6. Deleting Models

Delete a installed quantized model package from disk:

- **Via Web UI**: Click the red trash icon on the model card in the dashboard.
- **Via CLI**:
  ```bash
  ./alveare delete gemma3
  ```

---

## 🔧 Troubleshooting

- **Permission Denied on `/dev/accel/accel0`**: Add your user to the `render` group (`sudo usermod -aG render $USER`) and reboot or re-login.
- **`cannot import pyxrt`**: Ensure the `alveare-aie` Conda environment is active and running Python 3.14.
- **Port Conflict**: Override port with `--port 8999` or environment variable `ALVEARE_PORT=8999`.
