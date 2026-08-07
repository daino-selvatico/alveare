# 🚀 Quickstart Guide

Get Alveare up and running on your AMD Ryzen™ AI (XDNA2) NPU in minutes.

---

## 📋 Prerequisites

- **Processor**: AMD Ryzen AI 300 / 9 HX 470 series APU (Strix Point / Gorgon Point) with XDNA2 NPU (`/dev/accel/accel0`).
- **OS**: Linux (Ubuntu 24.04 / 26.04 LTS or compatible).
- **User Privileges**: Member of the `render` group.

---

## 🛠️ Step 1: Installation

Run the single-command automated installer from the project root:

```bash
./install.sh
```

This sets up:
1. Conda environment (`alveare-aie` with Python 3.14).
2. XRT runtime dependencies (`libxrt2`, `libxrt-npu2`).
3. Compiler toolchain (`mlir-aie`, `llvm-aie`).
4. Native C++ server binary (`alveare_runtime`).
5. Web UI frontend dependencies (`frontend/`).

---

## 🔍 Step 2: Verify NPU Access

Run the NPU smoke test to verify driver communication:

```bash
./alveare check
```

Expected output:
```text
✓ NPU smoke test completed successfully!
```

> **Note**: If you get a permission error on `/dev/accel/accel0`, add your user to the `render` group:
> ```bash
> sudo usermod -aG render "$USER"
> ```
> Then log out and log back in.

---

## 🖥️ Step 3: Launch the Web UI & Server

Start the Python control server and React Web UI dashboard:

```bash
./alveare serve
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 📦 Step 4: Pick or Add a Model

### Option A: In-Browser 1-Click Setup (Recommended)
1. In the Web UI at `http://127.0.0.1:8000`, click **"Aggiungi Modello"**.
2. Pick a model from the supported preset list:
   - **Gemma 4 12B Instruct** (`gemma4`)
   - **Gemma 4 E4B Edge 4B** (`gemma4-e4b`)
   - **Gemma 3 1B Instruct** (`gemma3`)
   - **Llama 3.2 1B Instruct** (`llama`)
3. Click **"Installa"**. Alveare will automatically download the GGUF from Hugging Face, quantize the weights into NPU Q4_0 layout, build the NPU hardware kernels, and register the package.

### Option B: Automated CLI Setup
You can also run setup from the terminal:

```bash
./alveare setup gemma4 --arch gemma4
```

---

## 💬 Step 5: Start Chatting

Once installed, select your model and start chatting through any of these interfaces:

### 1. React Web UI Playground
Go to `http://127.0.0.1:8000`:
- **Real-time Streaming Chat**: Supports markdown rendering, code blocks, and conversation history.
- **Multimodal File Upload**: Drag and drop images, audio, or document files (PDF/TXT/JSON) into the chat.
- **Generation Settings**: Adjust system prompt, temperature, top-p, top-k, max tokens, and thinking toggle in the side panel.
- **Dark/Light Theme**: Toggle between dark and light themes using the header switch.

### 2. Interactive Terminal REPL
In a separate terminal window:

```bash
./alveare chat
```

### 3. OpenAI Python SDK / REST API
Connect standard client libraries to `http://127.0.0.1:8000/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="gemma4",
    messages=[{"role": "user", "content": "Explain NPU acceleration in two sentences."}],
    temperature=0.7,
    max_tokens=128,
)

print(response.choices[0].message.content)
```

---

## 📚 Next Steps

- Learn how to add custom GGUF models or write quantizer plugins: [`docs/adding_models.md`](adding_models.md)
- Configure sampling parameters and OpenAI request bodies: [`docs/sampling.md`](sampling.md)
- Inspect full hardware and build requirements: [`docs/SETUP.md`](SETUP.md)
