# 🗣️ Alveare Text-to-Speech (TTS) Engine

Alveare includes an enterprise-grade, high-fidelity neural Text-to-Speech engine based on the **Audio8 DualAR Architecture**, optimized for **AMD Ryzen AI (XDNA2) NPUs** and multi-threaded CPU execution.

---

## 🌟 Highlights & Supported Models

| Model | Architecture | Size | Primary Acceleration | Features |
|---|---|---|---|---|
| **Audio8 TTS 0.1B Preview** | `audio8` | ~340 MB | AMD XDNA2 NPU (32 Cores) / CPU | Ultra-compact footprint, natural Italian prosody, low latency |
| **Audio8 TTS 0.6B Preview** | `audio8` | ~1.2 GB | AMD XDNA2 NPU (32 Cores) / CPU | SOTA multilingual acoustic fidelity, zero-shot voice cloning |

---

## ⚡ NPU Hardware Offload Architecture

The TTS engine executes with the DualAR hybrid framework:
- **Slow Branch (Semantic Transformer)**: Predicts discrete semantic tokens conditioned on phonetic text encodings. Linear projection layers are offloaded to AMD Ryzen AI NPU cores (`NPULinear` / `gemv`).
- **Fast Branch (Acoustic Transformer)**: Rapidly predicts 10 acoustic codebook tokens per step conditioned on slow branch hidden states.
- **Neural Codec Vocoder**: Converts discrete multi-codebook streams into high-resolution 44.1 kHz raw audio waveforms.

---

## 🚀 CLI Commands

### 1. One-Click Model Setup
```bash
# Setup Audio8 0.1B
./alveare setup audio8-0.1b

# Setup Audio8 0.6B
./alveare setup audio8-0.6b
```

### 2. Standalone Model Server
```bash
# Serve on AMD Ryzen AI NPU
./alveare serve audio8-0.1b --device npu --port 8000

# Serve on CPU
./alveare serve audio8-0.1b --device cpu --port 8000
```

---

## 🌐 API Endpoints & Usage

### 1. OpenAI-Compatible REST API (`POST /v1/audio/speech`)
Fully compatible with OpenAI official client SDKs:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "audio8-0.1b",
    "input": "Ciao da Alveare! La sintesi vocale neurale su AMD Ryzen AI è attiva.",
    "response_format": "wav"
  }' \
  --output speech.wav
```

#### Python OpenAI SDK Example:
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="audio8-0.1b",
    voice="default",
    input="Benvenuti nel futuro dell'elaborazione neurale su NPU."
)
response.stream_to_file("output.wav")
```

### 2. Alveare Studio API (`POST /api/tts/generate`)
High-level endpoint supporting zero-shot voice cloning with reference audio clips:

```bash
curl -X POST http://127.0.0.1:8080/api/tts/generate \
  -F "text=Testo da convertire in voce naturale con Alveare." \
  -F "model=audio8-0.1b" \
  -F "device=npu" \
  -F "temperature=0.8" \
  -F "top_p=0.95"
```

#### Zero-Shot Voice Cloning:
```bash
curl -X POST http://127.0.0.1:8080/api/tts/generate \
  -F "text=Questa frase verrà pronunciata con la voce clonata dal campione audio." \
  -F "model=audio8-0.1b" \
  -F "reference_audio=@my_voice_sample.wav" \
  -F "reference_text=Testo esatto pronunciato nel campione vocale di riferimento"
```

---

## 🎨 Web UI Studio

Open the Alveare Dashboard in your browser (`http://127.0.0.1:8080` or `http://daino-ai.local:8080`):
1. Select the **Audio** tab.
2. Switch to the **Sintesi Vocale (TTS)** tab.
3. Enter your prompt text, optionally enable **Voice Cloning Zero-Shot** by uploading a reference voice sample, and click **Genera Audio**.
4. Play or download the resulting 44.1 kHz WAV file with live latency and RTF telemetry metrics.
