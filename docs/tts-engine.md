# 🗣️ Alveare Text-to-Speech (TTS) Engine

Alveare provides a unified high-performance neural Text-to-Speech system featuring two complementary engines:
1. **Kokoro-82M**: Ultra-fast non-autoregressive neural synthesis (StyleTTS2 + ISTFTNet) delivering sub-500ms TTFA with natural Italian voices (`if_sara`, `im_nicola`, `af_heart`) on CPU Zen 5 and GPU CUDA. Recommended default for Live interactive conversation.
2. **Audio8 DualAR Architecture**: Enterprise-grade high-fidelity autoregressive neural synthesis (Slow Semantic + Fast Acoustic 10-codebook) with zero-shot voice cloning from audio samples, optimized for AMD Ryzen AI (XDNA2) NPU and CPU.

---

## 🌟 Highlights & Supported Models

| Model | Architecture | Size | Primary Acceleration | Sample Rate | Best For |
|---|---|---|---|---|---|
| **Kokoro 82M TTS** | `StyleTTS2 + ISTFTNet` | ~320 MB | CPU Zen 5 (RTF ~0.10x) / GPU CUDA | 24 kHz | **Full-duplex Live streaming, lowest TTFA (250–450ms)** |
| **Audio8 TTS 0.1B Preview** | `DualAR (Mamba+10 Codebooks)` | ~340 MB | AMD XDNA2 NPU (32 Cores) / CPU | 44.1 kHz | **Zero-shot cloning, rich emotional prosody** |
| **Audio8 TTS 0.6B Preview** | `DualAR (Mamba+10 Codebooks)` | ~1.2 GB | AMD XDNA2 NPU (32 Cores) / CPU | 44.1 kHz | **Studio-grade acoustic fidelity & cloning** |

---

## ⚡ Architecture Comparison

### Kokoro-82M (Non-Autoregressive StyleTTS2)
- **Single-Pass Parallel Synthesis**: Generates the entire audio waveform in one parallel forward pass using ISTFTNet vocoder.
- **Ultra-Fast RTF**: Generates 2–3 seconds of speech in ~250–400ms on AMD Ryzen AI 9 Zen 5 CPU without requiring GPU VRAM.
- **Presets**: Pre-computed style embeddings for Italian (`if_sara`, `im_nicola`) and English/melodic (`af_heart`).

### Audio8 (DualAR Hybrid Framework)
- **Slow Branch (Semantic Transformer)**: Predicts discrete semantic tokens conditioned on phonetic text encodings. Linear projection layers are offloaded to AMD Ryzen AI NPU cores (`NPULinear` / `gemv`).
- **Fast Branch (Acoustic Transformer)**: Rapidly predicts 10 acoustic codebook tokens per step conditioned on slow branch hidden states.
- **Neural Codec Vocoder**: Converts discrete multi-codebook streams into high-resolution 44.1 kHz raw audio waveforms.
- **Zero-Shot Voice Cloning**: Conditioned on reference WAV audio snippets.

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
