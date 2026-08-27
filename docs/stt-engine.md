# 🎙️ Alveare Speech-to-Text (STT) Engine

Alveare includes an enterprise-grade, low-latency Speech-to-Text engine optimized for **AMD Ryzen AI (XDNA2) NPUs** and high-efficiency CPU fallback.

---

## 🌟 Highlights & Supported Models

| Model | Architecture | Size | Primary Acceleration | Recommended Use Case |
|---|---|---|---|---|
| **Whisper Base** | `whisper` | ~145 MB | AMD XDNA2 NPU (32 Cores) / CPU | Real-time live dictation, minimal RAM footprint |
| **Whisper Large v3 Turbo** | `whisper` | ~1.6 GB | AMD XDNA2 NPU (32 Cores) / CPU | State-of-the-art multilingual accuracy & fast decoding |
| **Whisper Large v3** | `whisper` | ~3.1 GB | AMD XDNA2 NPU (32 Cores) / CPU | Ultra-deep 32-layer transcription |

---

## ⚡ NPU Hardware Offload Architecture

The STT runtime offloads all heavy transformer decoder projections to AMD Ryzen AI NPU hardware kernels:
- **Fused Self-Attention QKV**: Dispatched as a single kernel call (`gemv_1536x512` for Base, `gemv_3840x1280` for Large).
- **Self-Attention Output**: (`gemv_512x512` / `gemv_1280x1280`).
- **Cross-Attention Output & Projections**: Native hardware dispatch.
- **Feed-Forward Network (FFN)**: FC1 (`gemv_2048x512` / `gemv_5120x1280`) and FC2 (`gemv_512x2048` / `gemv_1280x5120`).

---

## 🚀 Usage & API Endpoints

### 1. Web UI Audio Playground
Navigate to the **Audio** tab in the Alveare Web Dashboard (`http://127.0.0.1:8000`):
- **Live Microphone Streaming**: Real-time voice transcription with audio visualizer and automatic Voice Activity Detection (VAD).
- **Audio File Upload**: Upload `.wav`, `.mp3`, `.ogg`, `.flac`, or `.m4a` files for instant transcription.
- **Device Selector**: Switch dynamically between `NPU (XDNA2)` and `CPU`.

### 2. WebSocket Real-Time Streaming (`/ws/stt`)
Stream raw audio chunks (PCM 16-bit 16kHz mono or WebM/OGG) over WebSocket for real-time live captions:

```javascript
const ws = new WebSocket("ws://127.0.0.1:8080/ws/stt");

ws.onopen = () => {
  ws.send(JSON.stringify({ type: "start", sample_rate: 16000, language: "it" }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "segment") {
    console.log("Transcribed segment:", data.text);
  }
};
```

### 3. OpenAI-Compatible REST API (`/v1/audio/transcriptions`)
Send standard multipart form-data requests compatible with OpenAI SDKs:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@meeting_recording.wav" \
  -F "model=whisper-large-v3-turbo" \
  -F "language=it" \
  -F "response_format=verbose_json"
```

---

## 💻 CLI Commands

```bash
# Automated 1-click model installation
./alveare setup whisper-large-v3-turbo

# Run STT server directly on NPU
./alveare run whisper-large-v3-turbo --device npu

# Run STT server on CPU
./alveare run whisper-large-v3-turbo --device cpu
```
