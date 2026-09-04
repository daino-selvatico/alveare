# 🎙️ Alveare 3.0: Full-Duplex "Live" Voice-to-Voice Engine

Alveare 3.0 features a low-latency, full-duplex conversational engine enabling natural, real-time voice-to-voice interaction without manual push-to-talk buttons.

```
       Microphone (16kHz PCM)
                 │
                 ▼
        ┌─────────────────┐
        │  RMS Energy VAD │ ◄── Detects speech onset & silence
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Whisper STT   │ ◄── Dispatched to NPU (XDNA2)
        └────────┬────────┘
                 │ (User Transcript)
                 ▼
        ┌─────────────────┐
        │   Gemma-3 LLM   │ ◄── Streaming generation on GPU (Vulkan 1.4)
        └────────┬────────┘
                 │ (Token Stream + Sentence Boundaries)
                 ▼
        ┌─────────────────┐
        │   Audio8 TTS    │ ◄── Multi-threaded neural synthesis on CPU
        └────────┬────────┘
                 │ (Chunked WAV Audio)
                 ▼
        Speaker Output & WebSocket Streaming
```

---

## ⚡ Key Capabilities

### 1. Voice Activity Detection (VAD) & Smart Turn-Taking
- Analyzes incoming 16-bit 16kHz linear PCM audio chunks (default: 40ms frames).
- Calculates root-mean-square (RMS) energy against an adaptive noise floor (default threshold: `0.012`).
- Requires consecutive frames exceeding the threshold (250ms) to trigger a speech onset.
- Detects utterance completion after 450ms of trailing silence, triggering immediate transcription.

### 2. Full-Duplex User Barge-In (Interruption Handling)
When the assistant is speaking, if the user starts speaking:
1. The VAD immediately detects speech energy.
2. An `interrupted` event is emitted.
3. Audio playback queues are instantly flushed and cleared.
4. Any running LLM text generation or TTS synthesis tasks are aborted immediately.
5. The system transitions seamlessly into listening mode for the user's new question.

### 3. Pipelined Sentence-Level Streaming
Instead of waiting for the LLM to complete its full response:
- Tokens are accumulated until a clause or sentence boundary is reached (`.`, `!`, `?`, `\n`, `:`, `;`).
- The sentence chunk is dispatched asynchronously to the Audio8 TTS worker while the LLM continues generating the subsequent sentence.
- Time-to-First-Audio (TTFA) drops to **under 800ms**.

---

## 🌐 WebSocket Protocol (`/ws/live`)

The live session communicates over bidirectional WebSocket at `/ws/live`.

### Client to Server Messages

#### 1. Binary Audio Chunk
Send raw 16-bit mono 16000Hz PCM audio directly as binary frames.

#### 2. Start Session
```json
{
  "type": "start_session",
  "voice": "valeria"
}
```

#### 3. Stop Session
```json
{
  "type": "stop_session"
}
```

#### 4. Interrupt
```json
{
  "type": "interrupt"
}
```

---

### Server to Client Messages

| Event Type | Payload Fields | Description |
|---|---|---|
| `session_started` | `session_id`, `state` | Confirms live session initialization. |
| `vad_speech_start` | `rms` | User started speaking. |
| `vad_speech_end` | `duration` | User finished speaking, STT triggered. |
| `transcription` | `text`, `latency_ms` | Whisper transcription result. |
| `llm_token` | `token`, `ttft_ms` | Streaming LLM token for transcript rendering. |
| `audio_out` | `audio_b64`, `text`, `ttfa_ms`, `e2e_ms` | Synthesized speech audio (Base64 WAV) to play. |
| `interrupted` | `reason` | Assistant playback interrupted by user barge-in. |
| `metrics` | `stt_latency_ms`, `ttft_ms`, `ttfa_ms`, `e2e_ms` | Real-time latency HUD telemetry update. |

---

## 🖥️ Using Live Mode

### Via Web UI
1. Launch Alveare dashboard: `./alveare start`.
2. Open `http://127.0.0.1:8000` in Chrome, Firefox, or Edge.
3. Click the **"Live Studio"** tab in the top navigation bar.
4. Click **"Start Live Session"** and grant microphone access.
5. Speak naturally; observe the audio waveform visualizer, hardware routing cards, and latency metrics.

### Via Command Line
```bash
# Start live session in terminal with interactive HUD
./alveare live

# Specify custom host or port
./alveare live --host 127.0.0.1 --port 8080
```
