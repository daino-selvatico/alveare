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
        │ Kokoro / Audio8 │ ◄── Sub-second neural synthesis on CPU (Kokoro) or NPU (Audio8)
        └────────┬────────┘
                 │ (Chunked Audio)
                 ▼
        Speaker Output & WebSocket Streaming
```

---

## ⚡ Key Capabilities

### 1. Voice Activity Detection (VAD) & Smart Turn-Taking
- Analyzes incoming 16-bit 16kHz linear PCM audio chunks.
- Calculates root-mean-square (RMS) energy against an adaptive noise floor (default threshold: `0.015`).
- Requires consecutive frames exceeding the threshold (250ms) to trigger a speech onset.
- Detects utterance completion after trailing silence (configurable from `0.20s` to `1.0s`, default: `0.35s`), triggering immediate transcription.

### 2. Full-Duplex User Barge-In (Interruption Handling)
When the assistant is speaking, if the user starts speaking:
1. The VAD immediately detects speech energy.
2. An `interrupted` event is emitted.
3. Audio playback queues are instantly flushed and cleared.
4. Any running LLM text generation or TTS synthesis tasks are aborted immediately.
5. The system transitions seamlessly into listening mode for the user's new question.

### 3. Pipelined Sentence-Level Streaming with Kokoro-82M
Instead of waiting for the LLM to complete its full response:
- Tokens are accumulated until a clause boundary is reached (min 3 words or punctuation `,`, `.`, `!`, `?`, `\n`, `:`, `;`).
- The clause chunk is dispatched asynchronously to the Kokoro TTS worker (~200–300ms on Zen 5 CPU) while the LLM continues generating the subsequent sentence.
- Time-to-First-Audio (TTFA) drops to **250–450ms**, ensuring conversational real-time responsiveness.

---

## 🎛️ Live Settings & Dynamic Tuning

The Live Studio includes a slide-out **Live Settings** drawer in the UI and supports real-time parameter tuning without restarting the session:

- **Context Window (Max History Turns)**: Regulates the number of conversational turns kept in memory (1 to 16 turns). Prevents context saturation and reduces prompt evaluation latency.
- **Max Tokens per Response**: Constrains output length (e.g., 40 to 300 tokens, default: 120) to guarantee crisp, concise verbal answers.
- **LLM Temperature**: Controls response variability (0.0 to 1.5, default: 0.5).
- **Live Voice Selection**: Switch instantly between Kokoro Italian voices (`if_sara`, `im_nicola`, `af_heart`) or Audio8 voices (`valeria`, `daino`, `bolt`).
- **Speech Rate / Speed**: Adjustable synthesis playback rate (0.75x to 1.5x).
- **VAD Silence Duration**: Fine-tunes how quickly turn-taking triggers (0.2s for snappy interruptions, 0.5s for thoughtful pauses).
- **Thinking Toggle**: Disables or enables internal reasoning tokens (e.g. `<thought>` tags in Gemma 4) to eliminate latency overhead during live speech.
- **System Prompt Presets**: Choose between Ultra-Fast Concise, Conversational, Technical Assistant, or custom instructions.

---

## 🌐 WebSocket Protocol (`/ws/live`)

The live session communicates over bidirectional WebSocket at `/ws/live`.

### Client to Server Messages

#### 1. Binary Audio Chunk
Send raw 16-bit mono 16000Hz PCM audio directly as binary frames.

#### 2. Dynamic Settings Update (`type: "config"`)
```json
{
  "type": "config",
  "settings": {
    "max_context_turns": 6,
    "max_tokens": 100,
    "temperature": 0.4,
    "voice": "if_sara",
    "speed": 1.05,
    "min_silence_duration_s": 0.30,
    "enable_thinking": false,
    "system_prompt": "Rispondi in modo conciso e naturale."
  }
}
```

#### 3. Interrupt
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
