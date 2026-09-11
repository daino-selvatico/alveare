# 🎙️ Alveare 3.0 Voice WebSocket Protocol Specification

This document details the bidirectional WebSocket protocol for Alveare Live Voice-to-Voice communication served at `/ws/live`. It specifies the additive wire-compatible contract implemented for the web frontend, Android satellite clients, and external agents.

---

## 1. Connection & Handshake

### WebSocket Endpoint
```
wss://<host>:<port>/ws/live[?tts_mode=server|client]
```

### Initial Event: `live_connected`
Immediately upon connection establishment, the server transmits `live_connected`.

```json
{
  "event": "live_connected",
  "session_id": "8a32b61f-...",
  "hardware_profile": {
    "llm": {"model": "gemma4-e4b", "device": "gpu", "status": "running"},
    "stt": {"model": "openai/whisper-large-v3-turbo", "device": "cpu", "status": "running"},
    "tts": {"model": "hexgrad/Kokoro-82M", "device": "cpu", "status": "running", "voice": "if_sara"}
  },
  "sample_rate": 16000,
  "input_sample_rate": 16000,
  "output_sample_rate": 24000,
  "audio_format": "wav",
  "settings": {
    "max_context_turns": 4,
    "max_tokens": 120,
    "temperature": 0.5,
    "voice": "if_sara",
    "speed": 1.0,
    "min_silence_duration_s": 0.35,
    "enable_thinking": false,
    "tts_mode": "server"
  }
}
```

#### Wire Metadata Fields
| Field | Type | Description |
|---|---|---|
| `sample_rate` | `int` | Input audio sample rate (16000Hz). Kept for backwards compatibility with legacy clients. |
| `input_sample_rate` | `int` | Explicit input audio sample rate (16000Hz, 16-bit linear PCM mono). |
| `output_sample_rate` | `int` | Output audio sample rate for synthesized speech (24000Hz for Kokoro-82M, 44100Hz for Audio8). |
| `audio_format` | `string` | Output audio format (`"wav"`). |
| `settings.tts_mode` | `string` | Synthesis mode: `"server"` (default, server synthesizes WAV chunks) or `"client"` (server sends text chunks, client synthesizes locally). |

---

## 2. Client-to-Server Messages

### A. Binary Audio Stream
- Raw 16-bit Linear PCM, Mono, 16000 Hz, Little-Endian.
- Sent as binary WebSocket frames in ~20ms chunks (640 bytes) or ~40ms chunks (1280 bytes).
- Server tolerates odd byte lengths by truncating to the nearest even byte boundary.

### B. Base64 Audio Frame (`type: "audio"`)
```json
{
  "type": "audio",
  "data": "<base64-encoded 16-bit 16000Hz mono PCM>"
}
```

### C. Configuration Update (`type: "config"` or `"update_settings"`)
```json
{
  "type": "config",
  "settings": {
    "tts_mode": "client",
    "max_context_turns": 6,
    "max_tokens": 150,
    "temperature": 0.7,
    "voice": "if_sara",
    "speed": 1.05,
    "min_silence_duration_s": 0.30,
    "enable_thinking": false,
    "system_prompt": "Rispondi in modo conciso in italiano."
  }
}
```
*Server validates and clamps parameter ranges, updating live session without reconnecting. Echoes updated settings via `settings_updated`.*

### D. Interruption (`type: "interrupt"`)
```json
{
  "type": "interrupt"
}
```
*Immediately aborts in-flight LLM token streaming and TTS synthesis, purges queues, increments turn ID, and emits `interrupted`.*

### E. End of Speech Flush (`type: "end_of_speech"` or `"flush"`)
```json
{
  "type": "end_of_speech"
}
```
*Triggers immediate turn completion without waiting for VAD silence duration.*

### F. Ping Check (`type: "ping"`)
```json
{
  "type": "ping"
}
```
*Server responds immediately with `{"event": "pong", "timestamp": <unix_ts>}`.*

### G. Memory Clear (`type: "clear_memory"`, `"reset_context"`, or `"reset_history"`)
```json
{
  "type": "clear_memory"
}
```
*Cancels any in-flight turn, purges audio/TTS queues, resets conversation history to system prompt, increments turn ID monotonically (never resets to 0), and emits `memory_cleared`.*

### H. Client & Satellite Registration (`type: "register_client"`, `"register"`, or `"handshake"`)
Transmitted immediately upon connection by mobile clients (e.g. Android satellite) or smart speaker nodes.
```json
{
  "type": "register_client",
  "client_id": "android-satellite",
  "room": "\"salotto\"",
  "mode": "voice_satellite",
  "device_name": "Speaker Salotto"
}
```
*Server validates and bounds fields (client_id, room, mode), unquotes quoted room strings, associates satellite metadata with the LiveSession, and returns a non-fatal `client_registered` ack.*

### I. Legacy Heartbeat / Keepalive (`type: "heartbeat"` or `"keepalive"`)
Accepted safely for backwards compatibility with legacy client polling loops; responds immediately with `{"event": "pong", "timestamp": <unix_ts>}`.

---

## 3. Server-to-Client Messages

### `client_registered`
Registration confirmation sent in response to `register_client`.
```json
{
  "event": "client_registered",
  "client_id": "android-satellite",
  "room": "salotto",
  "mode": "voice_satellite",
  "status": "ok",
  "timestamp": 1789049870.150
}
```

### `vad_speech_start`
User speech energy onset detected above calibrated noise floor.
```json
{
  "event": "vad_speech_start",
  "timestamp": 1789049870.123
}
```

### `vad_speech_end`
Speech completion detected after trailing silence.
```json
{
  "event": "vad_speech_end",
  "turn_id": 1,
  "timestamp": 1789049872.456
}
```

### `user_transcript`
Whisper STT transcription result for user utterance.
```json
{
  "event": "user_transcript",
  "text": "Che tempo fa oggi?",
  "turn_id": 1,
  "stt_latency_ms": 284.5
}
```

### `ttft`
Time-To-First-Token metric from LLM start.
```json
{
  "event": "ttft",
  "ttft_ms": 112.4,
  "turn_id": 1
}
```

### `llm_chunk`
Ordered natural clause or complete sentence chunk emitted for incremental UI transcript rendering and/or client-side TTS.
```json
{
  "event": "llm_chunk",
  "chunk_idx": 1,
  "text": "Oggi a Milano il tempo è soleggiato con temperature miti.",
  "turn_id": 1
}
```

### `audio_chunk` (Server TTS Mode only)
Synthesized audio chunk streamed concurrently as sentences are completed by the LLM.
```json
{
  "event": "audio_chunk",
  "chunk_idx": 1,
  "audio_b64": "<base64-encoded WAV>",
  "format": "wav",
  "sample_rate": 24000,
  "text": "Oggi a Milano il tempo è soleggiato con temperature miti.",
  "turn_id": 1,
  "duration_sec": 2.45,
  "ttfa_ms": 340.2,
  "chunk_latency_ms": 210.8
}
```

### `turn_complete`
Emitted once all streaming tokens and all TTS audio chunks have completed synthesis and streaming.
```json
{
  "event": "turn_complete",
  "turn_id": 1,
  "full_text": "Oggi a Milano il tempo è soleggiato con temperature miti.",
  "metrics": {
    "stt_latency_ms": 284.5,
    "ttft_ms": 112.4,
    "ttfa_ms": 340.2,
    "tts_total_ms": 210.8,
    "e2e_latency_ms": 551.0
  },
  "timestamp": 1789049873.123
}
```

### `interrupted`
Emitted when assistant speech is interrupted by user barge-in or explicit interrupt message.
```json
{
  "event": "interrupted",
  "turn_id": 2,
  "message": "User interruption detected (barge-in)",
  "timestamp": 1789049874.567
}
```

### `memory_cleared`
Emitted after memory reset, preserving monotonic turn identity.
```json
{
  "event": "memory_cleared",
  "turn_id": 3,
  "message": "Memoria conversazione azzerata.",
  "timestamp": 1789049875.000
}
```

### `error`
Actionable error message when malformed input is received. Does not leak sensitive internal traces or python file paths.
```json
{
  "event": "error",
  "error": "Invalid JSON format in text message"
}
```

---

## 4. Concurrency & Overlap Architecture

```
User Audio (PCM 16k)
        │
        ▼
   Whisper STT
        │ (Transcript)
        ▼
   Gemma 4 LLM ──[Tokens]──► Chunker ──[llm_chunk]──► UI / Client TTS
        │                                  │
        │ (reads ahead concurrently)       ▼ (tts_mode: 'server')
        │                         Bounded Queue (maxsize=8)
        │                                  │
        │                                  ▼
        │                         Single TTS Worker
        │                                  │
        │                                  ▼
        │                          [audio_chunk (WAV)]
        │                                  │
        ▼                                  ▼
   LLM Stream End ──► Await Queue Drain ──► turn_complete
```

1. **True Concurrency**: LLM token reading does not await TTS synthesis. Chunks enter a bounded FIFO queue (`asyncio.Queue(maxsize=8)`).
2. **Deterministic Ordering**: Exactly one consumer synthesizes TTS chunks in strict sequence `chunk_idx: 1, 2, 3...`.
3. **Cancellation Safety**: When interrupted, the active turn task and TTS consumer are cancelled; queue is cleared; worker lock serializes subprocess stdin/stdout safely.
4. **Tool & Thought Guard**: Thought blocks (`<thought>...</thought>`) and tool calls (`[TOOL_CALL: ...]`) are filtered in real-time and never spoken or leaked into TTS audio.
