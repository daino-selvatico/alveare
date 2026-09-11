# 🧪 Alveare Voice Backend Verification & Reliability Report

This document records the comprehensive verification results, regression tests, architectural invariants, and benchmarking procedures for the Alveare Voice Live WebSocket backend (`runtime/py/live_orchestrator.py` and `runtime/py/control_server.py`).

---

## 1. Test Suite Summary

The backend test suite was executed using the project's active conda environment:
`/home/daino/miniconda3/envs/alveare-aie/bin/pytest`

```bash
PYTHONPATH=. /home/daino/miniconda3/envs/alveare-aie/bin/pytest \
    tests/test_live_backend_qa_regression.py \
    tests/test_live_smoke_negative.py \
    tests/test_control_server_live.py \
    tests/test_live_orchestrator_reliability.py \
    tests/test_live_orchestrator_tools.py \
    tests/test_sandbox_tools.py
```

### Result: 56 passed in 3.52s (100% pass rate)

| Test File | Tests | Status | Verification Scope |
|---|---|---|---|
| `tests/test_live_backend_qa_regression.py` | 8 | PASSED | VAD duration threshold enforcement on capped buffers, onset audio frame preservation, oversized chunk slicing, odd-byte PCM boundaries, `tts_mode` turn snapshotting, Android `register_client` unquoting, turn cancellation identity guards |
| `tests/test_live_smoke_negative.py` | 13 | PASSED | Integration smoke runner negative cases: handshake failure, timeout exit codes, invalid audio format, empty response detection, audio leak detection on interrupt, client-mode audio rejection, missing payload errors |
| `tests/test_control_server_live.py` | 11 | PASSED | `/ws/live` wire metadata, `tts_mode` negotiation, `register_client` onOpen handshake, legacy heartbeat/keepalive ping-pong, malformed JSON/base64 error handling, WebSocket finally cleanup |
| `tests/test_live_orchestrator_reliability.py` | 10 | PASSED | Concurrency overlap, prosody chunking without arbitrary cuts, monotonic turn IDs across resets, cancellation mid-TTS, thought and tool call suppression, VAD noise floor adaptation |
| `tests/test_live_orchestrator_tools.py` | 4 | PASSED | Tool execution (`web_search`, `bash`), memory clear, conversation history pruning |
| `tests/test_sandbox_tools.py` | 10 | PASSED | Sandboxed shell tools, security boundaries, timeout handling |

---

## 2. Priority Verification & Regression Matrix

### Bug 1: VAD Maximum Duration Accounting & Audio Frame Fidelity
- **Baseline Reality**: In `live_orchestrator.py`, `handle_audio_chunk` only incremented `speech_samples_count` if `len(speech_buffer) + len(chunk) <= max_bytes`, then evaluated `duration > max_turn_duration_s`. As soon as speech buffer filled to `max_speech_buffer_bytes`, `speech_samples_count` stopped accumulating, causing long turns to freeze in speech mode without ever tripping the duration threshold. Additionally, audio frames before speech threshold crossing were dropped, and incoming chunks exceeding 64000 bytes were rejected outright.
- **Fix**:
  1. Implemented bounded continuous duration tracking: `self.speech_samples_count += len(samples)` unconditionally accumulates all active speech samples regardless of buffer byte capacity.
  2. Enforced maximum turn duration: when `(self.speech_samples_count / self.sample_rate) >= self.max_turn_duration_s` (or buffer reaches `max_speech_buffer_bytes`), forces immediate turn completion via `_handle_speech_end(forced=True)` and logs a warning.
  3. Added a rolling `_onset_buffer` (max 5 frames / 100ms) preserving audio frames immediately preceding threshold detection, prepending them to the speech buffer at onset to preserve initial consonants and syllables.
  4. Added `_pcm_remainder` to track odd-byte linear PCM frame boundaries across WebSocket chunks.
  5. Slices oversized frames (>64000 bytes) into standard chunks sequentially to prevent frame dropping.
- **Verification**: `test_vad_max_duration_capped_buffer_forces_turn_completion`, `test_vad_onset_buffer_preserves_initial_audio`, `test_oversized_audio_chunk_handling`, `test_odd_byte_pcm_boundary_handling`.

### Bug 2: `tts_mode` Mutation & Turn Lifetime Isolation
- **Baseline Reality**: `self.tts_mode` could be changed via `update_settings` while an active voice turn was in progress. If toggled from `client` to `server` mid-turn, LLM chunks were pushed into `current_tts_queue` with no consumer task running, creating an unrecoverable deadlock once 8 chunks were queued. If toggled from `server` to `client` mid-turn, consumer draining was bypassed, dropping in-flight synthesis and leaving orphaned audio tasks.
- **Fix**:
  1. `_process_voice_turn` captures an immutable snapshot `turn_tts_mode = self.tts_mode` at turn entry.
  2. All downstream token consumption, queue enqueuing, and consumer draining for that turn evaluate `turn_tts_mode`.
  3. Settings updates take effect cleanly on subsequent turns without disturbing active pipelines.
- **Verification**: `test_tts_mode_mutation_client_to_server_does_not_deadlock`, `test_tts_mode_mutation_server_to_client_preserves_ongoing_turn`.

### Bug 3: Android Satellite `register_client` onOpen Protocol Contract
- **Baseline Reality**: Android satellite clients transmit `{"type": "register_client", "client_id": "...", "room": "\"salotto\"", ...}` immediately upon connection opening. The server previously flagged this as an unknown message or rejected it. Additionally, Android `SharedPreferences` serialization commonly wrapped room names in literal escaped quotes.
- **Fix**:
  1. Added dedicated handler for `register_client`, `register`, and `handshake` in `control_server.py`.
  2. Sanitized string inputs: stripped surrounding quotes (`"\"salotto\""` -> `salotto`) and clamped string lengths to safe bounds.
  3. Stored satellite metadata (`client_id`, `room`, `mode`, `device_name`) on `LiveSession`.
  4. Emitted non-fatal `client_registered` confirmation event with `status: "ok"`.
  5. Added compatibility for legacy `heartbeat` and `keepalive` messages, replying with `pong`.
- **Verification**: `test_register_client_handshake_with_quoted_room`, `test_heartbeat_keepalive_handling`, `test_client_registered_event`.

### Bug 4: Fail-Closed Live Smoke Integration Runner & Negative Test Suite
- **Baseline Reality**: Prior smoke testing scripts had false-success paths (could exit code 0 despite timeouts or incomplete turns) and did not validate actual audio payload headers, sample rates, or post-interrupt audio leaks.
- **Fix**:
  1. Re-engineered `tests/integration/test_live_smoke.py` to be strictly fail-closed: exits with code 1 on any assertion failure, timeout, or unexpected exception.
  2. Validates input WAV file format (must be mono 16kHz PCM16) before streaming.
  3. Decodes synthesized audio chunks verifying RIFF header, format, and negotiated sample rate (24000Hz or 44100Hz).
  4. Enforces 0 audio chunks received in `client` mode.
  5. Executes mid-generation interruption while LLM and TTS workers are actively producing output, asserting that no stale audio chunks leak post-interrupt.
  6. Sanitizes event timeline (strips large `audio_b64` payloads) and writes `timeline.json` and concatenated `response_full.wav` to `--output-dir`.
  7. Added comprehensive negative test suite `tests/test_live_smoke_negative.py` (13 tests) verifying failure paths.
- **Verification**: `tests/test_live_smoke_negative.py` (13/13 passing).

### Bug 5: Monotonic Turn ID Guards & Exception Cleanup
- **Baseline Reality**:
  - Regarding the previously reported `full_assistant_text` bug: Inspection of the git history prior to the previous lane's refactoring shows that `full_assistant_text = ""` was originally defined in scope at line 597 of `live_orchestrator.py`. The `NameError: name 'full_assistant_text' is not defined` was an internal scoping regression introduced during that lane's refactor when `final_assistant_text` was adopted without updating the reference in `turn_complete`. In the original unpatched codebase, `full_assistant_text` was defined, though early tool-use exits could leave it empty.
  - When a turn was cancelled, `is_ai_speaking = False` in `finally` and `except` handlers could overwrite state for a newly initiated turn if the cancellation callback executed late.
- **Fix**:
  1. Bound `final_assistant_text` consistently and safely across all normal and tool-use code branches.
  2. Guarded all `is_ai_speaking = False` mutations in `except asyncio.CancelledError`, `except Exception`, and `finally` blocks with `if turn_id == self.turn_id:` to prevent stale turns from corrupting state.
  3. Purged `current_tts_queue` on interrupt and turn completion to prevent orphaned synthesis tasks from locking the worker.
  4. Wrapped session lifecycle in `try: session = LiveSession(...) ... finally: if session: session.close()`.
  5. Added `await websocket.close(code=1011, ...)` on unhandled exceptions in the WebSocket receive loop.
- **Verification**: `test_full_text_bug_regression`, `test_cancellation_mid_tts`, `test_interrupt_purges_tts_queue`, `test_monotonic_turn_id_across_clear_memory`.

---

## 3. Architectural Clarifications & System Limits

1. **Synchronous Worker Pre-warming (`initialize_workers`)**:
   `initialize_workers()` in `live_orchestrator.py` is a synchronous method that acquires the process-wide worker lock (`acquire_worker_lock()`) to pre-warm Kokoro, Whisper, and Gemma subprocesses. It is intentionally synchronous to guarantee deterministic hardware initialization before serving traffic.
2. **Worker Process Serialization**:
   Subprocess communications (Gemma LLM and Kokoro TTS) share standard input/output streams guarded by process-level locks. The bounded queue (`maxsize=8`) and single consumer architecture ensure that only one synthesis task communicates with the worker at any time, preventing stream corruption.
3. **Running Daemon State**:
   Per coordinator constraints, the live system service daemon at `wss://127.0.0.1:8443` was not restarted or redeployed during this corrective pass. All changes are validated via unit and regression suites and will take effect once the coordinator restarts the service.

---

## 4. Standalone Live Integration Smoke Runner

A fail-closed integration script is provided at `tests/integration/test_live_smoke.py`:

```bash
# Run against local live service with real audio tests/test_audio.wav:
python tests/integration/test_live_smoke.py

# Or specify custom endpoint / options:
python tests/integration/test_live_smoke.py \
    --url wss://127.0.0.1:8443/ws/live \
    --audio tests/test_audio.wav \
    --tts-mode server \
    --output-dir /tmp/alveare_smoke_run
```

### Exit Codes & Guardrails
- **Exit 0**: Successful execution of handshake, ping-pong, full audio streaming, transcription, synthesis, decoding, mid-turn interruption, and memory reset.
- **Exit 1**: Any failure, assertion violation, or timeout (>120s). Fail-closed behavior ensures CI/CD reliability.

