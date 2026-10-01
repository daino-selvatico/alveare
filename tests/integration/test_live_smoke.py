#!/usr/bin/env python3
"""
Live Integration Smoke & Benchmark Runner for Alveare.

Features:
- Fail-closed execution: any validation failure exits with non-zero code.
- Generous 120s turn completion timeout.
- Validates input 16kHz mono PCM WAV file prior to streaming.
- Verifies live_connected handshake wire metadata (input_sr=16000, output_sr=24000/44100, format='wav', tts_mode).
- Validates ping/pong matching response.
- Verifies nonempty transcript and full_text.
- In server mode: decodes and verifies actual WAV sample rate and data for each audio_chunk.
- In client mode: verifies no audio_chunk is emitted by server.
- Real interruption occurs WHILE synthesis/generation is active, verifying no post-interrupt stale audio.
- Saves sanitized event timeline (base64 stripped) and received WAVs to optional --output-dir.

Usage:
    python tests/integration/test_live_smoke.py [--url wss://127.0.0.1:8443/ws/live] [--audio tests/test_audio.wav] [--tts-mode server] [--output-dir out/]
"""
import os
import io
import sys
import time
import json
import ssl
import wave
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import soundfile as sf
import websockets


def parse_args():
    parser = argparse.ArgumentParser(description="Alveare Live Integration Smoke & Benchmark")
    parser.add_argument("--url", default="wss://127.0.0.1:8443/ws/live", help="Live WebSocket URL")
    parser.add_argument("--audio", default="tests/test_audio.wav", help="Path to 16kHz mono test WAV file")
    parser.add_argument("--tts-mode", default="server", choices=["server", "client"], help="TTS mode to request")
    parser.add_argument("--output-dir", default=None, help="Optional directory to save timeline.json and received WAVs")
    parser.add_argument("--skip-interrupt", action="store_true", help="Skip the mid-generation interruption test")
    parser.add_argument("--turn-timeout", type=float, default=120.0, help="Turn completion timeout in seconds (default 120s)")
    return parser.parse_args()


class LiveSmokeRunner:
    """
    Fail-closed smoke tester for Alveare Live WebSocket endpoint.
    Strictly validates all wire contracts, data payloads, and event sequences.
    """

    def __init__(
        self,
        url: str,
        audio_path: str,
        tts_mode: str = "server",
        output_dir: Optional[str] = None,
        skip_interrupt: bool = False,
        turn_timeout: float = 120.0
    ):
        self.raw_url = url
        self.url = f"{url}?tts_mode={tts_mode}" if "?" not in url else f"{url}&tts_mode={tts_mode}"
        self.audio_path = audio_path
        self.tts_mode = tts_mode
        self.output_dir = output_dir
        self.skip_interrupt = skip_interrupt
        self.turn_timeout = turn_timeout

        self.events_timeline: List[Dict[str, Any]] = []
        self.t0: float = 0.0
        self.expected_output_sr: int = 24000
        self.saved_audio_chunks: List[bytes] = []

    def validate_input_audio(self) -> Tuple[bytes, int]:
        """
        Fail-closed validation of input audio file:
        Must exist, must be mono (1 channel), and must be 16000Hz PCM.
        """
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"Required test audio file not found: '{self.audio_path}'")

        try:
            data, sr = sf.read(self.audio_path, dtype="int16")
        except Exception as e:
            raise ValueError(f"Failed to read input audio file '{self.audio_path}': {e}") from e

        if len(data) == 0:
            raise ValueError(f"Audio file '{self.audio_path}' contains zero samples")

        channels = 1 if data.ndim == 1 else data.shape[1]
        if channels != 1:
            raise ValueError(f"Input audio must be mono (1 channel), but got {channels} channels in '{self.audio_path}'")

        if sr != 16000:
            raise ValueError(f"Input audio must have sample_rate=16000 Hz, but got {sr} Hz in '{self.audio_path}'")

        raw_pcm = data.tobytes()
        return raw_pcm, sr

    def log_event(self, event_name: str, payload: Dict[str, Any]):
        """Logs an event to the internal timeline, sanitizing base64 payloads to keep logs readable."""
        elapsed_s = time.perf_counter() - self.t0 if self.t0 > 0 else 0.0

        clean_payload = dict(payload)
        if "audio_b64" in clean_payload:
            b64_len = len(clean_payload["audio_b64"]) if isinstance(clean_payload["audio_b64"], str) else 0
            clean_payload["audio_b64"] = f"<stripped: {b64_len} chars base64>"
        if "data" in clean_payload and isinstance(clean_payload["data"], str) and len(clean_payload["data"]) > 100:
            clean_payload["data"] = f"<stripped: {len(clean_payload['data'])} chars>"

        self.events_timeline.append({
            "rel_time_s": round(elapsed_s, 3),
            "event": event_name,
            "payload": clean_payload
        })

        extra = ""
        if event_name == "live_connected":
            extra = (
                f"in_sr={payload.get('input_sample_rate')} out_sr={payload.get('output_sample_rate')} "
                f"fmt={payload.get('audio_format')} tts_mode={payload.get('settings', {}).get('tts_mode')}"
            )
        elif event_name == "user_transcript":
            extra = f"text='{payload.get('text')}' stt_lat={payload.get('stt_latency_ms')}ms"
        elif event_name == "ttft":
            extra = f"ttft={payload.get('ttft_ms')}ms"
        elif event_name == "llm_chunk":
            extra = f"chunk={payload.get('chunk_idx')} text='{payload.get('text')}'"
        elif event_name == "audio_chunk":
            extra = f"chunk={payload.get('chunk_idx')} sr={payload.get('sample_rate')} dur={payload.get('duration_sec')}s"
        elif event_name == "turn_complete":
            extra = f"full_text='{payload.get('full_text')}'"
        elif event_name == "error":
            extra = f"error='{payload.get('error')}'"
        elif event_name in ("interrupted", "memory_cleared", "pong", "client_registered"):
            extra = f"turn_id={payload.get('turn_id', '-')}"

        print(f"[{elapsed_s:6.3f}s] {event_name.upper():<18} {extra}")

    async def validate_handshake(self, ws) -> Dict[str, Any]:
        """Awaits live_connected event and strictly validates all required handshake wire fields."""
        raw_msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
        try:
            payload = json.loads(raw_msg)
        except Exception as e:
            raise AssertionError(f"Initial message is not valid JSON: {raw_msg}") from e

        ev = payload.get("event")
        self.log_event(ev or "unknown", payload)

        if ev != "live_connected":
            raise AssertionError(f"Handshake failed: expected 'live_connected', got '{ev}'")

        if not payload.get("session_id"):
            raise AssertionError("Handshake failed: missing non-empty 'session_id'")

        legacy_sr = payload.get("sample_rate")
        input_sr = payload.get("input_sample_rate")
        if legacy_sr != 16000:
            raise AssertionError(f"Handshake failed: legacy sample_rate must be 16000, got {legacy_sr}")
        if input_sr != 16000:
            raise AssertionError(f"Handshake failed: input_sample_rate must be 16000, got {input_sr}")

        output_sr = payload.get("output_sample_rate")
        if not output_sr or not isinstance(output_sr, int) or output_sr <= 0:
            raise AssertionError(f"Handshake failed: output_sample_rate must be positive integer, got {output_sr}")
        self.expected_output_sr = output_sr

        fmt = payload.get("audio_format")
        if fmt != "wav":
            raise AssertionError(f"Handshake failed: audio_format must be 'wav', got '{fmt}'")

        settings = payload.get("settings", {})
        server_tts_mode = settings.get("tts_mode")
        if server_tts_mode != self.tts_mode:
            raise AssertionError(
                f"Handshake failed: server tts_mode '{server_tts_mode}' does not match requested '{self.tts_mode}'"
            )

        return payload

    async def validate_ping_pong(self, ws):
        """Sends ping and strictly verifies matching pong response."""
        t_ping = time.perf_counter()
        await ws.send(json.dumps({"type": "ping"}))
        resp_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        try:
            resp = json.loads(resp_raw)
        except Exception as e:
            raise AssertionError(f"Ping response is not valid JSON: {resp_raw}") from e

        ev = resp.get("event")
        self.log_event(ev or "unknown", resp)
        if ev != "pong":
            raise AssertionError(f"Ping failed: expected 'pong', got '{ev}'")

        rtt_ms = (time.perf_counter() - t_ping) * 1000.0
        print(f"[INFO] Ping/Pong RTT: {rtt_ms:.2f}ms")

    async def stream_audio_chunks(self, ws, raw_bytes: bytes, sr: int):
        """Streams audio in 20ms chunks followed by trailing silence and flush."""
        chunk_bytes_len = int(sr * 0.02 * 2)  # 640 bytes for 16kHz int16
        for offset in range(0, len(raw_bytes), chunk_bytes_len):
            chunk = raw_bytes[offset:offset + chunk_bytes_len]
            await ws.send(chunk)
            await asyncio.sleep(0.02)

        # Send trailing silence (0.4s) to allow VAD to detect turn completion naturally
        silence = b"\x00" * int(sr * 0.4 * 2)
        for offset in range(0, len(silence), chunk_bytes_len):
            await ws.send(silence[offset:offset + chunk_bytes_len])
            await asyncio.sleep(0.02)

        # Send flush to guarantee turn end
        await ws.send(json.dumps({"type": "end_of_speech"}))

    def validate_wav_chunk(self, raw_wav: bytes, chunk_idx: int) -> float:
        """Decodes raw WAV chunk and verifies valid header, sample rate, and non-empty samples."""
        if not raw_wav or len(raw_wav) < 44:
            raise AssertionError(f"Audio chunk {chunk_idx} has invalid WAV byte length: {len(raw_wav)}")

        bio = io.BytesIO(raw_wav)
        try:
            with wave.open(bio, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.getnframes()
                if sr != self.expected_output_sr:
                    raise AssertionError(
                        f"Chunk {chunk_idx} sample rate {sr} does not match expected {self.expected_output_sr}"
                    )
                if frames == 0:
                    raise AssertionError(f"Chunk {chunk_idx} has 0 frames in WAV header")
                dur_s = frames / float(sr)
                return dur_s
        except Exception as e:
            raise AssertionError(f"Chunk {chunk_idx} is not decodable WAV audio: {e}") from e

    async def execute_voice_turn(self, ws, raw_bytes: bytes, sr: int) -> Dict[str, Any]:
        """
        Executes a complete voice turn with generous bounded timeout (120s).
        Enforces:
        - Non-empty transcription
        - Non-empty full_text
        - Server mode: at least 1 decodable WAV chunk matching sample rate
        - Client mode: 0 audio_chunk events
        """
        print(f"\n[INFO] Streaming {len(raw_bytes)} bytes audio ({len(raw_bytes)/(sr*2):.2f}s) to WebSocket...")
        transcript = ""
        full_text = ""
        turn_complete_event = None
        audio_chunks_count = 0
        total_audio_sec = 0.0

        async def listen_loop():
            nonlocal transcript, full_text, turn_complete_event, audio_chunks_count, total_audio_sec
            while True:
                msg_raw = await ws.recv()
                try:
                    payload = json.loads(msg_raw)
                except Exception as e:
                    raise AssertionError(f"Received non-JSON message during turn: {msg_raw}") from e

                ev = payload.get("event")
                self.log_event(ev or "unknown", payload)

                if ev == "error":
                    raise RuntimeError(f"Server reported error during turn: {payload.get('error')}")

                elif ev == "user_transcript":
                    transcript = payload.get("text", "").strip()

                elif ev == "audio_chunk":
                    if self.tts_mode == "client":
                        raise AssertionError(
                            "Violation: Received 'audio_chunk' event from server while tts_mode is 'client'!"
                        )
                    audio_b64 = payload.get("audio_b64", "")
                    if not audio_b64:
                        raise AssertionError(f"audio_chunk {payload.get('chunk_idx')} missing audio_b64 data")

                    import base64
                    try:
                        raw_wav = base64.b64decode(audio_b64)
                    except Exception as e:
                        raise AssertionError(f"audio_chunk base64 decode failed: {e}") from e

                    chunk_idx = payload.get("chunk_idx", audio_chunks_count + 1)
                    dur_s = self.validate_wav_chunk(raw_wav, chunk_idx)
                    audio_chunks_count += 1
                    total_audio_sec += dur_s
                    self.saved_audio_chunks.append(raw_wav)

                    if self.output_dir:
                        os.makedirs(self.output_dir, exist_ok=True)
                        chunk_path = Path(self.output_dir) / f"chunk_{chunk_idx:03d}.wav"
                        with open(chunk_path, "wb") as f:
                            f.write(raw_wav)

                elif ev == "turn_complete":
                    full_text = payload.get("full_text", "").strip()
                    turn_complete_event = payload
                    break

        # Run streaming and listening concurrently under generous bounded timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    self.stream_audio_chunks(ws, raw_bytes, sr),
                    listen_loop()
                ),
                timeout=self.turn_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Turn execution timed out after {self.turn_timeout}s without turn_complete!")

        # Strict fail-closed assertions
        if not transcript:
            raise AssertionError("Turn failed: STT returned empty transcription!")
        if not full_text:
            raise AssertionError("Turn failed: turn_complete contained empty full_text!")
        if self.tts_mode == "server":
            if audio_chunks_count == 0:
                raise AssertionError("Turn failed: Server mode completed with 0 audio_chunk events!")
            if total_audio_sec <= 0:
                raise AssertionError("Turn failed: Total synthesized audio duration was 0s!")
        elif self.tts_mode == "client":
            if audio_chunks_count > 0:
                raise AssertionError(f"Turn failed: Client mode received {audio_chunks_count} audio chunks!")

        print("\n=== Verified Turn Summary ===")
        print(f"Transcription:      '{transcript}'")
        print(f"Full text:          '{full_text}'")
        print(f"Audio chunks count: {audio_chunks_count}")
        print(f"Total audio length: {total_audio_sec:.2f}s")
        if turn_complete_event:
            metrics = turn_complete_event.get("metrics", {})
            print(f"E2E Latency:        {metrics.get('e2e_latency_ms', '-')}ms")
            print(f"TTFT:               {metrics.get('ttft_ms', '-')}ms")
            print(f"TTFA:               {metrics.get('ttfa_ms', '-')}ms")
            print(f"STT Latency:        {metrics.get('stt_latency_ms', '-')}ms")

        return {
            "transcript": transcript,
            "full_text": full_text,
            "audio_chunks": audio_chunks_count,
            "total_audio_sec": total_audio_sec,
            "turn_complete": turn_complete_event
        }

    async def execute_mid_generation_interrupt(self, ws, raw_bytes: bytes, sr: int):
        """
        Exercises real interruption WHILE LLM synthesis/generation is active.
        Sends interrupt upon the first text or audio chunk, then validates that
        an interrupted event is returned and NO stale audio chunks arrive afterwards.
        """
        print("\n[INFO] Starting mid-generation interruption test...")
        interrupted_turn_id = None
        interrupt_sent = False

        # Start streaming turn 2
        stream_task = asyncio.create_task(self.stream_audio_chunks(ws, raw_bytes, sr))

        # Listen until first generation event arrives
        try:
            while True:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
                payload = json.loads(msg_raw)
                ev = payload.get("event")
                self.log_event(ev or "unknown", payload)

                if ev in ("llm_chunk", "ttft", "audio_chunk"):
                    interrupted_turn_id = payload.get("turn_id")
                    # Real interrupt triggered while active!
                    await ws.send(json.dumps({"type": "interrupt"}))
                    interrupt_sent = True
                    break
                elif ev == "turn_complete":
                    break
        finally:
            if not stream_task.done():
                stream_task.cancel()

        if not interrupt_sent:
            raise AssertionError("Interruption test could not trigger: no generation event was received to interrupt!")

        # Await interrupted event
        interr_resp_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        interr_resp = json.loads(interr_resp_raw)
        self.log_event(interr_resp.get("event", "unknown"), interr_resp)

        if interr_resp.get("event") != "interrupted":
            raise AssertionError(f"Expected 'interrupted' event, got: '{interr_resp.get('event')}'")

        # Drain check: verify no stale audio or text for the interrupted turn arrives
        drain_end = time.perf_counter() + 0.5
        while time.perf_counter() < drain_end:
            try:
                stale_msg = await asyncio.wait_for(ws.recv(), timeout=0.2)
                stale_p = json.loads(stale_msg)
                stale_ev = stale_p.get("event")
                self.log_event(stale_ev or "unknown", stale_p)
                if stale_ev in ("audio_chunk", "llm_chunk") and stale_p.get("turn_id") == interrupted_turn_id:
                    raise AssertionError(
                        f"Leak detected: received stale {stale_ev} for turn {interrupted_turn_id} after interruption!"
                    )
            except asyncio.TimeoutError:
                break

        print("[INFO] Mid-generation interruption verified cleanly (no stale audio leaked).")

    async def validate_memory_clear(self, ws):
        """Sends clear_memory and validates matching memory_cleared response."""
        print("\n[INFO] Validating clear_memory...")
        await ws.send(json.dumps({"type": "clear_memory"}))
        resp_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        resp = json.loads(resp_raw)
        ev = resp.get("event")
        self.log_event(ev or "unknown", resp)
        if ev != "memory_cleared":
            raise AssertionError(f"Expected 'memory_cleared' event, got: '{ev}'")

    def save_output_artifacts(self):
        """Saves sanitized timeline JSON and concatenated WAV if output_dir provided."""
        if not self.output_dir:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        timeline_path = Path(self.output_dir) / "timeline.json"
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(self.events_timeline, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved event timeline to {timeline_path}")

        # Combine all received WAV chunks into complete response audio
        if self.saved_audio_chunks:
            all_pcm = []
            for w in self.saved_audio_chunks:
                try:
                    with wave.open(io.BytesIO(w), "rb") as wf:
                        all_pcm.append(wf.readframes(wf.getnframes()))
                except Exception:
                    pass
            if all_pcm:
                combined_path = Path(self.output_dir) / "response_full.wav"
                with wave.open(str(combined_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.expected_output_sr)
                    wf.writeframes(b"".join(all_pcm))
                print(f"[INFO] Saved combined response audio to {combined_path}")

    async def run(self):
        """Main runner method: fail-closed sequence executing full suite."""
        print(f"=== Starting Alveare Live Smoke Test against {self.url} ===")
        self.t0 = time.perf_counter()

        # 1. Fail-closed audio validation before any connection
        raw_bytes, sr = self.validate_input_audio()
        print(f"[INFO] Input audio valid: {self.audio_path} ({len(raw_bytes)/(sr*2):.2f}s at {sr}Hz mono PCM)")

        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        async with websockets.connect(self.url, ssl=ssl_ctx) as ws:
            # 2. Handshake validation
            await self.validate_handshake(ws)

            # 3. Ping/Pong latency check
            await self.validate_ping_pong(ws)

            # 4. Turn execution with fail-closed bounds
            await self.execute_voice_turn(ws, raw_bytes, sr)

            # 5. Mid-generation interruption test
            if not self.skip_interrupt:
                await self.execute_mid_generation_interrupt(ws, raw_bytes, sr)

            # 6. Memory reset check
            await self.validate_memory_clear(ws)

        # 7. Save artifacts
        self.save_output_artifacts()

        elapsed = time.perf_counter() - self.t0
        print(f"\n=== Live Smoke Test Completed Successfully ({elapsed:.2f}s) ===")


def main():
    args = parse_args()
    runner = LiveSmokeRunner(
        url=args.url,
        audio_path=args.audio,
        tts_mode=args.tts_mode,
        output_dir=args.output_dir,
        skip_interrupt=args.skip_interrupt,
        turn_timeout=args.turn_timeout
    )
    try:
        asyncio.run(runner.run())
    except Exception as e:
        print(f"\n[FATAL SMOKE FAILURE] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
