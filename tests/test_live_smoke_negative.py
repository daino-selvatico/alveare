"""
Explicitly MOCKED unit and integration tests verifying the fail-closed behavior
of the LiveSmokeRunner against negative edge cases, network timeouts, and protocol violations.

All tests in this file are MOCKED and do NOT interact with any running service daemon.
"""
import os
import io
import wave
import json
import tempfile
import unittest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from tests.integration.test_live_smoke import LiveSmokeRunner


def create_mock_wav(path: str, sr: int = 16000, channels: int = 1, duration_s: float = 0.5) -> str:
    """Creates a temporary valid PCM WAV file for testing."""
    num_samples = int(sr * duration_s)
    samples = (np.ones((num_samples, channels), dtype=np.int16) * 3000)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


class MockWebSocket:
    def __init__(self, incoming_messages: list):
        self.incoming = list(incoming_messages)
        self.sent_messages = []

    async def recv(self):
        if not self.incoming:
            raise TimeoutError("MockWebSocket: no more messages in queue")
        item = self.incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def send(self, data):
        self.sent_messages.append(data)


class TestLiveSmokeNegativeCases(unittest.IsolatedAsyncioTestCase):
    """Explicitly mocked tests for LiveSmokeRunner negative failure scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.valid_wav = os.path.join(self.temp_dir.name, "valid_16k_mono.wav")
        create_mock_wav(self.valid_wav, sr=16000, channels=1, duration_s=0.5)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Missing Audio File
    def test_smoke_fails_on_missing_audio_file(self):
        runner = LiveSmokeRunner(
            url="wss://127.0.0.1:8443/ws/live",
            audio_path="/path/to/nonexistent_audio_file.wav"
        )
        with self.assertRaises(FileNotFoundError):
            runner.validate_input_audio()

    # 2. Stereo Audio Rejected
    def test_smoke_fails_on_stereo_audio(self):
        stereo_wav = os.path.join(self.temp_dir.name, "stereo.wav")
        create_mock_wav(stereo_wav, sr=16000, channels=2)
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=stereo_wav)
        with self.assertRaises(ValueError) as ctx:
            runner.validate_input_audio()
        self.assertIn("mono", str(ctx.exception).lower())

    # 3. Wrong Sample Rate Rejected
    def test_smoke_fails_on_wrong_sample_rate(self):
        sr_44k_wav = os.path.join(self.temp_dir.name, "44k.wav")
        create_mock_wav(sr_44k_wav, sr=44100, channels=1)
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=sr_44k_wav)
        with self.assertRaises(ValueError) as ctx:
            runner.validate_input_audio()
        self.assertIn("16000", str(ctx.exception))

    # 4. Handshake Missing Required Fields
    async def test_smoke_fails_on_handshake_missing_session_id(self):
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=self.valid_wav)
        bad_handshake = json.dumps({
            "event": "live_connected",
            "session_id": "",  # Empty
            "sample_rate": 16000,
            "input_sample_rate": 16000,
            "output_sample_rate": 24000,
            "audio_format": "wav",
            "settings": {"tts_mode": "server"}
        })
        ws = MockWebSocket([bad_handshake])
        with self.assertRaises(AssertionError) as ctx:
            await runner.validate_handshake(ws)
        self.assertIn("session_id", str(ctx.exception))

    # 5. Handshake Sample Rate Mismatch
    async def test_smoke_fails_on_handshake_sample_rate_mismatch(self):
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=self.valid_wav)
        bad_handshake = json.dumps({
            "event": "live_connected",
            "session_id": "session-123",
            "sample_rate": 44100,  # Invalid legacy rate
            "input_sample_rate": 44100,
            "output_sample_rate": 24000,
            "audio_format": "wav",
            "settings": {"tts_mode": "server"}
        })
        ws = MockWebSocket([bad_handshake])
        with self.assertRaises(AssertionError) as ctx:
            await runner.validate_handshake(ws)
        self.assertIn("sample_rate", str(ctx.exception))

    # 6. Ping Mismatch / Non-Pong Response
    async def test_smoke_fails_on_unexpected_ping_response(self):
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=self.valid_wav)
        # Server mistakenly sends something other than pong
        ws = MockWebSocket([json.dumps({"event": "error", "error": "Not pong"})])
        with self.assertRaises(AssertionError) as ctx:
            await runner.validate_ping_pong(ws)
        self.assertIn("pong", str(ctx.exception))

    # 7. Server Error Event During Turn
    async def test_smoke_fails_on_server_error_event(self):
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=self.valid_wav, turn_timeout=5.0)
        ws = MockWebSocket([
            json.dumps({"event": "error", "error": "NPU kernel memory fault"})
        ])
        with self.assertRaises(RuntimeError) as ctx:
            await runner.execute_voice_turn(ws, b"\x00" * 640, 16000)
        self.assertIn("NPU kernel memory fault", str(ctx.exception))

    # 8. Empty Transcript Fails Closed
    async def test_smoke_fails_on_empty_user_transcript(self):
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=self.valid_wav, turn_timeout=5.0)
        ws = MockWebSocket([
            json.dumps({"event": "user_transcript", "text": "   "}),  # Empty
            json.dumps({"event": "turn_complete", "full_text": "Risposta assistente"})
        ])
        with self.assertRaises(AssertionError) as ctx:
            await runner.execute_voice_turn(ws, b"\x00" * 640, 16000)
        self.assertIn("empty transcription", str(ctx.exception).lower())

    # 9. Empty Full Text Fails Closed
    async def test_smoke_fails_on_empty_full_text(self):
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=self.valid_wav, turn_timeout=5.0)
        ws = MockWebSocket([
            json.dumps({"event": "user_transcript", "text": "Grazie a tutti."}),
            json.dumps({"event": "turn_complete", "full_text": ""})  # Empty
        ])
        with self.assertRaises(AssertionError) as ctx:
            await runner.execute_voice_turn(ws, b"\x00" * 640, 16000)
        self.assertIn("empty full_text", str(ctx.exception).lower())

    # 10. Client Mode Receiving Audio Fails Closed
    async def test_smoke_fails_when_client_mode_receives_audio(self):
        runner = LiveSmokeRunner(
            url="wss://127.0.0.1:8443/ws/live",
            audio_path=self.valid_wav,
            tts_mode="client",
            turn_timeout=5.0
        )
        ws = MockWebSocket([
            json.dumps({"event": "user_transcript", "text": "Grazie a tutti."}),
            json.dumps({"event": "audio_chunk", "chunk_idx": 1, "audio_b64": "ABCD"}),
            json.dumps({"event": "turn_complete", "full_text": "Risposta"})
        ])
        with self.assertRaises(AssertionError) as ctx:
            await runner.execute_voice_turn(ws, b"\x00" * 640, 16000)
        self.assertIn("tts_mode is 'client'", str(ctx.exception))

    # 11. Server Mode Receiving Corrupt Audio Fails Closed
    async def test_smoke_fails_when_server_mode_receives_corrupt_wav(self):
        runner = LiveSmokeRunner(
            url="wss://127.0.0.1:8443/ws/live",
            audio_path=self.valid_wav,
            tts_mode="server",
            turn_timeout=5.0
        )
        import base64
        corrupt_b64 = base64.b64encode(b"THIS_IS_NOT_A_VALID_WAV_HEADER_DATA_1234567890").decode("utf-8")
        ws = MockWebSocket([
            json.dumps({"event": "user_transcript", "text": "Grazie a tutti."}),
            json.dumps({"event": "audio_chunk", "chunk_idx": 1, "audio_b64": corrupt_b64})
        ])
        with self.assertRaises(AssertionError) as ctx:
            await runner.execute_voice_turn(ws, b"\x00" * 640, 16000)
        self.assertIn("not decodable wav audio", str(ctx.exception).lower())

    # 12. Post-Interrupt Stale Audio Leak Detected
    async def test_smoke_fails_when_stale_audio_arrives_after_interrupt(self):
        runner = LiveSmokeRunner(url="wss://127.0.0.1:8443/ws/live", audio_path=self.valid_wav)
        ws = MockWebSocket([
            json.dumps({"event": "llm_chunk", "chunk_idx": 1, "text": "Prima frase", "turn_id": 5}),
            json.dumps({"event": "interrupted", "turn_id": 6}),
            # Stale audio chunk arriving for turn 5 after interrupt
            json.dumps({"event": "audio_chunk", "chunk_idx": 1, "turn_id": 5, "audio_b64": "AAA="})
        ])
        with self.assertRaises(AssertionError) as ctx:
            await runner.execute_mid_generation_interrupt(ws, b"\x00" * 640, 16000)
        self.assertIn("stale audio_chunk", str(ctx.exception).lower())

    # 13. Timeline Sanitizes Base64 and Output Dir Writes File
    def test_smoke_timeline_sanitizes_base64_and_saves_output_dir(self):
        out_dir = os.path.join(self.temp_dir.name, "out_test")
        runner = LiveSmokeRunner(
            url="wss://127.0.0.1:8443/ws/live",
            audio_path=self.valid_wav,
            output_dir=out_dir
        )
        # Log event with large base64
        runner.log_event("audio_chunk", {
            "chunk_idx": 1,
            "audio_b64": "A" * 5000,
            "sample_rate": 24000
        })
        runner.save_output_artifacts()

        timeline_file = os.path.join(out_dir, "timeline.json")
        self.assertTrue(os.path.exists(timeline_file))
        with open(timeline_file, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertIn("<stripped:", data[0]["payload"]["audio_b64"])
        self.assertNotIn("A" * 5000, data[0]["payload"]["audio_b64"])


if __name__ == "__main__":
    unittest.main()
