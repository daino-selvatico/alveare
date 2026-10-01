"""
Failing regression tests for Backend QA corrective pass.
Tests cover:
1. VAD maximum duration bug (continuous speech not finishing when buffer caps), onset audio dropping, oversized frames truncation.
2. tts_mode mutation during active turn (client->server deadlock >8 chunks, server->client skipping drain).
3. Android register_client with quoted room regressing with unknown-message error.
4. Turn identity guards: old cancelled turn resetting is_ai_speaking=False.
5. Integration smoke fail-closed validation.
"""
import unittest
import asyncio
import json
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient

from runtime.py.live_orchestrator import LiveSession, LiveTurnMetrics
from runtime.py.control_server import app, state


class MockWorkerSlot:
    def __init__(self, model="gemma4-e4b", device="gpu"):
        self.status = "running"
        self.model = model
        self.device = device
        self.host = "127.0.0.1"
        self.port = 8000
        self.process = MagicMock()
        self.process.poll = MagicMock(return_value=None)


class TestLiveBackendQARegression(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.sent_events = []
        async def mock_send(event):
            self.sent_events.append(event)
        self.mock_send = mock_send
        self.session = LiveSession(
            session_id="regression-session-1",
            send_event_fn=mock_send,
            max_context_turns=3
        )

    # -------------------------------------------------------------
    # 1. VAD Maximum Duration & Onset & Oversized Frames
    # -------------------------------------------------------------
    async def test_vad_maximum_duration_triggers_turn_completion(self):
        """
        VAD bug: handle_audio_chunk only increments speech_samples_count if
        speech_buffer + chunk <= max_bytes, then checks duration > max_turn_duration_s.
        This cannot trigger once buffer caps.
        Regression test: continuous speech exceeding max_turn_duration_s MUST trigger
        vad_speech_end and finish turn at bounded threshold without silence.
        """
        self.session.max_turn_duration_s = 1.0  # 1 second max
        self.session.sample_rate = 16000
        self.session.speech_energy_threshold = 0.01
        self.session.noise_floor = 0.005

        # Loud voice frame: 20ms = 320 samples = 640 bytes
        voice_chunk = (np.ones(320, dtype=np.int16) * 8000).tobytes()

        # Send 1.5s of continuous loud speech (75 chunks of 20ms = 1.5s > 1.0s)
        # without any silence
        with patch.object(self.session, "_process_voice_turn", new_callable=AsyncMock):
            for _ in range(75):
                await self.session.handle_audio_chunk(voice_chunk)

            end_events = [e for e in self.sent_events if e.get("event") == "vad_speech_end"]
            self.assertGreaterEqual(
                len(end_events), 1,
                "Continuous speech exceeding max_turn_duration_s must trigger vad_speech_end!"
            )
            # Buffer must be bounded to max_turn_duration_s * sample_rate * 2 bytes
            max_expected_bytes = int(1.0 * 16000 * 2)
            self.assertLessEqual(len(self.session.speech_buffer), max_expected_bytes)

    async def test_vad_onset_audio_not_silently_dropped(self):
        """
        VAD bug: first onset audio before min_speech_frames_onset is currently silently dropped.
        Regression test: audio from initial onset frames must be included in speech_buffer.
        """
        self.session.min_speech_frames_onset = 3
        self.session.speech_energy_threshold = 0.01
        self.session.noise_floor = 0.005

        # Frame 1: distinctive pattern
        frame1 = (np.ones(320, dtype=np.int16) * 5001).tobytes()
        # Frame 2: distinctive pattern
        frame2 = (np.ones(320, dtype=np.int16) * 5002).tobytes()
        # Frame 3: distinctive pattern (triggers onset)
        frame3 = (np.ones(320, dtype=np.int16) * 5003).tobytes()

        await self.session.handle_audio_chunk(frame1)
        self.assertFalse(self.session.in_speech, "Frame 1 alone should not trigger speech onset")
        await self.session.handle_audio_chunk(frame2)
        self.assertFalse(self.session.in_speech, "Frame 2 alone should not trigger speech onset")
        await self.session.handle_audio_chunk(frame3)
        self.assertTrue(self.session.in_speech, "Frame 3 should trigger speech onset")

        # The speech buffer must contain audio from frame 1 and frame 2
        self.assertIn(
            frame1, bytes(self.session.speech_buffer),
            "First onset audio (frame 1) was silently dropped!"
        )
        self.assertIn(
            frame2, bytes(self.session.speech_buffer),
            "Onset audio (frame 2) was silently dropped!"
        )

    async def test_vad_oversized_audio_frame_preserved(self):
        """
        VAD bug: frames > 64000 bytes are currently clamped with chunk[:64000],
        dropping trailing audio.
        Regression test: oversized chunks (e.g. 100,000 bytes) must not drop tail bytes.
        """
        self.session.speech_energy_threshold = 0.01
        self.session.noise_floor = 0.005
        # 100,000 bytes = 50,000 samples
        oversized = (np.ones(50000, dtype=np.int16) * 7000).tobytes()

        await self.session.handle_audio_chunk(oversized)
        self.assertTrue(self.session.in_speech)
        # All 50000 samples should be accounted for (within max duration)
        self.assertEqual(self.session.speech_samples_count, 50000)

    async def test_vad_odd_bytes_boundary_preservation(self):
        """
        PCM payload test: odd byte count across streaming chunks must be preserved
        in _pcm_remainder across chunk boundaries without losing samples.
        """
        self.session.speech_energy_threshold = 0.01
        self.session.noise_floor = 0.005
        self.session.in_speech = True

        # Chunk 1: 641 bytes (odd) -> 640 bytes processed, 1 byte kept in remainder
        chunk1 = (np.ones(320, dtype=np.int16) * 5000).tobytes() + b"\x42"
        await self.session.handle_audio_chunk(chunk1)
        self.assertEqual(self.session._pcm_remainder, b"\x42")
        self.assertEqual(self.session.speech_samples_count, 320)

        # Chunk 2: 641 bytes (odd) -> combined with 1 byte = 642 bytes (321 samples)
        chunk2 = b"\x13" + (np.ones(320, dtype=np.int16) * 5000).tobytes()
        await self.session.handle_audio_chunk(chunk2)
        self.assertEqual(self.session._pcm_remainder, b"")
        self.assertEqual(self.session.speech_samples_count, 641)

    # -------------------------------------------------------------
    # 2. tts_mode Mutation During Active Turn
    # -------------------------------------------------------------
    async def test_tts_mode_switch_client_to_server_no_deadlock(self):
        """
        tts_mode mutable during active turn:
        client->server can enqueue into queue WITHOUT consumer, deadlocking >8 chunks.
        Regression test: starting in client mode and switching to server mid-turn
        must NOT deadlock even when producing 12 chunks (>8 queue capacity).
        """
        self.session.tts_mode = "client"

        # Mock SSE generator with 12 ready sentence chunks
        chunks = [f"Questa e la frase numero {i} del discorso. " for i in range(12)]
        mock_lines = []
        for ch in chunks:
            data = {"choices": [{"delta": {"content": ch}}]}
            mock_lines.append(f"data: {json.dumps(data)}")
        mock_lines.append("data: [DONE]")

        class MockStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for idx, l in enumerate(self.lines):
                    if idx == 2:
                        # Switch tts_mode mid-stream to server
                        self_outer.session.update_settings({"tts_mode": "server"})
                    yield l

        self_outer = self
        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        self.session.turn_id = 1
        voice_pcm = (np.ones(16000, dtype=np.int16) * 6000).tobytes()
        with patch.object(self.session, "stt") as mock_stt:
            mock_stt.transcribe.return_value = {"text": "Parlami del progetto Alveare."}
            with patch("httpx.AsyncClient", return_value=MockClient(MockStreamContext(mock_lines))):
                # Must complete within 3.0s without deadlocking on queue.put
                await asyncio.wait_for(
                    self.session._process_voice_turn(voice_pcm, turn_id=1),
                    timeout=3.0
                )

        llm_chunks = [e for e in self.sent_events if e.get("event") == "llm_chunk"]
        self.assertEqual(len(llm_chunks), 12, "All 12 llm_chunks must be emitted without deadlock")

    async def test_tts_mode_switch_server_to_client_drains_cleanly(self):
        """
        tts_mode mutable during active turn:
        server->client can skip consumer drain, drop queued audio and change ongoing path.
        Regression test: starting in server mode and switching to client mid-turn
        must drain queued audio cleanly without dropping chunks or hanging.
        """
        self.session.tts_mode = "server"
        synthesized_chunks = []

        async def mock_synth(text, chunk_idx, turn_id, metrics, turn_start, tts_mode=None):
            synthesized_chunks.append((chunk_idx, text))
            await asyncio.sleep(0.01)

        self.session._synthesize_and_stream_chunk = mock_synth

        chunks = [
            "Frase breve prima del discorso. ",
            "Seconda frase con spiegazioni importanti. ",
            "Terza frase da ascoltare con calma. ",
            "Quarta frase conclusiva per la prova. "
        ]
        mock_lines = []
        for ch in chunks:
            data = {"choices": [{"delta": {"content": ch}}]}
            mock_lines.append(f"data: {json.dumps(data)}")
        mock_lines.append("data: [DONE]")

        class MockStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for idx, l in enumerate(self.lines):
                    if idx == 1:
                        # Switch tts_mode mid-stream to client
                        self_outer.session.update_settings({"tts_mode": "client"})
                    yield l

        self_outer = self
        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        self.session.turn_id = 1
        voice_pcm = (np.ones(16000, dtype=np.int16) * 6000).tobytes()
        with patch.object(self.session, "stt") as mock_stt:
            mock_stt.transcribe.return_value = {"text": "Parlami del progetto Alveare."}
            with patch("httpx.AsyncClient", return_value=MockClient(MockStreamContext(mock_lines))):
                await asyncio.wait_for(
                    self.session._process_voice_turn(voice_pcm, turn_id=1),
                    timeout=3.0
                )

        # In server mode turn, all 4 chunks should be synthesized without being dropped
        self.assertEqual(len(synthesized_chunks), 4, "Server turn must synthesize all chunks even if mode mutated mid-turn")

    # -------------------------------------------------------------
    # 3. Turn Identity Guards on Cancelled Turn
    # -------------------------------------------------------------
    async def test_cancelled_turn_does_not_clear_new_turn_ai_speaking(self):
        """
        Cancelled turn bug: except asyncio.CancelledError in generate_response_and_stream
        unconditionally sets self.is_ai_speaking = False, clobbering the new turn's state!
        """
        self.session.turn_id = 1
        self.session.is_ai_speaking = True

        # Run generate_response_and_stream for turn 1 with a slow stream
        async def slow_lines():
            yield "data: {\"choices\": [{\"delta\": {\"content\": \"Ciao\"}}]}\n"
            await asyncio.sleep(5.0)
            yield "data: [DONE]\n"

        class MockStreamContext:
            def __init__(self):
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def aiter_lines(self):
                return slow_lines()

        class MockClient:
            def __init__(self):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return MockStreamContext()

        fake_pcm = (np.ones(16000 * 2, dtype=np.int16) * 1000).tobytes()
        with patch.object(self.session, "_synthesize_and_stream_chunk", new_callable=AsyncMock):
            with patch("httpx.AsyncClient", return_value=MockClient()):
                with patch.object(self.session, "stt") as mock_stt:
                    mock_stt.transcribe.return_value = {"text": "Ciao come stai?"}
                    t1 = asyncio.create_task(
                        self.session._process_voice_turn(fake_pcm, turn_id=1)
                    )
                    await asyncio.sleep(0.05)

                    # Barge-in / Turn 2 starts
                    self.session.turn_id = 2
                    self.session.is_ai_speaking = True

                    # Cancel Turn 1
                    t1.cancel()
                    try:
                        await t1
                    except asyncio.CancelledError:
                        pass

                    # Turn 1 must NOT have set is_ai_speaking to False for Turn 2!
                    self.assertTrue(
                        self.session.is_ai_speaking,
                        "Turn 1 cancellation must not reset is_ai_speaking for Turn 2!"
                    )


class TestControlServerAndroidRegisterRegression(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.orig_slots = dict(state.slots)
        state.slots["llm"] = MockWorkerSlot("gemma4-e4b", "gpu")
        state.slots["stt"] = MockWorkerSlot("openai/whisper-large-v3-turbo", "cpu")
        state.slots["tts"] = MockWorkerSlot("hexgrad/Kokoro-82M", "cpu")

    def tearDown(self):
        state.slots.update(self.orig_slots)

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_android_register_client_exact_sequence(self, mock_init):
        """
        Android sends register_client immediately onOpen:
        {"type":"register_client","client_id":"android-satellite","room":"\"salotto\"","mode":"voice_satellite"}
        New backend unknown-message error regresses existing app.
        Accept register_client with bounded validated fields and optional ack,
        do not treat it as session-fatal error.
        """
        with self.client.websocket_connect("/ws/live") as ws:
            # 1. onOpen: server sends live_connected
            conn_raw = ws.receive_text()
            conn = json.loads(conn_raw)
            self.assertEqual(conn.get("event"), "live_connected")

            # 2. Android immediately sends register_client with quoted room
            android_msg = json.dumps({
                "type": "register_client",
                "client_id": "android-satellite",
                "room": '"salotto"',
                "mode": "voice_satellite"
            })
            ws.send_text(android_msg)

            # 3. Read response: MUST NOT be an error event!
            resp_raw = ws.receive_text()
            resp = json.loads(resp_raw)
            self.assertNotEqual(
                resp.get("event"), "error",
                f"register_client must not return error: {resp}"
            )
            # Should be client_registered ack
            self.assertEqual(resp.get("event"), "client_registered")
            self.assertEqual(resp.get("client_id"), "android-satellite")
            # Quoted room should be stripped/sanitized
            self.assertEqual(resp.get("room"), "salotto")
            self.assertEqual(resp.get("status"), "ok")

            # 4. Connection must still be alive and accept pings
            ws.send_text(json.dumps({"type": "ping"}))
            pong_raw = ws.receive_text()
            pong = json.loads(pong_raw)
            self.assertEqual(pong.get("event"), "pong")


if __name__ == "__main__":
    unittest.main()
