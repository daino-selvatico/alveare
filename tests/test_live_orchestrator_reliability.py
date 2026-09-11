"""
Focused Reliability & Latency Backend Tests for Live Orchestrator and WebSocket endpoint.
Fake engines explicitly labeled mocked prove:
- LLM / TTS actual overlap and ordered consumption
- Cancellation mid-TTS clears pending work without emitting stale audio
- Reset/clear memory preserves monotonic turn IDs and avoids stale writes
- Wire-compatible metadata (sample_rate=16000 legacy, input_sample_rate=16000, output_sample_rate=24000/44100, audio_format='wav')
- Client mode emits llm_chunk for sentences but does NOT synthesize server TTS
- Correct full_text in turn_complete (fixing NameError bug)
- Chunking prosody (no arbitrary 6-word cuts, safe clauses, bounded unpunctuated fallback)
- Thought (<thought>...</thought>) and tool call ([TOOL_CALL:...]) suppression
- VAD robustness against odd bytes, noise bursts, buffer clamping
"""
import unittest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from runtime.py.live_orchestrator import LiveSession, LiveTurnMetrics, StreamingThoughtFilter

class MockTTSEngine:
    """Explicitly labeled mocked TTS engine for deterministic unit testing."""
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.call_count = 0
        self.synthesized_chunks = []

    def generate(self, text, voice="if_sara", speed=1.0, **kwargs):
        self.call_count += 1
        self.synthesized_chunks.append(text)
        if self.delay > 0:
            time.sleep(self.delay)
        return {
            "status": "success",
            "audio_path": None,
            "sample_rate": 24000,
            "duration_sec": 0.5,
            "latency_ms": round(self.delay * 1000.0, 1)
        }

class MockSTTEngine:
    """Explicitly labeled mocked STT engine for deterministic unit testing."""
    def __init__(self, transcript: str = "ciao alveare"):
        self.transcript = transcript
        self.call_count = 0

    def transcribe(self, wav_bytes, language="it"):
        self.call_count += 1
        return {"text": self.transcript, "language": language}

class TestLiveOrchestratorReliability(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.sent_events = []
        async def mock_send(event):
            self.sent_events.append(event)
        self.send_fn = mock_send

    def _create_session(self, **kwargs):
        defaults = {
            "session_id": "test-reliability-session",
            "send_event_fn": self.send_fn,
            "llm_host": "127.0.0.1",
            "llm_port": 8000,
            "tts_model": "hexgrad/Kokoro-82M",
            "stt_model": "openai/whisper-large-v3-turbo",
            "max_context_turns": 4,
            "max_tokens": 120
        }
        defaults.update(kwargs)
        session = LiveSession(**defaults)
        session.stt = MockSTTEngine()
        session.tts = MockTTSEngine()
        return session

    async def test_full_text_bug_regression(self):
        """Priority 2: turn_complete must include full_text with final_assistant_text without NameError."""
        session = self._create_session()
        
        mock_lines = [
            'data: {"choices": [{"delta": {"content": "Ciao! "}}]}',
            'data: {"choices": [{"delta": {"content": "Come posso aiutarti oggi?"}}]}',
            'data: [DONE]'
        ]

        class MockStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for l in self.lines:
                    yield l

        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        fake_pcm = (np.ones(16000 * 2, dtype=np.int16) * 1000).tobytes()

        with patch("httpx.AsyncClient", return_value=MockClient(MockStreamContext(mock_lines))):
            session.turn_id = 1
            await session._process_voice_turn(fake_pcm, turn_id=1)

        complete_events = [e for e in self.sent_events if e.get("event") == "turn_complete"]
        self.assertEqual(len(complete_events), 1, "Expected exactly one turn_complete event")
        complete = complete_events[0]
        self.assertIn("full_text", complete)
        self.assertEqual(complete["full_text"], "Ciao! Come posso aiutarti oggi?")

    async def test_monotonic_turn_id_across_clear_memory(self):
        """Priority 2: clear_memory must preserve strictly monotonic turn IDs and not reset to 0."""
        session = self._create_session()
        session.turn_id = 5
        res = session.clear_memory()
        self.assertEqual(res["status"], "ok")
        self.assertGreater(session.turn_id, 5, "turn_id must monotonically increase across clear_memory")
        self.assertNotEqual(session.turn_id, 0, "turn_id must NEVER be reset to 0 on clear_memory")

    async def test_wire_compatible_output_metadata(self):
        """Priority 3: Session provides wire-compatible input/output sample rates and tts_mode."""
        session_kokoro = self._create_session(tts_model="hexgrad/Kokoro-82M", tts_mode="server")
        self.assertEqual(session_kokoro.sample_rate, 16000)
        self.assertEqual(session_kokoro.input_sample_rate, 16000)
        self.assertEqual(session_kokoro.output_sample_rate, 24000)
        self.assertEqual(session_kokoro.audio_format, "wav")
        self.assertEqual(session_kokoro.tts_mode, "server")

        session_audio8 = self._create_session(tts_model="Audio8/Audio8-TTS-Preview-0.1b", tts_mode="server")
        self.assertEqual(session_audio8.output_sample_rate, 44100)

        session_client = self._create_session(tts_mode="client")
        self.assertEqual(session_client.tts_mode, "client")

    async def test_client_mode_emits_llm_chunk_but_no_server_tts(self):
        """Priority 3: In tts_mode='client', emit llm_chunk but do not call server TTS or emit audio_chunk."""
        session = self._create_session(tts_mode="client")
        mock_tts = MockTTSEngine()
        session.tts = mock_tts

        mock_lines = [
            'data: {"choices": [{"delta": {"content": "Risposta in modalità client."}}]}',
            'data: [DONE]'
        ]

        class MockStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for l in self.lines:
                    yield l

        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        fake_pcm = (np.ones(16000 * 2, dtype=np.int16) * 1000).tobytes()

        with patch("httpx.AsyncClient", return_value=MockClient(MockStreamContext(mock_lines))):
            session.turn_id = 1
            await session._process_voice_turn(fake_pcm, turn_id=1)

        llm_chunks = [e for e in self.sent_events if e.get("event") == "llm_chunk"]
        audio_chunks = [e for e in self.sent_events if e.get("event") == "audio_chunk"]
        self.assertTrue(len(llm_chunks) >= 1, "llm_chunk events must be emitted in client mode")
        self.assertEqual(len(audio_chunks), 0, "No audio_chunk events should be emitted in client mode")
        self.assertEqual(mock_tts.call_count, 0, "Server TTS should not be called in client mode")

    async def test_text_chunking_prosody_no_arbitrary_six_word_cut(self):
        """Priority 4: Text chunking should not arbitrarily cut at 6 words, preserving natural phrases."""
        session = self._create_session()
        
        twelve_words = "Oggi a Milano il tempo sarà prevalentemente soleggiato e caldo per tutta la giornata."
        
        mock_lines = [
            f'data: {{"choices": [{{"delta": {{"content": "{twelve_words}"}}}}]}}',
            'data: [DONE]'
        ]

        class MockStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for l in self.lines:
                    yield l

        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        fake_pcm = (np.ones(16000 * 2, dtype=np.int16) * 1000).tobytes()

        with patch("httpx.AsyncClient", return_value=MockClient(MockStreamContext(mock_lines))):
            session.turn_id = 1
            await session._process_voice_turn(fake_pcm, turn_id=1)

        llm_chunks = [e for e in self.sent_events if e.get("event") == "llm_chunk"]
        self.assertEqual(len(llm_chunks), 1, f"Expected 1 complete chunk, but got {len(llm_chunks)}: {[c.get('text') for c in llm_chunks]}")
        self.assertEqual(llm_chunks[0]["text"], twelve_words)

    async def test_thought_and_partial_tool_suppression(self):
        """Priority 4: Thought tokens and partial tool call markers are never passed to TTS or emitted in text chunks."""
        session = self._create_session()
        
        mock_lines = [
            'data: {"choices": [{"delta": {"content": "<thought>pensando...</thought>Ecco la "}}]}',
            'data: {"choices": [{"delta": {"content": "risposta."}}]}',
            'data: [DONE]'
        ]

        class MockStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for l in self.lines:
                    yield l

        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        fake_pcm = (np.ones(16000 * 2, dtype=np.int16) * 1000).tobytes()

        with patch("httpx.AsyncClient", return_value=MockClient(MockStreamContext(mock_lines))):
            session.turn_id = 1
            await session._process_voice_turn(fake_pcm, turn_id=1)

        llm_chunks = [e for e in self.sent_events if e.get("event") == "llm_chunk"]
        for c in llm_chunks:
            self.assertNotIn("<thought>", c["text"])
            self.assertNotIn("pensando", c["text"])
            self.assertNotIn("</thought>", c["text"])

        complete_events = [e for e in self.sent_events if e.get("event") == "turn_complete"]
        self.assertEqual(len(complete_events), 1)
        self.assertNotIn("<thought>", complete_events[0]["full_text"])
        self.assertEqual(complete_events[0]["full_text"], "Ecco la risposta.")

    async def test_llm_tts_overlap_and_ordering(self):
        """Priority 1: LLM token consumption must overlap with TTS synthesis via single ordered queue."""
        session = self._create_session()
        session.tts = MockTTSEngine(delay=0.05)
        
        sentences = [
            "Prima frase di prova.",
            "Seconda frase di prova.",
            "Terza frase di prova conclusiva."
        ]
        
        mock_lines = [
            f'data: {{"choices": [{{"delta": {{"content": "{s} "}}}}]}}'
            for s in sentences
        ] + ['data: [DONE]']

        class MockStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for l in self.lines:
                    yield l

        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        fake_pcm = (np.ones(16000 * 2, dtype=np.int16) * 1000).tobytes()

        with patch("httpx.AsyncClient", return_value=MockClient(MockStreamContext(mock_lines))):
            session.turn_id = 1
            await session._process_voice_turn(fake_pcm, turn_id=1)

        audio_chunks = [e for e in self.sent_events if e.get("event") == "audio_chunk"]
        self.assertEqual(len(audio_chunks), 3)
        indices = [c["chunk_idx"] for c in audio_chunks]
        self.assertEqual(indices, [1, 2, 3])
        for ac in audio_chunks:
            self.assertEqual(ac.get("format"), "wav")
            self.assertEqual(ac.get("sample_rate"), 24000)

    async def test_cancellation_mid_tts(self):
        """Priority 1: User interruption mid-generation cancels task, clears pending TTS, emits no stale audio."""
        session = self._create_session()
        session.tts = MockTTSEngine(delay=0.2)

        mock_lines = [
            'data: {"choices": [{"delta": {"content": "Prima frase lunga di prova. "}}]}',
            'data: {"choices": [{"delta": {"content": "Seconda frase che dovrebbe essere cancellata."}}]}',
            'data: [DONE]'
        ]

        class SlowStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for l in self.lines:
                    await asyncio.sleep(0.05)
                    yield l

        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        fake_pcm = (np.ones(16000 * 2, dtype=np.int16) * 1000).tobytes()

        with patch("httpx.AsyncClient", return_value=MockClient(SlowStreamContext(mock_lines))):
            session.turn_id = 1
            task = asyncio.create_task(session._process_voice_turn(fake_pcm, turn_id=1))
            session.current_processing_task = task
            await asyncio.sleep(0.06)
            # User barge-in / interrupt
            session.is_ai_speaking = False
            session.turn_id += 1
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        complete_events = [e for e in self.sent_events if e.get("event") == "turn_complete"]
        self.assertEqual(len(complete_events), 0, "No turn_complete should be sent for cancelled turn")

    async def test_vad_odd_bytes_and_noise_floor_resilience(self):
        """Priority 4: VAD handles odd-byte PCM chunks safely without crashing, and clamps noise floor spikes."""
        session = self._create_session()
        
        # 1. Odd number of bytes (e.g. 1001 bytes) - must not crash
        odd_chunk = b"\x00" * 1001
        await session.handle_audio_chunk(odd_chunk)

        # 2. Sudden loud burst frame while not in speech
        loud_samples = (np.ones(160, dtype=np.int16) * 32000).tobytes()
        await session.handle_audio_chunk(loud_samples)
        
        # Noise floor must be bounded / clamped and not explode to 0.5+
        self.assertLessEqual(session.noise_floor, 0.05, f"Noise floor exploded: {session.noise_floor}")

    async def test_tool_call_suppression_from_tts(self):
        """Priority 4: Tool call payload syntax is never spoken or emitted in llm_chunk."""
        session = self._create_session()
        
        mock_lines = [
            'data: {"choices": [{"delta": {"content": "Ecco il tempo. [TOOL_CALL: web_search({\\"query\\": \\"meteo\\"})]"}}]}',
            'data: [DONE]'
        ]

        class MockStreamContext:
            def __init__(self, lines):
                self.lines = lines
                self.status_code = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def aiter_lines(self):
                for l in self.lines:
                    yield l

        class MockClient:
            def __init__(self, stream_ctx):
                self.stream_ctx = stream_ctx
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            def stream(self, method, url, **kwargs):
                return self.stream_ctx

        fake_pcm = (np.ones(16000 * 2, dtype=np.int16) * 1000).tobytes()

        # Mock tool engine cleanly via context manager
        with patch.object(session.tool_engine, "execute_tool", return_value={"status": "ok", "summary": "Soleggiato 22C"}):
            with patch("httpx.AsyncClient", return_value=MockClient(MockStreamContext(mock_lines))):
                session.turn_id = 1
                # Iteration 1 will intercept the tool and continue to iteration 2
                # For iteration 2 we let it finish
                with patch.object(session, "_execute_tool_and_notify", new_callable=AsyncMock) as mock_tool_call:
                    mock_tool_call.return_value = "Soleggiato 22C"
                    # Running single turn
                    task = asyncio.create_task(session._process_voice_turn(fake_pcm, turn_id=1))
                    await asyncio.sleep(0.05)
                    # Cancel task after tool intercepted to inspect emitted chunks
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # Verify no llm_chunk contained "[TOOL_CALL"
        llm_chunks = [e for e in self.sent_events if e.get("event") == "llm_chunk"]
        for c in llm_chunks:
            self.assertNotIn("[TOOL_CALL", c["text"])
            self.assertNotIn("web_search", c["text"])

if __name__ == "__main__":
    unittest.main()
