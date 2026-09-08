"""
Unit and Integration Tests for Kokoro-82M TTS in Alveare.

Covers:
1. TTS Engine Factory (is_kokoro_model, get_tts_engine, thread-safe singleton, stop_all_tts)
2. KokoroTTS Engine (warmup, synthesis, chunk generation, streaming, word timestamps, RTF, shutdown)
3. Server API Integration (discover_models, start_tts_slot, /v1/audio/speech, /api/tts/generate)
"""
import os
import sys
import unittest
import tempfile
import time
from pathlib import Path

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.py.tts_factory import is_kokoro_model, get_tts_engine, stop_all_tts
from runtime.py.kokoro_tts import KokoroTTS
from runtime.py.audio8_tts import Audio8TTS


class TestTTSFactory(unittest.TestCase):
    def tearDown(self):
        stop_all_tts()

    def test_is_kokoro_model_variants(self):
        """Verify is_kokoro_model correctly recognizes all variants and aliases."""
        positive_cases = [
            "kokoro",
            "Kokoro",
            "KOKORO",
            "kokoro-82m",
            "Kokoro-82M",
            "hexgrad/Kokoro-82M",
            "hexgrad/kokoro-82m",
            "cocore",
            "CoCore",
            "cocore-82m",
            "82m",
            "82M",
            "quantized_weights_kokoro-82m",
        ]
        for name in positive_cases:
            with self.subTest(name=name):
                self.assertTrue(is_kokoro_model(name), f"Failed for {name}")

        negative_cases = [
            "audio8-0.1b",
            "Audio8/Audio8-TTS-Preview-0.1b",
            "Audio8/Audio8-TTS-Preview-0.6b",
            "openai/whisper-base",
            "gemma3",
            "gemma4",
            "llama",
            "",
            None,
        ]
        for name in negative_cases:
            with self.subTest(name=name):
                self.assertFalse(is_kokoro_model(name), f"Should not be kokoro: {name}")

    def test_get_tts_engine_routing(self):
        """Verify get_tts_engine routes correctly to KokoroTTS or Audio8TTS."""
        kokoro_engine = get_tts_engine("hexgrad/Kokoro-82M", device="cpu")
        self.assertIsInstance(kokoro_engine, KokoroTTS)
        self.assertEqual(kokoro_engine.device, "cpu")

        # Verify singleton behavior
        kokoro_engine_2 = get_tts_engine("kokoro", device="cpu")
        self.assertIs(kokoro_engine, kokoro_engine_2)

        # Verify Audio8 routing
        audio8_engine = get_tts_engine("Audio8/Audio8-TTS-Preview-0.1b", device="cpu")
        self.assertIsInstance(audio8_engine, Audio8TTS)


class TestKokoroTTSEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = KokoroTTS.get_instance(model_id="hexgrad/Kokoro-82M", device="cpu")

    @classmethod
    def tearDownClass(cls):
        if cls.engine:
            cls.engine.shutdown()

    def test_engine_initialization(self):
        """Verify that KokoroTTS starts and initializes its persistent worker."""
        self.engine._ensure_loaded()
        self.assertTrue(self.engine.is_loaded)
        self.assertIsNotNone(self.engine._worker_proc)
        self.assertIsNone(self.engine._worker_proc.poll())

    def test_synthesize_real_speech_cpu(self):
        """
        Verify real end-to-end speech synthesis on CPU with Italian text,
        checking latency, RTF, duration, sample rate and word timestamps.
        """
        prompt = "Alveare integra Kokoro con successo. La sintesi vocale è ultra rapida e naturale."
        res = self.engine.synthesize(
            text=prompt,
            voice="if_sara",
            speed=1.0,
            language="it"
        )
        self.assertEqual(res.get("status"), "success", f"Synthesis failed: {res}")
        self.assertEqual(res.get("sample_rate"), 24000)
        self.assertGreater(res.get("duration_sec", 0), 1.5)
        self.assertGreater(res.get("latency_ms", 0), 0)

        # Real-time factor (RTF) check: non-autoregressive StyleTTS2 on Zen 5 CPU is well under 0.5x
        rtf = res.get("rtf", 1.0)
        self.assertLess(rtf, 0.5, f"RTF too high: {rtf} (expected < 0.5x)")

        # Verify audio file output
        audio_path = res.get("audio_path")
        self.assertTrue(audio_path and os.path.exists(audio_path), "Audio file does not exist")
        self.assertGreater(os.path.getsize(audio_path), 10000, "Audio file too small")

        # Verify word timestamps
        timestamps = res.get("word_timestamps", [])
        self.assertGreater(len(timestamps), 5, "Word timestamps list too short")
        self.assertEqual(timestamps[0]["word"], "Alveare")
        self.assertEqual(timestamps[0]["start_ms"], 0.0)

    def test_synthesize_chunk(self):
        """Verify chunk/clause synthesis returning raw PCM bytes."""
        chunk_text = "Questo è un frammento di test per Alveare."
        pcm_bytes, meta = self.engine.synthesize_chunk(
            text=chunk_text,
            seq=1,
            voice="im_nicola",
            language="it",
            format="pcm16"
        )
        self.assertIsInstance(pcm_bytes, bytes)
        self.assertGreater(len(pcm_bytes), 1000, "PCM bytes buffer is empty")
        self.assertEqual(meta.get("status"), "success")
        self.assertEqual(meta.get("sample_rate"), 24000)

    def test_synthesize_stream(self):
        """Verify streaming generator yields audio frames and ends properly."""
        full_text = "Prima frase completa di test. Seconda frase di verifica streaming."
        frames = []
        stream_end_seen = False

        for pcm_bytes, meta in self.engine.synthesize_stream(
            text=full_text,
            voice="if_sara",
            chunk_strategy="sentences"
        ):
            if meta.get("event") == "audio_frame":
                self.assertGreater(len(pcm_bytes), 0)
                frames.append((pcm_bytes, meta))
            elif meta.get("event") == "stream_end":
                stream_end_seen = True
                self.assertGreater(meta.get("total_duration", 0), 1.0)

        self.assertGreater(len(frames), 0, "No audio frames yielded")
        self.assertTrue(stream_end_seen, "Stream end event not received")

    def test_synthesize_bytes_wav(self):
        """Verify synthesize_bytes returns valid WAV binary format."""
        wav_bytes, meta = self.engine.synthesize_bytes(
            "Test per byte audio WAV.",
            voice="if_sara"
        )
        self.assertIsInstance(wav_bytes, bytes)
        self.assertTrue(wav_bytes.startswith(b"RIFF"), "Not a valid WAV header")
        self.assertIn(b"WAVE", wav_bytes[:16])


class TestControlServerKokoroIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from runtime.py.control_server import app, state
        cls.client = TestClient(app)
        cls.state = state

    @classmethod
    def tearDownClass(cls):
        stop_all_tts()

    def test_discover_models_includes_kokoro(self):
        """Verify discover_models returns Kokoro 82M with task text-to-speech."""
        res = self.client.get("/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.json().get("data", [])
        kokoro_models = [m for m in data if is_kokoro_model(m.get("id"))]
        self.assertGreater(len(kokoro_models), 0, "Kokoro model not found in /v1/models")
        k = kokoro_models[0]
        self.assertEqual(k.get("task"), "text-to-speech")

    def test_start_tts_slot_kokoro(self):
        """Verify starting the TTS slot with Kokoro 82M."""
        from runtime.py.control_server import start_tts_slot, stop_slot, state
        success = start_tts_slot(model="hexgrad/Kokoro-82M", device="cpu")
        self.assertTrue(success)

        # Wait briefly for async loading
        max_wait = 15.0
        t0 = time.time()
        while time.time() - t0 < max_wait:
            tts_slot = state.slots["tts"]
            if tts_slot.is_loaded and tts_slot.status == "running":
                break
            time.sleep(0.2)

        tts_slot = state.slots["tts"]
        self.assertTrue(tts_slot.is_loaded, f"TTS slot failed to load: {tts_slot.last_error}")
        self.assertEqual(tts_slot.status, "running")
        self.assertEqual(tts_slot.device, "cpu")

        # Stop slot
        stop_slot("tts")
        self.assertEqual(state.slots["tts"].status, "stopped")

    def test_v1_audio_speech_api(self):
        """Verify /v1/audio/speech generates valid WAV audio with Kokoro."""
        payload = {
            "model": "hexgrad/Kokoro-82M",
            "input": "Ciao da Alveare, il test dell'endpoint OpenAI compatibile è riuscito.",
            "voice": "if_sara",
            "response_format": "wav"
        }
        res = self.client.post("/v1/audio/speech", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "audio/wav")
        content = res.content
        self.assertTrue(content.startswith(b"RIFF"), "Output is not a WAV file")
        self.assertGreater(len(content), 10000, "WAV payload too small")


if __name__ == "__main__":
    unittest.main()
