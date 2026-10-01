"""
Unit tests for control_server.py live WebSocket endpoint (/ws/live).
Tests explicitly mock neural workers and prove:
- Wire-compatible metadata in live_connected (sample_rate=16000, input_sample_rate=16000, output_sample_rate=24000/44100, audio_format='wav')
- Settings echoing tts_mode ('server' | 'client')
- Monotonic turn IDs across clear_memory (not 0)
- Actionable error events for malformed JSON, unknown message types, invalid base64 (no sensitive internals leaked)
- Finally block session cleanup on disconnect
"""
import unittest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient

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

class TestControlServerLiveEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # Patch slots to avoid launching real subprocesses during test
        self.orig_slots = dict(state.slots)
        state.slots["llm"] = MockWorkerSlot("gemma4-e4b", "gpu")
        state.slots["stt"] = MockWorkerSlot("openai/whisper-large-v3-turbo", "cpu")
        state.slots["tts"] = MockWorkerSlot("hexgrad/Kokoro-82M", "cpu")

    def tearDown(self):
        state.slots.update(self.orig_slots)

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_live_connected_wire_metadata(self, mock_init_workers):
        """Priority 3: live_connected contains wire-compatible input/output rates and format."""
        with self.client.websocket_connect("/ws/live") as ws:
            raw = ws.receive_text()
            data = json.loads(raw)
            self.assertEqual(data.get("event"), "live_connected")
            self.assertEqual(data.get("sample_rate"), 16000, "Legacy sample_rate must stay 16000")
            self.assertEqual(data.get("input_sample_rate"), 16000)
            self.assertEqual(data.get("output_sample_rate"), 24000)
            self.assertEqual(data.get("audio_format"), "wav")
            self.assertIn("settings", data)
            self.assertEqual(data["settings"].get("tts_mode"), "server")

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_client_mode_via_query_param(self, mock_init_workers):
        """Priority 3: live_connected reflects tts_mode='client' when passed via query param."""
        with self.client.websocket_connect("/ws/live?tts_mode=client") as ws:
            raw = ws.receive_text()
            data = json.loads(raw)
            self.assertEqual(data.get("event"), "live_connected")
            self.assertEqual(data["settings"].get("tts_mode"), "client")

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_malformed_json_error_event(self, mock_init_workers):
        """Priority 2: Malformed JSON receives actionable error event without leaking internals."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_text()  # live_connected
            ws.send_text("THIS IS NOT JSON {{{")
            err_raw = ws.receive_text()
            err_data = json.loads(err_raw)
            self.assertEqual(err_data.get("event"), "error")
            self.assertIn("Invalid JSON", err_data.get("error", ""))
            # Ensure no stack trace or python paths leaked
            self.assertNotIn("Traceback", err_data.get("error", ""))
            self.assertNotIn(".py", err_data.get("error", ""))

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_unknown_message_type_error_event(self, mock_init_workers):
        """Priority 2: Unknown message type receives actionable error event."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_text()  # live_connected
            ws.send_text(json.dumps({"type": "nonexistent_action"}))
            err_raw = ws.receive_text()
            err_data = json.loads(err_raw)
            self.assertEqual(err_data.get("event"), "error")
            self.assertIn("Unknown message type", err_data.get("error", ""))

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_invalid_base64_audio_error_event(self, mock_init_workers):
        """Priority 2: Corrupted base64 audio receives actionable error event."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_text()  # live_connected
            ws.send_text(json.dumps({"type": "audio", "data": "not valid base64 !!!@@@"}))
            err_raw = ws.receive_text()
            err_data = json.loads(err_raw)
            self.assertEqual(err_data.get("event"), "error")
            self.assertIn("Invalid base64", err_data.get("error", ""))

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_clear_memory_monotonic_turn_id(self, mock_init_workers):
        """Priority 2: memory_cleared uses monotonically advancing turn_id, not 0."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_text()  # live_connected
            ws.send_text(json.dumps({"type": "clear_memory"}))
            cleared_raw = ws.receive_text()
            cleared_data = json.loads(cleared_raw)
            self.assertEqual(cleared_data.get("event"), "memory_cleared")
            self.assertGreater(cleared_data.get("turn_id", 0), 0, "turn_id in memory_cleared must be > 0 and monotonic")

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_update_settings_echoes_tts_mode(self, mock_init_workers):
        """Priority 3: config/update_settings updates and echoes tts_mode."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_text()  # live_connected
            ws.send_text(json.dumps({
                "type": "config",
                "settings": {
                    "tts_mode": "client",
                    "temperature": 0.7
                }
            }))
            resp_raw = ws.receive_text()
            resp_data = json.loads(resp_raw)
            self.assertEqual(resp_data.get("event"), "settings_updated")
            self.assertEqual(resp_data["settings"].get("tts_mode"), "client")
            self.assertEqual(resp_data["settings"].get("temperature"), 0.7)

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    @patch("runtime.py.live_orchestrator.LiveSession.close")
    def test_websocket_finally_cleanup(self, mock_close, mock_init_workers):
        """Priority 2: WebSocket disconnection calls session.close() in finally block."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_text()
        # Connection closed
        self.assertTrue(mock_close.called, "session.close() must be invoked on disconnect")

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_android_register_client_exact_sequence(self, mock_init_workers):
        """Requirement 3: Exact Android register_client onOpen sequence with quoted room."""
        with self.client.websocket_connect("/ws/live") as ws:
            conn_raw = ws.receive_text()
            conn = json.loads(conn_raw)
            self.assertEqual(conn.get("event"), "live_connected")

            # Exact string Android sends:
            # {"type":"register_client","client_id":"android-satellite","room":"$roomName","mode":"voice_satellite"}
            android_msg = '{"type":"register_client","client_id":"android-satellite","room":"\\"salotto\\"","mode":"voice_satellite"}'
            ws.send_text(android_msg)

            ack_raw = ws.receive_text()
            ack = json.loads(ack_raw)
            self.assertEqual(ack.get("event"), "client_registered")
            self.assertEqual(ack.get("client_id"), "android-satellite")
            self.assertEqual(ack.get("room"), "salotto")
            self.assertEqual(ack.get("mode"), "voice_satellite")
            self.assertEqual(ack.get("status"), "ok")

            # Validate next ping works cleanly
            ws.send_text(json.dumps({"type": "ping"}))
            pong_raw = ws.receive_text()
            pong = json.loads(pong_raw)
            self.assertEqual(pong.get("event"), "pong")

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers")
    def test_legacy_heartbeat_and_handshake(self, mock_init_workers):
        """Requirement 3: Accept legacy heartbeat/keepalive without error."""
        with self.client.websocket_connect("/ws/live") as ws:
            ws.receive_text()  # live_connected
            ws.send_text(json.dumps({"type": "heartbeat"}))
            resp = json.loads(ws.receive_text())
            self.assertEqual(resp.get("event"), "pong")

            ws.send_text(json.dumps({"type": "keepalive"}))
            resp2 = json.loads(ws.receive_text())
            self.assertEqual(resp2.get("event"), "pong")

    @patch("runtime.py.live_orchestrator.LiveSession.initialize_workers", side_effect=RuntimeError("Worker prewarm error"))
    @patch("runtime.py.live_orchestrator.LiveSession.close")
    def test_safe_cleanup_on_exception_before_receive_loop(self, mock_close, mock_init_workers):
        """Requirement 5: If error occurs before receive loop, session.close() is safely called."""
        try:
            with self.client.websocket_connect("/ws/live") as ws:
                ws.receive_text()
        except Exception:
            pass
        self.assertTrue(mock_close.called, "session.close() must be called even if initialization raises")


if __name__ == "__main__":
    unittest.main()
