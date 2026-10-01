"""
Unit tests for Qwen 3.8-27B GSQ-RCO, MTP speculative decoding integration,
and Control Server slot/telemetry/model setup architecture.
"""
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runtime.py.control_server import (
    discover_models,
    start_llm_slot,
    state,
    ModelSetupRequest,
    start_model_setup,
)
from tools.detect_arch import SUPPORTED
from tools.convert.base_quantizer import load_quantizer_plugin
from tools.setup_model import SUPPORTED_MODELS


class TestQwenControlAndArchitecture(unittest.TestCase):

    def setUp(self):
        self.orig_slots = dict(state.slots)

    def tearDown(self):
        if "llm" in state.slots:
            state.slots["llm"].process = None
            state.slots["llm"].status = "stopped"
        state.slots.update(self.orig_slots)

    def test_discover_models_qwen(self):
        """Verify discover_models handles qwen alias and extracts correct metadata."""
        # Mock Path.glob to return a fake qwen model directory
        fake_qwen_dir = MagicMock(spec=Path)
        fake_qwen_dir.name = "quantized_weights_qwen3.8-27b"
        fake_qwen_dir.exists.return_value = True
        fake_qwen_dir.resolve.return_value = fake_qwen_dir
        fake_qwen_dir.rglob.return_value = []
        fake_cfg = MagicMock()
        fake_cfg.exists.return_value = False
        fake_qwen_dir.__truediv__.return_value = fake_cfg

        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [fake_qwen_dir]
            models = discover_models()
            qwen_models = [m for m in models if "qwen" in m.get("id", "").lower()]
            self.assertGreater(len(qwen_models), 0, "Qwen model should be discovered")
            q = qwen_models[0]
            self.assertEqual(q["task"], "text-generation")
            self.assertEqual(q["arch"], "qwen")
            self.assertIn("Qwen", q["name"])
            self.assertIn("GSQ-RCO", q["description"])
            self.assertIn("MTP", q["description"])
            self.assertEqual(q["size_mb"], 11400.0)

    def test_llm_slot_layer_telemetry(self):
        """Verify layer telemetry allocates correct layer counts per model family."""
        # Mock subprocess.Popen and threading.Thread to prevent spawning background loops
        with patch("subprocess.Popen") as mock_popen, \
             patch("threading.Thread") as mock_thread:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_proc.stdout.readline.return_value = ""
            mock_popen.return_value = mock_proc

            # Qwen 27B -> 64 layers
            start_llm_slot(model="qwen3.8-27b", device="gpu")
            self.assertEqual(state.slots["llm"].total_layers, 64)

            # Gemma 4 E4B -> 42 layers
            start_llm_slot(model="gemma4-e4b", device="gpu")
            self.assertEqual(state.slots["llm"].total_layers, 42)

            # Gemma 4 12B -> 48 layers
            start_llm_slot(model="gemma4-12b", device="gpu")
            self.assertEqual(state.slots["llm"].total_layers, 48)

            # Llama 3.2 1B -> 26 layers
            start_llm_slot(model="llama", device="gpu")
            self.assertEqual(state.slots["llm"].total_layers, 26)

    def test_detect_arch_supported_qwen(self):
        """Verify detect_arch.py maps qwen, qwen2, and qwen3 to qwen quantizer key."""
        self.assertEqual(SUPPORTED.get("qwen"), "qwen")
        self.assertEqual(SUPPORTED.get("qwen2"), "qwen")
        self.assertEqual(SUPPORTED.get("qwen3"), "qwen")

    def test_load_quantizer_plugin_qwen(self):
        """Verify base_quantizer loads quantize_qwen for all Qwen aliases."""
        for alias in ["qwen", "qwen2", "qwen3", "qwen2.5", "qwen3.8"]:
            plugin = load_quantizer_plugin(alias)
            self.assertTrue(hasattr(plugin, "main") or hasattr(plugin, "quantize"))

    def test_setup_model_supported_models(self):
        """Verify qwen3.8-27b entry is properly declared in setup_model.py."""
        qwen_entries = [m for m in SUPPORTED_MODELS if m["id"] == "qwen3.8-27b"]
        self.assertEqual(len(qwen_entries), 1)
        entry = qwen_entries[0]
        self.assertEqual(entry["arch"], "qwen")
        self.assertIn("MTP Speculative", entry["name"])


if __name__ == "__main__":
    unittest.main()
