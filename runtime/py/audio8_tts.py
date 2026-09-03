"""
Audio8 Text-to-Speech (TTS) Engine for Alveare.

Provides low-latency speech synthesis, natural multilingual Italian prosody,
and zero-shot voice cloning accelerated on AMD Ryzen AI NPU and CPU.
"""
import os
import sys
import io
import time
import base64
import tempfile
import subprocess
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union

def find_torch_python() -> Optional[str]:
    """Find a Python interpreter that has torch and transformers installed."""
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)

    candidates = [
        Path(os.path.expanduser("~/miniconda3/envs/gemma4-ref/bin/python")),
        Path(os.path.expanduser("~/progetti/voice-studio/.venv/bin/python")),
        Path(os.path.expanduser("~/miniconda3/envs/alveare-aie/bin/python")),
        Path(sys.executable),
    ]
    for cand in candidates:
        if cand.exists():
            try:
                res = subprocess.run(
                    [str(cand), "-c", "import torch, transformers, soundfile; print('ok')"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=clean_env
                )
                if res.returncode == 0 and "ok" in res.stdout:
                    return str(cand)
            except Exception:
                pass
    return None

VOICE_PRESETS = {
    "female_expressive": (Path("/home/daino/progetti/voice-studio/voices/valeria_expressive.wav"), "Questo è un campione di voce espressivo per la sintesi vocale."),
    "valeria": (Path("/home/daino/progetti/voice-studio/voices/valeria_expressive.wav"), "Questo è un campione di voce espressivo per la sintesi vocale."),
    "male_deep": (Path("/home/daino/progetti/voice-studio/voices/daino_raw.wav"), "Questo è un campione di voce registrato per il test."),
    "daino": (Path("/home/daino/progetti/voice-studio/voices/daino_raw.wav"), "Questo è un campione di voce registrato per il test."),
    "default": (Path("/home/daino/progetti/voice-studio/voices/valeria_expressive.wav"), "Questo è un campione di voce espressivo per la sintesi vocale."),
}

class Audio8TTS:
    _instance = None
    _worker_proc = None
    _worker_lock = threading.Lock()
    _loading = False

    def __init__(self, model_id: str = "Audio8/Audio8-TTS-Preview-0.1b", device: str = "npu"):
        # Map common shorthands
        if model_id in ("audio8-0.1b", "audio8-100m", "0.1b"):
            model_id = "Audio8/Audio8-TTS-Preview-0.1b"
        elif model_id in ("audio8-0.6b", "audio8-600m", "0.6b"):
            model_id = "Audio8/Audio8-TTS-Preview-0.6b"

        self.model_id = model_id
        self.device = device or "npu"
        self._external_py = None

    @classmethod
    def get_instance(cls, model_id: str = "Audio8/Audio8-TTS-Preview-0.1b", device: str = "npu"):
        dev = device or "npu"
        if model_id in ("audio8-0.1b", "audio8-100m", "0.1b"):
            norm_model = "Audio8/Audio8-TTS-Preview-0.1b"
        elif model_id in ("audio8-0.6b", "audio8-600m", "0.6b"):
            norm_model = "Audio8/Audio8-TTS-Preview-0.6b"
        else:
            norm_model = model_id

        if cls._instance is None or cls._instance.model_id != norm_model or cls._instance.device != dev:
            if cls._instance and cls._instance._worker_proc:
                try:
                    cls._instance._worker_proc.terminate()
                except Exception:
                    pass
            cls._instance = cls(model_id=norm_model, device=dev)
        return cls._instance

    def _ensure_loaded(self):
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return
        if self._loading:
            while self._loading:
                time.sleep(0.1)
            return

        self._loading = True
        try:
            print(f"[Audio8TTS] Initializing {self.model_id} on {self.device}...")
            ext_py = find_torch_python() or sys.executable
            self._external_py = ext_py
            worker_script = str(Path(__file__).resolve().parent / "tts_worker.py")
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONPATH", None)

            cmd = [ext_py, worker_script, self.model_id, self.device]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=clean_env
            )
            # Wait for worker to load weights and output READY
            ready_line = proc.stdout.readline().strip()
            if ready_line != "READY":
                err = proc.stderr.read()
                raise RuntimeError(f"TTS worker failed to initialize: {ready_line} | {err}")

            self._worker_proc = proc
            print(f"[TTS Engine] Persistent worker started for {self.model_id} on {self.device}.")
        finally:
            self._loading = False

    def generate(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate synthetic speech for the input text.
        Returns dictionary with audio_path, sample_rate, duration_sec, latency_ms, rtf.
        """
        self._ensure_loaded()
        ref_audio_str = reference_audio
        ref_text_str = reference_text
        if not ref_audio_str or ref_audio_str in VOICE_PRESETS:
            preset_key = ref_audio_str if ref_audio_str in VOICE_PRESETS else "female_expressive"
            preset_path, preset_text = VOICE_PRESETS[preset_key]
            if preset_path.exists():
                ref_audio_str = str(preset_path)
                if not ref_text_str:
                    ref_text_str = preset_text

        with self._worker_lock:
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_loaded()

            req = {
                "action": "generate",
                "text": text,
                "reference_audio": ref_audio_str,
                "reference_text": ref_text_str,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "output_path": output_path
            }

            try:
                self._worker_proc.stdin.write(json.dumps(req) + "\n")
                self._worker_proc.stdin.flush()
                resp_line = self._worker_proc.stdout.readline()
                if not resp_line:
                    raise RuntimeError("TTS worker process closed unexpectedly.")
                res = json.loads(resp_line)
                return res
            except Exception as e:
                # Terminate and reset on error
                try:
                    if self._worker_proc:
                        self._worker_proc.terminate()
                except Exception:
                    pass
                self._worker_proc = None
                return {"status": "error", "error": str(e)}

    def generate_bytes(self, text: str, **kwargs) -> tuple[bytes, Dict[str, Any]]:
        """Synthesizes speech and returns raw WAV bytes alongside metadata."""
        res = self.generate(text, **kwargs)
        if res.get("status") == "success" and res.get("audio_path") and os.path.exists(res["audio_path"]):
            with open(res["audio_path"], "rb") as f:
                wav_bytes = f.read()
            return wav_bytes, res
        raise RuntimeError(res.get("error", "TTS synthesis failed"))
