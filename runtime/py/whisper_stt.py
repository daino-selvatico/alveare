"""
Whisper Speech-to-Text (STT) Engine for Alveare.

Provides high-accuracy speech transcription and multilingual recognition (Italian, English,
Spanish, French, German, Chinese, Japanese, etc.) accelerated on AMD NPU.
"""
import os
import sys
import io
import time
import re
import tempfile
import subprocess
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np

def find_torch_python() -> Optional[str]:
    """Find a Python interpreter that has torch installed."""
    # 1. Check current interpreter
    try:
        import torch
        return sys.executable
    except Exception:
        pass

    # 2. Check known conda envs
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)

    candidates = [
        Path(os.path.expanduser("~/miniconda3/envs/gemma4-ref/bin/python")),
        Path(os.path.expanduser("~/miniconda3/envs/comfyui/bin/python")),
        Path(os.path.expanduser("~/anaconda3/envs/gemma4-ref/bin/python")),
        Path("/opt/conda/envs/gemma4-ref/bin/python")
    ]
    for cand in candidates:
        if cand.exists():
            try:
                res = subprocess.run([str(cand), "-c", "import torch; print('ok')"], capture_output=True, text=True, timeout=5, env=clean_env)
                if res.returncode == 0 and "ok" in res.stdout:
                    return str(cand)
            except Exception:
                pass
    return None

class WhisperSTT:
    _instance = None
    _model = None
    _worker_proc = None
    _worker_lock = threading.Lock()
    _loading = False
    _external_py = None

    def __init__(self, model_id: str = "openai/whisper-base", device: str = "npu"):
        self.model_id = model_id
        self.device = device or "npu"
        self._external_py = None

    @classmethod
    def get_instance(cls, model_id: str = "openai/whisper-base", device: str = "npu"):
        dev = device or "npu"
        if cls._instance is None or cls._instance.model_id != model_id or cls._instance.device != dev:
            if cls._instance and cls._instance._worker_proc:
                try:
                    cls._instance._worker_proc.terminate()
                except Exception:
                    pass
            cls._instance = cls(model_id=model_id, device=dev)
        return cls._instance

    def _ensure_loaded(self):
        if self._model is not None or (self._worker_proc is not None and self._worker_proc.poll() is None):
            return
        if self._loading:
            while self._loading:
                time.sleep(0.1)
            return

        self._loading = True
        try:
            print(f"[WhisperSTT] Initializing {self.model_id} on {self.device}...")
            ext_py = find_torch_python() or sys.executable
            self._external_py = ext_py
            worker_script = str(Path(__file__).resolve().parent / "stt_worker.py")
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
            if "READY" in ready_line:
                self._worker_proc = proc
                print(f"[STT Engine] Persistent STT worker started with {ext_py} (model {self.model_id} ready in RAM).")
                return
            else:
                err_out = proc.stderr.read()
                raise RuntimeError(f"STT worker failed to start: {err_out}")
        except BaseException as e:
            print(f"[WhisperSTT] Error loading STT model: {e}")
            raise e
        finally:
            self._loading = False

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Strip special tokens and clean whitespace."""
        clean = re.sub(r"<\|[^|>]+(?:\|>)?", "", raw_text).strip()
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def transcribe(
        self,
        audio_input: Union[str, bytes, np.ndarray, Path],
        language: str = "auto",
        use_itn: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio input to text.
        audio_input can be:
          - A file path string / Path
          - Raw audio file bytes (WAV, MP3, OGG, WebM, FLAC)
          - Numpy float32 array sampled at 16kHz
        """
        self._ensure_loaded()
        t0 = time.perf_counter()

        temp_path = None
        input_target = audio_input

        if isinstance(audio_input, bytes):
            suffix = ".wav"
            if audio_input.startswith(b"RIFF"):
                suffix = ".wav"
            elif audio_input.startswith(b"\x1a\x45\xdf\xa3"):
                suffix = ".webm"
            elif audio_input.startswith(b"OggS"):
                suffix = ".ogg"
            elif audio_input.startswith(b"ID3") or audio_input.startswith(b"\xff\xfb") or audio_input.startswith(b"\xff\xf3"):
                suffix = ".mp3"
            elif audio_input.startswith(b"fLaC"):
                suffix = ".flac"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(audio_input)
                temp_path = tf.name
            input_target = temp_path
        elif isinstance(audio_input, np.ndarray):
            import wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_path = tf.name
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                int_pcm = (np.clip(audio_input, -1.0, 1.0) * 32767.0).astype(np.int16)
                wf.writeframes(int_pcm.tobytes())
            input_target = temp_path
        elif isinstance(audio_input, Path):
            input_target = str(audio_input)

        try:
            raw_text = ""
            detected_lang = language if language not in ("auto", "unknown", "") else "it"

            if self._model is not None:
                res = self._model.generate(
                    input=input_target,
                    cache={},
                    language=language if language != "auto" else "auto",
                    use_itn=use_itn
                )
                if res and isinstance(res, list) and len(res) > 0:
                    raw_text = res[0].get("text", "")
            elif self._worker_proc is not None:
                with self._worker_lock:
                    if self._worker_proc.poll() is not None:
                        # Process died, restart
                        self._ensure_loaded()
                    req_line = json.dumps({
                        "input": input_target,
                        "language": language if language != "auto" else "auto",
                        "use_itn": use_itn
                    })
                    self._worker_proc.stdin.write(req_line + "\n")
                    self._worker_proc.stdin.flush()
                    resp_line = self._worker_proc.stdout.readline()
                    data = json.loads(resp_line.strip())
                    if data.get("status") == "error":
                        raise RuntimeError(data.get("error", "Worker error"))
                    raw_text = data.get("raw_text", "")
                    detected_lang = data.get("language", detected_lang)
            else:
                raise RuntimeError("No STT model or worker available.")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            clean = self.clean_text(raw_text)
            return {
                "text": clean,
                "raw_text": raw_text,
                "language": detected_lang,
                "latency_ms": round(elapsed_ms, 2),
                "status": "success"
            }

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": "",
                "raw_text": "",
                "language": "unknown",
                "latency_ms": round(elapsed_ms, 2),
                "status": "error",
                "error": str(e)
            }
        finally:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass
