"""
Kokoro-82M Text-to-Speech (TTS) Engine for Alveare.

Provides ultra-fast speech synthesis, natural multilingual Italian prosody,
and lightweight footprint accelerated on CPU (multi-threaded Zen 5) and GPU.
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
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, Generator, Tuple

_CACHED_KOKORO_PY = None

def find_kokoro_python() -> Optional[str]:
    """Find a Python interpreter that has kokoro, torch, and soundfile installed."""
    global _CACHED_KOKORO_PY
    if _CACHED_KOKORO_PY and os.path.exists(_CACHED_KOKORO_PY):
        return _CACHED_KOKORO_PY

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)

    candidates = [
        Path(os.path.expanduser("~/progetti/voice-studio/.venv/bin/python")),
        Path(os.path.expanduser("~/miniconda3/envs/alveare-aie/bin/python")),
        Path(os.path.expanduser("~/miniconda3/envs/comfyui/bin/python")),
        Path(os.path.expanduser("~/miniconda3/envs/gemma4-ref/bin/python")),
        Path(os.path.expanduser("~/anaconda3/envs/comfyui/bin/python")),
        Path("/opt/conda/envs/comfyui/bin/python"),
    ]

    # Fast path: check site-packages directory existence first
    for cand in candidates:
        if cand.exists():
            sp = list((cand.parent.parent / "lib").glob("python*/site-packages/kokoro"))
            if sp and sp[0].exists():
                _CACHED_KOKORO_PY = str(cand)
                return str(cand)

    # Subprocess check with generous timeout
    for cand in candidates:
        if cand.exists():
            try:
                res = subprocess.run(
                    [str(cand), "-c", "import kokoro; print('ok')"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    env=clean_env
                )
                if res.returncode == 0 and "ok" in res.stdout:
                    _CACHED_KOKORO_PY = str(cand)
                    return str(cand)
            except Exception:
                pass

    # Robust fallback: check sys.executable
    try:
        res = subprocess.run(
            [sys.executable, "-c", "import kokoro; print('ok')"],
            capture_output=True,
            text=True,
            timeout=15,
            env=clean_env
        )
        if res.returncode == 0 and "ok" in res.stdout:
            _CACHED_KOKORO_PY = sys.executable
            return sys.executable
    except Exception:
        pass

    return None

def split_into_chunks(
    text: str,
    strategy: str = "sentences",
    words_per_chunk: int = 8,
    min_words_first: int = 3,
    max_words_per_chunk: int = 22
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if strategy == "words":
        words = text.split()
        if not words:
            return []
        chunks = []
        wpc = max(1, words_per_chunk)
        for i in range(0, len(words), wpc):
            chunk = " ".join(words[i:i + wpc])
            if chunk:
                chunks.append(chunk)
        return chunks

    # Default & "sentences": Split on complete sentences and clauses
    parts = re.split(r'([.!?;\n:…]+\s*)', text)
    chunks = []
    for i in range(0, len(parts) - 1, 2):
        seg = (parts[i] + parts[i+1]).strip()
        if seg:
            chunks.append(seg)
    if len(parts) % 2 == 1 and parts[-1].strip():
        chunks.append(parts[-1].strip())

    return chunks if chunks else [text]

def extract_completed_chunks(
    text: str,
    strategy: str = "sentences",
    words_per_chunk: int = 8,
    is_first: bool = False
) -> tuple[list[str], str]:
    if not text:
        return [], ""

    if strategy == "words":
        words = text.split()
        wpc = max(1, words_per_chunk)
        if len(words) < wpc:
            return [], text
        ready = []
        idx = 0
        while idx + wpc <= len(words):
            ready.append(" ".join(words[idx:idx + wpc]))
            idx += wpc
        remaining = " ".join(words[idx:])
        return ready, remaining

    # Default & "sentences": Extract complete sentences
    parts = re.split(r'([.!?;\n:…]+\s*)', text)
    ready = []
    for i in range(0, len(parts) - 1, 2):
        seg = (parts[i] + parts[i+1]).strip()
        if seg:
            ready.append(seg)
    remaining = parts[-1] if len(parts) % 2 == 1 else ""
    return ready, remaining

class KokoroTTS:
    _instance = None
    _instance_lock = threading.Lock()
    _worker_proc = None
    _worker_lock = threading.Lock()
    _loading = False

    def __init__(self, model_id: str = "hexgrad/Kokoro-82M", device: str = "cpu"):
        if model_id in ("kokoro", "kokoro-82m", "cocore", "cocore-82m", "82m"):
            model_id = "hexgrad/Kokoro-82M"
        self.model_id = model_id
        self.device = device or "cpu"
        self._external_py = None

    @classmethod
    def get_instance(cls, model_id: str = "hexgrad/Kokoro-82M", device: str = "cpu"):
        with cls._instance_lock:
            dev = (device or "cpu").lower()
            if model_id in ("kokoro", "kokoro-82m", "cocore", "cocore-82m", "82m"):
                norm_model = "hexgrad/Kokoro-82M"
            else:
                norm_model = model_id

            if cls._instance is None or cls._instance.model_id != norm_model or cls._instance.device != dev:
                if cls._instance:
                    cls._instance.shutdown()
                cls._instance = cls(model_id=norm_model, device=dev)
            return cls._instance

    def _ensure_loaded(self):
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return
        if self._loading:
            while self._loading:
                time.sleep(0.05)
            return

        self._loading = True
        try:
            print(f"[KokoroTTS] Initializing {self.model_id} on {self.device}...")
            ext_py = find_kokoro_python()
            if not ext_py:
                raise RuntimeError(
                    "Python environment with kokoro, torch, and soundfile not found for Kokoro TTS worker. "
                    "Please ensure ~/progetti/voice-studio/.venv or ~/miniconda3/envs/comfyui is available."
                )
            self._external_py = ext_py
            worker_script = str(Path(__file__).resolve().parent / "kokoro_worker.py")
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONPATH", None)
            venv_dir = Path(ext_py).parent.parent
            if (venv_dir / "pyvenv.cfg").exists():
                clean_env["VIRTUAL_ENV"] = str(venv_dir)
                clean_env["PATH"] = str(Path(ext_py).parent) + os.pathsep + clean_env.get("PATH", "")

            cmd = [ext_py, worker_script, self.model_id, self.device]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=clean_env,
                start_new_session=True
            )
            ready_line = proc.stdout.readline().strip()
            if ready_line != "READY":
                err = proc.stderr.read()
                raise RuntimeError(f"Kokoro TTS worker failed to initialize: {ready_line} | {err}")

            self._worker_proc = proc
            print(f"[KokoroTTS] Persistent worker started for {self.model_id} on {self.device}.")
        finally:
            self._loading = False

    def generate(
        self,
        text: str,
        voice: Optional[str] = "if_sara",
        speed: float = 1.0,
        pitch: float = 0.0,
        language: str = "it",
        output_path: Optional[str] = None,
        sample_rate: int = 24000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate synthetic speech for the input text.
        Returns dictionary with audio_path, sample_rate, duration_sec, latency_ms, rtf, word_timestamps.
        """
        self._ensure_loaded()
        with self._worker_lock:
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_loaded()

            req = {
                "action": "generate",
                "text": text,
                "voice": voice or "if_sara",
                "language": language or "it",
                "speed": float(speed) if speed else 1.0,
                "pitch": float(pitch) if pitch is not None else 0.0,
                "output_path": output_path,
                "sample_rate": sample_rate or 24000,
                "format": "pcm16"
            }

            try:
                self._worker_proc.stdin.write(json.dumps(req) + "\n")
                self._worker_proc.stdin.flush()
                resp_line = self._worker_proc.stdout.readline()
                if not resp_line:
                    raise RuntimeError("Kokoro TTS worker process closed unexpectedly.")
                res = json.loads(resp_line)
                return res
            except Exception as e:
                try:
                    if self._worker_proc:
                        self._worker_proc.terminate()
                except Exception:
                    pass
                self._worker_proc = None
                return {"status": "error", "error": str(e)}

    def generate_chunk(
        self,
        text: str,
        seq: int = 0,
        voice: Optional[str] = "if_sara",
        speed: float = 1.0,
        pitch: float = 0.0,
        language: str = "it",
        sample_rate: int = 24000,
        format: str = "pcm16",
        **kwargs
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Synthesizes a single chunk/clause of text and returns raw PCM bytes + metadata.
        """
        self._ensure_loaded()
        with self._worker_lock:
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_loaded()

            req = {
                "action": "generate_chunk",
                "text": text,
                "seq": seq,
                "voice": voice or "if_sara",
                "language": language or "it",
                "speed": float(speed) if speed else 1.0,
                "pitch": float(pitch) if pitch is not None else 0.0,
                "sample_rate": sample_rate or 24000,
                "format": format or "pcm16"
            }

            try:
                self._worker_proc.stdin.write(json.dumps(req) + "\n")
                self._worker_proc.stdin.flush()
                resp_line = self._worker_proc.stdout.readline()
                if not resp_line:
                    raise RuntimeError("Kokoro TTS worker process closed unexpectedly.")
                res = json.loads(resp_line)
                if res.get("status") == "error":
                    raise RuntimeError(res.get("error", "Chunk synthesis error"))
                pcm_b64 = res.get("pcm_b64", "")
                pcm_bytes = base64.b64decode(pcm_b64) if pcm_b64 else b""
                return pcm_bytes, res
            except Exception as e:
                try:
                    if self._worker_proc:
                        self._worker_proc.terminate()
                except Exception:
                    pass
                self._worker_proc = None
                raise e

    def generate_stream(
        self,
        text: str,
        voice: Optional[str] = "if_sara",
        speed: float = 1.0,
        pitch: float = 0.0,
        language: str = "it",
        chunk_strategy: str = "sentences",
        words_per_chunk: int = 8,
        sample_rate: int = 24000,
        format: str = "pcm16",
        **kwargs
    ) -> Generator[Tuple[bytes, Dict[str, Any]], None, None]:
        """
        Streaming generator yielding (pcm_bytes, meta_dict) for each acoustic chunk.
        """
        self._ensure_loaded()
        with self._worker_lock:
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_loaded()

            req = {
                "action": "generate_stream",
                "text": text,
                "chunk_strategy": chunk_strategy,
                "words_per_chunk": words_per_chunk,
                "voice": voice or "if_sara",
                "language": language or "it",
                "speed": float(speed) if speed else 1.0,
                "pitch": float(pitch) if pitch is not None else 0.0,
                "sample_rate": sample_rate or 24000,
                "format": format or "pcm16"
            }

            try:
                self._worker_proc.stdin.write(json.dumps(req) + "\n")
                self._worker_proc.stdin.flush()

                while True:
                    line = self._worker_proc.stdout.readline()
                    if not line:
                        break
                    data = json.loads(line.strip())
                    event = data.get("event")
                    if event == "audio_frame":
                        pcm_b64 = data.get("pcm_b64", "")
                        pcm_bytes = base64.b64decode(pcm_b64) if pcm_b64 else b""
                        yield pcm_bytes, data
                    elif event == "stream_end":
                        yield b"", data
                        break
                    elif data.get("status") == "error":
                        raise RuntimeError(data.get("error", "Kokoro TTS stream synthesis failed"))
            except Exception as e:
                try:
                    if self._worker_proc:
                        self._worker_proc.terminate()
                except Exception:
                    pass
                self._worker_proc = None
                raise e

    def generate_bytes(self, text: str, **kwargs) -> tuple[bytes, Dict[str, Any]]:
        """Synthesizes speech and returns raw WAV bytes alongside metadata."""
        res = self.generate(text, **kwargs)
        if res.get("status") == "success" and res.get("audio_path") and os.path.exists(res["audio_path"]):
            with open(res["audio_path"], "rb") as f:
                wav_bytes = f.read()
            return wav_bytes, res
        raise RuntimeError(res.get("error", "Kokoro TTS synthesis failed"))

    # High-level standard API aliases
    def synthesize(self, text: str, **kwargs) -> Dict[str, Any]:
        """Alias to generate() for standard TTS synthesis API."""
        return self.generate(text, **kwargs)

    def synthesize_stream(self, text: str, **kwargs) -> Generator[Tuple[bytes, Dict[str, Any]], None, None]:
        """Alias to generate_stream() for streaming synthesis API."""
        return self.generate_stream(text, **kwargs)

    def synthesize_chunk(self, text: str, **kwargs) -> Tuple[bytes, Dict[str, Any]]:
        """Alias to generate_chunk() for clause-based synthesis."""
        return self.generate_chunk(text, **kwargs)

    def synthesize_bytes(self, text: str, **kwargs) -> tuple[bytes, Dict[str, Any]]:
        """Alias to generate_bytes() for raw WAV binary output."""
        return self.generate_bytes(text, **kwargs)

    @property
    def is_loaded(self) -> bool:
        """Checks if the persistent worker process is running and alive."""
        return self._worker_proc is not None and self._worker_proc.poll() is None

    def shutdown(self):
        """Gracefully shuts down the persistent Kokoro TTS worker process."""
        with self._worker_lock:
            if self._worker_proc is not None:
                try:
                    if self._worker_proc.poll() is None:
                        try:
                            self._worker_proc.stdin.write(json.dumps({"action": "stop"}) + "\n")
                            self._worker_proc.stdin.flush()
                            self._worker_proc.wait(timeout=0.8)
                        except Exception:
                            pass
                        if self._worker_proc.poll() is None:
                            self._worker_proc.terminate()
                            self._worker_proc.wait(timeout=1.0)
                except Exception:
                    try:
                        self._worker_proc.kill()
                    except Exception:
                        pass
                finally:
                    try:
                        if self._worker_proc.stdin:
                            self._worker_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        if self._worker_proc.stdout:
                            self._worker_proc.stdout.close()
                    except Exception:
                        pass
                    try:
                        if self._worker_proc.stderr:
                            self._worker_proc.stderr.close()
                    except Exception:
                        pass
                    self._worker_proc = None

    def stop(self):
        """Alias to shutdown()."""
        self.shutdown()
