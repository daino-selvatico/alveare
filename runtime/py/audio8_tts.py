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
    "valeria": (
        Path("/home/daino/progetti/voice-studio/voices/valeria_expressive.wav"),
        "Ma dai, non ci credo, davvero, hai fatto tutto questo. Guarda che è venuto benissimo, dobbiamo assolutamente organizzarci per il weekend, voglio fare qualcosa di divertente insieme. Che ne dici, usciamo fuori a pranzo?"
    ),
    "daino": (
        Path("/home/daino/progetti/voice-studio/voices/daino_raw.wav"),
        "Ciao, Bolt! Oggi è stata una giornata bella piena di lavoro e di codice ma finalmente ci prendiamo un momento per fare qualche test sui modelli vocali. Guarda che set up incredibile abbiamo tirato su direttamente dalla chat Telegram. Fammi sentire cosa viene fuori con la mia voce."
    ),
    "narratore": (
        Path("/home/daino/progetti/voice-studio/voices/valeria_master.wav"),
        "Ciao, oggi ho fatto una bella passeggiata e c'era un'aria davvero piacevole. Più tardi mi metto a cucinare con calma qualcosa di buono per cena, magari un piatto semplice ma saporito. Mi rilassa molto prendermi del tempo per me e staccare la spina."
    ),
    "female_expressive": (
        Path("/home/daino/progetti/voice-studio/voices/valeria_expressive.wav"),
        "Ma dai, non ci credo, davvero, hai fatto tutto questo. Guarda che è venuto benissimo, dobbiamo assolutamente organizzarci per il weekend, voglio fare qualcosa di divertente insieme. Che ne dici, usciamo fuori a pranzo?"
    ),
    "default": (
        Path("/home/daino/progetti/voice-studio/voices/valeria_expressive.wav"),
        "Ma dai, non ci credo, davvero, hai fatto tutto questo. Guarda che è venuto benissimo, dobbiamo assolutamente organizzarci per il weekend, voglio fare qualcosa di divertente insieme. Che ne dici, usciamo fuori a pranzo?"
    ),
}

import re

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

    # Default & "sentences": Split exclusively on complete sentences (delimiters: '.', '!', '?', '\n', ':', ';', '…')
    # avoiding splits at commas or partial words to preserve Audio8 prosody without artificial pauses or repetitions.
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

    # Default & "sentences": Extract exclusively complete sentences with delimiters '.', '!', '?', '\n', ':', ';', '…'
    parts = re.split(r'([.!?;\n:…]+\s*)', text)
    ready = []
    for i in range(0, len(parts) - 1, 2):
        seg = (parts[i] + parts[i+1]).strip()
        if seg:
            ready.append(seg)
    remaining = parts[-1] if len(parts) % 2 == 1 else ""
    return ready, remaining

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

    def _resolve_reference(
        self,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        if reference_audio and os.path.exists(reference_audio):
            return reference_audio, reference_text or ""

        preset_key = (voice or reference_audio or "valeria").lower().strip()
        if preset_key in VOICE_PRESETS:
            preset_path, preset_text = VOICE_PRESETS[preset_key]
            if preset_path.exists():
                return str(preset_path), reference_text or preset_text
        elif "female" in preset_key:
            preset_path, preset_text = VOICE_PRESETS["female_expressive"]
            if preset_path.exists():
                return str(preset_path), reference_text or preset_text
        elif "daino" in preset_key:
            preset_path, preset_text = VOICE_PRESETS["daino"]
            if preset_path.exists():
                return str(preset_path), reference_text or preset_text
        elif "narrat" in preset_key:
            preset_path, preset_text = VOICE_PRESETS["narratore"]
            if preset_path.exists():
                return str(preset_path), reference_text or preset_text

        # Default fallback
        preset_path, preset_text = VOICE_PRESETS["default"]
        if preset_path.exists():
            return str(preset_path), reference_text or preset_text
        return None, None

    def generate(
        self,
        text: str,
        voice: Optional[str] = "valeria",
        speed: float = 1.0,
        pitch: float = 0.0,
        language: str = "it",
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
        ref_audio_str, ref_text_str = self._resolve_reference(voice, reference_audio, reference_text)

        with self._worker_lock:
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_loaded()

            req = {
                "action": "generate",
                "text": text,
                "voice": voice or "valeria",
                "language": language or "it",
                "speed": float(speed) if speed else 1.0,
                "pitch": float(pitch) if pitch is not None else 0.0,
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

    def generate_chunk(
        self,
        text: str,
        seq: int = 0,
        voice: Optional[str] = "valeria",
        speed: float = 1.0,
        pitch: float = 0.0,
        language: str = "it",
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        sample_rate: int = 44100,
        format: str = "pcm16"
    ) -> tuple[bytes, Dict[str, Any]]:
        """
        Synthesizes a single chunk/clause of text and returns raw PCM bytes + metadata.
        """
        self._ensure_loaded()
        ref_audio_str, ref_text_str = self._resolve_reference(voice, reference_audio, reference_text)

        with self._worker_lock:
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_loaded()

            req = {
                "action": "generate_chunk",
                "text": text,
                "seq": seq,
                "voice": voice or "valeria",
                "language": language or "it",
                "speed": float(speed) if speed else 1.0,
                "pitch": float(pitch) if pitch is not None else 0.0,
                "reference_audio": ref_audio_str,
                "reference_text": ref_text_str,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "sample_rate": sample_rate,
                "format": format
            }

            try:
                self._worker_proc.stdin.write(json.dumps(req) + "\n")
                self._worker_proc.stdin.flush()
                resp_line = self._worker_proc.stdout.readline()
                if not resp_line:
                    raise RuntimeError("TTS worker process closed unexpectedly.")
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
        voice: Optional[str] = "valeria",
        speed: float = 1.0,
        pitch: float = 0.0,
        language: str = "it",
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        chunk_strategy: str = "sentences",
        words_per_chunk: int = 8,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        sample_rate: int = 44100,
        format: str = "pcm16"
    ):
        """
        Streaming generator yielding (pcm_bytes, meta_dict) for each acoustic chunk.
        """
        self._ensure_loaded()
        ref_audio_str, ref_text_str = self._resolve_reference(voice, reference_audio, reference_text)

        with self._worker_lock:
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_loaded()

            req = {
                "action": "generate_stream",
                "text": text,
                "chunk_strategy": chunk_strategy,
                "words_per_chunk": words_per_chunk,
                "voice": voice or "valeria",
                "language": language or "it",
                "speed": float(speed) if speed else 1.0,
                "pitch": float(pitch) if pitch is not None else 0.0,
                "reference_audio": ref_audio_str,
                "reference_text": ref_text_str,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "sample_rate": sample_rate,
                "format": format
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
                        raise RuntimeError(data.get("error", "TTS stream synthesis failed"))
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
        raise RuntimeError(res.get("error", "TTS synthesis failed"))
