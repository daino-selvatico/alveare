"""
SenseVoice Small Speech-to-Text (STT) Engine for Alveare.

Provides ultra-fast non-autoregressive speech transcription, multilingual recognition
(Italian, English, Chinese, Japanese, Korean, etc.), emotion detection, and audio event detection.
"""
import os
import sys
import io
import time
import re
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np

def find_torch_python() -> Optional[str]:
    """Find a Python interpreter that has torch and funasr installed."""
    # 1. Check current interpreter
    try:
        import torch
        from funasr import AutoModel
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
                res = subprocess.run([str(cand), "-c", "import torch, funasr; print('ok')"], capture_output=True, text=True, timeout=5, env=clean_env)
                if res.returncode == 0 and "ok" in res.stdout:
                    return str(cand)
            except Exception:
                pass
    return None

class SenseVoiceSTT:
    _instance = None
    _model = None
    _loading = False
    _external_py = None

    def __init__(self, model_id: str = "FunAudioLLM/SenseVoiceSmall", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._external_py = None

    @classmethod
    def get_instance(cls, model_id: str = "FunAudioLLM/SenseVoiceSmall", device: str = "cpu"):
        if cls._instance is None:
            cls._instance = cls(model_id=model_id, device=device)
        return cls._instance

    def _ensure_loaded(self):
        if self._model is not None or self._external_py is not None:
            return
        if self._loading:
            while self._loading:
                time.sleep(0.1)
            return

        self._loading = True
        try:
            print(f"[SenseVoiceSTT] Initializing {self.model_id} on {self.device}...")
            # 1. Try in-process if torch is importable
            try:
                import torch
                from funasr import AutoModel
                self._model = AutoModel(
                    model=self.model_id,
                    hub="hf",
                    device=self.device,
                    disable_update=True
                )
                print("[SenseVoiceSTT] SenseVoiceSmall loaded in-process.")
                return
            except BaseException as in_proc_err:
                # 2. Try external python environment
                ext_py = find_torch_python()
                if ext_py and ext_py != sys.executable:
                    self._external_py = ext_py
                    print(f"[SenseVoiceSTT] Using external PyTorch environment: {ext_py}")
                    return
                else:
                    raise in_proc_err
        except BaseException as e:
            print(f"[SenseVoiceSTT] Error loading SenseVoice model: {e}")
            raise e
        finally:
            self._loading = False

    @staticmethod
    def clean_text(raw_text: str) -> Dict[str, Any]:
        """Parse language, emotion, event tags and return clean text."""
        lang_match = re.search(r"<\|([a-z]{2,5})\|>", raw_text)
        detected_lang = lang_match.group(1) if lang_match else "unknown"

        emo_match = re.search(r"<\|(EMO_[A-Z_]+|NEUTRAL|HAPPY|SAD|ANGRY|FEARFUL|DISGUSTED|SURPRISED)\|>", raw_text, re.IGNORECASE)
        emotion = emo_match.group(1) if emo_match else "NEUTRAL"

        event_match = re.search(r"<\|(BGM|Speech|Applause|Laughter|Cry|Sneeze|Breath|Cough)\|>", raw_text, re.IGNORECASE)
        event = event_match.group(1) if event_match else "Speech"

        # Strip all special tokens <|...|>
        clean = re.sub(r"<\|[^|>]+(?:\|>)?", "", raw_text).strip()
        # Clean any remaining extra whitespace
        clean = re.sub(r"\s+", " ", clean).strip()

        return {
            "text": clean,
            "language": detected_lang,
            "emotion": emotion,
            "event": event
        }

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
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
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
            if self._model is not None:
                res = self._model.generate(
                    input=input_target,
                    cache={},
                    language=language if language != "auto" else "auto",
                    use_itn=use_itn
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                raw_text = ""
                if res and isinstance(res, list) and len(res) > 0:
                    raw_text = res[0].get("text", "")

                parsed = self.clean_text(raw_text)
                parsed["raw_text"] = raw_text
                parsed["latency_ms"] = round(elapsed_ms, 2)
                parsed["status"] = "success"
                return parsed
            elif self._external_py:
                # Subprocess worker execution
                worker_code = f"""
import sys, json
from funasr import AutoModel
model = AutoModel(model='{self.model_id}', hub='hf', device='{self.device}', disable_update=True)
res = model.generate(input='{input_target}', cache={{}}, language='{language}', use_itn={use_itn})
raw = res[0].get('text', '') if res and len(res) > 0 else ''
print(json.dumps({{'raw_text': raw}}))
"""
                cmd = [self._external_py, "-c", worker_code]
                clean_env = os.environ.copy()
                clean_env.pop("PYTHONPATH", None)
                proc = subprocess.run(cmd, capture_output=True, text=True, env=clean_env)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if proc.returncode == 0:
                    lines = [l for l in proc.stdout.strip().split("\n") if l.strip()]
                    last_json = lines[-1] if lines else "{}"
                    data = json.loads(last_json)
                    raw_text = data.get("raw_text", "")
                    parsed = self.clean_text(raw_text)
                    parsed["raw_text"] = raw_text
                    parsed["latency_ms"] = round(elapsed_ms, 2)
                    parsed["status"] = "success"
                    return parsed
                else:
                    raise RuntimeError(f"External STT worker failed: {proc.stderr}")
            else:
                raise RuntimeError("No SenseVoice STT model or worker available.")

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": "",
                "raw_text": "",
                "language": "unknown",
                "emotion": "unknown",
                "event": "unknown",
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
