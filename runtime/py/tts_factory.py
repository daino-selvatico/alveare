"""
Unified TTS Engine Factory for Alveare.

Dispatches transparently to Audio8TTS (NPU/CPU/GPU) or KokoroTTS (CPU/GPU)
based on model_id, with thread-safe singleton management and clean lifecycle controls.
"""
import threading
from typing import Any, Union, Optional
from runtime.py.audio8_tts import Audio8TTS, split_into_chunks, extract_completed_chunks
from runtime.py.kokoro_tts import KokoroTTS

_FACTORY_LOCK = threading.Lock()

KOKORO_KEYWORDS = ("kokoro", "cocore", "82m", "hexgrad/kokoro-82m")

def is_kokoro_model(model_id: Optional[str]) -> bool:
    """
    Checks whether a model ID or alias refers to the Kokoro-82M TTS engine.
    Recognizes variants: 'kokoro', 'cocore', '82m', 'hexgrad/kokoro-82m', 'kokoro-82m', etc.
    """
    if not model_id:
        return False
    m = model_id.lower().strip()
    return any(k in m for k in KOKORO_KEYWORDS)

def get_tts_engine(
    model_id: Optional[str] = "Audio8/Audio8-TTS-Preview-0.1b",
    device: Optional[str] = None
) -> Union[KokoroTTS, Audio8TTS]:
    """
    Returns the thread-safe singleton instance of the appropriate TTS engine.
    - For Kokoro-82M ('kokoro', 'cocore', '82m'): routed to KokoroTTS on CPU (default) or GPU (CUDA).
    - For Audio8: routed to Audio8TTS on NPU (default) or CPU/GPU.
    """
    with _FACTORY_LOCK:
        m = model_id or "Audio8/Audio8-TTS-Preview-0.1b"
        if is_kokoro_model(m):
            dev = (device or "cpu").lower().strip()
            dev = "gpu" if dev in ("gpu", "cuda") else "cpu"
            return KokoroTTS.get_instance(model_id=m, device=dev)
        else:
            dev = (device or "npu").lower().strip()
            return Audio8TTS.get_instance(model_id=m, device=dev)

def stop_all_tts():
    """
    Gracefully shuts down any active TTS worker processes across all engines.
    """
    with _FACTORY_LOCK:
        if KokoroTTS._instance is not None:
            try:
                KokoroTTS._instance.shutdown()
            except Exception:
                pass
            KokoroTTS._instance = None

        if Audio8TTS._instance is not None:
            try:
                if hasattr(Audio8TTS._instance, "shutdown"):
                    Audio8TTS._instance.shutdown()
                elif Audio8TTS._instance._worker_proc:
                    Audio8TTS._instance._worker_proc.terminate()
                    Audio8TTS._instance._worker_proc.wait(timeout=1.0)
            except Exception:
                pass
            Audio8TTS._instance = None
