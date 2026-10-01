import sys
import os
import io
import json
import time
import math
import tempfile
import base64
import re
from pathlib import Path
import numpy as np

def compute_word_timestamps(text_segment: str, duration_sec: float) -> list:
    """
    Calculates word-by-word timestamps based on actual audio duration.
    Returns: [{'word': w, 'start_ms': s, 'end_ms': e}]
    """
    words = text_segment.strip().split()
    if not words or duration_sec <= 0:
        return []

    duration_ms = duration_sec * 1000.0
    if len(words) == 1:
        return [{
            "word": words[0],
            "start_ms": 0.0,
            "end_ms": round(duration_ms, 2)
        }]

    weights = []
    for w in words:
        clean = re.sub(r'[^\w]', '', w)
        weight = max(1.0, float(len(clean)))
        if w and w[-1] in {'.', '!', '?', ';', ':', '…'}:
            weight += 2.0
        elif w and w[-1] in {',', '-', '—'}:
            weight += 1.0
        weights.append(weight)

    total_weight = sum(weights) if sum(weights) > 0 else float(len(words))

    timestamps = []
    current_ms = 0.0
    for i, w in enumerate(words):
        if i == len(words) - 1:
            end_ms = duration_ms
        else:
            w_dur = (weights[i] / total_weight) * duration_ms
            end_ms = current_ms + w_dur

        timestamps.append({
            "word": w,
            "start_ms": round(current_ms, 2),
            "end_ms": round(end_ms, 2)
        })
        current_ms = end_ms

    return timestamps

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

def resolve_voice_and_lang(voice_name: str, language: str) -> tuple[str, str]:
    """Resolves voice alias and language code for Kokoro."""
    v = (voice_name or "if_sara").lower().strip()
    lang = (language or "it").lower().strip()

    # Mappings for Alveare voice presets
    if v in ("valeria", "female_expressive", "narratore", "default", "sara", "f"):
        chosen_voice = "if_sara"
        lang_code = "i"
    elif v in ("daino", "male", "nicola", "m"):
        chosen_voice = "im_nicola"
        lang_code = "i"
    elif v in ("bolt", "assistant", "conciso"):
        chosen_voice = "im_nicola"
        lang_code = "i"
    elif v == "if_sara":
        chosen_voice = "if_sara"
        lang_code = "i"
    elif v == "im_nicola":
        chosen_voice = "im_nicola"
        lang_code = "i"
    else:
        chosen_voice = voice_name.strip()
        # Strictly check exact Kokoro prefixes
        if chosen_voice.startswith(("if_", "im_")):
            lang_code = "i"
        elif chosen_voice.startswith(("af_", "am_")):
            lang_code = "a"
        elif chosen_voice.startswith(("bf_", "bm_")):
            lang_code = "b"
        elif chosen_voice.startswith(("jf_", "jm_")):
            lang_code = "j"
        elif chosen_voice.startswith(("zf_", "zm_")):
            lang_code = "z"
        elif chosen_voice.startswith(("ef_", "em_")):
            lang_code = "e"
        elif chosen_voice.startswith(("ff_", "fm_")):
            lang_code = "f"
        elif chosen_voice.startswith(("hf_", "hm_")):
            lang_code = "h"
        elif chosen_voice.startswith(("pf_", "pm_")):
            lang_code = "p"
        else:
            lang_code = "i" if lang.startswith("it") else "a"
            if lang_code == "i" and chosen_voice not in ("if_sara", "im_nicola"):
                chosen_voice = "if_sara"

    return chosen_voice, lang_code

def main():
    model_id = "hexgrad/Kokoro-82M"
    device = "cpu"
    if len(sys.argv) > 1 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2]:
        device = sys.argv[2].lower()

    # Save real IPC stdout descriptor and redirect sys.stdout to sys.stderr during model load
    ipc_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    import torch
    import torchaudio
    import soundfile as sf
    import kokoro
    from kokoro import KPipeline

    torch.set_num_threads(8)

    torch_device = "cpu"
    if device in ("gpu", "cuda"):
        if torch.cuda.is_available():
            torch_device = "cuda"
            sys.stderr.write(f"[Kokoro Worker] Using CUDA GPU acceleration: {torch.cuda.get_device_name(0)}\n")
        else:
            sys.stderr.write("[Kokoro Worker] CUDA requested but not available in PyTorch, falling back to CPU (Zen 5 AVX-512).\n")

    sys.stderr.write(f"[Kokoro Worker] Loading {model_id} on {torch_device}...\n")
    pipelines = {}

    def get_pipeline(lang_code: str) -> KPipeline:
        if lang_code not in pipelines:
            sys.stderr.write(f"[Kokoro Worker] Initializing KPipeline for lang='{lang_code}' on {torch_device}...\n")
            repo_id = model_id if "/" in model_id else "hexgrad/Kokoro-82M"
            try:
                p = KPipeline(lang_code=lang_code, repo_id=repo_id, device=torch_device)
            except Exception as pe:
                sys.stderr.write(f"[Kokoro Worker] Error loading pipeline for '{lang_code}': {pe}, falling back to Italian 'i'\n")
                if "i" in pipelines:
                    return pipelines["i"]
                p = KPipeline(lang_code="i", repo_id=repo_id, device=torch_device)
            pipelines[lang_code] = p
        return pipelines[lang_code]

    # Pre-warm Italian pipeline and default voice
    try:
        p_it = get_pipeline("i")
        _ = list(p_it("Inizializzazione completata.", voice="if_sara", speed=1.0))
        sys.stderr.write("[Kokoro Worker] Warmup completed for voice 'if_sara' (Italian).\n")
    except Exception as we:
        sys.stderr.write(f"[Kokoro Worker] Warmup notice: {we}\n")

    sys.stderr.write(f"[Kokoro Worker] Kokoro-82M fully loaded and ready on {torch_device}.\n")
    ipc_out.write("READY\n")
    ipc_out.flush()

    def synthesize_segment(
        text_segment: str,
        voice: str = "if_sara",
        speed: float = 1.0,
        pitch: float = 0.0,
        language: str = "it",
        target_sample_rate: int = 24000,
        format_type: str = "pcm16"
    ):
        chosen_voice, lang_code = resolve_voice_and_lang(voice, language)
        try:
            pipeline = get_pipeline(lang_code)
            chunks = list(pipeline(text_segment, voice=chosen_voice, speed=speed))
        except Exception as pe:
            sys.stderr.write(f"[Kokoro Worker] Synthesis error with voice '{chosen_voice}' lang '{lang_code}': {pe}, falling back to Italian pipeline\n")
            try:
                p_fallback = get_pipeline("i")
                chunks = list(p_fallback(text_segment, voice="if_sara", speed=speed))
            except Exception as pe2:
                sys.stderr.write(f"[Kokoro Worker] Critical synthesis fallback error: {pe2}\n")
                chunks = []
        if not chunks:
            audio_tensor = torch.zeros(0, dtype=torch.float32)
        elif len(chunks) == 1:
            audio_tensor = chunks[0][2]
        else:
            audio_tensor = torch.cat([c[2] for c in chunks], dim=0)

        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.from_numpy(audio_tensor)

        native_sr = 24000

        # Pitch Shift Adjustment (torchaudio)
        if pitch != 0.0 and pitch != 1.0 and audio_tensor.numel() > 0:
            if 0.2 < pitch <= 2.0 and pitch != 1.0:
                n_steps = 12.0 * math.log2(pitch)
            else:
                n_steps = float(pitch)
            if abs(n_steps) > 0.01:
                try:
                    audio_tensor = torchaudio.functional.pitch_shift(
                        audio_tensor.unsqueeze(0), sample_rate=native_sr, n_steps=n_steps
                    ).squeeze(0)
                except Exception as pe:
                    sys.stderr.write(f"[Kokoro Worker] Pitch shift error: {pe}\n")

        # Resample if target sample rate differs
        effective_sr = native_sr
        if target_sample_rate > 0 and target_sample_rate != native_sr and audio_tensor.numel() > 0:
            try:
                resampler = torchaudio.transforms.Resample(orig_freq=native_sr, new_freq=target_sample_rate)
                audio_tensor = resampler(audio_tensor)
                effective_sr = target_sample_rate
            except Exception as re_err:
                sys.stderr.write(f"[Kokoro Worker] Resampling error: {re_err}\n")
                effective_sr = native_sr

        audio_samples = audio_tensor.float().cpu().numpy()

        # Format conversion
        fmt = (format_type or "pcm16").lower().strip()
        if fmt in ("float32", "f32"):
            pcm_bytes = audio_samples.astype(np.float32).tobytes()
        elif fmt in ("wav", "audio/wav"):
            buf = io.BytesIO()
            sf.write(buf, audio_samples, effective_sr, format="WAV")
            pcm_bytes = buf.getvalue()
        else:  # "pcm16", "pcm_s16le", etc.
            clipped = np.clip(audio_samples, -1.0, 1.0)
            pcm_bytes = (clipped * 32767.0).astype(np.int16).tobytes()

        duration_sec = len(audio_samples) / effective_sr if effective_sr > 0 else 0.0
        return pcm_bytes, audio_samples, effective_sr, duration_sec

    # Main IPC Command Loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            action = req.get("action", "generate")

            if action in ("stop", "exit", "shutdown"):
                ipc_out.write(json.dumps({"status": "bye"}) + "\n")
                ipc_out.flush()
                break

            if action == "ping":
                ipc_out.write(json.dumps({"status": "pong"}) + "\n")
                ipc_out.flush()
                continue

            voice = req.get("voice", "if_sara")
            speed = float(req.get("speed", 1.0))
            pitch = float(req.get("pitch", 0.0))
            language = req.get("language", "it")
            req_sr = int(req.get("sample_rate", 24000))
            req_format = str(req.get("format", "pcm16")).lower()

            if action == "generate_chunk":
                chunk_text = req.get("text", "").strip()
                if not chunk_text:
                    ipc_out.write(json.dumps({"status": "error", "error": "Empty chunk text"}) + "\n")
                    ipc_out.flush()
                    continue

                seq_id = int(req.get("seq", 0))
                t0 = time.perf_counter()
                pcm_bytes, audio_samples, sr, duration_sec = synthesize_segment(
                    chunk_text,
                    voice=voice,
                    speed=speed,
                    pitch=pitch,
                    language=language,
                    target_sample_rate=req_sr,
                    format_type=req_format
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                pcm_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                word_timestamps = compute_word_timestamps(chunk_text, duration_sec)

                words_count = len(chunk_text.split())
                res = {
                    "status": "success",
                    "event": "audio_frame",
                    "seq": seq_id,
                    "sample_rate": sr,
                    "num_samples": len(audio_samples),
                    "duration_sec": round(duration_sec, 3),
                    "duration_ms": round(duration_sec * 1000.0, 2),
                    "latency_ms": round(elapsed_ms, 2),
                    "pcm_b64": pcm_b64,
                    "text": chunk_text,
                    "word_timestamps": word_timestamps,
                    "num_frames": 1,
                    "num_tokens": words_count
                }
                ipc_out.write(json.dumps(res) + "\n")
                ipc_out.flush()
                continue

            if action == "generate_stream":
                text = req.get("text", "").strip()
                if not text:
                    ipc_out.write(json.dumps({"status": "error", "error": "Empty text prompt"}) + "\n")
                    ipc_out.flush()
                    continue

                strategy = req.get("chunk_strategy", "sentences")
                words_per_chunk = int(req.get("words_per_chunk", 8))
                chunks = split_into_chunks(text, strategy=strategy, words_per_chunk=words_per_chunk)

                t_stream_start = time.perf_counter()
                first_latency_ms = None
                total_samples = 0
                total_duration = 0.0

                for seq_id, chunk in enumerate(chunks):
                    t_chunk_start = time.perf_counter()
                    pcm_bytes, audio_samples, sr, dur_sec = synthesize_segment(
                        chunk,
                        voice=voice,
                        speed=speed,
                        pitch=pitch,
                        language=language,
                        target_sample_rate=req_sr,
                        format_type=req_format
                    )
                    chunk_elapsed_ms = (time.perf_counter() - t_chunk_start) * 1000.0
                    if first_latency_ms is None:
                        first_latency_ms = (time.perf_counter() - t_stream_start) * 1000.0

                    total_samples += len(audio_samples)
                    total_duration += dur_sec
                    pcm_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                    word_timestamps = compute_word_timestamps(chunk, dur_sec)

                    frame_meta = {
                        "event": "audio_frame",
                        "seq": seq_id,
                        "num_samples": len(audio_samples),
                        "sample_rate": sr,
                        "duration_sec": round(dur_sec, 3),
                        "duration_ms": round(dur_sec * 1000.0, 2),
                        "chunk_latency_ms": round(chunk_elapsed_ms, 2),
                        "ttfa_ms": round(first_latency_ms, 2) if seq_id == 0 else None,
                        "pcm_b64": pcm_b64,
                        "text_segment": chunk,
                        "word_timestamps": word_timestamps
                    }
                    ipc_out.write(json.dumps(frame_meta) + "\n")
                    ipc_out.flush()

                total_elapsed_ms = (time.perf_counter() - t_stream_start) * 1000.0
                total_elapsed_sec = total_elapsed_ms / 1000.0
                rtf = total_elapsed_sec / total_duration if total_duration > 0 else 0.0

                end_meta = {
                    "event": "stream_end",
                    "total_samples": total_samples,
                    "total_duration": round(total_duration, 3),
                    "total_duration_sec": round(total_duration, 3),
                    "latency_ms": round(total_elapsed_ms, 2),
                    "ttfa_ms": round(first_latency_ms, 2) if first_latency_ms else 0.0,
                    "rtf": round(rtf, 3),
                    "num_chunks": len(chunks),
                    "sample_rate": sr,
                    "format": req_format
                }
                ipc_out.write(json.dumps(end_meta) + "\n")
                ipc_out.flush()
                continue

            if action == "generate":
                text = req.get("text", "").strip()
                if not text:
                    ipc_out.write(json.dumps({"status": "error", "error": "Empty text prompt"}) + "\n")
                    ipc_out.flush()
                    continue

                t0 = time.perf_counter()
                pcm_bytes, audio_samples, sr, duration_sec = synthesize_segment(
                    text,
                    voice=voice,
                    speed=speed,
                    pitch=pitch,
                    language=language,
                    target_sample_rate=req_sr,
                    format_type=req_format
                )

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                elapsed_sec = elapsed_ms / 1000.0
                rtf = elapsed_sec / duration_sec if duration_sec > 0 else 0.0
                word_timestamps = compute_word_timestamps(text, duration_sec)

                out_path = req.get("output_path")
                if not out_path:
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    out_path = tmp_file.name
                    tmp_file.close()

                sf.write(out_path, audio_samples, sr)

                words_count = len(text.split())
                res = {
                    "status": "success",
                    "audio_path": out_path,
                    "sample_rate": sr,
                    "duration_sec": round(duration_sec, 3),
                    "latency_ms": round(elapsed_ms, 2),
                    "rtf": round(rtf, 3),
                    "num_samples": len(audio_samples),
                    "num_frames": 1,
                    "num_tokens": words_count,
                    "tokens_per_sec": round(words_count / elapsed_sec, 2) if elapsed_sec > 0 else 0.0,
                    "frames_per_sec": round(1.0 / elapsed_sec, 2) if elapsed_sec > 0 else 0.0,
                    "word_timestamps": word_timestamps,
                    "device": str(torch_device),
                    "model": model_id
                }
                ipc_out.write(json.dumps(res) + "\n")
                ipc_out.flush()

        except Exception as e:
            sys.stderr.write(f"[Kokoro Worker] Error during synthesis: {e}\n")
            ipc_out.write(json.dumps({"status": "error", "error": str(e)}) + "\n")
            ipc_out.flush()

if __name__ == "__main__":
    main()
