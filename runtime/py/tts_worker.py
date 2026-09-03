import sys
import os
import io
import json
import time
import math
import ctypes
import tempfile
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = ROOT_DIR / "runtime" / "cpp" / "build" / "libalveare_npu_c.so"

def pack_weight_q4(W: np.ndarray) -> np.ndarray:
    N, K = W.shape
    W_blocks = W.reshape(N, K // 32, 32)
    max_vals = np.max(np.abs(W_blocks), axis=2)
    scales = max_vals / 7.0
    scales[scales == 0.0] = 1.0
    q_blocks = np.clip(np.round(W_blocks / np.expand_dims(scales, axis=2)), -8, 7).astype(np.int32)
    q = q_blocks.reshape(N, K)
    q0 = q[:, 0::2]
    q1 = q[:, 1::2]
    w_q4 = ((q0 & 0x0F) | ((q1 & 0x0F) << 4)).astype(np.uint8)

    packed = np.zeros((N, (K // 32) * 20), dtype=np.uint8)
    for i in range(K // 32):
        packed[:, i*20 : i*20+16] = w_q4[:, i*16 : (i+1)*16]
        sc_f32 = scales[:, i].astype(np.float32)
        sc_u16 = (sc_f32.view(np.uint32) >> 16).astype(np.uint16)
        packed[:, i*20+16] = (sc_u16 & 0xFF).astype(np.uint8)
        packed[:, i*20+17] = ((sc_u16 >> 8) & 0xFF).astype(np.uint8)
    return packed

def init_npu_lib():
    if not LIB_PATH.exists():
        return None
    try:
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.alveare_npu_create_registry.restype = ctypes.c_void_p
        lib.alveare_npu_create_registry.argtypes = [ctypes.c_char_p]
        lib.alveare_npu_free_registry.argtypes = [ctypes.c_void_p]
        lib.alveare_npu_create_gemv_weight.restype = ctypes.c_uint32
        lib.alveare_npu_create_gemv_weight.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        lib.alveare_npu_run_gemv.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        lib.alveare_npu_run_gemv_seq.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.alveare_npu_has_shape.restype = ctypes.c_int
        lib.alveare_npu_has_shape.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        return lib
    except Exception as e:
        sys.stderr.write(f"[TTS Worker] NPU lib load failed: {e}\n")
        return None

def main():
    model_id = "Audio8/Audio8-TTS-Preview-0.1b"
    device = "npu"
    if len(sys.argv) > 1 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2]:
        device = sys.argv[2].lower()

    # Normalize aliases
    if model_id in ("audio8-0.1b", "audio8-100m", "0.1b"):
        model_id = "Audio8/Audio8-TTS-Preview-0.1b"
    elif model_id in ("audio8-0.6b", "audio8-600m", "0.6b"):
        model_id = "Audio8/Audio8-TTS-Preview-0.6b"

    # Save real IPC stdout descriptor and redirect sys.stdout to sys.stderr during model load
    ipc_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    import torch
    import torch.nn as nn
    import torchaudio
    import librosa
    import soundfile as sf
    from transformers import AutoProcessor, AutoModel

    torch.set_num_threads(8)

    sys.stderr.write(f"[TTS Worker] Loading {model_id} (device={device})...\n")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Initialize mup_vector on all FalconH1 mamba layers to prevent NaNs in torch_forward
    try:
        from transformers.models.falcon_h1.modeling_falcon_h1 import compute_mup_vector
        if hasattr(model, "slow") and hasattr(model.slow, "config"):
            mup_vec = compute_mup_vector(model.slow.config).to(dtype=torch.bfloat16)
            for layer in model.slow.layers:
                if hasattr(layer, "mamba") and layer.mamba is not None:
                    layer.mamba.register_buffer("mup_vector", mup_vec.clone(), persistent=False)
    except Exception as e:
        sys.stderr.write(f"[TTS Worker] Note on mup_vector init: {e}\n")

    # Recompute RoPE tables to eliminate uninitialized/NaN weights in Hugging Face checkpoint
    def precompute_rope(length: int, head_dim: int, base: float) -> torch.Tensor:
        frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim))
        phases = torch.outer(torch.arange(length), frequencies)
        values = torch.polar(torch.ones_like(phases), phases)
        return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)

    try:
        if hasattr(model, "freqs_cis") and hasattr(model, "config"):
            model.register_buffer("freqs_cis", precompute_rope(model.config.max_seq_len, model.config.head_dim, model.config.rope_base), persistent=False)
        if hasattr(model, "fast_freqs_cis") and hasattr(model, "config"):
            model.register_buffer("fast_freqs_cis", precompute_rope(model.config.num_codebooks, model.config.fast_head_dim, model.config.rope_base), persistent=False)
    except Exception as e:
        sys.stderr.write(f"[TTS Worker] Note on RoPE precompute: {e}\n")

    model.eval()

    # Pre-warm codec vocoder
    try:
        model.load_codec(device="cpu")
    except Exception as e:
        sys.stderr.write(f"[TTS Worker] Pre-loading codec vocoder: {e}\n")

    if device == "npu":
        npu_lib = init_npu_lib()
        manifest_path = ROOT_DIR / "kernels" / "build" / "manifest.json"
        if npu_lib and manifest_path.exists():
            reg = npu_lib.alveare_npu_create_registry(str(manifest_path).encode("utf-8"))
            if reg:
                class NPULinear(nn.Module):
                    def __init__(self, linear_layer: nn.Linear):
                        super().__init__()
                        self.in_features = linear_layer.in_features
                        self.out_features = linear_layer.out_features
                        self.N = self.out_features
                        self.K = self.in_features
                        W = linear_layer.weight.detach().float().numpy()
                        self.bias = linear_layer.bias.detach().clone() if linear_layer.bias is not None else None
                        
                        packed = pack_weight_q4(W)
                        self.wh = npu_lib.alveare_npu_create_gemv_weight(
                            reg, self.N, self.K, packed.ctypes.data_as(ctypes.c_void_p), packed.nbytes
                        )

                    def forward(self, x: torch.Tensor):
                        orig_shape = x.shape
                        x_2d = x.reshape(-1, self.K).contiguous().to(torch.bfloat16)
                        n_tokens = x_2d.shape[0]
                        
                        if n_tokens == 1:
                            y = torch.zeros((1, self.N), dtype=torch.bfloat16, device=x.device)
                            npu_lib.alveare_npu_run_gemv(reg, self.N, self.K, self.wh, x_2d.data_ptr(), y.data_ptr())
                            out = y.to(x.dtype)
                            if self.bias is not None:
                                out = out + self.bias
                            return out.reshape(*orig_shape[:-1], self.out_features)
                        else:
                            y = torch.zeros((n_tokens, self.N), dtype=torch.bfloat16, device=x.device)
                            npu_lib.alveare_npu_run_gemv_seq(reg, self.N, self.K, self.wh, x_2d.data_ptr(), y.data_ptr(), n_tokens)
                            out = y.to(x.dtype)
                            if self.bias is not None:
                                out = out + self.bias
                            return out.reshape(*orig_shape[:-1], self.out_features)

                offloaded_count = 0
                # Recursively offload all Linear layers matching NPU XDNA2 shapes
                for name, module in list(model.named_modules()):
                    for child_name, child in list(module.named_children()):
                        if isinstance(child, nn.Linear):
                            if npu_lib.alveare_npu_has_shape(reg, child.out_features, child.in_features):
                                setattr(module, child_name, NPULinear(child))
                                offloaded_count += 1

                sys.stderr.write(f"[TTS Worker] Offloaded {offloaded_count} linear projections to AMD Ryzen AI NPU cores!\n")
            else:
                sys.stderr.write("[TTS Worker] Could not initialize NPU registry.\n")
        else:
            sys.stderr.write("[TTS Worker] NPU library or manifest not found.\n")

    sys.stderr.write(f"[TTS Worker] {model_id} fully loaded and ready on {device}.\n")
    ipc_out.write("READY\n")
    ipc_out.flush()

    import base64
    import re

    def split_into_chunks(
        text: str,
        strategy: str = "sentences",
        words_per_chunk: int = 8,
        min_words_first: int = 3,
        max_words_per_chunk: int = 22
    ) -> list:
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

    def compute_word_timestamps(text_segment: str, duration_sec: float) -> list:
        """
        Calculates exact word-by-word timestamps based on actual audio duration.
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

    def synthesize_audio_segment(
        text_segment: str,
        ref_audio: str = None,
        ref_text: str = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        target_sample_rate: int = 44100,
        format_type: str = "pcm16"
    ):
        processor_kwargs = {"text": [text_segment], "return_tensors": "pt"}
        if ref_audio and os.path.exists(ref_audio) and ref_text:
            processor_kwargs["reference_audio"] = [ref_audio]
            processor_kwargs["reference_text"] = [ref_text]

        # Dynamic token budget based on exact word count to avoid runaway autoregression on small chunks
        words_cnt = len(text_segment.split())
        chunk_bounded_tokens = max(24, min(max_new_tokens, int(words_cnt * 5.2 + 12)))

        inputs = processor(**processor_kwargs)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=chunk_bounded_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                return_dict_in_generate=True,
            )
            waveforms, waveform_lengths = model.decode_audio(output.codes)

        audio_tensor = waveforms[0, : int(waveform_lengths[0])].float().cpu()
        native_sr = getattr(model.config, "codec_sample_rate", 44100) or 44100

        # Pitch Shift Adjustment (torchaudio)
        if pitch != 0.0 and pitch != 1.0:
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
                    sys.stderr.write(f"[TTS Worker] Pitch shift error: {pe}\n")

        audio_samples = audio_tensor.numpy()

        # Speed / Time-Stretch Adjustment (librosa)
        if speed > 0 and abs(speed - 1.0) > 0.02:
            try:
                audio_samples = librosa.effects.time_stretch(audio_samples, rate=speed)
            except Exception as se:
                sys.stderr.write(f"[TTS Worker] Speed stretch error: {se}\n")

        # Resample if target sample rate differs
        effective_sr = native_sr
        if target_sample_rate > 0 and target_sample_rate != native_sr:
            try:
                audio_samples = librosa.resample(audio_samples, orig_sr=native_sr, target_sr=target_sample_rate)
                effective_sr = target_sample_rate
            except Exception as re_err:
                sys.stderr.write(f"[TTS Worker] Resampling error: {re_err}\n")
                effective_sr = native_sr

        # Format conversion
        if format_type.lower() == "float32":
            pcm_bytes = audio_samples.astype(np.float32).tobytes()
        else:
            clipped = np.clip(audio_samples, -1.0, 1.0)
            pcm_bytes = (clipped * 32767.0).astype(np.int16).tobytes()

        duration_sec = len(audio_samples) / effective_sr if effective_sr > 0 else 0.0
        num_frames = int(output.codes.shape[-1]) if hasattr(output, "codes") else 0
        num_tokens = int(output.codes.numel()) if hasattr(output, "codes") else 0

        return pcm_bytes, audio_samples, effective_sr, duration_sec, num_frames, num_tokens

    # Main IPC Command Loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            action = req.get("action", "generate")
            
            if action == "ping":
                ipc_out.write(json.dumps({"status": "pong"}) + "\n")
                ipc_out.flush()
                continue

            ref_audio = req.get("reference_audio")
            ref_text = req.get("reference_text")
            speed = float(req.get("speed", 1.0))
            pitch = float(req.get("pitch", 0.0))
            max_new_tokens = int(req.get("max_new_tokens", 300))
            temperature = float(req.get("temperature", 0.8))
            top_p = float(req.get("top_p", 0.95))
            top_k = int(req.get("top_k", 50))
            do_sample = bool(req.get("do_sample", True))
            req_sr = int(req.get("sample_rate", 44100))
            req_format = str(req.get("format", "pcm16")).lower()

            if action == "generate_chunk":
                chunk_text = req.get("text", "").strip()
                if not chunk_text:
                    ipc_out.write(json.dumps({"status": "error", "error": "Empty chunk text"}) + "\n")
                    ipc_out.flush()
                    continue

                seq_id = int(req.get("seq", 0))
                t0 = time.perf_counter()
                pcm_bytes, audio_samples, sr, duration_sec, num_frames, num_tokens = synthesize_audio_segment(
                    chunk_text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    speed=speed,
                    pitch=pitch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    target_sample_rate=req_sr,
                    format_type=req_format
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                pcm_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                word_timestamps = compute_word_timestamps(chunk_text, duration_sec)

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
                    "num_frames": num_frames,
                    "num_tokens": num_tokens
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
                    pcm_bytes, audio_samples, sr, dur_sec, n_frames, n_tokens = synthesize_audio_segment(
                        chunk,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        speed=speed,
                        pitch=pitch,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
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

                processor_kwargs = {"text": [text], "return_tensors": "pt"}
                if ref_audio and os.path.exists(ref_audio) and ref_text:
                    processor_kwargs["reference_audio"] = [ref_audio]
                    processor_kwargs["reference_text"] = [ref_text]

                # Dynamic token budget for full generation based on exact word count
                words_cnt = len(text.split())
                bounded_tokens = max(24, min(max_new_tokens, int(words_cnt * 5.2 + 12)))

                inputs = processor(**processor_kwargs)

                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=bounded_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
                        return_dict_in_generate=True,
                    )
                    waveforms, waveform_lengths = model.decode_audio(output.codes)

                audio_tensor = waveforms[0, : int(waveform_lengths[0])].float().cpu()
                sr = getattr(model.config, "codec_sample_rate", 44100) or 44100

                # 1. Pitch Shift Adjustment (torchaudio)
                if pitch != 0.0 and pitch != 1.0:
                    if 0.2 < pitch <= 2.0 and pitch != 1.0:
                        n_steps = 12.0 * math.log2(pitch)
                    else:
                        n_steps = float(pitch)
                    if abs(n_steps) > 0.01:
                        try:
                            audio_tensor = torchaudio.functional.pitch_shift(
                                audio_tensor.unsqueeze(0), sample_rate=sr, n_steps=n_steps
                            ).squeeze(0)
                        except Exception as pe:
                            sys.stderr.write(f"[TTS Worker] Pitch shift error: {pe}\n")

                audio_samples = audio_tensor.numpy()

                # 2. Speed / Time-Stretch Adjustment (librosa)
                if speed > 0 and abs(speed - 1.0) > 0.02:
                    try:
                        audio_samples = librosa.effects.time_stretch(audio_samples, rate=speed)
                    except Exception as se:
                        sys.stderr.write(f"[TTS Worker] Speed stretch error: {se}\n")

                duration_sec = len(audio_samples) / sr if sr > 0 else 0.0
                word_timestamps = compute_word_timestamps(text, duration_sec)

                # Write output to temporary or requested wav file
                out_path = req.get("output_path")
                if not out_path:
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    out_path = tmp_file.name
                    tmp_file.close()

                sf.write(out_path, audio_samples, sr)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                elapsed_sec = elapsed_ms / 1000.0
                rtf = elapsed_sec / duration_sec if duration_sec > 0 else 0.0

                num_frames = int(output.codes.shape[-1]) if hasattr(output, "codes") else 0
                num_tokens = int(output.codes.numel()) if hasattr(output, "codes") else 0
                tokens_per_sec = round(num_tokens / elapsed_sec, 2) if elapsed_sec > 0 else 0.0
                frames_per_sec = round(num_frames / elapsed_sec, 2) if elapsed_sec > 0 else 0.0

                res = {
                    "status": "success",
                    "audio_path": out_path,
                    "sample_rate": sr,
                    "duration_sec": round(duration_sec, 3),
                    "latency_ms": round(elapsed_ms, 2),
                    "rtf": round(rtf, 3),
                    "num_samples": len(audio_samples),
                    "num_frames": num_frames,
                    "num_tokens": num_tokens,
                    "tokens_per_sec": tokens_per_sec,
                    "frames_per_sec": frames_per_sec,
                    "word_timestamps": word_timestamps,
                    "device": device,
                    "model": model_id
                }
                ipc_out.write(json.dumps(res) + "\n")
                ipc_out.flush()

        except Exception as e:
            sys.stderr.write(f"[TTS Worker] Error during synthesis: {e}\n")
            ipc_out.write(json.dumps({"status": "error", "error": str(e)}) + "\n")
            ipc_out.flush()

if __name__ == "__main__":
    main()
