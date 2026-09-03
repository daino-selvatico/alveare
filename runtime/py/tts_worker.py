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
                
            if action == "generate":
                text = req.get("text", "").strip()
                if not text:
                    ipc_out.write(json.dumps({"status": "error", "error": "Empty text prompt"}) + "\n")
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

                t0 = time.perf_counter()

                processor_kwargs = {"text": [text], "return_tensors": "pt"}
                if ref_audio and os.path.exists(ref_audio) and ref_text:
                    processor_kwargs["reference_audio"] = [ref_audio]
                    processor_kwargs["reference_text"] = [ref_text]

                inputs = processor(**processor_kwargs)

                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
                        return_dict_in_generate=True,
                    )
                    waveforms, waveform_lengths = model.decode_audio(output.codes)

                audio_tensor = waveforms[0, : int(waveform_lengths[0])].float().cpu()
                sr = model.config.codec_sample_rate

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
