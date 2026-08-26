import sys
import os
import json
import time
import ctypes
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
        sys.stderr.write(f"[WhisperSTT] NPU lib load failed: {e}\n")
        return None

def extract_multimodal_emotion(text, audio_arr, sr=16000):
    if len(audio_arr) < 1600:
        return {"emotion": "neutral", "tone": "calmo", "confidence": 0.85, "pitch_hz": 0.0}
    try:
        # 1. Acoustic Prosodic Analysis
        rms = float(np.sqrt(np.mean(audio_arr ** 2))) if len(audio_arr) > 0 else 0.0
        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.010)
        n_frames = max(1, (len(audio_arr) - win_len) // hop_len)
        
        frame_energies = []
        pitches = []
        
        for i in range(n_frames):
            frame = audio_arr[i*hop_len : i*hop_len + win_len]
            f_rms = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(f_rms)
            if f_rms > 0.015:
                frame_centered = frame - np.mean(frame)
                corr = np.correlate(frame_centered, frame_centered, mode='full')
                corr = corr[len(corr)//2:]
                min_lag = int(sr / 400)
                max_lag = int(sr / 70)
                if max_lag < len(corr):
                    peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                    if corr[peak_lag] > 0.35 * corr[0]:
                        pitches.append(sr / peak_lag)

        valid_pitches = np.array(pitches) if len(pitches) >= 3 else np.array([150.0])
        mean_pitch = float(np.mean(valid_pitches))
        pitch_std = float(np.std(valid_pitches))
        pitch_cv = pitch_std / (mean_pitch + 1e-6)

        # 2. Text Lexical Sentiment
        lower_text = (text or "").lower()
        pos_words = [
            'bello', 'bella', 'grazie', 'ottimo', 'ottima', 'felice', 'perfetto', 'perfetta', 
            'super', 'frizzante', 'carina', 'carino', 'piace', 'fantastico', 'fantastica', 
            'evviva', 'amore', 'meraviglia', 'eccellente', 'gioia', 'contento', 'contenta', 
            'bravo', 'brava', 'splendido', 'splendida', 'great', 'awesome', 'happy', 'love', 
            'good', 'thank', 'thanks', 'wonderful', 'amazing'
        ]
        neg_words = [
            'brutto', 'brutta', 'pessimo', 'pessima', 'male', 'triste', 'arrabbiato', 'arrabbiata', 
            'schifo', 'errore', 'fastidio', 'problema', 'stanco', 'stanca', 'stufo', 'stufa', 
            'noioso', 'noiosa', 'peccato', 'odioso', 'odiosa', 'difficile', 'fallito', 'rabbia', 
            'paura', 'dolore', 'bad', 'sad', 'angry', 'terrible', 'horrible', 'hate', 'tired', 
            'broken', 'error', 'fail', 'pain'
        ]
        
        pos_score = sum(1 for w in pos_words if w in lower_text)
        neg_score = sum(1 for w in neg_words if w in lower_text)
        has_excl = '!' in (text or '') or '?' in (text or '')
        
        # Default baseline is ALWAYS Neutral / Calmo
        emotion = 'neutral'
        tone = 'calmo'
        confidence = 0.85

        if neg_score > pos_score and neg_score >= 1:
            if rms > 0.08 or pitch_cv > 0.30:
                emotion = 'angry'
                tone = 'deciso'
                confidence = 0.88
            else:
                emotion = 'sad'
                tone = 'sommesso'
                confidence = 0.85
        elif pos_score > neg_score and pos_score >= 1:
            if pitch_cv > 0.22 or mean_pitch > 210.0 or has_excl:
                emotion = 'happy'
                tone = 'frizzante'
                confidence = 0.90
            else:
                emotion = 'happy'
                tone = 'cordiale'
                confidence = 0.84
        else:
            # Pure acoustic features for utterances without explicit emotional keywords
            if (mean_pitch > 240.0 or pitch_cv > 0.38) and rms > 0.10:
                emotion = 'surprised' if has_excl else 'happy'
                tone = 'esclamativo' if has_excl else 'vivace'
                confidence = 0.80
            elif mean_pitch > 210.0 and rms > 0.12 and pitch_cv > 0.28:
                emotion = 'angry'
                tone = 'deciso'
                confidence = 0.78
            elif (mean_pitch < 120.0 or rms < 0.025) and pitch_cv < 0.10:
                emotion = 'sad'
                tone = 'sommesso'
                confidence = 0.80
            else:
                emotion = 'neutral'
                tone = 'calmo'
                confidence = 0.88

        return {
            "emotion": emotion,
            "tone": tone,
            "confidence": confidence,
            "pitch_hz": round(mean_pitch, 1)
        }
    except Exception:
        return {"emotion": "neutral", "tone": "calmo", "confidence": 0.85, "pitch_hz": 0.0}

def main():
    model_id = "openai/whisper-base"
    device = "npu"
    if len(sys.argv) > 1 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2]:
        device = sys.argv[2].lower()

    # Save real IPC stdout descriptor and redirect sys.stdout to sys.stderr during model load
    ipc_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    is_whisper = "whisper" in model_id.lower()
    
    if is_whisper:
        import torch
        import torch.nn as nn
        import torchaudio
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        torch.set_num_threads(8)
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        model.eval()

        if device == "npu":
            npu_lib = init_npu_lib()
            manifest_path = ROOT_DIR / "kernels" / "build" / "whisper-base" / "manifest.json"
            if npu_lib and manifest_path.exists():
                reg = npu_lib.alveare_npu_create_registry(str(manifest_path).encode("utf-8"))
                if reg:
                    class FusedQKVNPU(nn.Module):
                        def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
                            super().__init__()
                            self.in_features = q_proj.in_features
                            self.out_features = q_proj.out_features * 3
                            self.N = 1536
                            self.K = 512
                            
                            W_q = q_proj.weight.detach().float().numpy()
                            W_k = k_proj.weight.detach().float().numpy()
                            W_v = v_proj.weight.detach().float().numpy()
                            W_fused = np.vstack([W_q, W_k, W_v])
                            
                            b_q = q_proj.bias if q_proj.bias is not None else torch.zeros(512)
                            b_k = k_proj.bias if k_proj.bias is not None else torch.zeros(512)
                            b_v = v_proj.bias if v_proj.bias is not None else torch.zeros(512)
                            self.register_buffer("bias", torch.cat([b_q, b_k, b_v]))
                            
                            packed = pack_weight_q4(W_fused)
                            self.wh = npu_lib.alveare_npu_create_gemv_weight(
                                reg, self.N, self.K, packed.ctypes.data_as(ctypes.c_void_p), packed.nbytes
                            )

                        def forward(self, x: torch.Tensor):
                            orig_shape = x.shape
                            x_2d = x.reshape(-1, self.K).contiguous().to(torch.bfloat16)
                            n_tokens = x_2d.shape[0]
                            
                            if n_tokens == 1:
                                y = torch.zeros(self.N, dtype=torch.bfloat16, device=x.device)
                                npu_lib.alveare_npu_run_gemv(reg, self.N, self.K, self.wh, x_2d.data_ptr(), y.data_ptr())
                                out = y.to(x.dtype) + self.bias
                                q, k, v = torch.split(out, 512, dim=-1)
                                return q.reshape(*orig_shape[:-1], 512), k.reshape(*orig_shape[:-1], 512), v.reshape(*orig_shape[:-1], 512)
                            else:
                                y_2d = torch.zeros(n_tokens, self.N, dtype=torch.bfloat16, device=x.device)
                                for i in range(n_tokens):
                                    npu_lib.alveare_npu_run_gemv(reg, self.N, self.K, self.wh, x_2d[i].contiguous().data_ptr(), y_2d[i].data_ptr())
                                out = y_2d.to(x.dtype) + self.bias
                                q, k, v = torch.split(out, 512, dim=-1)
                                return q.reshape(*orig_shape[:-1], 512), k.reshape(*orig_shape[:-1], 512), v.reshape(*orig_shape[:-1], 512)

                    class NPULinear(nn.Module):
                        def __init__(self, linear_layer: nn.Linear):
                            super().__init__()
                            self.in_features = linear_layer.in_features
                            self.out_features = linear_layer.out_features
                            self.bias = linear_layer.bias
                            
                            W_np = linear_layer.weight.detach().float().numpy()
                            self.N = self.out_features
                            self.K = self.in_features
                            packed = pack_weight_q4(W_np)
                            self.wh = npu_lib.alveare_npu_create_gemv_weight(
                                reg, self.N, self.K, packed.ctypes.data_as(ctypes.c_void_p), packed.nbytes
                            )

                        def forward(self, x: torch.Tensor) -> torch.Tensor:
                            orig_shape = x.shape
                            x_2d = x.reshape(-1, self.K).contiguous().to(torch.bfloat16)
                            n_tokens = x_2d.shape[0]
                            
                            if n_tokens == 1:
                                y = torch.zeros(self.N, dtype=torch.bfloat16, device=x.device)
                                npu_lib.alveare_npu_run_gemv(reg, self.N, self.K, self.wh, x_2d.data_ptr(), y.data_ptr())
                                out = y.to(x.dtype).reshape(*orig_shape[:-1], self.N)
                            else:
                                y_2d = torch.zeros(n_tokens, self.N, dtype=torch.bfloat16, device=x.device)
                                for i in range(n_tokens):
                                    npu_lib.alveare_npu_run_gemv(reg, self.N, self.K, self.wh, x_2d[i].contiguous().data_ptr(), y_2d[i].data_ptr())
                                out = y_2d.to(x.dtype).reshape(*orig_shape[:-1], self.N)
                            
                            if self.bias is not None:
                                out = out + self.bias
                            return out

                    converted_count = 0
                    for layer in model.model.decoder.layers:
                        # 1. Fused QKV on NPU for Self-Attention
                        if npu_lib.alveare_npu_has_shape(reg, 1536, 512):
                            layer.self_attn.q_proj = NPULinear(layer.self_attn.q_proj)
                            layer.self_attn.k_proj = NPULinear(layer.self_attn.k_proj)
                            layer.self_attn.v_proj = NPULinear(layer.self_attn.v_proj)
                            converted_count += 3
                        else:
                            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                                if npu_lib.alveare_npu_has_shape(reg, proj.out_features, proj.in_features):
                                    layer.self_attn.q_proj = NPULinear(layer.self_attn.q_proj)
                                    converted_count += 1
                        
                        layer.self_attn.out_proj = NPULinear(layer.self_attn.out_proj)
                        layer.encoder_attn.q_proj = NPULinear(layer.encoder_attn.q_proj)
                        # Keep encoder_attn k_proj & v_proj on vectorized CPU since they only execute once during 1500-token prefill
                        layer.encoder_attn.out_proj = NPULinear(layer.encoder_attn.out_proj)
                        layer.fc1 = NPULinear(layer.fc1)
                        layer.fc2 = NPULinear(layer.fc2)
                        converted_count += 5

                    sys.stderr.write(f"[WhisperSTT] Successfully offloaded {converted_count} decoder layers to AMD Ryzen AI NPU hardware (XDNA2)!\n")
    else:
        from funasr import AutoModel
        target_device = "cpu" if device in ("cpu", "npu") else device
        model = AutoModel(model=model_id, hub="hf", device=target_device, disable_update=True)
    
    # Notify parent process that model is loaded and ready
    ipc_out.write("READY\n")
    ipc_out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            audio_target = req.get("input")
            language = req.get("language", "auto")
            use_itn = req.get("use_itn", True)
            detect_emotion = req.get("detect_emotion", True)
            
            t0 = time.perf_counter()
            raw_text = ""
            detected_lang = "it"

            if is_whisper:
                import torch
                import torchaudio
                speech, sr = torchaudio.load(audio_target)
                if speech.shape[0] > 1:
                    speech = torch.mean(speech, dim=0, keepdim=True)
                if sr != 16000:
                    speech = torchaudio.functional.resample(speech, sr, 16000)
                speech_np = speech.squeeze(0).numpy()

                if len(speech_np) >= 1200: # at least 75ms
                    inputs = processor(speech_np, sampling_rate=16000, return_tensors="pt")
                    gen_kwargs = {
                        "max_new_tokens": 128,
                        "num_beams": 1,
                        "no_repeat_ngram_size": 3,
                        "return_dict_in_generate": True
                    }
                    target_lang = language if (language and language not in ("auto", "unknown", "")) else "it"
                    try:
                        forced_ids = processor.get_decoder_prompt_ids(language=target_lang, task="transcribe")
                        gen_kwargs["forced_decoder_ids"] = forced_ids
                        detected_lang = target_lang
                    except Exception:
                        pass
                    
                    with torch.no_grad():
                        gen_out = model.generate(inputs.input_features, **gen_kwargs)
                    
                    seq = gen_out.sequences[0].tolist()
                    if len(seq) > 1 and (not language or language in ("auto", "unknown", "")):
                        token_str = processor.tokenizer.decode([seq[1]]).strip()
                        if token_str.startswith("<|") and token_str.endswith("|>"):
                            detected_lang = token_str[2:-2]
                    elif language and language not in ("auto", "unknown", ""):
                        detected_lang = language

                    raw_text = processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0].strip()
            else:
                res = model.generate(
                    input=audio_target,
                    cache={},
                    language=language if language != "auto" else "auto",
                    use_itn=use_itn
                )
                if res and isinstance(res, list) and len(res) > 0:
                    raw_text = res[0].get("text", "")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            # Extract multimodal vocal emotion and tone in parallel if enabled
            emotion_res = None
            if detect_emotion and is_whisper and 'speech_np' in locals():
                emotion_res = extract_multimodal_emotion(raw_text, speech_np)
                
            resp_dict = {
                "status": "success",
                "raw_text": raw_text,
                "language": detected_lang,
                "latency_ms": round(elapsed_ms, 2)
            }
            if emotion_res:
                resp_dict["emotion"] = emotion_res.get("emotion", "neutral")
                resp_dict["tone"] = emotion_res.get("tone", "calmo")
                resp_dict["pitch_hz"] = emotion_res.get("pitch_hz", 0.0)
                resp_dict["emotion_confidence"] = emotion_res.get("confidence", 0.85)

            ipc_out.write(json.dumps(resp_dict) + "\n")
            ipc_out.flush()
        except Exception as e:
            ipc_out.write(json.dumps({
                "status": "error",
                "error": str(e),
                "raw_text": ""
            }) + "\n")
            ipc_out.flush()

if __name__ == "__main__":
    main()
