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
                            
                            y_2d = torch.zeros(n_tokens, self.N, dtype=torch.bfloat16, device=x.device)
                            npu_lib.alveare_npu_run_gemv_seq(
                                reg, self.N, self.K, self.wh, x_2d.data_ptr(), y_2d.data_ptr(), n_tokens
                            )
                            
                            out = y_2d.to(x.dtype).reshape(*orig_shape[:-1], self.N)
                            if self.bias is not None:
                                out = out + self.bias
                            return out

                    converted_count = 0
                    for name, module in list(model.model.named_modules()):
                        for child_name, child in list(module.named_children()):
                            if isinstance(child, nn.Linear):
                                if npu_lib.alveare_npu_has_shape(reg, child.out_features, child.in_features):
                                    setattr(module, child_name, NPULinear(child))
                                    converted_count += 1
                    sys.stderr.write(f"[WhisperSTT] Successfully offloaded {converted_count} linear layers to AMD Ryzen AI NPU hardware (XDNA2)!\n")
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
                        "max_new_tokens": 256,
                        "no_repeat_ngram_size": 3,
                        "return_dict_in_generate": True
                    }
                    if language and language not in ("auto", "unknown", ""):
                        try:
                            forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
                            gen_kwargs["forced_decoder_ids"] = forced_ids
                            detected_lang = language
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
                
            ipc_out.write(json.dumps({
                "status": "success",
                "raw_text": raw_text,
                "language": detected_lang,
                "latency_ms": round(elapsed_ms, 2)
            }) + "\n")
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
