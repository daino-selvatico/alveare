import sys
import os
import json
import time

def main():
    model_id = "openai/whisper-base"
    device = "cpu"
    if len(sys.argv) > 1 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2]:
        device = sys.argv[2]

    # Save real IPC stdout descriptor and redirect sys.stdout to sys.stderr during model load
    ipc_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    is_whisper = "whisper" in model_id.lower()
    
    if is_whisper:
        import torch
        import torchaudio
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        # Optimize CPU threads for Zen 5
        torch.set_num_threads(8)
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        model.to(device)
        model.eval()
    else:
        from funasr import AutoModel
        model = AutoModel(model=model_id, hub="hf", device=device, disable_update=True)
    
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
                        "no_repeat_ngram_size": 3
                    }
                    if language and language not in ("auto", "unknown"):
                        lang_map = {
                            "it": "italian", "en": "english", "es": "spanish", "fr": "french",
                            "de": "german", "zh": "chinese", "ja": "japanese", "ko": "korean",
                            "pt": "portuguese", "ru": "russian"
                        }
                        gen_kwargs["language"] = lang_map.get(language, language)
                    
                    with torch.no_grad():
                        gen_ids = model.generate(inputs.input_features.to(device), **gen_kwargs)
                    raw_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
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
