import sys
import os
import json
import time

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
        import torchaudio
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        
        # Optimize CPU threads for Zen 5
        torch.set_num_threads(8)
        
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        target_device = "cpu" if device in ("cpu", "npu") else device
        model.to(target_device)
        model.eval()
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
                        gen_out = model.generate(inputs.input_features.to(target_device), **gen_kwargs)
                    
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
