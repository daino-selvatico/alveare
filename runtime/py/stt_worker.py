import sys
import os
import json
import time

def main():
    model_id = "FunAudioLLM/SenseVoiceSmall"
    device = "cpu"
    if len(sys.argv) > 1 and sys.argv[1]:
        model_id = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2]:
        device = sys.argv[2]

    # Save real IPC stdout descriptor and redirect sys.stdout to sys.stderr during model load
    ipc_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

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
            res = model.generate(
                input=audio_target,
                cache={},
                language=language if language != "auto" else "auto",
                use_itn=use_itn
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            raw_text = ""
            if res and isinstance(res, list) and len(res) > 0:
                raw_text = res[0].get("text", "")
                
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
