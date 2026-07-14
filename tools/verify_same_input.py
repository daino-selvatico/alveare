import sys
import os
import time
import subprocess
import requests
import json
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from runtime.py.model import LlamaNPUModel
from runtime.py.tokenizer_glue import TokenizerGlue

def main():
    print("=== Same-Input Token Fidelity Comparison ===")
    
    weights_dir = Path(__file__).resolve().parents[1] / "quantized_weights_gemma4"
    if not weights_dir.exists():
        print(f"Error: Weights dir {weights_dir} does not exist!")
        sys.exit(1)
        
    # 1. Initialize Alveare NPU Model & Tokenizer
    print("Initializing Alveare model...")
    model = LlamaNPUModel(weights_dir)
    tokenizer = TokenizerGlue("google/gemma-4-12b-it")
    
    prompt_messages = [{"role": "user", "content": "The capital of France is"}]
    input_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
    print(f"Alveare Chat Template Token IDs: {input_ids}")
    
    # 2. Run greedy generation on NPU (Alveare)
    print("\nRunning greedy generation on NPU (Alveare)...")
    model.reset_caches()
    
    # Prefill
    for pos in range(len(input_ids) - 1):
        _ = model.forward(input_ids[pos], pos, return_logits=False, use_npu=True)
        
    # Decode
    current_token_id = input_ids[-1]
    pos = len(input_ids) - 1
    
    npu_tokens = []
    npu_strings = []
    for i in range(8):
        logits = model.forward(current_token_id, pos, use_npu=True)
        next_token_id = int(np.argmax(logits))
        npu_tokens.append(next_token_id)
        npu_strings.append(tokenizer.decode([next_token_id]))
        print(f"  Step {i}: token={next_token_id} ({npu_strings[-1]!r})")
        if next_token_id in [tokenizer.eos_token_id, tokenizer.eot_token_id]:
            break
        current_token_id = next_token_id
        pos += 1
        
    # 3. Start llama-server and feed the same input token IDs
    print("\nStarting llama-server...")
    server_bin = "/home/daino/llama-mtp/llama.cpp/build/bin/llama-server"
    model_path = "/home/daino/llama-mtp/models/gemma-4-12b-it-UD-Q4_K_XL.gguf"
    port = 18081
    
    cmd = [
        server_bin,
        "-m", model_path,
        "-c", "2048",
        "--port", str(port),
        "-ngl", "0"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    url = f"http://127.0.0.1:{port}/completion"
    server_ready = False
    for _ in range(100):
        try:
            res = requests.get(f"http://127.0.0.1:{port}/health")
            if res.status_code == 200:
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)
        
    if not server_ready:
        print("Error: llama-server failed to start!")
        proc.terminate()
        sys.exit(1)
        
    print("llama-server is ready. Sending `/completion` request with token IDs...")
    
    payload = {
        "prompt": input_ids,
        "temperature": 0.0,
        "n_predict": 8
    }
    
    try:
        res = requests.post(url, json=payload)
        data = res.json()
        print("Raw completion response keys:", data.keys())
        server_generated_tokens = [t.get("id") if isinstance(t, dict) else t for t in data.get("tokens", [])]
        print("llama-server generated tokens in response:", server_generated_tokens)
        
        # In llama.cpp, if prompt is an array of IDs, the completion is generated.
        # Let's extract the generated tokens/text
        text_out = data.get("content", "")
        # Wait, does the server return the token IDs generated?
        # llama-server /completion response has "generation_settings" and "content"
        # Let's tokenise the content back using the tokenizer to check token IDs,
        # or see if llama-server response has a "tokens" or similar field.
        print(f"llama-server Text Output: {text_out!r}")
        
        # Tokenize llama.cpp's text output back to IDs for comparison
        # (Since we query via /completion, it generates the response text)
        ref_tokens = tokenizer.tokenizer.encode(text_out, add_special_tokens=False)
        ref_strings = [tokenizer.decode([t]) for t in ref_tokens]
        print(f"llama-server Token IDs: {ref_tokens}")
        for i, (t, s) in enumerate(zip(ref_tokens, ref_strings)):
            print(f"  Step {i}: token={t} ({s!r})")
            
        print("\n=== Side-by-Side Token Match ===")
        print(f"{'NPU (Alveare)':<30} | {'llama.cpp':<30}")
        print("-" * 63)
        max_len = max(len(npu_tokens), len(ref_tokens))
        for i in range(max_len):
            n_tok = f"{npu_tokens[i]} ({npu_strings[i]!r})" if i < len(npu_tokens) else ""
            r_tok = f"{ref_tokens[i]} ({ref_strings[i]!r})" if i < len(ref_tokens) else ""
            print(f"{n_tok:<30} | {r_tok:<30}")
            
    except Exception as e:
        print(f"Error querying llama-server: {e}")
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
