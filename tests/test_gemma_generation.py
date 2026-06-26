import unittest
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

class TestGemmaGeneration(unittest.TestCase):
    
    def test_greedy_generation(self):
        print("\n=== Testing Gemma-3 Greedy Generation on NPU ===")
        weights_dir = Path(__file__).resolve().parents[1] / "quantized_weights_gemma"
        self.assertTrue(weights_dir.exists(), f"Weights dir {weights_dir} does not exist. Run quantize_gemma.py first!")
        
        # Load model
        model = LlamaNPUModel(weights_dir)
        tokenizer = TokenizerGlue("unsloth/gemma-3-1b-it")
        
        prompt_messages = [{"role": "user", "content": "The capital of France is"}]
        
        # 1. Run generation on NPU
        print("Running generation on NPU...")
        npu_text, latencies = self.generate(model, tokenizer, prompt_messages, max_new_tokens=5)
        print(f"NPU Output:\n{npu_text}")
        print(f"Per-token latencies (ms): {[f'{l*1000:.1f}' for l in latencies]}")
        mean_latency = np.mean(latencies[1:]) * 1000 if len(latencies) > 1 else latencies[0] * 1000
        print(f"Average latency after first token: {mean_latency:.1f} ms")
        
        # 2. Run reference generation on llama-server (llama.cpp)
        print("\nRunning reference generation on llama.cpp (llama-server)...")
        llama_cpp_text = self.run_llama_cpp_reference(prompt_messages, max_new_tokens=5)
        print(f"llama.cpp Output:\n{llama_cpp_text}")
        
        # 3. Print side-by-side comparison
        print("\n=== Side-by-Side Comparison ===")
        print(f"{'NPU (Alveare Gemma-3)':<40} | {'llama.cpp (Gemma-3 GGUF)':<40}")
        print("-" * 83)
        npu_lines = npu_text.strip().split("\n")
        ref_lines = llama_cpp_text.strip().split("\n")
        max_lines = max(len(npu_lines), len(ref_lines))
        for i in range(max_lines):
            npu_l = npu_lines[i] if i < len(npu_lines) else ""
            ref_l = ref_lines[i] if i < len(ref_lines) else ""
            print(f"{npu_l[:40]:<40} | {ref_l[:40]:<40}")
            
        # Write results to test notes
        notes_path = Path(__file__).resolve().parents[1] / "tests" / "gemma_generation_test_results.txt"
        with open(notes_path, "w") as f:
            f.write("=== NPU vs llama.cpp Side-by-Side Greedy Continuation ===\n")
            f.write(f"Prompt: {prompt_messages[0]['content']}\n\n")
            f.write(f"NPU Output:\n{npu_text}\n\n")
            f.write(f"llama.cpp Output:\n{llama_cpp_text}\n\n")
            f.write(f"NPU Latencies (ms) after first token: {[f'{l*1000:.1f}' for l in latencies[1:]]}\n")
            f.write(f"Average latency after first token: {mean_latency:.1f} ms\n")
            
        print(f"\nSaved test results to {notes_path}")
        
        # Simple coherence check
        self.assertTrue(any(word in npu_text.lower() for word in ["paris", "capital", "france"]), 
                        f"NPU output {npu_text!r} should contain relevant continuation words!")
        
    def generate(self, model, tokenizer, prompt_messages, max_new_tokens=20):
        input_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
        model.reset_caches()
        
        # Prefill prompt tokens
        for pos in range(len(input_ids) - 1):
            _ = model.forward(input_ids[pos], pos, return_logits=False, use_npu=False)
            
        # Start generation
        current_token_id = input_ids[-1]
        pos = len(input_ids) - 1
        
        generated_tokens = []
        latencies = []
        
        for i in range(max_new_tokens):
            t0 = time.perf_counter()
            logits = model.forward(current_token_id, pos, use_npu=True)
            # Argmax for greedy sampling
            next_token_id = int(np.argmax(logits))
            t1 = time.perf_counter()
            
            latencies.append(t1 - t0)
            
            if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.eot_token_id:
                break
                
            generated_tokens.append(next_token_id)
            current_token_id = next_token_id
            pos += 1
            
        return tokenizer.decode(generated_tokens), latencies

    def run_llama_cpp_reference(self, prompt_messages, max_new_tokens=20):
        server_bin = "/home/daino/llama-mtp/llama.cpp/build/bin/llama-server"
        model_path = "/home/daino/llama-mtp/models/google_gemma-3-1b-it-bf16.gguf"
        port = 18080
        
        # Start llama-server in the background
        cmd = [
            server_bin,
            "-m", model_path,
            "-c", "2048",
            "--port", str(port),
            "-ngl", "0"  # Run on CPU
        ]
        
        print(f"Starting llama-server: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        for _ in range(30):
            try:
                res = requests.get(f"http://127.0.0.1:{port}/health")
                if res.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
            
        # Send request
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": prompt_messages,
            "temperature": 0.0,
            "max_tokens": max_new_tokens
        }
        
        try:
            res = requests.post(url, headers=headers, json=payload)
            data = res.json()
            output_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error querying llama-server: {e}")
            output_text = ""
        finally:
            proc.terminate()
            proc.wait()
            
        return output_text

if __name__ == "__main__":
    unittest.main()
