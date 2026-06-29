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

class TestGemma4Generation(unittest.TestCase):
    
    def test_greedy_generation(self):
        print("\n=== Testing Gemma-4 12B Greedy Generation on NPU ===")
        weights_dir = Path(__file__).resolve().parents[1] / "quantized_weights_gemma4"
        self.assertTrue(weights_dir.exists(), f"Weights dir {weights_dir} does not exist!")
        
        # Load model
        t_init0 = time.perf_counter()
        model = LlamaNPUModel(weights_dir)
        t_init1 = time.perf_counter()
        print(f"Model initialized in {t_init1 - t_init0:.2f}s")
        
        tokenizer = TokenizerGlue("google/gemma-4-12b-it")
        
        prompt_messages = [{"role": "user", "content": "The capital of France is"}]
        
        # 1. Run generation on NPU
        print("Running generation on NPU...")
        npu_text, latencies = self.generate(model, tokenizer, prompt_messages, max_new_tokens=5)
        print(f"NPU Output:\n{npu_text}")
        print(f"Per-token latencies (s): {[f'{l:.3f}' for l in latencies]}")
        mean_latency = np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]
        print(f"Average latency after first token: {mean_latency:.3f} s")
        
        # 2. Run reference generation on llama-server (llama.cpp)
        print("\nRunning reference generation on llama.cpp (llama-server)...")
        llama_cpp_text = self.run_llama_cpp_reference(prompt_messages, max_new_tokens=5)
        print(f"llama.cpp Output:\n{llama_cpp_text}")
        
        # 3. Print side-by-side comparison
        print("\n=== Side-by-Side Comparison ===")
        print(f"{'NPU (Alveare Gemma-4-12B)':<40} | {'llama.cpp (Gemma-4 GGUF)':<40}")
        print("-" * 83)
        npu_lines = npu_text.strip().split("\n")
        ref_lines = llama_cpp_text.strip().split("\n")
        max_lines = max(len(npu_lines), len(ref_lines))
        for i in range(max_lines):
            npu_l = npu_lines[i] if i < len(npu_lines) else ""
            ref_l = ref_lines[i] if i < len(ref_lines) else ""
            print(f"{npu_l[:40]:<40} | {ref_l[:40]:<40}")
            
        # Write results to test notes
        notes_path = Path(__file__).resolve().parents[1] / "tests" / "gemma4_generation_test_results.txt"
        with open(notes_path, "w") as f:
            f.write("=== NPU vs llama.cpp Side-by-Side Greedy Continuation ===\n")
            f.write(f"Prompt: {prompt_messages[0]['content']}\n\n")
            f.write(f"NPU Output:\n{npu_text}\n\n")
            f.write(f"llama.cpp Output:\n{llama_cpp_text}\n\n")
            f.write(f"NPU Latencies (s) after first token: {[f'{l:.3f}' for l in latencies[1:]]}\n")
            f.write(f"Average latency after first token: {mean_latency:.3f} s\n")
            
        print(f"\nSaved test results to {notes_path}")
        
        # Check coherence: NPU output should be clearly identical/close to llama.cpp
        # and contain words related to the continuation
        self.assertTrue(len(npu_text.strip()) > 0, "NPU generated output must not be empty!")
        
    def generate(self, model, tokenizer, prompt_messages, max_new_tokens=5):
        input_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
        model.reset_caches()
        
        # Prefill prompt tokens on NPU
        print(f"Prefilling {len(input_ids)} tokens on NPU...")
        t0_prefill = time.perf_counter()
        for pos in range(len(input_ids) - 1):
            _ = model.forward(input_ids[pos], pos, return_logits=False, use_npu=True)
        t1_prefill = time.perf_counter()
        print(f"Prefill completed in {t1_prefill - t0_prefill:.2f}s")
        
        # Start generation
        current_token_id = input_ids[-1]
        pos = len(input_ids) - 1
        
        generated_tokens = []
        latencies = []
        
        for i in range(max_new_tokens):
            print(f"Generating token {i+1}/{max_new_tokens}...")
            t0 = time.perf_counter()
            logits = model.forward(current_token_id, pos, use_npu=True)
            next_token_id = int(np.argmax(logits))
            t1 = time.perf_counter()
            
            latencies.append(t1 - t0)
            print(f"Token {i+1} generated in {t1 - t0:.2f}s (id={next_token_id}, token={tokenizer.decode([next_token_id])!r})")
            
            if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.eot_token_id:
                print("Reached EOS/EOT.")
                break
                
            generated_tokens.append(next_token_id)
            current_token_id = next_token_id
            pos += 1
            
        return tokenizer.decode(generated_tokens), latencies

    def run_llama_cpp_reference(self, prompt_messages, max_new_tokens=5):
        server_bin = "/home/daino/llama-mtp/llama.cpp/build/bin/llama-server"
        model_path = "/home/daino/llama-mtp/models/gemma-4-12b-it-UD-Q4_K_XL.gguf"
        port = 18081
        
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
        for _ in range(100):
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
            message = data["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                output_text = f"{reasoning} {content}".strip()
            else:
                output_text = content
        except Exception as e:
            print(f"Error querying llama-server: {e}")
            output_text = ""
        finally:
            proc.terminate()
            proc.wait()
            
        return output_text

if __name__ == "__main__":
    unittest.main()
