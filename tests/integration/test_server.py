import subprocess
import time
import requests
import sys
import json
from pathlib import Path

def wait_for_server(url, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False

def test_chat():
    print("Starting server...")
    root_dir = Path(__file__).resolve().parents[2]
    # Use stdbuf to disable stdout buffering
    server_proc = subprocess.Popen(
        ["./alveare", "serve", "gemma4"],
        cwd=str(root_dir),
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    try:
        print("Waiting for server to become ready...")
        if not wait_for_server("http://127.0.0.1:8000/v1/models", timeout=300):
            print("Server failed to start within timeout.")
            sys.exit(1)

        print("Server is ready! Sending test prompt...")
        prompt = "Spiegami cos'è l'acqua in una sola riga."
        
        t0 = time.time()
        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "gemma-4-12b-it",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "stream": True
            },
            stream=True
        )
        
        if response.status_code != 200:
            print(f"Error from server: {response.status_code}")
            print(response.text)
            sys.exit(1)

        first_token_time = None
        tokens = 0
        
        print("\nResponse: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    data = json.loads(data_str)
                    content = data["choices"][0]["delta"].get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        print(content, end="", flush=True)
                        tokens += 1
                        
        t1 = time.time()
        print("\n")
        
        ttft = first_token_time - t0 if first_token_time else 0
        gen_time = t1 - first_token_time if first_token_time else 0
        tps = tokens / gen_time if gen_time > 0 else 0
        
        print(f"TTFT (Time To First Token): {ttft:.2f}s")
        print(f"Tokens generated: {tokens}")
        print(f"Generation speed: {tps:.2f} tokens/s")
        print("Test passed successfully!")
        
    finally:
        print("Shutting down server...")
        server_proc.terminate()
        server_proc.wait(timeout=10)

if __name__ == "__main__":
    test_chat()
