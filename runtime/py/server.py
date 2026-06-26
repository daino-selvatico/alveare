import os
import sys
import time
import uuid
import json
import asyncio
from pathlib import Path
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from runtime.py.model import LlamaNPUModel
from runtime.py.tokenizer_glue import TokenizerGlue
from runtime.py.sampler import sample

app = FastAPI(title="Alveare NPU Server")

# Global states
model = None
tokenizer = None
lock = asyncio.Lock()
model_id = "Llama-3.2-1B-Instruct"

@app.on_event("startup")
def startup_event():
    global model, tokenizer, model_id
    weights_dir = Path(__file__).resolve().parents[2] / "quantized_weights"
    if not weights_dir.exists():
        print(f"Error: weights directory {weights_dir} does not exist!")
        sys.exit(1)
        
    config_path = weights_dir / "config.json"
    hf_model_id = "unsloth/Llama-3.2-1B-Instruct"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        if config.get("model_type") == "gemma3":
            model_id = "gemma-3-1b-it"
            hf_model_id = "unsloth/gemma-3-1b-it"
            
    print(f"Initializing model on NPU ({model_id})...")
    model = LlamaNPUModel(weights_dir)
    print("Loading tokenizer...")
    tokenizer = TokenizerGlue(model_id=hf_model_id)
    print("Server ready.")

class ChatCompletionMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = model_id
    messages: List[ChatCompletionMessage]
    temperature: float = Field(1.0, ge=0.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(128, ge=1)
    stream: bool = False

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "alveare"
            }
        ]
    }

async def generate_stream(
    input_ids: List[int],
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
    request_id: str
):
    # Lock is held during the entire generation process for this request
    async with lock:
        model.reset_caches()
        
        # 1. Prefill prompt
        num_prompt_tokens = len(input_ids)
        print(f"[{request_id}] Starting prefill of {num_prompt_tokens} tokens...")
        t0_prefill = time.perf_counter()
        for pos in range(num_prompt_tokens - 1):
            # Run without returning logits (saves time)
            model.forward(input_ids[pos], pos, return_logits=False, use_npu=False)
            # Yield to other async tasks briefly
            await asyncio.sleep(0)
        t1_prefill = time.perf_counter()
        print(f"[{request_id}] Prefill completed in {t1_prefill - t0_prefill:.2f}s")
        
        # 2. Generation
        current_token_id = input_ids[-1]
        pos = num_prompt_tokens - 1
        
        generated_count = 0
        finish_reason = "length"
        
        for i in range(max_tokens):
            t0_step = time.perf_counter()
            logits = model.forward(current_token_id, pos, use_npu=True)
            next_token_id = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            t1_step = time.perf_counter()
            
            print(f"[{request_id}] Token {i+1}/{max_tokens} generated in {(t1_step - t0_step)*1000:.1f}ms (id={next_token_id})")
            
            if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.eot_token_id:
                finish_reason = "stop"
                break
                
            token_text = tokenizer.decode(next_token_id)
            generated_count += 1
            
            # Format as SSE chunk
            chunk = {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
            current_token_id = next_token_id
            pos += 1
            await asyncio.sleep(0)
            
        # Yield the final chunk
        final_chunk = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = str(uuid.uuid4())
    
    # Map pydantic messages to dicts
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Encode with tokenizer template
    try:
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenizer error: {str(e)}")
        
    if request.stream:
        return StreamingResponse(
            generate_stream(
                input_ids,
                request.temperature,
                request.top_k,
                request.top_p,
                request.max_tokens,
                request_id
            ),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming implementation
        async with lock:
            model.reset_caches()
            
            # Prefill prompt
            num_prompt_tokens = len(input_ids)
            print(f"[{request_id}] Starting prefill of {num_prompt_tokens} tokens...")
            for pos in range(num_prompt_tokens - 1):
                model.forward(input_ids[pos], pos, return_logits=False, use_npu=False)
                await asyncio.sleep(0)
                
            # Generation loop
            current_token_id = input_ids[-1]
            pos = num_prompt_tokens - 1
            
            generated_tokens = []
            finish_reason = "length"
            
            for i in range(request.max_tokens):
                t0_step = time.perf_counter()
                logits = model.forward(current_token_id, pos, use_npu=True)
                next_token_id = sample(logits, temperature=request.temperature, top_k=request.top_k, top_p=request.top_p)
                t1_step = time.perf_counter()
                
                print(f"[{request_id}] Token {i+1}/{request.max_tokens} generated in {(t1_step - t0_step)*1000:.1f}ms (id={next_token_id})")
                
                if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.eot_token_id:
                    finish_reason = "stop"
                    break
                    
                generated_tokens.append(next_token_id)
                current_token_id = next_token_id
                pos += 1
                await asyncio.sleep(0)
                
            content = tokenizer.decode(generated_tokens)
            
            return {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": num_prompt_tokens,
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": num_prompt_tokens + len(generated_tokens)
                }
            }

if __name__ == "__main__":
    import uvicorn
    # Read host and port from environment or use defaults
    host = os.getenv("ALVEARE_HOST", "127.0.0.1")
    port = int(os.getenv("ALVEARE_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
