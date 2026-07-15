import sys
import time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from runtime.py.model import LlamaNPUModel
from runtime.py.tokenizer_glue import TokenizerGlue

def main():
    weights_dir = Path(__file__).resolve().parents[2] / "quantized_weights_gemma4"
    print("Initializing model...")
    model = LlamaNPUModel(weights_dir)
    tokenizer = TokenizerGlue("google/gemma-4-12b-it")
    
    prompt_messages = [{"role": "user", "content": "The capital of France is"}]
    input_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
    
    print(f"Prompt length: {len(input_ids)}")
    
    # Run 1D prefill
    model.reset_caches()
    t0 = time.perf_counter()
    for pos in range(len(input_ids) - 1):
        model.forward(input_ids[pos], pos, return_logits=False, use_npu=False)
    # Get last logit
    logits_1d = model.forward(input_ids[-1], len(input_ids) - 1, use_npu=False)
    t1 = time.perf_counter()
    print(f"1D prefill took {t1 - t0:.2f}s")
    pred_1d = int(np.argmax(logits_1d))
    
    # Run 2D batched prefill on NPU
    model.reset_caches()
    t0 = time.perf_counter()
    batch = input_ids[:-1]
    model.forward_batch(batch, 0, use_npu=True)
    logits_2d = model.forward(input_ids[-1], len(input_ids) - 1, use_npu=True)
    t1 = time.perf_counter()
    print(f"2D batched prefill (NPU) took {t1 - t0:.2f}s")
    pred_2d = int(np.argmax(logits_2d))
    
    print(f"1D prediction: {tokenizer.decode([pred_1d])} ({pred_1d})")
    print(f"2D prediction (NPU): {tokenizer.decode([pred_2d])} ({pred_2d})")
    
    if np.allclose(logits_1d, logits_2d, atol=1e-3):
        print("Logits MATCH perfectly!")
    else:
        print("Logits MISMATCH!")
        diff = np.abs(logits_1d - logits_2d)
        print(f"Max diff: {np.max(diff)}")

if __name__ == "__main__":
    main()
