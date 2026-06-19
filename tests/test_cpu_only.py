import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from runtime.py.model import LlamaNPUModel
from runtime.py.tokenizer_glue import TokenizerGlue

def main():
    weights_dir = Path(__file__).resolve().parents[1] / "quantized_weights"
    model = LlamaNPUModel(weights_dir)
    tokenizer = TokenizerGlue()
    
    prompt_messages = [{"role": "user", "content": "capital of France is"}]
    input_ids = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
    print(f"Input tokens: {input_ids}")
    print(f"Input text decoded: {tokenizer.decode(input_ids)}")
    
    model.reset_caches()
    
    # 1. Prefill
    for pos in range(len(input_ids) - 1):
        model.forward(input_ids[pos], pos, return_logits=False, use_npu=False)
        
    # 2. Generation
    current_token_id = input_ids[-1]
    pos = len(input_ids) - 1
    
    generated_tokens = []
    for i in range(10):
        logits = model.forward(current_token_id, pos, use_npu=False)
        top_indices = np.argsort(logits)[-5:][::-1]
        print(f"Step {i}: logits max={np.max(logits):.3f}")
        for idx in top_indices:
            print(f"  - token={idx} ({tokenizer.decode([int(idx)])!r}): logit={logits[idx]:.3f}")
        next_token_id = int(top_indices[0])
        
        if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.eot_token_id:
            print("Reached EOS/EOT.")
            break
            
        generated_tokens.append(next_token_id)
        current_token_id = next_token_id
        pos += 1
        
    print(f"Generated text: {tokenizer.decode(generated_tokens)}")

if __name__ == "__main__":
    main()
