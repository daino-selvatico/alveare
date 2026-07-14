# M8 — Close the Gemma-4-12B Fidelity Gap

**Status**: **Completed**.

## Goal
Diagnose and resolve the greedy output divergence between Alveare's NPU runtime and `llama.cpp` on Gemma-4-12B-it under the same input token sequence.

---

## 1. The Diagnosis

We isolated two distinct causes for the greedy output divergence at token 1:

### A. Proportional RoPE Bug in Global Attention Layers
We validated a global full-attention layer (Layer 5) against the HuggingFace oracle.
* **Before the fix**: The Query RoPE relative error was **16.73%** (Max Absolute Error: `2.49658`).
* **Root cause**: Gemma-4 global layers use **proportional RoPE** with `partial_rotary_factor = 0.25`. This means for a head dimension of `512`, only 25% of the dimensions (128 elements: the first 64 and the middle 64) are rotated by RoPE, while the remaining 384 dimensions remain completely unrotated. Alveare was incorrectly rotating the full 512 dimensions.

### B. Chat Template Mismatch
We compared the formatted prompt token IDs between Alveare's HF tokenizer call and `llama.cpp`:
* Alveare default prompt tokens (18 tokens):
  `[2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107, 100, 45518, 107, 101]`
  * Decoded: `<bos><|turn>user\nThe capital of France is<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`
* llama.cpp formatted prompt tokens (21 tokens):
  `[2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107]`
  * Decoded: `<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nThe capital of France is<turn|>\n<|turn>model\n`
* **Root cause**: `llama.cpp` applies the template with `enable_thinking=True` by default, ending the prompt with `<|turn>model\n` to start the `thought` channel. Alveare's default tokenizer call omitted this flag, resulting in the prompt ending with `<|channel>thought\n<channel|>`, which closed the thinking channel immediately and forced the model to jump directly to the final answer `Paris`.

---

## 2. The Fixes

### A. Proportional RoPE Fix
We updated `precompute_cos_sin_table_gemma` in [model.py](file:///home/daino/progetti/alveare/runtime/py/model.py):
* When `self.model_type == "gemma4"` and `dim == 512`, we calculate `rope_angles = 64` (representing 25% partial rotary factor) and set the remaining 192 frequency values in `inv_freq` to 0.0.
* Because `cos(0) = 1.0` and `sin(0) = 0.0`, the existing vectorized `run_rope_cpu_gemma` helper automatically preserves the unrotated dimensions.
* **Result**: Query RoPE relative error dropped from **16.73%** to **0.15%** (Max Absolute Error: `0.03125`), matching HF within standard float16/bfloat16 precision limits.

### B. Chat Template Alignment
We updated `apply_chat_template` in [tokenizer_glue.py](file:///home/daino/progetti/alveare/runtime/py/tokenizer_glue.py):
* We dynamically check the tokenizer's template JINJA string. If `"enable_thinking"` is found, we set `enable_thinking=True`.
* This ensures that Alveare generates the exact same 21 prompt tokens as `llama.cpp` and correctly starts generation within the `thought` channel.

---

## 3. Same-Input Token-by-Token Match

To verify correct execution, we fed the identical 21 token IDs to both Alveare (NPU) and `llama-server` (llama.cpp) via `/completion`. The first 8 greedy tokens generated match **exactly**:

| Step | NPU (Alveare) Token ID | llama.cpp Token ID | Decoded Token String |
|---|---|---|---|
| 0 | **100** | **100** | `<|channel>` |
| 1 | **45518** | **45518** | `thought` |
| 2 | **107** | **107** | `\n` |
| 3 | **818** | **818** | `The` |
| 4 | **2430** | **2430** | ` user` |
| 5 | **563** | **563** | ` is` |
| 6 | **10980** | **10980** | ` asking` |
| 7 | **573** | **573** | ` for` |

---

## 4. Gates (Stay Green)

All Unified Alveare runtime verification tests passed successfully:
* `tests/test_gemv_q.py`: **PASS**
* `tests/test_cpu_only.py`: **PASS** (Correctly generated `The capital of France is Paris.`)
* `tests/test_gemma_layer.py`: **PASS**
* `tests/test_gemma4_layer.py`: **PASS**
* `tests/test_gemma4_global_layer.py` (New): **PASS** (Validated Layer 5 against HF oracle)
* `tests/test_gemma4_generation.py`: **PASS**
