# Gemma-4-E4B Reference Output Dumps (Task 0.1)

Generated using `llama.cpp` (commit `adb541a`, binary `llama-server` / `llama-completion`) on model `/home/daino/llama-mtp/models/gemma-4-E4B-it-UD-Q4_K_XL.gguf`.

## Architecture Recognition Confirmation
- `print_info: arch = gemma4`
- `print_info: model type = E4B`
- `print_info: model params = 7,52 B`
- `embedding_length_per_layer_input = 256` (PLE)
- Zero warnings/errors about unrecognized or ignored PLE tensors (`per_layer_token_embd`, `per_layer_model_proj`, `per_layer_proj_norm`, `blk.N.inp_gate`, `blk.N.proj`, `blk.N.post_norm`).

## Reference Prompts & Deterministic Token Sequences (temp=0)

### Prompt 1: `"Hello world"`
- **Prompt Tokens**: `[2, 9259, 1902]`
- **Generated Text**: `"! This is a test.\n\n***\n\nThis is a test.\n\n***\n\nThis is a test.\n\n***\n\nThis is a test.\n\n***"`
- **Token IDs**: `[236888, 1174, 563, 496, 1594, 236761, 108, 13513, 108, 2094, 563, 496, 1594, 236761, 108, 13513, 108, 2094, 563, 496, 1594, 236761, 108, 13513, 108, 2094, 563, 496, 1594, 236761, 108, 13513]`
- **Details**: `prompt_01.json`

### Prompt 2: `"The capital of France is"`
- **Prompt Tokens**: `[2, 818, 6088, 529, 9363, 603]`
- **Generated Text**: `" Paris.\n"`
- **Token IDs**: `[9079, 236761, 106]`
- **Details**: `prompt_02.json`

### Prompt 3: `"Write a python function to add two numbers:"`
- **Prompt Tokens**: `[2, 17293, 476, 15694, 4832, 531, 1184, 1548, 4867, 235334]`
- **Generated Text**: `"\n\`\`\`python\ndef add_numbers(a, b):\n    return a + b\n\`\`\`\n"`
- **Token IDs**: `[108, 2717, 6719, 107, 2063, 1184, 46194, 235282, 477, 235269, 522, 235334, 107, 1269, 1699, 477, 665, 522, 107, 1018, 108, 107, 2094, 1085, 235272, 603, 476, 1594, 1260, 531, 1184, 1548]`
- **Details**: `prompt_03.json`

### Prompt 4: `"What is the speed of light?"`
- **Prompt Tokens**: `[2, 3910, 603, 576, 5621, 529, 3680, 235336]`
- **Generated Text**: `"\n**299,792,458 meters per second** (or approximately **186,282 miles per second**).\n"`
- **Token IDs**: `[108, 1018, 7925, 53121, 669, 235300, 16867, 989, 2196, 1018, 584, 1373, 20436, 1018, 235270, 87692, 7064, 989, 2196, 1018, 235305, 108, 2094, 603, 476, 14357, 3680, 529, 903, 107, 108, 1174]`
- **Details**: `prompt_04.json`
