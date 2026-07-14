# runtime/

Host-side runtime — the "easy 70%". Owns the inference loop and talks to the NPU via XRT.
Implemented in Python (`pyxrt` + MLIR-AIE/IRON); see [`../docs/decisions/0004-python-first-runtime.md`](../docs/decisions/0004-python-first-runtime.md).

## `py/` components

- `model.py` — per-architecture model definitions (Llama / Gemma-3 / Gemma-4), the
  forward graph of kernel calls, weight streaming, and KV cache.
- `layer.py` — decoder-layer building blocks.
- `sampler.py` — greedy / temperature / top-k / top-p sampling.
- `tokenizer_glue.py` — thin wrapper over the HF tokenizer + chat template.
- `server.py` — OpenAI-compatible HTTP server (`/v1/models`, `/v1/chat/completions`, SSE).

Launch the server via the top-level `alveare serve <model>` command (see [`../docs/SETUP.md`](../docs/SETUP.md)).
