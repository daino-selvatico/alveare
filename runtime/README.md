# runtime/

Host-side C++ runtime — the "easy 70%". Owns the inference loop and talks to the NPU via XRT.

Planned components (see `../docs/architecture.md`):

- `xrt/` — device, xclbin loading, buffer objects, submit/sync
- `streamer/` — weight streaming DRAM → NPU buffers (sync first, prefetch later)
- `kvcache/` — per-layer K/V storage
- `model/` — per-architecture model definitions (graph of kernel calls)
- `sampler/` — greedy / temperature / top-k/p
- `tokenizer/` — reuse existing tokenizer.json
- `server/` — OpenAI-compatible HTTP (`/v1/models`, `/v1/chat/completions`, SSE)

Empty until M2–M3.
