# tools/

Helper scripts and offline tooling.

- `check_npu.sh` — NPU pre-flight + smoke test (device node, `render` group, XRT, conda
  env, memcpy kernel). Wrapped by `alveare check`.
- `detect_arch.py` — read a GGUF's `general.architecture` and map it to a quantizer key
  (`llama`/`gemma3`/`gemma4`). Used by `alveare quantize` for auto-detection.
- `quantize_model.py` — quantize a **llama**-arch GGUF into Alveare's Q4 layout.
- `quantize_gemma.py` — quantize a **gemma3**-arch GGUF.
- `quantize_gemma4.py` — quantize a **gemma4**-arch GGUF.
- `chat.py` — minimal streaming terminal chat client (OpenAI endpoint). Wrapped by `alveare chat`.
- `convert/` — GGUF → Alveare tiled Q4 weight-layout helpers (the shared Q4_0 packing).
- `ref/` — CPU/numpy/HF reference implementations (the correctness oracles).
- `verify_same_input.py` — side-by-side greedy token comparison vs `llama.cpp`.

Prefer the top-level CLI: `alveare quantize [alias] <gguf> [--arch A]` auto-detects the
architecture and calls the right quantizer here. The quantizers also run standalone —
`python tools/quantize_gemma4.py <gguf> --out <dir>` — and default to the author's local
paths if run with no arguments. Outputs are large and git-ignored (never commit weights).

**All three quantizers share the same Q4_0 quantization algorithm** (`convert/gemv_q_convert.py`);
they differ only in per-architecture tensor wiring and the `config.json` they emit.
