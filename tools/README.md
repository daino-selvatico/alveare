# tools/

Helper scripts and offline tooling.

- `check_npu.sh` — NPU pre-flight + smoke test (device node, `render` group, XRT, conda
  env, memcpy kernel). Wrapped by `alveare check`.
- `quantize_model.py` — quantize Llama-3.2-1B GGUF → `quantized_weights/`.
- `quantize_gemma.py` — quantize Gemma-3-1B GGUF → `quantized_weights_gemma/`.
- `quantize_gemma4.py` — quantize Gemma-4-12B GGUF → `quantized_weights_gemma4/`.
- `convert/` — GGUF → Alveare tiled Q4 weight-layout helpers (packing/quantization).
- `ref/` — CPU/numpy/HF reference implementations (the correctness oracles).
- `verify_same_input.py` — side-by-side greedy token comparison vs `llama.cpp`.

The quantize scripts read a source GGUF path defined at the top of each script — edit it
to point at your GGUF. Outputs are large and git-ignored (never commit weights).
