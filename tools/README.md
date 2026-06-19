# tools/

Helper scripts and offline tooling.

- `check_npu.sh` — NPU pre-flight + smoke test (created in M0).
- `convert/` — GGUF → Alveare quantized/tiled weight layout (created in M1).
- `ref/` — CPU/numpy reference implementations of each kernel (the correctness oracles).

Reuse the GGUF models already on disk (Gemma 4, Qwen) as test data rather than re-downloading/re-quantizing.

Empty until M0/M1.
