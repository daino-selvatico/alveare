# M1 — Quantized matrix-vector kernel

**Blocked by:** M0.

## Goal

Implement the dominant LLM-decode primitive on the NPU: a **quantized weight × bf16 activation matrix-vector multiply** (GEMV), with on-chip (or fused) dequantization.

## Definition of done

For a realistic matrix shape (start ~2560×2560, a Gemma-4-E-class hidden size):

- A kernel taking int4 (block-quantized, e.g. Q4_0-style) weights + a bf16 input vector produces an output vector matching a numpy/CPU reference within tolerance (define a relative-error threshold).
- A microbenchmark records NPU latency vs a CPU baseline for that shape.
- Host ABI documented in `kernels/gemv_q/README.md`.

## Sub-steps

1. **CPU reference first** (`tools/ref/gemv_q.py`): dequantize + matvec in numpy. This is the oracle for all correctness checks.
2. **Weight layout**: decide block size (32) and the on-device tiling; implement the converter that turns a GGUF tensor into that layout (`tools/convert/`). Reuse a real Gemma/Qwen GGUF tensor as test data.
3. **Dequant**: validate int4→bf16 unpack on-device matches the reference (may be standalone first, then fused).
4. **GEMV kernel**: IRON design, single tile first (correctness), then spread across cores.
5. **Correctness harness** (`tests/`): random + real-weight vectors, compare to reference.
6. **Bench**: latency across a few shapes; write numbers to `tests/bench/`.

## Notes

- Correctness over speed. A single-core, synchronous, "obviously correct" version is the M1 deliverable. Multi-core tiling and DMA overlap are optimizations for later.
- Tolerance: bf16 accumulation differs from fp32 reference — pick a sensible relative error bound and document the rationale.

## Why this shape/primitive first

At decode time (batch=1), every projection and MLP matmul is a GEMV against quantized weights. Get this right and fast and you've addressed the bulk of inference cost. Everything else (norm, rope, attention) is comparatively cheap.
