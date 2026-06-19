# M2 — One transformer layer on NPU

**Blocked by:** M1.

## Goal

Run a complete decoder layer on the NPU for a **small, dense, simple** model: RMSNorm → QKV → attention (with KV cache) → output proj → RMSNorm → MLP → residuals.

## Target model

Pick the simplest well-documented dense model to avoid architecture-specific complications:

- **Candidate A:** Llama-3.2-1B (plain dense, standard attention, well documented, GGUF available locally).
- **Candidate B:** Gemma-3 270M (smaller, but Gemma-family quirks).

Decision recorded in an ADR once chosen. **Not** Gemma-4 dense yet (its sliding/full alternation, QK-norm, softcapping are deferred).

## Definition of done

For the chosen model, one decoder layer's output (given a real input hidden state) matches a reference (HF transformers or llama.cpp) within tolerance, **with weights streamed from DRAM** rather than assumed resident.

## New kernels needed beyond M1

- `rmsnorm`
- `rope`
- `attn` (QKᵀ, `softmax`, ·V) with GQA grouping and a real KV cache buffer
- residual add (trivial, may be host-side initially)

## Sub-steps

1. CPU reference for the whole layer (numpy / pull from HF) — the oracle.
2. Implement + validate each new kernel against its own reference.
3. Wire kernels into a layer in `runtime/`, reading weights via the streamer.
4. KV cache: allocate, write current token's K/V, let attention read it.
5. Full-layer correctness test.

## Risks

- Attention is the first kernel with non-trivial data-dependent control (softmax, variable KV length). Budget time here.
- Numerical drift accumulates across sub-ops; validate sub-op by sub-op, not just end-to-end.
