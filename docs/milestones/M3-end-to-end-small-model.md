# M3 — End-to-end small model + server

**Blocked by:** M2.

## Goal

A complete small model generating coherent text on the NPU, served over an OpenAI-compatible HTTP endpoint, and routable from the existing `lemonade_router`.

## Definition of done

- `POST /v1/chat/completions` (non-streaming first, then streaming) returns coherent tokens for the M2 model, fully generated on the NPU.
- Tokenizer + chat template produce correct prompts.
- Greedy and basic temperature sampling work.
- The model is reachable as a backend in `~/lemonade_router/rules.yaml`.

## Sub-steps

1. Stack all layers into a full forward pass (loop over M2 layer, all weights streamed).
2. Embedding lookup + final norm + LM head (big GEMV → logits).
3. Sampler (greedy, temperature, top-k/top-p as needed).
4. Tokenizer integration (reuse existing tokenizer.json from the model).
5. Chat template (minja/jinja or a minimal templater).
6. OpenAI-compatible server (`runtime/server/`): `/v1/models`, `/v1/chat/completions`.
7. Streaming responses (SSE).
8. Add to the router as a new NPU backend; smoke-test through it.

## Definition of "coherent"

Not benchmark-quality — just: given a simple prompt, the small model produces on-topic, grammatical continuation matching what the same GGUF produces under llama.cpp (allowing for sampling differences). Side-by-side vs llama.cpp output is the sanity check.

## After M3

This is the first genuinely *usable* (if slow) milestone — an open NPU LLM runtime that serves a real model. Everything after is scale + speed (M4).
