#include "alveare/generator.h"
#include "alveare/prompt_lookup.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>

namespace alveare {

Generator::Generator(Model& model, const ModelWeights& weights, const Tokenizer& tokenizer)
    : model_(model), weights_(weights), tokenizer_(tokenizer) {}

int Generator::sample(const std::vector<float>& logits, const GenerationParams& params) {
    // Greedy search for now
    int best_token = -1;
    float best_val = -1e9f;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_token = static_cast<int>(i);
        }
    }
    return best_token;
}

void Generator::run_lm_head(const bf16* x, std::vector<float>& logits) {
    const ModelConfig& cfg = model_.get_config();
    int hidden_size = cfg.hidden_size;

    // NPU path: the packed LM head was uploaded as row-tiles; run one quantized
    // gemv per tile and concatenate. x is zero-padded to the kernel's K.
    if (!weights_.lm_head_chunks.empty()) {
        int K = weights_.lm_head_K;
        int chunk_N = weights_.lm_head_chunk_N;
        logits.resize(weights_.lm_head_vocab);

        std::vector<bf16> x_pad(K, bf16(0.0f));
        for (int i = 0; i < hidden_size && i < K; ++i) x_pad[i] = x[i];

        std::vector<bf16> y(chunk_N);
        for (size_t c = 0; c < weights_.lm_head_chunks.size(); ++c) {
            model_.registry().run_gemv(chunk_N, K, weights_.lm_head_chunks[c],
                                       x_pad.data(), y.data());
            int base = static_cast<int>(c) * chunk_N;
            for (int i = 0; i < chunk_N; ++i) {
                logits[base + i] = y[i].to_float(); // raw logits (softcap is monotonic; skip for greedy argmax)
            }
        }
        return;
    }

    // Tied embeddings (no packed lm_head): dense fp32 matmul against token_embd.
    if (weights_.lm_head.empty()) {
        int vocab_size = weights_.token_embd.size() / hidden_size;
        logits.resize(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            const float* w_row = &weights_.token_embd[static_cast<size_t>(v) * hidden_size];
            float dot = 0.0f;
            for (int i = 0; i < hidden_size; ++i) {
                dot += x[i].to_float() * w_row[i];
            }
            logits[v] = dot;
        }
        return;
    }

    // Packed Q4_0 lm_head: on-disk layout is (vocab, K_blocks * 20) uint8, where
    // each 20-byte block holds 16 bytes of interleaved int4 quants, a 2-byte bf16
    // scale (bytes 16..17) and 2 pad bytes. K = K_blocks * 32 is padded (4096 for
    // Gemma-4, vs hidden_size 3840), so x is treated as zero-padded past hidden_size.
    const int block_bytes = 20;
    const int K_padded = cfg.get_padded_hidden_size();
    const int K_blocks = K_padded / 32;
    const int row_bytes = K_blocks * block_bytes;
    int vocab_size = static_cast<int>(weights_.lm_head.size() / row_bytes);
    logits.resize(vocab_size);

    std::vector<float> xf(K_padded, 0.0f);
    for (int i = 0; i < hidden_size && i < K_padded; ++i) xf[i] = x[i].to_float();

    const uint8_t* base = weights_.lm_head.data();
    for (int v = 0; v < vocab_size; ++v) {
        const uint8_t* row = base + static_cast<size_t>(v) * row_bytes;
        float dot = 0.0f;
        for (int bk = 0; bk < K_blocks; ++bk) {
            const uint8_t* blk = row + bk * block_bytes;
            alveare::bf16 sc;
            sc.v = static_cast<uint16_t>(blk[16]) | (static_cast<uint16_t>(blk[17]) << 8);
            const float* xb = &xf[bk * 32];
            float bsum = 0.0f;
            for (int j = 0; j < 16; ++j) {
                int lo = blk[j] & 0x0F; if (lo >= 8) lo -= 16;
                int hi = (blk[j] >> 4) & 0x0F; if (hi >= 8) hi -= 16;
                bsum += lo * xb[2 * j] + hi * xb[2 * j + 1];
            }
            dot += bsum * sc.to_float();
        }
        logits[v] = dot;
    }
}

void Generator::generate(const std::string& prompt, const GenerationParams& params, std::function<bool(const std::string&)> on_token) {
    std::lock_guard<std::mutex> gen_lock(gen_mutex_);
    using clock = std::chrono::steady_clock;
    static std::atomic<int> req_counter{0};
    int req = ++req_counter;
    auto tag = [&]() -> std::ostream& { return std::cout << "[req-" << req << "] "; };

    const ModelConfig& cfg = model_.get_config();
    int hidden_size = cfg.hidden_size;
    bool is_gemma = (cfg.model_type == "gemma3" || cfg.model_type == "gemma4");
    float embed_scale = is_gemma ? std::sqrt(static_cast<float>(hidden_size)) : 1.0f;

    std::vector<int> input_tokens = tokenizer_.encode(prompt);
    int num_prompt_tokens = static_cast<int>(input_tokens.size());
    if (num_prompt_tokens == 0) {
        tag() << "empty prompt, nothing to generate\n" << std::flush;
        return;
    }

    // KV-cache reuse: the model's cache already holds valid state for the tokens
    // of the previous request (cached_tokens_). Reuse the longest common prefix
    // and only prefill from there — for multi-turn chat this skips re-prefilling
    // the whole conversation history. We never need the last prompt token in the
    // reused prefix (the decode loop processes it), so cap at num_prompt-1.
    int reuse = 0;
    {
        int maxP = std::min(static_cast<int>(cached_tokens_.size()), num_prompt_tokens - 1);
        while (reuse < maxP && input_tokens[reuse] == cached_tokens_[reuse]) ++reuse;
    }
    // Rebuild the cached sequence for this request; the decode loop appends the
    // fed-back generated tokens as their KV is written.
    cached_tokens_ = input_tokens;

    std::vector<bf16> x(hidden_size);
    std::vector<bf16> out(hidden_size);
    std::vector<float> logits;

    double lm_head_ms = 0.0;  // profiling: last forward's LM-head wall time

    // Run one token through the embedding + all transformer layers. When
    // want_logits is set, also apply the final norm and LM head into `logits`.
    auto forward = [&](int token, int pos, bool want_logits) {
        for (int i = 0; i < hidden_size; ++i) {
            x[i] = bf16(weights_.token_embd[static_cast<size_t>(token) * hidden_size + i] * embed_scale);
        }
        for (int l = 0; l < cfg.num_hidden_layers; ++l) {
            model_.run_layer(x.data(), pos, l, out.data());
            x = out;
        }
        if (!want_logits) return;

        float variance = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float val = x[i].to_float();
            variance += val * val;
        }
        variance /= hidden_size;
        float inv_denom = 1.0f / std::sqrt(variance + cfg.rms_norm_eps);

        std::vector<bf16> normed(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            float w = weights_.output_norm.empty() ? 1.0f : weights_.output_norm[i];
            normed[i] = bf16(x[i].to_float() * inv_denom * w);
        }
        auto t_lm = clock::now();
        run_lm_head(normed.data(), logits);
        lm_head_ms = std::chrono::duration<double, std::milli>(clock::now() - t_lm).count();
    };

    // 1. Prefill: process every prompt token except the last (no logits needed).
    // gemma4 uses the batched GEMM path (B=16 chunks); other models fall back to
    // the per-token decode path.
    int prefill_count = num_prompt_tokens - 1;
    if (reuse > 0)
        tag() << "Reusing " << reuse << " cached tokens; ";
    else
        tag();
    std::cout << "prefilling " << (prefill_count - reuse) << " new of "
              << num_prompt_tokens << " prompt tokens...\n" << std::flush;
    auto t0_prefill = clock::now();
    // Batched (B=16 GEMM) prefill is CORRECT but not faster than the per-token
    // fused path on this runtime: the NPU is already compute-bound per token
    // (no interpreter overhead to amortize), and batching trades the efficient
    // fused FFN kernel for separate streamed gate/up/down GEMMs. Kept behind a
    // flag for experimentation; default is the per-token path.
    bool use_batched = (cfg.model_type == "gemma4") && std::getenv("ALVEARE_BATCH_PREFILL");
    if (use_batched) {
        const int PB = 16;
        std::vector<bf16> xb, ob;
        for (int start = reuse; start < prefill_count; start += PB) {
            int nrows = std::min(PB, prefill_count - start);
            xb.assign(static_cast<size_t>(nrows) * hidden_size, bf16(0.0f));
            ob.assign(static_cast<size_t>(nrows) * hidden_size, bf16(0.0f));
            for (int b = 0; b < nrows; ++b) {
                int token = input_tokens[start + b];
                for (int i = 0; i < hidden_size; ++i)
                    xb[static_cast<size_t>(b) * hidden_size + i] =
                        bf16(weights_.token_embd[static_cast<size_t>(token) * hidden_size + i] * embed_scale);
            }
            for (int l = 0; l < cfg.num_hidden_layers; ++l) {
                model_.run_layer_batch(xb.data(), nrows, start, l, ob.data());
                std::swap(xb, ob);
            }
            tag() << "  prefill chunk " << (start / PB + 1) << " ["
                  << start << ".." << (start + nrows - 1) << "] done\n" << std::flush;
        }
    } else {
        for (int pos = reuse; pos < prefill_count; ++pos) {
            forward(input_tokens[pos], pos, false);
        }
    }
    double prefill_s = std::chrono::duration<double>(clock::now() - t0_prefill).count();
    tag() << "Prefill completed in " << std::fixed << std::setprecision(2) << prefill_s << "s\n" << std::flush;

    // 2. Decode: the last prompt token produces the first generated token.
    int current_token = input_tokens.back();
    int pos = num_prompt_tokens - 1;

    // Speculative decode (gemma4, ALVEARE_SPECULATIVE): a prompt-lookup n-gram
    // drafter proposes up to K tokens, verified in ONE batched forward; accept the
    // matching prefix + one correction. When the drafter finds no match it falls
    // back to the normal single-token forward, so it never decodes slower than the
    // default path. The batched forward pays a fixed B=16 GEMM cost, so it only
    // wins when a draft is found AND partly accepted (repetitive / structured
    // text); the per-step log prints draft/accepted so the trade-off is visible.
    bool use_spec = (cfg.model_type == "gemma4") && std::getenv("ALVEARE_SPECULATIVE");
    if (use_spec) {
        // max draft length. The batched verify pads the GEMM to B=8 regardless of
        // nd (see run_layer_batch), so the mmul cost is IDENTICAL for any nd<=7 —
        // a longer draft fills the already-paid-for B=8 pad for free. K=7 => B=8,
        // so on repetitive/structured text a verify can emit up to 8 tokens at the
        // same ~3s cost as one that emitted 5 (profiled: 3060ms batched forward).
        const int K = 7;                 // max draft length (fills the B=8 pad)
        const int max_seq_len = 2048;
        std::vector<int> seq = input_tokens;  // committed sequence (drafter context)
        int generated = 0, step = 0;

        auto emit = [&](int t) -> bool {  // returns false to stop
            if (tokenizer_.is_stop_token(t)) return false;
            if (!on_token(tokenizer_.decode(t))) return false;
            seq.push_back(t);
            ++generated;
            return true;
        };

        while (generated < params.max_tokens) {
            auto t0_step = clock::now();
            // min_ngram=3: require a 3-token context match before drafting. A
            // rejected draft still runs the full B=8 verify (~3-4s, ~4x the 910ms
            // fallback), so mis-fires on novel text are costly — but raising the gate
            // to 4-grams did NOT remove them (matches into earlier repeated context
            // are >=4-gram) and fired later on genuine repetition, so 3 is kept.
            // Speculative is opt-in (ALVEARE_SPECULATIVE) and situational: a clear
            // win on repetitive/structured runs, roughly neutral-to-negative on prose.
            std::vector<int> draft = propose_draft(seq, K, 3, 3);
            while (!draft.empty() && pos + static_cast<int>(draft.size()) >= max_seq_len)
                draft.pop_back();
            int nd = static_cast<int>(draft.size());

            if (nd == 0) {
                // Fallback: normal single-token decode (fused FFN, no B=16 penalty).
                forward(current_token, pos, true);
                if (pos >= num_prompt_tokens) cached_tokens_.push_back(current_token);
                int t = sample(logits, params);
                double ms = std::chrono::duration<double, std::milli>(clock::now() - t0_step).count();
                tag() << "spec " << ++step << ": draft=0 fallback -> 1 tok in "
                      << std::fixed << std::setprecision(1) << ms << "ms (id=" << t << ")\n" << std::flush;
                current_token = t; ++pos;
                if (!emit(t)) break;
                continue;
            }

            // Batched verify: rows = [current_token, draft...] at [pos..pos+nd].
            int B = nd + 1;
            std::vector<bf16> xb(static_cast<size_t>(B) * hidden_size);
            std::vector<bf16> ob(static_cast<size_t>(B) * hidden_size);
            std::vector<int> btoks(B);
            btoks[0] = current_token;
            for (int j = 0; j < nd; ++j) btoks[j + 1] = draft[j];
            for (int b = 0; b < B; ++b)
                for (int i = 0; i < hidden_size; ++i)
                    xb[static_cast<size_t>(b) * hidden_size + i] =
                        bf16(weights_.token_embd[static_cast<size_t>(btoks[b]) * hidden_size + i] * embed_scale);
            for (int l = 0; l < cfg.num_hidden_layers; ++l) {
                model_.run_layer_batch(xb.data(), B, pos, l, ob.data());
                std::swap(xb, ob);
            }
            // Per-row: final norm + LM head + argmax.
            std::vector<int> preds(B);
            std::vector<bf16> normed(hidden_size);
            std::vector<float> rl;
            for (int b = 0; b < B; ++b) {
                const bf16* xrow = &xb[static_cast<size_t>(b) * hidden_size];
                float var = 0.0f;
                for (int i = 0; i < hidden_size; ++i) { float v = xrow[i].to_float(); var += v * v; }
                var /= hidden_size;
                float inv = 1.0f / std::sqrt(var + cfg.rms_norm_eps);
                for (int i = 0; i < hidden_size; ++i) {
                    float w = weights_.output_norm.empty() ? 1.0f : weights_.output_norm[i];
                    normed[i] = bf16(xrow[i].to_float() * inv * w);
                }
                run_lm_head(normed.data(), rl);
                preds[b] = sample(rl, params);
            }
            // Accept draft[j] while it matches the model's argmax at row j.
            int accept = 0;
            while (accept < nd && preds[accept] == draft[accept]) ++accept;
            double ms = std::chrono::duration<double, std::milli>(clock::now() - t0_step).count();
            tag() << "spec " << ++step << ": draft=" << nd << " accepted=" << accept
                  << " -> " << (accept + 1) << " tok in " << std::fixed << std::setprecision(1) << ms
                  << "ms (" << std::setprecision(1) << (ms / (accept + 1)) << "ms/tok)\n" << std::flush;

            // KV is valid for rows 0..accept (current_token + accepted draft). Emit
            // the accepted draft tokens then the correction preds[accept]; advance
            // pos to pos+accept+1 (correction's KV is written by the next forward).
            // Stale KV at rejected positions is overwritten next iteration.
            bool stop = false;
            for (int j = 0; j < accept; ++j) {
                if (generated >= params.max_tokens) { stop = true; break; }
                if (!emit(draft[j])) { stop = true; break; }
            }
            if (stop) break;
            int corr = preds[accept];
            current_token = corr;
            pos = pos + accept + 1;
            if (generated >= params.max_tokens) break;
            if (!emit(corr)) break;
        }
        // KV is in the cache for positions [0, pos); current_token (seq.back()) has
        // not been forwarded yet. Record only the cached tokens for cross-request reuse.
        cached_tokens_.assign(seq.begin(), seq.begin() + std::min<size_t>(pos, seq.size()));
        return;
    }

    for (int i = 0; i < params.max_tokens; ++i) {
        double npu_s0 = model_.registry().npu_seconds();
        double ffn_s0 = model_.registry().ffn_seconds();
        long npu_c0 = model_.registry().npu_calls();
        auto t0_step = clock::now();
        forward(current_token, pos, true);
        // current_token's KV is now in the cache at position `pos`. Positions
        // [0, num_prompt) are already in cached_tokens_ (== input_tokens); record
        // the fed-back generated tokens that extend the cache beyond the prompt.
        if (pos >= num_prompt_tokens) cached_tokens_.push_back(current_token);
        int next_token = sample(logits, params);
        double step_ms = std::chrono::duration<double, std::milli>(clock::now() - t0_step).count();
        double npu_ms = (model_.registry().npu_seconds() - npu_s0) * 1000.0;
        double ffn_ms = (model_.registry().ffn_seconds() - ffn_s0) * 1000.0;
        long npu_calls = model_.registry().npu_calls() - npu_c0;
        double gemv_ms = npu_ms - lm_head_ms - ffn_ms;  // attention/proj GEMVs
        double cpu_ms = step_ms - npu_ms;
        tag() << "Token " << (i + 1) << "/" << params.max_tokens
              << " in " << std::fixed << std::setprecision(1) << step_ms << "ms"
              << " [ffn=" << ffn_ms << " gemv=" << gemv_ms << " lm_head=" << lm_head_ms
              << " cpu=" << cpu_ms << " | " << npu_calls << " launches]"
              << " (id=" << next_token << ")\n" << std::flush;

        if (tokenizer_.is_stop_token(next_token)) break;
        if (!on_token(tokenizer_.decode(next_token))) break;

        current_token = next_token;
        ++pos;
    }
}

} // namespace alveare
