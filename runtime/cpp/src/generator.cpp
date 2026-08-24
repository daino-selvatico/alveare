#include "alveare/generator.h"
#include "alveare/prompt_lookup.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <fstream>

namespace alveare {

Generator::Generator(Model& model, const ModelWeights& weights, const Tokenizer& tokenizer)
    : model_(model), weights_(weights), tokenizer_(tokenizer) {
    int hidden_size = model_.get_config().hidden_size;
    int vocab_size = weights_.lm_head_vocab > 0 ? weights_.lm_head_vocab : model_.get_config().vocab_size;
    int chunk_N = weights_.lm_head_chunk_N > 0 ? weights_.lm_head_chunk_N : 16384;
    int lm_K = weights_.lm_head_K > 0 ? weights_.lm_head_K : hidden_size;

    inpL_f_.assign(hidden_size, 0.0f);
    x_.assign(hidden_size, bf16(0.0f));
    out_.assign(hidden_size, bf16(0.0f));
    normed_.assign(hidden_size, bf16(0.0f));
    lm_x_pad_.assign(lm_K, bf16(0.0f));
    lm_y_.assign(chunk_N, bf16(0.0f));
    logits_.assign(vocab_size, 0.0f);
    if (model_.get_config().per_layer_input > 0) {
        inp_per_layer_.assign(model_.get_config().num_hidden_layers * model_.get_config().per_layer_input, 0.0f);
    }
}

void Generator::reset_cache() {
    std::lock_guard<std::mutex> gen_lock(gen_mutex_);
    model_.reset_caches();
    cached_tokens_.clear();
}

int Generator::sample(const std::vector<float>& logits, const GenerationParams& params) {
    const int n = static_cast<int>(logits.size());
    if (n == 0) return -1;

    if (std::getenv("ALVEARE_DUMP_TOPK")) {
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 6, idx.end(),
                          [&](int a, int b){ return logits[a] > logits[b]; });
        std::cerr << "[topk]";
        for (int j = 0; j < 6; ++j) std::cerr << " " << idx[j] << ":" << logits[idx[j]];
        std::cerr << "\n";
    }

    // Greedy (deterministic, bit-exact) when temperature ~ 0 — the default path,
    // keeps the 12B/e4b/gemma3 outputs reproducible.
    if (params.temperature <= 1e-6f) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < n; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

    // --- temperature / top-k / top-p (nucleus) sampling ---
    // rng_ is seeded once per request in generate() (not here), so successive
    // tokens draw independently while staying reproducible for a fixed seed.
    const float inv_t = 1.0f / params.temperature;

    // Candidate set: top-k highest logits when top_k>0 or top_p<1 (needs ranking);
    // otherwise the whole vocab (pure temperature sampling). Cap the ranked set at
    // 2048 when only top_p is set — beyond that the tail prob is negligible.
    const bool need_rank = (params.top_k > 0) || (params.top_p < 1.0f);
    std::vector<int> cand(n);
    for (int i = 0; i < n; ++i) cand[i] = i;
    if (need_rank) {
        int k = params.top_k > 0 ? std::min(params.top_k, n) : std::min(n, 2048);
        std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                          [&](int a, int b){ return logits[a] > logits[b]; });
        cand.resize(k);
    }

    // Temperature-scaled softmax over the candidates (numerically stable).
    float maxl = logits[cand[0]];
    for (int idx : cand) maxl = std::max(maxl, logits[idx]);
    std::vector<float> probs(cand.size());
    float sum = 0.0f;
    for (size_t i = 0; i < cand.size(); ++i) {
        float p = std::exp((logits[cand[i]] - maxl) * inv_t);
        probs[i] = p;
        sum += p;
    }
    for (float& p : probs) p /= sum;

    // Nucleus (top-p): cand is sorted desc when need_rank, so keep the smallest
    // prefix whose cumulative probability reaches top_p.
    size_t keep = cand.size();
    if (params.top_p < 1.0f) {
        float cum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            if (cum >= params.top_p) { keep = i + 1; break; }
        }
    }

    // Sample from the kept prefix (renormalized) via inverse CDF.
    float norm = 0.0f;
    for (size_t i = 0; i < keep; ++i) norm += probs[i];
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float r = uni(rng_) * norm;
    float cum = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        cum += probs[i];
        if (r <= cum) return cand[i];
    }
    return cand[keep - 1];
}

void Generator::run_lm_head(const bf16* x, std::vector<float>& logits) {
    const ModelConfig& cfg = model_.get_config();
    int hidden_size = cfg.hidden_size;

    // NPU path: the packed LM head was uploaded as row-tiles; run one quantized
    // gemv per tile and concatenate. x is zero-padded to the kernel's K.
    if (!weights_.lm_head_chunks.empty()) {
        int K = weights_.lm_head_K;
        int chunk_N = weights_.lm_head_chunk_N;
        if (logits.size() != static_cast<size_t>(weights_.lm_head_vocab)) {
            logits.resize(weights_.lm_head_vocab);
        }

        std::fill(lm_x_pad_.begin(), lm_x_pad_.end(), bf16(0.0f));
        for (int i = 0; i < hidden_size && i < K; ++i) lm_x_pad_[i] = x[i];

        for (size_t c = 0; c < weights_.lm_head_chunks.size(); ++c) {
            model_.registry().run_gemv(chunk_N, K, weights_.lm_head_chunks[c],
                                       lm_x_pad_.data(), lm_y_.data());
            int base = static_cast<int>(c) * chunk_N;
            for (int i = 0; i < chunk_N; ++i) {
                logits[base + i] = lm_y_[i].to_float(); // raw logits
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

GenerationStats Generator::generate(
    const std::string& prompt,
    const GenerationParams& params,
    std::function<bool(const std::string&)> on_token,
    const std::vector<std::vector<float>>& visual_embeddings,
    const std::vector<std::vector<float>>& audio_embeddings
) {
    GenerationStats stats;
    std::lock_guard<std::mutex> gen_lock(gen_mutex_);
    // Reseed once per request for reproducible sampling with a fixed seed; when
    // seed==0 leave rng_ advancing so repeated requests stay nondeterministic.
    if (params.seed != 0) rng_.seed(params.seed);
    using clock = std::chrono::steady_clock;
    static std::atomic<int> req_counter{0};
    int req = ++req_counter;
    auto tag = [&]() -> std::ostream& { return std::cout << "[req-" << req << "] "; };

    const ModelConfig& cfg = model_.get_config();
    int hidden_size = cfg.hidden_size;
    bool is_gemma = (cfg.model_type == "gemma3" || cfg.is_gemma4());
    float embed_scale = is_gemma ? std::sqrt(static_cast<float>(hidden_size)) : 1.0f;

    std::vector<int> input_tokens = tokenizer_.encode(prompt);
    int num_prompt_tokens = static_cast<int>(input_tokens.size());
    stats.prompt_tokens = num_prompt_tokens;
    if (num_prompt_tokens == 0) {
        tag() << "empty prompt, nothing to generate\n" << std::flush;
        return stats;
    }

    std::unordered_map<int, const float*> custom_emb_map;
    if (!visual_embeddings.empty()) {
        size_t v_idx = 0;
        for (size_t i = 0; i < input_tokens.size(); ++i) {
            if (input_tokens[i] == 258880 && v_idx < visual_embeddings.size()) {
                custom_emb_map[static_cast<int>(i)] = visual_embeddings[v_idx++].data();
            }
        }
        tag() << "Injected " << v_idx << " visual token embeddings into prompt sequence\n" << std::flush;
    }
    if (!audio_embeddings.empty()) {
        size_t a_idx = 0;
        for (size_t i = 0; i < input_tokens.size(); ++i) {
            if (input_tokens[i] == 258881 && a_idx < audio_embeddings.size()) {
                custom_emb_map[static_cast<int>(i)] = audio_embeddings[a_idx++].data();
            }
        }
        tag() << "Injected " << a_idx << " audio token embeddings into prompt sequence\n" << std::flush;
    }

    std::cout << "[input_tokens]";
    for (int t : input_tokens) std::cout << " " << t;
    std::cout << "\n" << std::flush;

    // KV-cache reuse: the model's cache already holds valid state for the tokens
    // of the previous request (cached_tokens_). Reuse the longest common prefix
    // and only prefill from there — for multi-turn chat this skips re-prefilling
    // the whole conversation history. We never need the last prompt token in the
    // reused prefix (the decode loop processes it), so cap at num_prompt-1.
    int reuse = 0;
    if (visual_embeddings.empty() && audio_embeddings.empty()) {
        int maxP = std::min(static_cast<int>(cached_tokens_.size()), num_prompt_tokens - 1);
        while (reuse < maxP && input_tokens[reuse] == cached_tokens_[reuse]) ++reuse;
    }
    if (reuse == 0) {
        model_.reset_caches();
    }
    cached_tokens_ = input_tokens;

    bf16* cur_x = x_.data();
    bf16* cur_out = out_.data();

    double lm_head_ms = 0.0;  // profiling: last forward's LM-head wall time

    // Run one token through the embedding + all transformer layers. When
    // want_logits is set, also apply the final norm and LM head into `logits_`.
    auto forward = [&](int token, int pos, bool want_logits) {
        float* inpL_ptr = inpL_f_.data();
        auto it = custom_emb_map.find(pos);
        if (it != custom_emb_map.end()) {
            const float* emb_ptr = it->second;
            for (int i = 0; i < hidden_size; ++i) {
                float val = emb_ptr[i];
                cur_x[i] = bf16(val);
                inpL_ptr[i] = val;
            }
        } else {
            const float* emb_ptr = &weights_.token_embd[static_cast<size_t>(token) * hidden_size];
            for (int i = 0; i < hidden_size; ++i) {
                float val = emb_ptr[i] * embed_scale;
                cur_x[i] = bf16(val);
                inpL_ptr[i] = val;
            }
        }
        static const bool no_ple = (std::getenv("ALVEARE_NO_PLE") != nullptr);
        if (cfg.per_layer_input > 0 && !no_ple) {
            model_.compute_per_layer_inputs(token, inpL_ptr, inp_per_layer_);
        }
        const float* ple_ptr = inp_per_layer_.empty() ? nullptr : inp_per_layer_.data();
        for (int l = 0; l < cfg.num_hidden_layers; ++l) {
            model_.run_layer(cur_x, pos, l, cur_out, ple_ptr);
            std::swap(cur_x, cur_out);
        }
        if (!want_logits) return;

        float variance = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float val = cur_x[i].to_float();
            variance += val * val;
        }
        variance /= hidden_size;
        float inv_denom = 1.0f / std::sqrt(variance + cfg.rms_norm_eps);

        bf16* normed_ptr = normed_.data();
        for (int i = 0; i < hidden_size; ++i) {
            float w = weights_.output_norm.empty() ? 1.0f : weights_.output_norm[i];
            normed_ptr[i] = bf16(cur_x[i].to_float() * inv_denom * w);
        }
        auto t_lm = clock::now();
        run_lm_head(normed_ptr, logits_);
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
    // Batched (B=16 GEMM) prefill uses the resident ONESHAPE GEMM tiles (zero weight streaming)
    // and achieves 4.35x faster prompt prefill (~60-96ms/tok vs ~350-420ms/tok).
    // Enabled by default for all Gemma-4 models when prefill length >= 4 tokens.
    bool use_batched = cfg.is_gemma4() && (prefill_count - reuse >= 4) && (std::getenv("ALVEARE_NO_BATCH_PREFILL") == nullptr);
    if (use_batched) {
        const int PB = 16;
        std::vector<bf16> xb(static_cast<size_t>(PB) * hidden_size, bf16(0.0f));
        std::vector<bf16> ob(static_cast<size_t>(PB) * hidden_size, bf16(0.0f));
        std::vector<float> batch_ple;
        std::vector<float> inpL_tmp(hidden_size);
        if (cfg.per_layer_input > 0) {
            batch_ple.resize(static_cast<size_t>(PB) * cfg.num_hidden_layers * cfg.per_layer_input);
        }
        for (int start = reuse; start < prefill_count; start += PB) {
            int nrows = std::min(PB, prefill_count - start);
            std::fill(xb.begin(), xb.end(), bf16(0.0f));
            for (int b = 0; b < nrows; ++b) {
                int pos = start + b;
                int token = input_tokens[pos];
                auto it = custom_emb_map.find(pos);
                float cur_scale = (it != custom_emb_map.end()) ? 1.0f : embed_scale;
                const float* emb_ptr = (it != custom_emb_map.end()) ? it->second : &weights_.token_embd[static_cast<size_t>(token) * hidden_size];
                for (int i = 0; i < hidden_size; ++i) {
                    float val = emb_ptr[i] * cur_scale;
                    xb[static_cast<size_t>(b) * hidden_size + i] = bf16(val);
                    inpL_tmp[i] = val;
                }
                static const bool no_ple = (std::getenv("ALVEARE_NO_PLE") != nullptr);
                if (cfg.per_layer_input > 0 && !no_ple) {
                    float* ple_dst = &batch_ple[static_cast<size_t>(b) * cfg.num_hidden_layers * cfg.per_layer_input];
                    model_.compute_per_layer_inputs(token, inpL_tmp.data(), ple_dst);
                }
            }
            const float* ple_ptr = batch_ple.empty() ? nullptr : batch_ple.data();
            for (int l = 0; l < cfg.num_hidden_layers; ++l) {
                model_.run_layer_batch(xb.data(), nrows, start, l, ob.data(), ple_ptr);
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
    stats.prefill_time_ms = prefill_s * 1000.0;
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
    bool use_spec = cfg.is_gemma4() && (std::getenv("ALVEARE_NO_SPECULATIVE") == nullptr);
    if (use_spec) {
        auto t0_decode = clock::now();
        const int K_draft = 7;                 // max draft length (fills the B=16 pad)
        const int max_seq_len = model_.get_config().max_position_embeddings;
        std::vector<int> seq = input_tokens;  // committed sequence (drafter context)
        int generated = 0, step = 0;

        auto emit = [&](int t) -> bool {  // returns false to stop
            if (tokenizer_.is_stop_token(t)) return false;
            std::string text = tokenizer_.decode(t);
            for (const auto& s : params.stop) {
                if (!s.empty() && text.find(s) != std::string::npos) return false;
            }
            if (!on_token(text)) return false;
            seq.push_back(t);
            ++generated;
            return true;
        };

        const int max_B = 16;
        std::vector<bf16> xb(static_cast<size_t>(max_B) * hidden_size, bf16(0.0f));
        std::vector<bf16> ob(static_cast<size_t>(max_B) * hidden_size, bf16(0.0f));
        std::vector<int> btoks(max_B);
        std::vector<int> preds(max_B);
        std::vector<bf16> normed(hidden_size);
        std::vector<float> rl;
        std::vector<float> batch_ple;
        std::vector<float> inpL_tmp(hidden_size);
        if (cfg.per_layer_input > 0) {
            batch_ple.resize(static_cast<size_t>(max_B) * cfg.num_hidden_layers * cfg.per_layer_input);
        }

        while (generated < params.max_tokens) {
            auto t0_step = clock::now();
            std::vector<int> draft = propose_draft(seq, K_draft, 3, 3);
            while (!draft.empty() && pos + static_cast<int>(draft.size()) >= max_seq_len)
                draft.pop_back();
            int nd = static_cast<int>(draft.size());

            if (nd == 0) {
                // Fallback: normal single-token decode (fused FFN, zero B=16 penalty).
                forward(current_token, pos, true);
                if (pos >= num_prompt_tokens) cached_tokens_.push_back(current_token);
                int t = sample(logits_, params);
                double ms = std::chrono::duration<double, std::milli>(clock::now() - t0_step).count();
                tag() << "spec " << ++step << ": draft=0 fallback -> 1 tok in "
                      << std::fixed << std::setprecision(1) << ms << "ms (id=" << t << ")\n" << std::flush;
                current_token = t; ++pos;
                if (!emit(t)) break;
                continue;
            }

            // Batched verify: rows = [current_token, draft...] at [pos..pos+nd].
            int B = nd + 1;
            btoks[0] = current_token;
            for (int j = 0; j < nd; ++j) btoks[j + 1] = draft[j];
            for (int b = 0; b < B; ++b) {
                const float* emb_ptr = &weights_.token_embd[static_cast<size_t>(btoks[b]) * hidden_size];
                for (int i = 0; i < hidden_size; ++i) {
                    float val = emb_ptr[i] * embed_scale;
                    xb[static_cast<size_t>(b) * hidden_size + i] = bf16(val);
                    inpL_tmp[i] = val;
                }
                static const bool no_ple = (std::getenv("ALVEARE_NO_PLE") != nullptr);
                if (cfg.per_layer_input > 0 && !no_ple) {
                    float* ple_dst = &batch_ple[static_cast<size_t>(b) * cfg.num_hidden_layers * cfg.per_layer_input];
                    model_.compute_per_layer_inputs(btoks[b], inpL_tmp.data(), ple_dst);
                }
            }
            const float* ple_ptr = batch_ple.empty() ? nullptr : batch_ple.data();
            for (int l = 0; l < cfg.num_hidden_layers; ++l) {
                model_.run_layer_batch(xb.data(), B, pos, l, ob.data(), ple_ptr);
                std::swap(xb, ob);
            }
            // Per-row: final norm + LM head + argmax.
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
        stats.completion_tokens = generated;
        stats.decode_time_ms = std::chrono::duration<double, std::milli>(clock::now() - t0_decode).count();
        return stats;
    }

    auto t0_decode = clock::now();
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
        int next_token = sample(logits_, params);
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
        std::string token_str = tokenizer_.decode(next_token);
        bool user_stop = false;
        for (const auto& s : params.stop) {
            if (!s.empty() && token_str.find(s) != std::string::npos) {
                user_stop = true; break;
            }
        }
        if (user_stop) break;
        if (!on_token(token_str)) break;
        stats.completion_tokens++;

        current_token = next_token;
        ++pos;
    }
    stats.decode_time_ms = std::chrono::duration<double, std::milli>(clock::now() - t0_decode).count();
    return stats;
}

} // namespace alveare

