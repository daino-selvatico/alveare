#include "alveare/generator.h"
#include <cmath>
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
    const int K_padded = (cfg.model_type == "gemma4") ? 4096 : hidden_size;
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
    using clock = std::chrono::steady_clock;
    static std::atomic<int> req_counter{0};
    int req = ++req_counter;
    auto tag = [&]() -> std::ostream& { return std::cout << "[req-" << req << "] "; };

    const ModelConfig& cfg = model_.get_config();
    int hidden_size = cfg.hidden_size;
    bool is_gemma = (cfg.model_type == "gemma3" || cfg.model_type == "gemma4");
    float embed_scale = is_gemma ? std::sqrt(static_cast<float>(hidden_size)) : 1.0f;

    model_.reset_caches();
    std::vector<int> input_tokens = tokenizer_.encode(prompt);
    int num_prompt_tokens = static_cast<int>(input_tokens.size());
    if (num_prompt_tokens == 0) {
        tag() << "empty prompt, nothing to generate\n" << std::flush;
        return;
    }

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
    tag() << "Starting prefill of " << num_prompt_tokens << " tokens...\n" << std::flush;
    auto t0_prefill = clock::now();
    if (cfg.model_type == "gemma4") {
        const int PB = 16;
        std::vector<bf16> xb, ob;
        for (int start = 0; start < prefill_count; start += PB) {
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
        for (int pos = 0; pos < prefill_count; ++pos) {
            forward(input_tokens[pos], pos, false);
        }
    }
    double prefill_s = std::chrono::duration<double>(clock::now() - t0_prefill).count();
    tag() << "Prefill completed in " << std::fixed << std::setprecision(2) << prefill_s << "s\n" << std::flush;

    // 2. Decode: the last prompt token produces the first generated token.
    int current_token = input_tokens.back();
    int pos = num_prompt_tokens - 1;
    for (int i = 0; i < params.max_tokens; ++i) {
        double npu_s0 = model_.registry().npu_seconds();
        double ffn_s0 = model_.registry().ffn_seconds();
        long npu_c0 = model_.registry().npu_calls();
        auto t0_step = clock::now();
        forward(current_token, pos, true);
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
