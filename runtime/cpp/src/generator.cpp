#include "alveare/generator.h"
#include <cmath>
#include <algorithm>
#include <iostream>

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
    int hidden_size = model_.get_config().hidden_size;
    int vocab_size = weights_.token_embd.size() / hidden_size;
    logits.resize(vocab_size);
    
    // For CPU fallback we assume weights_.lm_head contains bf16 data.
    // If lm_head is empty, we might use token_embd as tied weights.
    bool tied = weights_.lm_head.empty();
    
    for (int v = 0; v < vocab_size; ++v) {
        float dot = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            float w_val = 0.0f;
            if (tied) {
                w_val = weights_.token_embd[v * hidden_size + i];
            } else {
                // assume lm_head is packed bf16
                // each bf16 is 2 bytes
                const uint16_t* p = reinterpret_cast<const uint16_t*>(weights_.lm_head.data());
                alveare::bf16 b;
                b.v = p[v * hidden_size + i];
                w_val = b.to_float();
            }
            dot += x[i].to_float() * w_val;
        }
        logits[v] = dot;
    }
}

void Generator::generate(const std::string& prompt, const GenerationParams& params, std::function<bool(const std::string&)> on_token) {
    model_.reset_caches();
    std::vector<int> input_tokens = tokenizer_.encode(prompt);
    
    int hidden_size = model_.get_config().hidden_size;
    std::vector<bf16> x(hidden_size);
    std::vector<bf16> out(hidden_size);
    std::vector<float> logits;

    int pos = 0;
    int next_token = -1;

    for (int step = 0; step < params.max_tokens; ++step) {
        int token = -1;
        if (step < input_tokens.size()) {
            token = input_tokens[step];
        } else {
            token = next_token;
        }

        if (token == tokenizer_.eos_token_id()) {
            break;
        }

        // 1. Embedding lookup
        for (int i = 0; i < hidden_size; ++i) {
            x[i] = bf16(weights_.token_embd[token * hidden_size + i]);
        }

        // 2. Run layers
        for (int l = 0; l < model_.get_config().num_hidden_layers; ++l) {
            model_.run_layer(x.data(), pos, l, out.data());
            x = out; // input for next layer
        }

        // Only compute logits and sample if we are generating (i.e. past the prompt)
        // or if it's the last token of the prompt
        if (step >= static_cast<int>(input_tokens.size()) - 1) {
            // 3. Final Output Norm
            float variance = 0.0f;
            for (int i = 0; i < hidden_size; ++i) {
                float val = x[i].to_float();
                variance += val * val;
            }
            variance /= hidden_size;
            float inv_denom = 1.0f / std::sqrt(variance + model_.get_config().rms_norm_eps);

            std::vector<bf16> normed(hidden_size);
            for (int i = 0; i < hidden_size; ++i) {
                float w = weights_.output_norm.empty() ? 1.0f : weights_.output_norm[i];
                normed[i] = bf16(x[i].to_float() * inv_denom * w);
            }

            // 4. LM Head matmul
            run_lm_head(normed.data(), logits);

            // 5. Sample
            next_token = sample(logits, params);

            // 6. Callback
            if (step >= static_cast<int>(input_tokens.size()) - 1) {
                std::string token_str = tokenizer_.decode(next_token);
                if (!on_token(token_str)) {
                    break;
                }
            }
        }
        pos++;
    }
}

} // namespace alveare
