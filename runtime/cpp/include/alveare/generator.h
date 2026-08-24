#pragma once
#include "alveare/model.h"
#include "alveare/tokenizer.h"
#include <string>
#include <functional>
#include <mutex>
#include <vector>
#include <random>

namespace alveare {

struct GenerationParams {
    int max_tokens = 100;
    float temperature = 0.0f; // 0.0 means greedy (deterministic, bit-exact)
    float top_p = 1.0f;       // nucleus sampling; 1.0 = disabled
    int top_k = 0;            // keep only the k highest-logit tokens; 0 = disabled
    unsigned seed = 0;        // RNG seed; 0 = nondeterministic (random_device)
    std::vector<std::string> stop; // custom stop sequences
};

struct GenerationStats {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double prefill_time_ms = 0.0;
    double decode_time_ms = 0.0;
};

class Generator {
public:
    Generator(Model& model, const ModelWeights& weights, const Tokenizer& tokenizer);

    // Generates text and calls the callback for each new token generated.
    // Callback should return true to continue, false to stop.
    // Accepts optional visual_embeddings to replace <|image|> placeholder tokens.
    // Returns generation statistics.
    GenerationStats generate(
        const std::string& prompt,
        const GenerationParams& params,
        std::function<bool(const std::string&)> on_token,
        const std::vector<std::vector<float>>& visual_embeddings = {},
        const std::vector<std::vector<float>>& audio_embeddings = {}
    );

    // Resets both the underlying model KV cache and the generator's cached_tokens_ tracker.
    void reset_cache();

    // Model config (used e.g. by the server to pick the chat template).
    const ModelConfig& config() const { return model_.get_config(); }

private:
    Model& model_;
    const ModelWeights& weights_;
    const Tokenizer& tokenizer_;

    // Token sequence currently represented in the model's KV cache (prompt +
    // fed-back generated tokens of the previous request). The next request
    // reuses the longest common prefix instead of re-prefilling it, so
    // multi-turn chat does not re-prefill the whole conversation each turn.
    std::vector<int> cached_tokens_;

    // generate() mutates the single shared KV cache + cached_tokens_, so it is
    // not reentrant; serialize concurrent server requests (the NPU runs one
    // forward at a time anyway).
    std::mutex gen_mutex_;

    // Sampling RNG. Seeded once per generate() call (from params.seed when set,
    // else left advancing for nondeterministic output), so draws are independent
    // across tokens within a request while staying reproducible for a fixed seed.
    std::mt19937 rng_{std::random_device{}()};

    int sample(const std::vector<float>& logits, const GenerationParams& params);
    void run_lm_head(const bf16* x, std::vector<float>& logits);

    // Pre-allocated scratchpad buffers (avoids malloc per token)
    std::vector<float> inpL_f_;
    std::vector<float> inp_per_layer_;
    std::vector<bf16> x_;
    std::vector<bf16> out_;
    std::vector<bf16> normed_;
    std::vector<bf16> lm_x_pad_;
    std::vector<bf16> lm_y_;
    std::vector<float> logits_;
};

} // namespace alveare
