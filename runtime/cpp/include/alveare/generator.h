#pragma once
#include "alveare/model.h"
#include "alveare/tokenizer.h"
#include <string>
#include <functional>
#include <vector>

namespace alveare {

struct GenerationParams {
    int max_tokens = 100;
    float temperature = 0.0f; // 0.0 means greedy
    float top_p = 1.0f;
};

class Generator {
public:
    Generator(Model& model, const ModelWeights& weights, const Tokenizer& tokenizer);

    // Generates text and calls the callback for each new token generated.
    // Callback should return true to continue, false to stop.
    void generate(const std::string& prompt, const GenerationParams& params, std::function<bool(const std::string&)> on_token);

private:
    Model& model_;
    const ModelWeights& weights_;
    const Tokenizer& tokenizer_;

    int sample(const std::vector<float>& logits, const GenerationParams& params);
    void run_lm_head(const bf16* x, std::vector<float>& logits);
};

} // namespace alveare
