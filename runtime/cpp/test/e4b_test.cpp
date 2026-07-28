#include "alveare/model.h"
#include "alveare/weights.h"
#include "alveare/config.h"
#include "alveare/npu.h"
#include "alveare/tokenizer.h"
#include "alveare/generator.h"
#include <iostream>
#include <vector>
#include <string>

using namespace alveare;

struct TestCase {
    std::string name;
    std::string prompt;
    std::vector<int> expected_tokens;
};

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "quantized_weights_gemma4-e4b";
    std::string manifest_path = (argc > 2) ? argv[2] : "kernels/build/manifest.json";

    std::vector<TestCase> test_cases = {
        {"Prompt 1", "Hello world", {236888, 1174, 563, 496, 1594, 236761, 108, 13513}},
        {"Prompt 2", "The capital of France is", {9079, 236761, 106}},
        {"Prompt 3", "Write a python function to add two numbers:", {108, 2717, 6719, 107, 2063, 1184, 46194, 235282, 477, 235269, 522, 235334}},
        {"Prompt 4", "What is the speed of light?", {108, 1018, 7925, 53121, 669, 235300, 16867, 989, 2196, 1018}}
    };

    try {
        std::cout << "Loading config from " << model_dir << "...\n";
        ModelConfig config = load_config(model_dir + "/config.json");
        NpuRegistry reg(manifest_path);

        std::cout << "Loading weights from " << model_dir << "...\n";
        ModelWeights mw = load_weights(model_dir, config, reg);
        Model model(config, mw, reg);

        std::string tok_path = model_dir + "/tokenizer.json";
        std::unique_ptr<Tokenizer> tokenizer;
        try {
            tokenizer = std::make_unique<GemmaTokenizer>(tok_path);
            std::cout << "Loaded tokenizer from " << tok_path << "\n";
        } catch (const std::exception& e) {
            std::cout << "Fallback to StubTokenizer: " << e.what() << "\n";
            tokenizer = std::make_unique<StubTokenizer>();
        }

        Generator generator(model, mw, *tokenizer);

        for (const auto& tc : test_cases) {
            std::cout << "\n========================================\n";
            std::cout << "Running " << tc.name << ": \"" << tc.prompt << "\"\n";
            
            // Reset KV cache between prompts to ensure clean state
            model.reset_caches();

            GenerationParams params;
            params.max_tokens = static_cast<int>(tc.expected_tokens.size());

            std::cout << "OUTPUT: " << std::flush;
            generator.generate(tc.prompt, params, [&](const std::string& text) {
                std::cout << text << std::flush;
                return true;
            });
            std::cout << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
