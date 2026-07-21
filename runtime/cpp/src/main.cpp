#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "alveare/config.h"
#include "alveare/bf16.h"
#include "alveare/weights.h"
#include "alveare/npu.h"
#include "alveare/model.h"
#include "alveare/tokenizer.h"
#include "alveare/generator.h"
#include "alveare/server.h"

using namespace alveare;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: alveare_runtime <model_dir> <manifest.json> [port]\n";
        return 1;
    }

    std::string model_dir = argv[1];
    std::string manifest_path = argv[2];
    int port = 8080;
    if (argc >= 4) {
        port = std::stoi(argv[3]);
    }

    try {
        std::cout << "Loading config from " << model_dir << "/config.json\n";
        ModelConfig config = load_config(model_dir + "/config.json");
        
        std::cout << "Initializing NPU Registry with manifest: " << manifest_path << "\n";
        NpuRegistry reg(manifest_path);

        std::cout << "Loading model weights...\n";
        ModelWeights mw = load_weights(model_dir, config, reg);
        
        Model model(config, mw, reg);

        // Validation hook: check run_gemm (batched) against run_gemv on a resident
        // weight. Confirms the gemv weight BO is usable by the gemm kernel.
        if (std::getenv("ALVEARE_TEST_GEMM")) {
            const int N = 4096, K = 4096, B = 16;  // layer 0 attn_q (sliding)
            std::vector<bf16> x(K, bf16(0.02f));
            std::vector<bf16> y_gemv(N);
            reg.run_gemv(N, K, mw.layers[0].w_q, x.data(), y_gemv.data());
            std::vector<bf16> xb(static_cast<size_t>(B) * K);
            for (int b = 0; b < B; ++b)
                for (int i = 0; i < K; ++i) xb[static_cast<size_t>(b) * K + i] = x[i];
            std::vector<bf16> yb(static_cast<size_t>(B) * N);
            reg.run_gemm(B, N, K, mw.layers[0].w_q, xb.data(), yb.data());
            float d0 = 0.0f, d15 = 0.0f, mag = 0.0f;
            for (int i = 0; i < N; ++i) {
                float g = y_gemv[i].to_float();
                mag = std::max(mag, std::fabs(g));
                d0 = std::max(d0, std::fabs(yb[i].to_float() - g));
                d15 = std::max(d15, std::fabs(yb[static_cast<size_t>(15) * N + i].to_float() - g));
            }
            std::cout << "GEMM-vs-GEMV: signal_max=" << mag
                      << " row0_maxdiff=" << d0 << " row15_maxdiff=" << d15 << "\n" << std::flush;
            return 0;
        }

        std::unique_ptr<Tokenizer> tokenizer;
        std::string tok_path = model_dir + "/tokenizer.json";
        try {
            tokenizer = std::make_unique<GemmaTokenizer>(tok_path);
            std::cout << "Loaded tokenizer from " << tok_path << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Warning: no usable tokenizer (" << e.what()
                      << "), falling back to byte StubTokenizer.\n";
            tokenizer = std::make_unique<StubTokenizer>();
        }

        Generator generator(model, mw, *tokenizer);
        std::cout << "Model ready.\n";

        // In-process self-test: run generate() on a fixed prompt and print the
        // decoded text, then exit. Lets us validate the batched prefill in a
        // single foreground process (no server/curl). ALVEARE_SELFTEST may hold
        // the user message; defaults to a short greeting.
        if (const char* st = std::getenv("ALVEARE_SELFTEST")) {
            std::string user_msg = (st[0] != '\0') ? st : "Ciao! Come stai? Raccontami una breve storia.";
            std::string prompt;
            bool is_gemma = (config.model_type == "gemma3" || config.model_type == "gemma4");
            if (is_gemma) {
                prompt = "<bos><|turn>user\n" + user_msg + "<turn|>\n"
                         "<|turn>model\n<|channel>thought\n<channel|>";
            } else {
                prompt = user_msg;
            }
            GenerationParams gp;
            gp.max_tokens = 32;
            std::cout << "\n=== SELFTEST prompt: " << user_msg << " ===\nOUTPUT: " << std::flush;
            generator.generate(prompt, gp, [](const std::string& tok) {
                std::cout << tok << std::flush;
                return true;
            });
            std::cout << "\n=== SELFTEST done ===\n" << std::flush;
            return 0;
        }

        ApiServer server(generator);
        server.start(port);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
