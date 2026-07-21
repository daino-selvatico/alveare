#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
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

        // Benchmark hook: is a gemm(B=16) bandwidth-bound (~= one gemv, so
        // batching wins) or compute-bound (~= 16 gemv, so batching is futile)?
        // Times the big FFN gate shape (16384x4096) with a RESIDENT weight, plus
        // the streamed variant to isolate the host->device upload cost.
        if (std::getenv("ALVEARE_TEST_GEMM")) {
            using clk = std::chrono::steady_clock;
            const int N = 16384, K = 4096, B = 16;  // FFN gate
            auto& gate = mw.layers[0].ffn_gate_bytes;
            WeightHandle wg = reg.create_gemv_weight(N, K, gate.data(), gate.size());
            std::vector<bf16> x(K, bf16(0.02f)), y_gemv(N);
            std::vector<bf16> xb(static_cast<size_t>(B) * K, bf16(0.02f));
            std::vector<bf16> yb(static_cast<size_t>(B) * N);
            auto ms = [](clk::time_point a, clk::time_point b, int n) {
                return std::chrono::duration<double, std::milli>(b - a).count() / n;
            };
            const int IT = 20;
            reg.run_gemv(N, K, wg, x.data(), y_gemv.data());        // warmup
            auto t0 = clk::now();
            for (int i = 0; i < IT; ++i) reg.run_gemv(N, K, wg, x.data(), y_gemv.data());
            double gemv_ms = ms(t0, clk::now(), IT);

            reg.run_gemm(B, N, K, wg, xb.data(), yb.data());        // warmup
            t0 = clk::now();
            for (int i = 0; i < IT; ++i) reg.run_gemm(B, N, K, wg, xb.data(), yb.data());
            double gemm_ms = ms(t0, clk::now(), IT);

            reg.run_gemm_streamed(B, N, K, gate.data(), gate.size(), xb.data(), yb.data());
            t0 = clk::now();
            for (int i = 0; i < IT; ++i)
                reg.run_gemm_streamed(B, N, K, gate.data(), gate.size(), xb.data(), yb.data());
            double gemm_str_ms = ms(t0, clk::now(), IT);

            // Fused FFN (whole gate+up+gelu+down) in isolation, vs the isolated
            // gemvs it replaces. If fused >> ~3x gemv, decode should switch to
            // separate gemvs.
            const int H = 4096, I = 16384;
            std::vector<bf16> xh(H, bf16(0.02f)), yh(H);
            reg.run_ffn_fused(H, I, "gelu", mw.layers[0].w_ffn_fused, xh.data(), yh.data());
            t0 = clk::now();
            for (int i = 0; i < IT; ++i)
                reg.run_ffn_fused(H, I, "gelu", mw.layers[0].w_ffn_fused, xh.data(), yh.data());
            double fused_ms = ms(t0, clk::now(), IT);

            // up gemv (same 16384x4096 shape as gate) for the separate estimate.
            auto& up = mw.layers[0].ffn_up_bytes;
            WeightHandle wu = reg.create_gemv_weight(N, K, up.data(), up.size());
            reg.run_gemv(N, K, wu, x.data(), y_gemv.data());
            t0 = clk::now();
            for (int i = 0; i < IT; ++i) reg.run_gemv(N, K, wu, x.data(), y_gemv.data());
            double up_ms = ms(t0, clk::now(), IT);

            std::cout << "\nFFN gate 16384x4096 timing (avg over " << IT << "):\n"
                      << "  gemv(1 tok)         = " << gemv_ms << " ms\n"
                      << "  gemm(16) resident   = " << gemm_ms << " ms  (per-tok "
                      << gemm_ms / B << " ms)\n"
                      << "  gemm(16) streamed   = " << gemm_str_ms << " ms  (per-tok "
                      << gemm_str_ms / B << " ms)\n"
                      << "  => batch speedup vs gemv (resident): "
                      << (gemv_ms * B) / gemm_ms << "x\n\n"
                      << "Decode FFN, fused vs separate (per token):\n"
                      << "  fused (gate+up+gelu+down) = " << fused_ms << " ms\n"
                      << "  gate gemv                 = " << gemv_ms << " ms\n"
                      << "  up gemv                   = " << up_ms << " ms\n"
                      << "  (down gemv ~ gate; +CPU gelu) est separate ~ "
                      << (gemv_ms + up_ms + gemv_ms) << " ms\n"
                      << "  => fused / separate est   = "
                      << fused_ms / (gemv_ms + up_ms + gemv_ms) << "x\n" << std::flush;
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
