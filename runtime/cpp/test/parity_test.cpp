#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "alveare/bf16.h"
#include "alveare/npu.h"
#include "alveare/npy.h"

using namespace alveare;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: parity_test <manifest.json> <golden_dir>\n";
        return 2;
    }
    const std::string manifest = argv[1];
    const std::string gd = argv[2];

    try {
        NpuRegistry reg(manifest);
        std::cout << "manifest: model=" << reg.model_type()
                  << " kernels=" << reg.kernels().size() << "\n";

        size_t total_mismatches = 0;
        int shapes_tested = 0;

        for (const auto& spec : reg.kernels()) {
            if (spec.kind == "gemm") continue; // Skip gemm for now

            std::string base;
            if (spec.kind == "gemv") {
                base = gd + "/gemv_" + std::to_string(spec.N) + "x" + std::to_string(spec.K);
            } else if (spec.kind == "ffn_fused") {
                base = gd + "/ffn_fused_" + std::to_string(spec.H) + "x" + std::to_string(spec.I) + "_" + spec.activation;
            } else {
                continue;
            }

            if (!std::filesystem::exists(base + "_W.npy")) {
                std::cout << "SKIP: " << spec.kind << " (no golden found at " << base << "_W.npy)\n";
                continue;
            }

            std::cout << "Testing " << spec.kind << " " << base.substr(base.find_last_of('/') + 1) << " ... ";

            NpyArray W = load_npy(base + "_W.npy");
            NpyArray X = load_npy(base + "_x.npy");
            NpyArray NPU = load_npy(base + "_npu.npy");

            WeightHandle wh;
            std::vector<uint16_t> y;

            if (spec.kind == "gemv") {
                wh = reg.create_gemv_weight(spec.N, spec.K, W.data, W.data_size);
                reg.pin_gemv(spec.N, spec.K);
                y.resize(static_cast<size_t>(spec.N));
                reg.run_gemv(spec.N, spec.K, wh, X.data, y.data());
            } else if (spec.kind == "ffn_fused") {
                wh = reg.create_ffn_fused_weight(spec.H, spec.I, spec.activation, W.data, W.data_size);
                reg.pin_ffn_fused(spec.H, spec.I, spec.activation);
                y.resize(static_cast<size_t>(spec.H));
                reg.run_ffn_fused(spec.H, spec.I, spec.activation, wh, X.data, y.data());
            }

            const bf16* npu = static_cast<const bf16*>(NPU.data);
            float max_vs_npu = 0.0f;
            size_t npu_mismatches = 0;
            size_t out_len = (spec.kind == "gemv") ? spec.N : spec.H;

            for (size_t i = 0; i < out_len; ++i) {
                bf16 a;
                a.v = y[i];
                const float av = a.to_float();
                const float d_npu = std::fabs(av - npu[i].to_float());
                if (d_npu > max_vs_npu) max_vs_npu = d_npu;
                
                if (y[i] != npu[i].v) {
                    if (std::isnan(av) && std::isnan(npu[i].to_float())) {
                        // Both are NaN, ignore bit representation differences
                        continue;
                    }
                    if (npu_mismatches < 5)
                        std::cerr << "\nNPU mismatch @" << i << " python "
                                  << npu[i].to_float() << " cpp " << av;
                    ++npu_mismatches;
                }
            }

            free_npy(W);
            free_npy(X);
            free_npy(NPU);

            if (npu_mismatches == 0) {
                std::cout << "PASS (max_diff=0)\n";
            } else {
                std::cout << "FAIL (" << npu_mismatches << " mismatches)\n";
            }
            total_mismatches += npu_mismatches;
            shapes_tested++;
        }

        std::cout << "\nTested " << shapes_tested << " shapes. Total loaded contexts: " << reg.loaded_contexts() << "\n";

        if (total_mismatches == 0 && shapes_tested > 0) {
            std::cout << "ALL PASS!\n";
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
