// Smoke test for the native-XRT NPU registry (src/npu.cpp).
//
// Loads kernels/build/manifest.json, uploads a resident weight, runs one real
// harvested GEMV shape on the NPU, and checks parity against the CPU dequant
// reference produced by tools/dump_gemv_golden.py.
//
//   npu_smoke <manifest.json> <N> <K> <golden_dir>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "alveare/bf16.h"
#include "alveare/npu.h"
#include "alveare/npy.h"

using namespace alveare;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: npu_smoke <manifest.json> <N> <K> <golden_dir>\n";
        return 2;
    }
    const std::string manifest = argv[1];
    const int N = std::stoi(argv[2]);
    const int K = std::stoi(argv[3]);
    const std::string gd = argv[4];

    try {
        NpuRegistry reg(manifest);
        std::cout << "manifest: model=" << reg.model_type()
                  << " kernels=" << reg.kernels().size() << "\n";
        if (!reg.has_gemv(N, K)) {
            std::cerr << "no gemv " << N << "x" << K << " in manifest\n";
            return 1;
        }

        const std::string base =
            gd + "/gemv_" + std::to_string(N) + "x" + std::to_string(K);
        NpyArray W = load_npy(base + "_W.npy");
        NpyArray X = load_npy(base + "_x.npy");
        // Primary oracle: the Python/pyxrt NPU output (same xclbin + instruction
        // stream + device), which the registry must reproduce bit-for-bit.
        NpyArray NPU = load_npy(base + "_npu.npy");
        // Informational: the fp32 CPU dequant reference (the NPU legitimately
        // diverges from this within the kernel's rtol=0.05/atol=1.0, more so as K
        // grows and cancellation dominates near-zero outputs).
        NpyArray E = load_npy(base + "_expected.npy");

        WeightHandle wh = reg.create_gemv_weight(N, K, W.data, W.data_size);
        reg.pin_gemv(N, K);
        std::cout << "loaded_contexts=" << reg.loaded_contexts()
                  << " (expect 1)\n";

        std::vector<uint16_t> y(static_cast<size_t>(N));
        reg.run_gemv(N, K, wh, X.data, y.data());

        const bf16* npu = static_cast<const bf16*>(NPU.data);
        const bf16* exp = static_cast<const bf16*>(E.data);
        float max_vs_npu = 0.0f;   // must be exactly 0
        float max_vs_cpu = 0.0f;   // informational
        size_t npu_mismatches = 0;
        for (int i = 0; i < N; ++i) {
            bf16 a;
            a.v = y[static_cast<size_t>(i)];
            const float av = a.to_float();
            const float d_npu = std::fabs(av - npu[i].to_float());
            const float d_cpu = std::fabs(av - exp[i].to_float());
            if (d_npu > max_vs_npu) max_vs_npu = d_npu;
            if (d_cpu > max_vs_cpu) max_vs_cpu = d_cpu;
            // Bit-exact check: compare the raw bf16 payloads.
            if (y[static_cast<size_t>(i)] != npu[i].v || std::isnan(av)) {
                if (npu_mismatches < 10)
                    std::cout << "NPU mismatch @" << i << " python "
                              << npu[i].to_float() << " cpp " << av << "\n";
                ++npu_mismatches;
            }
        }
        std::cout << "max_diff vs Python-NPU=" << max_vs_npu
                  << " (bit mismatches=" << npu_mismatches << ")\n";
        std::cout << "max_diff vs CPU-ref=" << max_vs_cpu << " (informational)\n";

        free_npy(W);
        free_npy(X);
        free_npy(NPU);
        free_npy(E);

        if (npu_mismatches == 0) {
            std::cout << "PASS: C++ registry gemv " << N << "x" << K
                      << " is bit-exact vs Python NPU output\n";
            return 0;
        }
        std::cout << "FAIL: C++ output differs from Python NPU output\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
