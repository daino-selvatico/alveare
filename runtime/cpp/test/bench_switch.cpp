// Micro-benchmark: is the per-layer NPU cost dominated by kernel context switches?
//
// Decode alternates kernel shapes every layer (qkv/o gemv -> fused FFN -> qkv ...),
// and each shape lives in its own xrt::hw_context. This times:
//   (a) N back-to-back FFN calls        (same context, no switch)
//   (b) N alternating FFN <-> gemv calls (one switch per call)
// The difference is the per-switch cost. Weights are dummy (zero) buffers — we only
// measure dispatch/streaming time, not numerics.
#include "alveare/npu.h"
#include "alveare/bf16.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace alveare;
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: bench_switch <manifest.json> [H] [I] [gemvN] [gemvK] [iters]\n");
        return 1;
    }
    const std::string manifest = argv[1];
    const int H     = argc > 2 ? std::atoi(argv[2]) : 2560;   // e4b hidden
    const int I     = argc > 3 ? std::atoi(argv[3]) : 10240;  // e4b intermediate
    const int gN    = argc > 4 ? std::atoi(argv[4]) : 3072;   // e4b fused-QKV N
    const int gK    = argc > 5 ? std::atoi(argv[5]) : 2560;   // e4b fused-QKV K
    const int iters = argc > 6 ? std::atoi(argv[6]) : 30;

    try {
        NpuRegistry reg(manifest);
        if (!reg.has_ffn_fused(H, I, "gelu")) { std::fprintf(stderr, "no ffn kernel %dx%d\n", H, I); return 1; }
        if (!reg.has_gemv(gN, gK))            { std::fprintf(stderr, "no gemv kernel %dx%d\n", gN, gK); return 1; }

        // Dummy resident weights of the right sizes (Q4_0: 20 bytes per 32 values).
        const size_t ffn_bytes  = size_t(3) * H * I / 32 * 20;
        const size_t gemv_bytes = size_t(gN) * (gK / 32) * 20;
        std::vector<uint8_t> ffn_w(ffn_bytes, 0), gemv_w(gemv_bytes, 0);
        WeightHandle hf = reg.create_ffn_fused_weight(H, I, "gelu", ffn_w.data(), ffn_w.size());
        WeightHandle hg = reg.create_gemv_weight(gN, gK, gemv_w.data(), gemv_w.size());

        std::vector<bf16> x_ffn(H, bf16(0.01f)), y_ffn(H);
        std::vector<bf16> x_gv(gK, bf16(0.01f)), y_gv(gN);

        // warm up both contexts
        reg.run_ffn_fused(H, I, "gelu", hf, x_ffn.data(), y_ffn.data());
        reg.run_gemv(gN, gK, hg, x_gv.data(), y_gv.data());

        auto t0 = clk::now();
        for (int i = 0; i < iters; ++i)
            reg.run_ffn_fused(H, I, "gelu", hf, x_ffn.data(), y_ffn.data());
        double ffn_same = ms_since(t0) / iters;

        t0 = clk::now();
        for (int i = 0; i < iters; ++i)
            reg.run_gemv(gN, gK, hg, x_gv.data(), y_gv.data());
        double gemv_same = ms_since(t0) / iters;

        t0 = clk::now();
        for (int i = 0; i < iters; ++i) {
            reg.run_ffn_fused(H, I, "gelu", hf, x_ffn.data(), y_ffn.data());
            reg.run_gemv(gN, gK, hg, x_gv.data(), y_gv.data());
        }
        double alt_pair = ms_since(t0) / iters;   // one FFN + one gemv, with 2 switches

        std::printf("ffn  same-context : %7.3f ms/call\n", ffn_same);
        std::printf("gemv same-context : %7.3f ms/call\n", gemv_same);
        std::printf("alternating pair  : %7.3f ms  (ffn+gemv)\n", alt_pair);
        std::printf("no-switch pair    : %7.3f ms\n", ffn_same + gemv_same);
        std::printf("=> switch overhead: %7.3f ms per pair (%.3f ms per switch)\n",
                    alt_pair - (ffn_same + gemv_same), (alt_pair - (ffn_same + gemv_same)) / 2.0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
