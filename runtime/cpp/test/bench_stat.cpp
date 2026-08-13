// Statistical NPU kernel benchmark.
//
// The earlier ad-hoc timing loop could not resolve sub-10% kernel changes: the same
// unmodified kernel measured 0.531 / 0.556 / 0.572 ms across runs, so a "5% faster"
// rewrite was indistinguishable from noise. This harness reports a MEDIAN of batch
// means plus the spread, so a change can be called significant (or not) honestly.
//
//   bench_stat <manifest.json> gemv <N> <K> [batches] [iters]
//   bench_stat <manifest.json> ffn  <H> <I> [batches] [iters]
//
// Each batch times `iters` back-to-back calls (same hw context, no switches) and keeps
// the mean; we then report the median batch, the min, and the p25..p75 band. Compare two
// builds by their MEDIAN and check the bands overlap before claiming a win.
#include "alveare/npu.h"
#include "alveare/bf16.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace alveare;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: bench_stat <manifest.json> <gemv|ffn> <N|H> <K|I> [batches=15] [iters=40]\n");
        return 1;
    }
    const std::string manifest = argv[1];
    const std::string kind = argv[2];
    const int A = std::atoi(argv[3]);
    const int B = std::atoi(argv[4]);
    const int batches = argc > 5 ? std::atoi(argv[5]) : 15;
    const int iters   = argc > 6 ? std::atoi(argv[6]) : 40;

    try {
        NpuRegistry reg(manifest);
        std::vector<double> means;
        means.reserve(batches);

        if (kind == "gemv") {
            if (!reg.has_gemv(A, B)) { std::fprintf(stderr, "no gemv %dx%d\n", A, B); return 1; }
            // nbuf > 1 rotates through DISTINCT resident weights, the way decode does:
            // every weight is touched once per token, so reusing one hot buffer (nbuf=1)
            // can report a bandwidth the real decode never sees.
            const int nbuf = argc > 7 ? std::atoi(argv[7]) : 1;
            std::vector<uint8_t> w(size_t(A) * (B / 32) * 20, 0);
            std::vector<WeightHandle> hs;
            for (int i = 0; i < nbuf; ++i) {
                w[(size_t(i) * 977) % w.size()] = uint8_t(i + 1);   // keep buffers distinct
                hs.push_back(reg.create_gemv_weight(A, B, w.data(), w.size()));
            }
            std::vector<bf16> x(B, bf16(0.01f)), y(A);
            for (int i = 0; i < 20; ++i) reg.run_gemv(A, B, hs[i % nbuf], x.data(), y.data());
            // hostwork_us simulates decode: real layers do norms/attention/GELU on the
            // host BETWEEN dispatches, so the NPU goes idle instead of being hammered
            // back-to-back. If a dispatch is more expensive after an idle gap, a tight
            // loop under-reports what decode actually pays.
            const int hostwork_us = argc > 8 ? std::atoi(argv[8]) : 0;
            long call = 0;
            volatile double sink = 0.0;
            for (int b = 0; b < batches; ++b) {
                auto t0 = clk::now();
                for (int i = 0; i < iters; ++i) {
                    reg.run_gemv(A, B, hs[call++ % nbuf], x.data(), y.data());
                    if (hostwork_us > 0) {
                        auto hw = clk::now();
                        while (std::chrono::duration<double, std::micro>(clk::now() - hw).count() < hostwork_us)
                            sink += 1.0;
                    }
                }
                double el = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / iters;
                means.push_back(el - hostwork_us / 1000.0);   // subtract the injected host time
            }
            std::printf("  (nbuf=%d distinct weight buffers)\n", nbuf);
        } else if (kind == "ffn") {
            if (!reg.has_ffn_fused(A, B, "gelu")) { std::fprintf(stderr, "no ffn %dx%d\n", A, B); return 1; }
            std::vector<uint8_t> w(size_t(3) * A * B / 32 * 20, 0);
            WeightHandle h = reg.create_ffn_fused_weight(A, B, "gelu", w.data(), w.size());
            std::vector<bf16> x(A, bf16(0.01f)), y(A);
            for (int i = 0; i < 10; ++i) reg.run_ffn_fused(A, B, "gelu", h, x.data(), y.data());
            for (int b = 0; b < batches; ++b) {
                auto t0 = clk::now();
                for (int i = 0; i < iters; ++i) reg.run_ffn_fused(A, B, "gelu", h, x.data(), y.data());
                means.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count() / iters);
            }
        } else {
            std::fprintf(stderr, "kind must be gemv or ffn\n");
            return 1;
        }

        std::vector<double> s = means;
        std::sort(s.begin(), s.end());
        const double med = s[s.size() / 2];
        const double p25 = s[s.size() / 4];
        const double p75 = s[(3 * s.size()) / 4];
        double sum = 0.0, sq = 0.0;
        for (double v : s) { sum += v; sq += v * v; }
        const double mean = sum / s.size();
        const double sd = std::sqrt(std::max(0.0, sq / s.size() - mean * mean));

        std::printf("%s %dx%d  batches=%d iters=%d\n", kind.c_str(), A, B, batches, iters);
        std::printf("  median  %8.4f ms/call\n", med);
        std::printf("  p25-p75 %8.4f .. %.4f  (band %.1f%%)\n", p25, p75, 100.0 * (p75 - p25) / med);
        std::printf("  min     %8.4f    sd %.4f (%.1f%%)\n", s.front(), sd, 100.0 * sd / med);
        std::printf("  => report the MEDIAN; a change is only meaningful if it moves it\n"
                    "     by more than the p25-p75 band above.\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
