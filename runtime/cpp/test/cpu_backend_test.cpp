#include "alveare/cpu_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <random>
#include <algorithm>

using namespace alveare;

static inline int clamp_i(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// Packs float weights (N, K) into Q4_0 packed layout: N * (K / 32) * 20
static std::vector<uint8_t> pack_reference_q4_0(const std::vector<float>& W, int N, int K) {
    const int K_blocks = K / 32;
    std::vector<uint8_t> packed(size_t(N) * K_blocks * 20, 0);

    for (int r = 0; r < N; ++r) {
        uint8_t* row_out = packed.data() + static_cast<size_t>(r) * K_blocks * 20;
        const float* row_in = W.data() + static_cast<size_t>(r) * K;

        for (int bk = 0; bk < K_blocks; ++bk) {
            const float* blk_in = row_in + bk * 32;
            uint8_t* blk_out = row_out + bk * 20;

            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::abs(blk_in[i]));
            }
            float scale = (amax == 0.0f) ? 1.0f : (amax / 7.0f);
            bf16 sc_bf16(scale);
            blk_out[16] = static_cast<uint8_t>(sc_bf16.v & 0xFF);
            blk_out[17] = static_cast<uint8_t>((sc_bf16.v >> 8) & 0xFF);

            float inv_scale = (scale == 0.0f) ? 0.0f : (1.0f / sc_bf16.to_float());
            for (int j = 0; j < 16; ++j) {
                int q0 = clamp_i(static_cast<int>(std::round(blk_in[2 * j] * inv_scale)), -8, 7);
                int q1 = clamp_i(static_cast<int>(std::round(blk_in[2 * j + 1] * inv_scale)), -8, 7);
                blk_out[j] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            }
        }
    }
    return packed;
}

int main() {
    std::cout << "=== Running CpuBackend Correctness & Benchmark Test ===\n";
    CpuBackend cpu;
    assert(cpu.type() == DeviceType::CPU);
    std::cout << "[ok] Device name: " << cpu.name() << "\n";

    const int N = 256;
    const int K = 512;
    std::vector<float> W(size_t(N) * K);
    std::vector<bf16> x(K);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < W.size(); ++i) W[i] = dist(rng);
    for (int i = 0; i < K; ++i) x[i] = bf16(dist(rng));

    auto packed = pack_reference_q4_0(W, N, K);
    WeightHandle wh = cpu.create_gemv_weight(N, K, packed.data(), packed.size());
    assert(wh != kInvalidWeight);
    std::cout << "[ok] Weight handle created: " << wh << "\n";

    std::vector<bf16> y(N);
    cpu.run_gemv(N, K, wh, x.data(), y.data());

    // Reference computation: dequantize on the fly and compute dot product
    float max_err = 0.0f;
    for (int r = 0; r < N; ++r) {
        float ref_dot = 0.0f;
        const uint8_t* row_bytes = packed.data() + static_cast<size_t>(r) * (K / 32) * 20;
        for (int bk = 0; bk < K / 32; ++bk) {
            const uint8_t* blk = row_bytes + bk * 20;
            bf16 sc;
            sc.v = static_cast<uint16_t>(blk[16]) | (static_cast<uint16_t>(blk[17]) << 8);
            float s = sc.to_float();
            for (int j = 0; j < 16; ++j) {
                int q0 = blk[j] & 0x0F; if (q0 >= 8) q0 -= 16;
                int q1 = (blk[j] >> 4) & 0x0F; if (q1 >= 8) q1 -= 16;
                ref_dot += (q0 * s) * x[bk * 32 + 2 * j].to_float();
                ref_dot += (q1 * s) * x[bk * 32 + 2 * j + 1].to_float();
            }
        }
        float err = std::abs(y[r].to_float() - ref_dot);
        max_err = std::max(max_err, err);
    }

    std::cout << "[ok] GEMV max parity difference vs dequant reference: " << max_err << "\n";
    assert(max_err < 1e-2f);

    // Test batched GEMM
    const int B = 4;
    std::vector<bf16> x_batch(size_t(B) * K);
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < K; ++i) x_batch[b * K + i] = bf16(x[i].to_float() * (b + 1));
    }
    std::vector<bf16> y_batch(size_t(B) * N);
    cpu.run_gemm(B, N, K, wh, x_batch.data(), y_batch.data());

    for (int b = 0; b < B; ++b) {
        for (int r = 0; r < N; ++r) {
            float expected = y[r].to_float() * (b + 1);
            float actual = y_batch[b * N + r].to_float();
            float diff = std::abs(actual - expected);
            assert(diff < 0.05f);
        }
    }
    std::cout << "[ok] GEMM batch parity verified across " << B << " rows.\n";

    // Benchmark loop
    const int IT = 500;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < IT; ++i) {
        cpu.run_gemv(N, K, wh, x.data(), y.data());
    }
    double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    double avg_us = (total_ms * 1000.0) / IT;
    double gmacs = (double(N) * K) / (avg_us * 1e-6) / 1e9;
    std::cout << "[BENCHMARK] CpuBackend GEMV (" << N << "x" << K << "): "
              << avg_us << " us/call (" << gmacs << " GMAC/s)\n";

    std::cout << "=== ALL CPU BACKEND TESTS PASSED ===\n";
    return 0;
}
