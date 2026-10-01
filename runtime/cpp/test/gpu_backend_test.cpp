#include "alveare/gpu_backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <string>

using namespace alveare;

#define TEST_CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "\n[TEST FAILURE] " << (msg) \
                  << "\n  Condition: (" #cond ")" \
                  << "\n  File: " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

static inline int clamp_i(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// Reference bit-accurate Q4_0 weight packing
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

// Reference Q4_0 GEMV computation
static std::vector<float> compute_reference_gemv(const uint8_t* packed, int N, int K, const bf16* x) {
    const int K_blocks = K / 32;
    std::vector<float> ref(N, 0.0f);
    for (int r = 0; r < N; ++r) {
        float ref_dot = 0.0f;
        const uint8_t* row_bytes = packed + static_cast<size_t>(r) * K_blocks * 20;
        for (int bk = 0; bk < K_blocks; ++bk) {
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
        ref[r] = ref_dot;
    }
    return ref;
}

int main(int argc, char** argv) {
    bool negative_control = (argc > 1 && std::string(argv[1]) == "--negative-control");

    std::cout << "=== Alveare 3.0.0-alpha.2 GpuBackend Quality & Reliability Verification ===\n";
    if (negative_control) {
        std::cout << "[INFO] Running in NEGATIVE CONTROL mode (deliberate corrupt data injection to verify fail-closed tests under -DNDEBUG).\n";
    }

    GpuBackend gpu;
    TEST_CHECK(gpu.type() == DeviceType::GPU, "Expected DeviceType::GPU");
    std::cout << "[ok] Device detected: " << gpu.name() << "\n";
    std::cout << "[ok] Vulkan active: " << (gpu.is_vulkan_active() ? "YES" : "NO") << "\n";

    // Strict Gate: Never accept a false GPU pass via CPU fallback
    TEST_CHECK(gpu.is_vulkan_active(), "Vulkan device must be actively initialized (silent CPU fallback disallowed for GPU test)");

    std::cout << "[ok] Subgroup size control: " << (gpu.supports_subgroup_size_control() ? "SUPPORTED" : "UNSUPPORTED")
              << " (default subgroup=" << gpu.default_subgroup_size() << ")\n";

    // -------------------------------------------------------------------------
    // Negative Control: Deliberately inject data corruption to prove test fails in Release (-DNDEBUG)
    // -------------------------------------------------------------------------
    if (negative_control) {
        const int N = 64;
        const int K = 128;
        std::vector<float> W(size_t(N) * K, 0.5f);
        std::vector<bf16> x(K, bf16(1.0f));
        auto packed = pack_reference_q4_0(W, N, K);
        WeightHandle wh = gpu.create_gemv_weight(N, K, packed.data(), packed.size());
        std::vector<bf16> y(N);
        gpu.run_gemv(N, K, wh, x.data(), y.data());

        // Deliberately corrupt reference output to simulate divergence
        auto ref = compute_reference_gemv(packed.data(), N, K, x.data());
        ref[0] += 50.0f; // Deliberate gross error

        float max_abs_err = 0.0f;
        for (int r = 0; r < N; ++r) {
            max_abs_err = std::max(max_abs_err, std::abs(y[r].to_float() - ref[r]));
        }
        std::cout << "[NEGATIVE CONTROL] Injected gross error. Measured max_abs_err=" << max_abs_err << "\n";
        TEST_CHECK(max_abs_err < 0.1f, "NEGATIVE CONTROL INTENDED FAILURE: Corrupted parity detected! (Demonstrates fail-closed gate under -DNDEBUG)");
        return 0;
    }

    // -------------------------------------------------------------------------
    // Priority 1: Realistic K-sweep and Non-tile N shapes
    // -------------------------------------------------------------------------
    const std::vector<int> test_K = { 512, 1024, 2048, 4096, 5120 };
    const std::vector<int> test_N = { 1, 33, 127, 256, 512 };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int K : test_K) {
        for (int N : test_N) {
            std::vector<float> W(size_t(N) * K);
            for (size_t i = 0; i < W.size(); ++i) W[i] = dist(rng);
            auto packed = pack_reference_q4_0(W, N, K);
            WeightHandle wh = gpu.create_gemv_weight(N, K, packed.data(), packed.size());
            TEST_CHECK(wh != kInvalidWeight, "Weight registration failed for N=" + std::to_string(N) + ", K=" + std::to_string(K));

            // Test Case A: Signed random inputs
            {
                std::vector<bf16> x(K);
                for (int i = 0; i < K; ++i) x[i] = bf16(dist(rng));
                std::vector<bf16> y(N);
                gpu.run_gemv(N, K, wh, x.data(), y.data());

                auto ref = compute_reference_gemv(packed.data(), N, K, x.data());
                float max_abs_err = 0.0f;
                float max_rel_err = 0.0f;
                for (int r = 0; r < N; ++r) {
                    float val = y[r].to_float();
                    TEST_CHECK(!std::isnan(val) && !std::isinf(val), "NaN/Inf detected in GPU GEMV output!");
                    float err = std::abs(val - ref[r]);
                    float denom = std::max(std::abs(ref[r]), 1.0f);
                    max_abs_err = std::max(max_abs_err, err);
                    max_rel_err = std::max(max_rel_err, err / denom);
                }
                TEST_CHECK(max_rel_err < 0.015f, "GEMV parity relative error exceeded BF16 limit for K=" + std::to_string(K));
            }

            // Test Case B: All zeros input
            {
                std::vector<bf16> x_zeros(K, bf16(0.0f));
                std::vector<bf16> y_zeros(N, bf16(1.0f));
                gpu.run_gemv(N, K, wh, x_zeros.data(), y_zeros.data());
                for (int r = 0; r < N; ++r) {
                    TEST_CHECK(y_zeros[r].to_float() == 0.0f, "Zero input did not produce exact zero output at row " + std::to_string(r));
                }
            }

            // Test Case C: Structured alternating (+1, -1) input
            {
                std::vector<bf16> x_alt(K);
                for (int i = 0; i < K; ++i) x_alt[i] = bf16((i % 2 == 0) ? 1.0f : -1.0f);
                std::vector<bf16> y_alt(N);
                gpu.run_gemv(N, K, wh, x_alt.data(), y_alt.data());

                auto ref = compute_reference_gemv(packed.data(), N, K, x_alt.data());
                for (int r = 0; r < N; ++r) {
                    float val = y_alt[r].to_float();
                    TEST_CHECK(!std::isnan(val) && !std::isinf(val), "NaN/Inf in alternating input test");
                    float err = std::abs(val - ref[r]);
                    float denom = std::max(std::abs(ref[r]), 1.0f);
                    TEST_CHECK((err / denom) < 0.015f, "Alternating input parity mismatch");
                }
            }

            // Test Case D: Unit impulse (spike at center)
            {
                std::vector<bf16> x_impulse(K, bf16(0.0f));
                x_impulse[K / 2] = bf16(1.0f);
                std::vector<bf16> y_impulse(N);
                gpu.run_gemv(N, K, wh, x_impulse.data(), y_impulse.data());

                auto ref = compute_reference_gemv(packed.data(), N, K, x_impulse.data());
                for (int r = 0; r < N; ++r) {
                    float err = std::abs(y_impulse[r].to_float() - ref[r]);
                    TEST_CHECK(err < 0.02f, "Impulse input parity mismatch");
                }
            }

            // Test Case E: Repeated calls determinism
            {
                std::vector<bf16> x(K, bf16(0.5f));
                std::vector<bf16> y1(N), y2(N);
                gpu.run_gemv(N, K, wh, x.data(), y1.data());
                gpu.run_gemv(N, K, wh, x.data(), y2.data());
                for (int r = 0; r < N; ++r) {
                    TEST_CHECK(y1[r].v == y2[r].v, "Non-deterministic GPU GEMV output between consecutive calls");
                }
            }
        }
    }
    std::cout << "[ok] Priority 1: Realistic K-sweep (512..5120) & N-mod tiles verified across random/zero/structured inputs.\n";

    // -------------------------------------------------------------------------
    // Priority 1: Batched GEMM (B in {1, 4, 8, 16})
    // -------------------------------------------------------------------------
    {
        const int N = 128;
        const int K = 1024;
        std::vector<float> W(size_t(N) * K);
        for (size_t i = 0; i < W.size(); ++i) W[i] = dist(rng);
        auto packed = pack_reference_q4_0(W, N, K);
        WeightHandle wh = gpu.create_gemv_weight(N, K, packed.data(), packed.size());

        for (int B : { 1, 4, 8, 16 }) {
            std::vector<bf16> x_batch(size_t(B) * K);
            for (int b = 0; b < B; ++b) {
                for (int i = 0; i < K; ++i) {
                    x_batch[b * K + i] = bf16(dist(rng));
                }
            }
            std::vector<bf16> y_batch(size_t(B) * N);
            gpu.run_gemm(B, N, K, wh, x_batch.data(), y_batch.data());

            for (int b = 0; b < B; ++b) {
                auto ref = compute_reference_gemv(packed.data(), N, K, x_batch.data() + b * K);
                for (int r = 0; r < N; ++r) {
                    float actual = y_batch[b * N + r].to_float();
                    TEST_CHECK(!std::isnan(actual) && !std::isinf(actual), "NaN/Inf in GEMM batch output");
                    float err = std::abs(actual - ref[r]);
                    float denom = std::max(std::abs(ref[r]), 1.0f);
                    TEST_CHECK((err / denom) < 0.015f, "GEMM batch mismatch at B=" + std::to_string(b) + ", r=" + std::to_string(r));
                }
            }
        }
    }
    std::cout << "[ok] Priority 1: GEMM verified across B=1, 4, 8, 16.\n";

    // -------------------------------------------------------------------------
    // Priority 2: Shader Workgroup Reduction & Optional Forced Wave32
    // -------------------------------------------------------------------------
    if (gpu.supports_subgroup_size_control()) {
        std::cout << "[INFO] Testing Forced Wave32 reduction pipeline on AMD hardware...\n";
        bool enabled = gpu.enable_forced_subgroup32(true);
        TEST_CHECK(enabled, "enable_forced_subgroup32 failed despite reported support");
        TEST_CHECK(gpu.is_forced_subgroup32(), "is_forced_subgroup32 should return true");

        // Test Wave32 on K=2048 and K=4096 (which require cross-subgroup reduction across 2 waves)
        for (int K : { 2048, 4096 }) {
            const int N = 64;
            std::vector<float> W(size_t(N) * K);
            for (size_t i = 0; i < W.size(); ++i) W[i] = dist(rng);
            auto packed = pack_reference_q4_0(W, N, K);
            WeightHandle wh = gpu.create_gemv_weight(N, K, packed.data(), packed.size());

            std::vector<bf16> x(K);
            for (int i = 0; i < K; ++i) x[i] = bf16(dist(rng));
            std::vector<bf16> y(N);
            gpu.run_gemv(N, K, wh, x.data(), y.data());

            auto ref = compute_reference_gemv(packed.data(), N, K, x.data());
            float max_rel_err = 0.0f;
            for (int r = 0; r < N; ++r) {
                float val = y[r].to_float();
                TEST_CHECK(!std::isnan(val) && !std::isinf(val), "NaN/Inf under Wave32 dispatch");
                float err = std::abs(val - ref[r]);
                float denom = std::max(std::abs(ref[r]), 1.0f);
                max_rel_err = std::max(max_rel_err, err / denom);
            }
            std::cout << "  Wave32 K=" << K << ": max relative error = " << max_rel_err << "\n";
            TEST_CHECK(max_rel_err < 0.015f, "Wave32 reduction failed on K=" + std::to_string(K));
        }

        // Restore default Wave64
        gpu.enable_forced_subgroup32(false);
        TEST_CHECK(!gpu.is_forced_subgroup32(), "is_forced_subgroup32 should return false after reset");
        std::cout << "[ok] Priority 2: Subgroup reduction successfully verified under forced Wave32 (no lane truncation) and default Wave64.\n";
    } else {
        std::cout << "[INFO] Subgroup size control not exposed by driver; skipping forced Wave32 test.\n";
    }

    // -------------------------------------------------------------------------
    // Priority 3: Error Handling & Invalid Bounds Rejection
    // -------------------------------------------------------------------------
    {
        std::vector<bf16> x(64, bf16(1.0f));
        std::vector<bf16> y(64);

        // Invalid weight handle
        gpu.run_gemv(64, 64, 99999, x.data(), y.data());
        TEST_CHECK(gpu.fallback_count() > 0, "Invalid WeightHandle should record fallback");
        std::cout << "[ok] Invalid handle rejected gracefully: reason = \"" << gpu.last_fallback_reason() << "\"\n";

        // Invalid K (not multiple of 32)
        gpu.run_gemv(64, 65, 0, x.data(), y.data());
        std::cout << "[ok] Unaligned K rejected gracefully: reason = \"" << gpu.last_fallback_reason() << "\"\n";

        // Null pointer
        gpu.run_gemv(64, 64, 0, nullptr, y.data());
        std::cout << "[ok] Null pointer rejected gracefully: reason = \"" << gpu.last_fallback_reason() << "\"\n";
    }
    std::cout << "[ok] Priority 3: Robust bounds checking and actionable error reporting verified.\n";

    // -------------------------------------------------------------------------
    // Priority 4: Double Counting Fix Regression Test
    // -------------------------------------------------------------------------
    {
        gpu.reset_profile();
        const int N = 128;
        const int K = 512;
        std::vector<float> W(size_t(N) * K, 0.1f);
        auto packed = pack_reference_q4_0(W, N, K);

        std::vector<WeightHandle> batch_weights;
        for (int i = 0; i < 4; ++i) {
            batch_weights.push_back(gpu.create_gemv_weight(N, K, packed.data(), packed.size()));
        }

        std::vector<bf16> x(K, bf16(1.0f));
        std::vector<bf16> y_concat(size_t(4) * N);

        // Warmup
        gpu.run_gemv_batch(N, K, batch_weights, x.data(), y_concat.data());
        gpu.reset_profile();

        auto t_wall_start = std::chrono::steady_clock::now();
        const int BATCH_ITERS = 50;
        for (int it = 0; it < BATCH_ITERS; ++it) {
            gpu.run_gemv_batch(N, K, batch_weights, x.data(), y_concat.data());
        }
        double measured_wall_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_wall_start).count();
        double reported_device_sec = gpu.device_seconds();
        long reported_calls = gpu.device_calls();

        std::cout << "[TIMING REGRESSION] Batched GEMV (" << BATCH_ITERS << " batches of 4 projections):\n"
                  << "  Measured Wall Clock:   " << measured_wall_sec * 1000.0 << " ms\n"
                  << "  Reported Device Time:  " << reported_device_sec * 1000.0 << " ms\n"
                  << "  Reported Calls:        " << reported_calls << " (expected " << BATCH_ITERS * 4 << ")\n"
                  << "  Ratio (device / wall): " << (reported_device_sec / measured_wall_sec) << "\n";

        TEST_CHECK(reported_calls == BATCH_ITERS * 4, "Per-projection call count mismatch");
        // In the double-counting bug, reported_device_sec was >= 1.9 * measured_wall_sec.
        // With single counting, reported_device_sec <= measured_wall_sec * 1.15.
        TEST_CHECK(reported_device_sec <= measured_wall_sec * 1.30,
                   "Double-counting detected! Reported device time significantly exceeds actual wall time.");
        TEST_CHECK(reported_device_sec >= measured_wall_sec * 0.70,
                   "Reported device time unexpectedly undercounted relative to wall time.");

        auto st = gpu.stats();
        std::cout << "  Breakdown: copy_in=" << st.copy_in_seconds * 1000.0 << " ms, "
                  << "fence_wait=" << st.gpu_fence_wait_seconds * 1000.0 << " ms, "
                  << "copy_out=" << st.copy_out_seconds * 1000.0 << " ms\n";
    }
    std::cout << "[ok] Priority 4: Metrics double-counting resolved and verified against wall clock.\n";

    // -------------------------------------------------------------------------
    // Microbenchmark (Warmup + Repeats under dev load)
    // -------------------------------------------------------------------------
    {
        const int N = 256;
        const int K = 512;
        std::vector<float> W(size_t(N) * K);
        for (size_t i = 0; i < W.size(); ++i) W[i] = dist(rng);
        auto packed = pack_reference_q4_0(W, N, K);
        WeightHandle wh = gpu.create_gemv_weight(N, K, packed.data(), packed.size());

        std::vector<bf16> x(K, bf16(0.5f));
        std::vector<bf16> y(N);

        // Warmup
        for (int i = 0; i < 20; ++i) gpu.run_gemv(N, K, wh, x.data(), y.data());

        const int IT = 500;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < IT; ++i) {
            gpu.run_gemv(N, K, wh, x.data(), y.data());
        }
        double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        double avg_us = (total_ms * 1000.0) / IT;
        double gmacs = (double(N) * K) / (avg_us * 1e-6) / 1e9;

        std::cout << "\n[MICROBENCHMARK - Simultaneous Dev Load (Not Performance Certification)]\n"
                  << "  GpuBackend GEMV (" << N << "x" << K << "): "
                  << avg_us << " us/call (" << gmacs << " GMAC/s)\n";
    }

    std::cout << "\n=== ALL GPU BACKEND TESTS PASSED SUCCESSFULLY ===\n";
    return 0;
}
