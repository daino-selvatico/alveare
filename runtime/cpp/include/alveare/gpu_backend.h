#pragma once

#include "alveare/device.h"
#include "alveare/bf16.h"
#include <vector>
#include <memory>
#include <mutex>
#include <string>

namespace alveare {

// High-performance GPU compute backend targeting AMD Radeon 890M (RDNA 3.5)
// and Vulkan 1.4 compute pipelines.
class GpuBackend : public ComputeDevice {
public:
    GpuBackend();
    ~GpuBackend() override;

    DeviceType type() const override { return DeviceType::GPU; }
    std::string name() const override;

    bool has_gemv(int N, int K) const override { return true; }
    bool has_gemm(int B, int N, int K) const override { return true; }
    bool has_ffn_fused(int H, int I, const std::string& activation) const override { return true; }

    WeightHandle create_gemv_weight(int N, int K, const void* packed, size_t nbytes) override;
    WeightHandle create_ffn_fused_weight(int H, int I, const std::string& activation,
                                         const void* packed, size_t nbytes) override;

    void run_gemv(int N, int K, WeightHandle w, const void* x_bf16, void* y_bf16) override;
    void run_gemv_batch(int N, int K, const std::vector<WeightHandle>& weights,
                        const void* x_bf16, void* y_bf16_concat) override;
    void run_gemv_multi_in_batch(int N, int K, const std::vector<WeightHandle>& weights,
                                 const std::vector<const void*>& x_ptrs, void* y_bf16_concat) override;

    void run_gemm(int B, int N, int K, WeightHandle w, const void* x_bf16, void* y_bf16) override;
    void run_gemm_streamed(int B, int N, int K, const void* packed, size_t nbytes,
                           const void* x_bf16, void* y_bf16) override;

    void run_ffn_fused(int H, int I, const std::string& activation, WeightHandle w,
                       const void* x_bf16, void* y_bf16) override;

    void pin_gemv(int N, int K) override {}
    void pin_ffn_fused(int H, int I, const std::string& activation) override {}

    struct GpuStats {
        uint64_t gemv_gpu_calls = 0;
        uint64_t gemm_gpu_calls = 0;
        uint64_t fault_fallback_calls = 0;
        uint64_t intentional_cpu_ffn_calls = 0;
        std::string last_fallback_reason;
        double wall_seconds = 0.0;
        double gpu_fence_wait_seconds = 0.0;
        double copy_in_seconds = 0.0;
        double copy_out_seconds = 0.0;
    };

    double device_seconds() const override { return elapsed_seconds_; }
    double ffn_seconds() const override { return ffn_seconds_; }
    long device_calls() const override { return total_calls_; }
    void reset_profile() override;

    bool is_vulkan_active() const;
    GpuStats stats() const;
    uint64_t fallback_count() const;
    std::string last_fallback_reason() const;

    // Subgroup control inspection & test configuration
    bool supports_subgroup_size_control() const;
    uint32_t default_subgroup_size() const;
    bool enable_forced_subgroup32(bool enable);
    bool is_forced_subgroup32() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable double elapsed_seconds_ = 0.0;
    mutable double ffn_seconds_ = 0.0;
    mutable long total_calls_ = 0;
};

} // namespace alveare
