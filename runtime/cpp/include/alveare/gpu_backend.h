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

    double device_seconds() const override { return elapsed_seconds_; }
    double ffn_seconds() const override { return ffn_seconds_; }
    long device_calls() const override { return total_calls_; }
    void reset_profile() override { elapsed_seconds_ = 0.0; ffn_seconds_ = 0.0; total_calls_ = 0; }

    bool is_vulkan_active() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable double elapsed_seconds_ = 0.0;
    mutable double ffn_seconds_ = 0.0;
    mutable long total_calls_ = 0;
};

} // namespace alveare
