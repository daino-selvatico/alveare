#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "alveare/bf16.h"

namespace alveare {

enum class DeviceType {
    NPU,
    CPU,
    GPU
};

inline const char* device_type_to_string(DeviceType t) {
    switch (t) {
        case DeviceType::NPU: return "npu";
        case DeviceType::CPU: return "cpu";
        case DeviceType::GPU: return "gpu";
        default: return "unknown";
    }
}

inline DeviceType string_to_device_type(const std::string& s) {
    std::string lower = s;
    for (char& c : lower) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
    if (lower == "cpu") return DeviceType::CPU;
    if (lower == "gpu" || lower == "vulkan" || lower == "rocm") return DeviceType::GPU;
    return DeviceType::NPU;
}

using WeightHandle = uint32_t;
inline constexpr WeightHandle kInvalidWeight = 0xFFFFFFFFu;

// Unified abstract compute device interface.
// Decouples model execution logic from hardware-specific implementations (XDNA2 NPU, x86 CPU, Vulkan GPU).
class ComputeDevice {
public:
    virtual ~ComputeDevice() = default;

    virtual DeviceType type() const = 0;
    virtual std::string name() const = 0;

    // Capability queries
    virtual bool has_gemv(int N, int K) const = 0;
    virtual bool has_gemm(int B, int N, int K) const = 0;
    virtual bool has_ffn_fused(int H, int I, const std::string& activation) const = 0;

    // Weight buffer registration
    virtual WeightHandle create_gemv_weight(int N, int K, const void* packed, size_t nbytes) = 0;
    virtual WeightHandle create_ffn_fused_weight(int H, int I, const std::string& activation,
                                                 const void* packed, size_t nbytes) = 0;

    // Decode GEMV execution: y[N] = W[N, K] @ x[K]
    virtual void run_gemv(int N, int K, WeightHandle w, const void* x_bf16, void* y_bf16) = 0;

    // Multi-tile pipelined GEMV execution (concatenated output)
    virtual void run_gemv_batch(int N, int K, const std::vector<WeightHandle>& weights,
                                const void* x_bf16, void* y_bf16_concat) = 0;

    // Multi-tile pipelined GEMV execution with distinct input chunks
    virtual void run_gemv_multi_in_batch(int N, int K, const std::vector<WeightHandle>& weights,
                                         const std::vector<const void*>& x_ptrs, void* y_bf16_concat) = 0;

    // Prefill GEMM execution: Y[B, N] = X[B, K] @ W^T
    virtual void run_gemm(int B, int N, int K, WeightHandle w, const void* x_bf16, void* y_bf16) = 0;

    // Streamed batched GEMM execution (for weights streamed from host memory)
    virtual void run_gemm_streamed(int B, int N, int K, const void* packed, size_t nbytes,
                                   const void* x_bf16, void* y_bf16) = 0;

    // Fused Feed-Forward Network: Gate/Up GEMV + GeGLU/SiLU + Down GEMV
    virtual void run_ffn_fused(int H, int I, const std::string& activation, WeightHandle w,
                               const void* x_bf16, void* y_bf16) = 0;

    // Context pinning (hints for resident working sets)
    virtual void pin_gemv(int N, int K) = 0;
    virtual void pin_ffn_fused(int H, int I, const std::string& activation) = 0;

    // Telemetry & profiling
    virtual double device_seconds() const { return 0.0; }
    virtual double ffn_seconds() const { return 0.0; }
    virtual long device_calls() const { return 0; }
    virtual void reset_profile() {}

    // Backward compatibility aliases
    double npu_seconds() const { return device_seconds(); }
    long npu_calls() const { return device_calls(); }
};

} // namespace alveare
