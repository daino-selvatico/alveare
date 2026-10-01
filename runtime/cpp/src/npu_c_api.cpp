#include "alveare/device.h"
#include "alveare/npu.h"
#include "alveare/cpu_backend.h"
#include "alveare/gpu_backend.h"
#include <cstring>
#include <iostream>
#include <memory>

extern "C" {

void* alveare_device_create(const char* device_type, const char* manifest_path) {
    try {
        std::string dev_str = device_type ? device_type : "npu";
        for (char& c : dev_str) if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';

        if (dev_str == "cpu") {
            return static_cast<void*>(new alveare::CpuBackend());
        } else if (dev_str == "gpu" || dev_str == "vulkan") {
            return static_cast<void*>(new alveare::GpuBackend());
        } else {
            std::string mp = manifest_path ? manifest_path : "";
            return static_cast<void*>(new alveare::NpuRegistry(mp));
        }
    } catch (const std::exception& e) {
        std::cerr << "[Device C-API] Failed to initialize device (" << (device_type ? device_type : "null")
                  << "): " << e.what() << std::endl;
        return nullptr;
    }
}

void* alveare_npu_create_registry(const char* manifest_path) {
    return alveare_device_create("npu", manifest_path);
}

void alveare_device_free(void* device_ptr) {
    if (!device_ptr) return;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    delete dev;
}

void alveare_npu_free_registry(void* registry_ptr) {
    alveare_device_free(registry_ptr);
}

uint32_t alveare_device_create_gemv_weight(void* device_ptr, int N, int K, const void* packed_data, size_t nbytes) {
    if (!device_ptr) return alveare::kInvalidWeight;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    try {
        return dev->create_gemv_weight(N, K, packed_data, nbytes);
    } catch (const std::exception& e) {
        std::cerr << "[Device C-API] create_gemv_weight failed: " << e.what() << std::endl;
        return alveare::kInvalidWeight;
    }
}

uint32_t alveare_npu_create_gemv_weight(void* registry_ptr, int N, int K, const void* packed_data, size_t nbytes) {
    return alveare_device_create_gemv_weight(registry_ptr, N, K, packed_data, nbytes);
}

void alveare_device_run_gemv(void* device_ptr, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    if (!device_ptr) return;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    dev->run_gemv(N, K, weight_handle, x_bf16, y_bf16);
}

void alveare_npu_run_gemv(void* registry_ptr, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    alveare_device_run_gemv(registry_ptr, N, K, weight_handle, x_bf16, y_bf16);
}

void alveare_device_run_gemv_seq(void* device_ptr, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16, int n_tokens) {
    if (!device_ptr || n_tokens <= 0) return;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    const uint8_t* x_ptr = static_cast<const uint8_t*>(x_bf16);
    uint8_t* y_ptr = static_cast<uint8_t*>(y_bf16);
    const size_t in_stride = static_cast<size_t>(K) * 2;
    const size_t out_stride = static_cast<size_t>(N) * 2;

    for (int i = 0; i < n_tokens; ++i) {
        dev->run_gemv(N, K, weight_handle, x_ptr + i * in_stride, y_ptr + i * out_stride);
    }
}

void alveare_npu_run_gemv_seq(void* registry_ptr, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16, int n_tokens) {
    alveare_device_run_gemv_seq(registry_ptr, N, K, weight_handle, x_bf16, y_bf16, n_tokens);
}

void alveare_device_run_gemm(void* device_ptr, int B, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    if (!device_ptr) return;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    dev->run_gemm(B, N, K, weight_handle, x_bf16, y_bf16);
}

void alveare_npu_run_gemm(void* registry_ptr, int B, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    alveare_device_run_gemm(registry_ptr, B, N, K, weight_handle, x_bf16, y_bf16);
}

void alveare_device_run_gemm_streamed(void* device_ptr, int B, int N, int K, const void* packed_data, size_t nbytes, const void* x_bf16, void* y_bf16) {
    if (!device_ptr) return;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    dev->run_gemm_streamed(B, N, K, packed_data, nbytes, x_bf16, y_bf16);
}

void alveare_npu_run_gemm_streamed(void* registry_ptr, int B, int N, int K, const void* packed_data, size_t nbytes, const void* x_bf16, void* y_bf16) {
    alveare_device_run_gemm_streamed(registry_ptr, B, N, K, packed_data, nbytes, x_bf16, y_bf16);
}

int alveare_device_has_shape(void* device_ptr, int N, int K) {
    if (!device_ptr) return 0;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    return dev->has_gemv(N, K) ? 1 : 0;
}

int alveare_npu_has_shape(void* registry_ptr, int N, int K) {
    return alveare_device_has_shape(registry_ptr, N, K);
}

int alveare_device_has_ffn(void* device_ptr, int H, int I, const char* activation) {
    if (!device_ptr) return 0;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    return dev->has_ffn_fused(H, I, std::string(activation ? activation : "gelu")) ? 1 : 0;
}

int alveare_npu_has_ffn(void* registry_ptr, int H, int I, const char* activation) {
    return alveare_device_has_ffn(registry_ptr, H, I, activation);
}

uint32_t alveare_device_create_ffn_weight(void* device_ptr, int H, int I, const char* activation, const void* packed_data, size_t nbytes) {
    if (!device_ptr) return alveare::kInvalidWeight;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    try {
        return dev->create_ffn_fused_weight(H, I, std::string(activation ? activation : "gelu"), packed_data, nbytes);
    } catch (const std::exception& e) {
        std::cerr << "[Device C-API] create_ffn_fused_weight failed: " << e.what() << std::endl;
        return alveare::kInvalidWeight;
    }
}

uint32_t alveare_npu_create_ffn_weight(void* registry_ptr, int H, int I, const char* activation, const void* packed_data, size_t nbytes) {
    return alveare_device_create_ffn_weight(registry_ptr, H, I, activation, packed_data, nbytes);
}

void alveare_device_run_ffn_fused(void* device_ptr, int H, int I, const char* activation, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    if (!device_ptr) return;
    auto* dev = static_cast<alveare::ComputeDevice*>(device_ptr);
    dev->run_ffn_fused(H, I, std::string(activation ? activation : "gelu"), weight_handle, x_bf16, y_bf16);
}

void alveare_npu_run_ffn_fused(void* registry_ptr, int H, int I, const char* activation, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    alveare_device_run_ffn_fused(registry_ptr, H, I, activation, weight_handle, x_bf16, y_bf16);
}

} // extern "C"
