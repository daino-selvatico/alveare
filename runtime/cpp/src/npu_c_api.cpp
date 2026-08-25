#include "alveare/npu.h"
#include <cstring>
#include <iostream>
#include <memory>

extern "C" {

void* alveare_npu_create_registry(const char* manifest_path) {
    try {
        auto* reg = new alveare::NpuRegistry(std::string(manifest_path));
        return static_cast<void*>(reg);
    } catch (const std::exception& e) {
        std::cerr << "[NPU C-API] Failed to initialize registry: " << e.what() << std::endl;
        return nullptr;
    }
}

void alveare_npu_free_registry(void* registry_ptr) {
    if (!registry_ptr) return;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    delete reg;
}

uint32_t alveare_npu_create_gemv_weight(void* registry_ptr, int N, int K, const void* packed_data, size_t nbytes) {
    if (!registry_ptr) return alveare::kInvalidWeight;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    try {
        return reg->create_gemv_weight(N, K, packed_data, nbytes);
    } catch (const std::exception& e) {
        std::cerr << "[NPU C-API] create_gemv_weight failed: " << e.what() << std::endl;
        return alveare::kInvalidWeight;
    }
}

void alveare_npu_run_gemv(void* registry_ptr, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    if (!registry_ptr) return;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    reg->run_gemv(N, K, weight_handle, x_bf16, y_bf16);
}

void alveare_npu_run_gemv_seq(void* registry_ptr, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16, int n_tokens) {
    if (!registry_ptr || n_tokens <= 0) return;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    const uint8_t* x_ptr = static_cast<const uint8_t*>(x_bf16);
    uint8_t* y_ptr = static_cast<uint8_t*>(y_bf16);
    const size_t in_stride = static_cast<size_t>(K) * 2;
    const size_t out_stride = static_cast<size_t>(N) * 2;

    for (int i = 0; i < n_tokens; ++i) {
        reg->run_gemv(N, K, weight_handle, x_ptr + i * in_stride, y_ptr + i * out_stride);
    }
}

void alveare_npu_run_gemm(void* registry_ptr, int B, int N, int K, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    if (!registry_ptr) return;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    reg->run_gemm(B, N, K, weight_handle, x_bf16, y_bf16);
}

void alveare_npu_run_gemm_streamed(void* registry_ptr, int B, int N, int K, const void* packed_data, size_t nbytes, const void* x_bf16, void* y_bf16) {
    if (!registry_ptr) return;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    reg->run_gemm_streamed(B, N, K, packed_data, nbytes, x_bf16, y_bf16);
}

int alveare_npu_has_shape(void* registry_ptr, int N, int K) {
    if (!registry_ptr) return 0;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    return reg->has_gemv(N, K) ? 1 : 0;
}

int alveare_npu_has_ffn(void* registry_ptr, int H, int I, const char* activation) {
    if (!registry_ptr) return 0;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    return reg->has_ffn_fused(H, I, std::string(activation ? activation : "gelu")) ? 1 : 0;
}

uint32_t alveare_npu_create_ffn_weight(void* registry_ptr, int H, int I, const char* activation, const void* packed_data, size_t nbytes) {
    if (!registry_ptr) return alveare::kInvalidWeight;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    try {
        return reg->create_ffn_fused_weight(H, I, std::string(activation ? activation : "gelu"), packed_data, nbytes);
    } catch (const std::exception& e) {
        std::cerr << "[NPU C-API] create_ffn_fused_weight failed: " << e.what() << std::endl;
        return alveare::kInvalidWeight;
    }
}

void alveare_npu_run_ffn_fused(void* registry_ptr, int H, int I, const char* activation, uint32_t weight_handle, const void* x_bf16, void* y_bf16) {
    if (!registry_ptr) return;
    auto* reg = static_cast<alveare::NpuRegistry*>(registry_ptr);
    reg->run_ffn_fused(H, I, std::string(activation ? activation : "gelu"), weight_handle, x_bf16, y_bf16);
}

} // extern "C"
