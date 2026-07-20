#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace alveare {

// One AOT-compiled kernel as harvested by tools/build_kernels.py, mirrored from
// kernels/build/manifest.json. A gemv entry has B == 0; a gemm entry has B > 0.
struct KernelSpec {
    std::string kind;   // "gemv" (decode), "gemm" (prefill), "ffn_fused"
    int N = 0;
    int K = 0;
    int B = 0;          // batch (gemm only)
    int H = 0;          // ffn_fused
    int I = 0;          // ffn_fused
    std::string activation; // ffn_fused
    int m = 0;
    int k_tile = 0;
    int n_cores = 0;
    std::string xclbin; // filename, relative to the manifest directory
    std::string insts;  // filename, relative to the manifest directory
};

// Opaque handle to a resident device weight buffer owned by the registry.
using WeightHandle = uint32_t;
inline constexpr WeightHandle kInvalidWeight = 0xFFFFFFFFu;

// Native-XRT kernel registry for the decode/prefill matmuls.
//
// Loads the AOT manifest, and on first use of a shape registers its xclbin as a
// hardware context and caches the kernel + instruction BO. XDNA2 allows only a
// bounded number of concurrent hardware contexts, so the registry keeps a
// resident set (never evicted inside the decode loop) and, if the manifest has
// more shapes than the context budget, evicts the least-recently-used
// non-resident context on the XRT "out of contexts" (errno 22) failure and
// retries (plan decision #2).
//
// XRT types are kept out of this header (pimpl) so model/weights code need not
// depend on XRT.
class NpuRegistry {
public:
    // manifest_path points at kernels/build/manifest.json. max_contexts is the
    // hardware context budget (XDNA2 ~= 8).
    explicit NpuRegistry(const std::string& manifest_path,
                         unsigned device_index = 0,
                         int max_contexts = 8);
    ~NpuRegistry();

    NpuRegistry(const NpuRegistry&) = delete;
    NpuRegistry& operator=(const NpuRegistry&) = delete;

    const std::vector<KernelSpec>& kernels() const;
    const std::string& model_type() const;

    // True if the manifest contains a gemv/gemm/ffn_fused kernel of the given shape.
    bool has_gemv(int N, int K) const;
    bool has_gemm(int B, int N, int K) const;
    bool has_ffn_fused(int H, int I, const std::string& activation) const;

    // Upload packed weights of logical shape (N, K) -- laid out (N, K/32*20)
    // uint8, Q4_0 -- into a resident device BO once. The returned handle is
    // reused for every subsequent run_gemv of that weight (zero re-upload). This
    // loads the (N,K) gemv context if it is not already resident.
    WeightHandle create_gemv_weight(int N, int K, const void* packed,
                                    size_t nbytes);

    WeightHandle create_ffn_fused_weight(int H, int I, const std::string& activation,
                                         const void* packed, size_t nbytes);

    // y[N] = W @ x[K], all bf16 on the host boundary. w must come from
    // create_gemv_weight with the same (N, K). x_bf16 points at K bf16 values,
    // y_bf16 receives N bf16 values. Activation/output BOs are pinned and
    // reused across calls.
    void run_gemv(int N, int K, WeightHandle w, const void* x_bf16,
                  void* y_bf16);

    void run_ffn_fused(int H, int I, const std::string& activation, WeightHandle w,
                       const void* x_bf16, void* y_bf16);

    // Mark a shape's context as resident (pinned): never evicted. Call for the
    // decode working-set shapes so the token loop issues zero xclbin reloads.
    void pin_gemv(int N, int K);
    void pin_ffn_fused(int H, int I, const std::string& activation);

    // Number of hardware contexts currently loaded (for tests / diagnostics).
    int loaded_contexts() const;

    // Profiling: cumulative wall time (seconds) and count of NPU kernel launches
    // (run_gemv + run_ffn_fused). reset_profile() zeroes them.
    double npu_seconds() const;
    double ffn_seconds() const;
    long npu_calls() const;
    void reset_profile();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace alveare
