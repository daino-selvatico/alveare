#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "alveare/device.h"

namespace alveare {

// Gated gemv sub-profiler (ALVEARE_PROFILE_GEMV=1): splits a gemv call into activation
// upload, kernel dispatch+wait, and result download.
bool npu_gemv_prof_on();
void npu_gemv_prof_add(double up_ms, double run_ms, double down_ms);
void npu_gemv_prof_report();


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
class NpuRegistry : public ComputeDevice {
public:
    DeviceType type() const override { return DeviceType::NPU; }
    std::string name() const override { return "AMD Ryzen AI XDNA2 NPU (32 AIE Cores)"; }
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
    bool has_gemv(int N, int K) const override;
    bool has_gemm(int B, int N, int K) const override;
    bool has_ffn_fused(int H, int I, const std::string& activation) const override;

    // Upload packed weights of logical shape (N, K) -- laid out (N, K/32*20)
    // uint8, Q4_0 -- into a resident device BO once. The returned handle is
    // reused for every subsequent run_gemv of that weight (zero re-upload). This
    // loads the (N,K) gemv context if it is not already resident.
    WeightHandle create_gemv_weight(int N, int K, const void* packed,
                                    size_t nbytes) override;

    WeightHandle create_ffn_fused_weight(int H, int I, const std::string& activation,
                                         const void* packed, size_t nbytes) override;

    // y[N] = W @ x[K], all bf16 on the host boundary. w must come from
    // create_gemv_weight with the same (N, K). x_bf16 points at K bf16 values,
    // y_bf16 receives N bf16 values. Activation/output BOs are pinned and
    // reused across calls.
    void run_gemv(int N, int K, WeightHandle w, const void* x_bf16,
                  void* y_bf16) override;

    // Asynchronously pipelined multi-tile GEMV execution: uploads activation once,
    // enqueues all kernel runs to the NPU command queue, and syncs output in batch.
    void run_gemv_batch(int N, int K, const std::vector<WeightHandle>& weights,
                        const void* x_bf16, void* y_bf16_concat) override;

    // Asynchronously pipelined multi-tile GEMV with distinct input chunks per tile
    // (e.g. for K-chunked down projections): uploads input chunks, enqueues all runs,
    // and syncs output in batch.
    void run_gemv_multi_in_batch(int N, int K, const std::vector<WeightHandle>& weights,
                                 const std::vector<const void*>& x_ptrs, void* y_bf16_concat) override;

    // Batched matmul Y[B,N] = X[B,K] @ W^T, for prefill. w is a gemv weight of
    // the same (N, K) (the Q4_0 packing is shared); this runs the gemm kernel of
    // shape (B, N, K). x_bf16 points at B*K bf16 values, y_bf16 receives B*N.
    void run_gemm(int B, int N, int K, WeightHandle w, const void* x_bf16,
                  void* y_bf16) override;

    // Batched matmul like run_gemm, but the (N, K) Q4_0 weight is streamed from
    // host `packed` (nbytes) into a reused scratch device BO instead of coming
    // from a resident handle. For prefill of weights we keep host-resident (the
    // FFN gate/up/down, which stay device-resident only in fused form for decode).
    void run_gemm_streamed(int B, int N, int K, const void* packed, size_t nbytes,
                           const void* x_bf16, void* y_bf16) override;

    void run_ffn_fused(int H, int I, const std::string& activation, WeightHandle w,
                       const void* x_bf16, void* y_bf16) override;

    // Mark a shape's context as resident (pinned): never evicted. Call for the
    // decode working-set shapes so the token loop issues zero xclbin reloads.
    void pin_gemv(int N, int K) override;
    void pin_ffn_fused(int H, int I, const std::string& activation) override;

    // Number of hardware contexts currently loaded (for tests / diagnostics).
    int loaded_contexts() const;

    // Profiling: cumulative wall time (seconds) and count of NPU kernel launches
    // (run_gemv + run_ffn_fused). reset_profile() zeroes them.
    double npu_seconds() const;
    double ffn_seconds() const override;
    long npu_calls() const;
    double device_seconds() const override { return npu_seconds(); }
    long device_calls() const override { return npu_calls(); }
    void reset_profile() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace alveare
