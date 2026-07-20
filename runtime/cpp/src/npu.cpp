#include "alveare/npu.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

#include "nlohmann/json.hpp"

#include "alveare/bf16.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

namespace alveare {

// Lightweight profiling counters (single decode thread): wall time and call
// count spent inside NPU kernel launches. Read via NpuRegistry::npu_seconds().
namespace prof {
double npu_seconds = 0.0;
double ffn_seconds = 0.0;
long npu_calls = 0;
struct ScopedTimer {
    double* target;
    std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
    explicit ScopedTimer(double* t) : target(t) {}
    ~ScopedTimer() {
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        npu_seconds += dt;
        if (target) *target += dt;
        ++npu_calls;
    }
};
}  // namespace prof

double NpuRegistry::npu_seconds() const { return prof::npu_seconds; }
double NpuRegistry::ffn_seconds() const { return prof::ffn_seconds; }
long NpuRegistry::npu_calls() const { return prof::npu_calls; }
void NpuRegistry::reset_profile() { prof::npu_seconds = 0.0; prof::ffn_seconds = 0.0; prof::npu_calls = 0; }

namespace {

std::string shape_key(const std::string& kind, int N, int K, int B, int H = 0, int I = 0, const std::string& act = "") {
    return kind + ':' + std::to_string(N) + ':' + std::to_string(K) + ':' +
           std::to_string(B) + ':' + std::to_string(H) + ':' + std::to_string(I) + ':' + act;
}

std::string dirname_of(const std::string& path) {
    auto p = path.find_last_of('/');
    return p == std::string::npos ? std::string(".") : path.substr(0, p);
}

// One AOT xclbin resolved to a live hardware context + cached kernel/instr BO,
// plus its pinned activation/output BOs.
struct LoadedKernel {
    KernelSpec spec;
    xrt::xclbin xclbin;
    xrt::hw_context ctx;
    xrt::kernel kernel;
    xrt::bo instr;
    uint32_t ninstr = 0;
    xrt::bo x_bo;   // pinned bf16 activation buffer
    xrt::bo y_bo;   // pinned bf16 output buffer
    bool pinned = false;
    uint64_t last_used = 0;
};

struct ResidentWeight {
    int N = 0;
    int K = 0;
    xrt::bo bo;     // resident packed-weight BO, synced once
};

} // namespace

struct NpuRegistry::Impl {
    xrt::device device;
    std::string manifest_dir;
    std::string model_type;
    std::string kernel_name = "MLIR_AIE";
    uint32_t opcode = 3;
    int max_contexts = 8;
    uint64_t tick = 0;

    std::vector<KernelSpec> specs;
    std::map<std::string, LoadedKernel> loaded;
    std::vector<ResidentWeight> weights;

    const KernelSpec* find_spec(const std::string& kind, int N, int K,
                                int B, int H = 0, int I = 0, const std::string& act = "") const {
        for (const auto& s : specs)
            if (s.kind == kind && s.N == N && s.K == K && s.B == B &&
                s.H == H && s.I == I && s.activation == act) return &s;
        return nullptr;
    }

    std::vector<uint32_t> load_instrs(const std::string& fname) {
        std::string path = manifest_dir + '/' + fname;
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("npu: cannot open insts " + path);
        std::streamsize sz = f.tellg();
        f.seekg(0);
        std::vector<uint32_t> buf(static_cast<size_t>(sz) / sizeof(uint32_t));
        if (!f.read(reinterpret_cast<char*>(buf.data()), sz))
            throw std::runtime_error("npu: failed reading insts " + path);
        return buf;
    }

    // Free the least-recently-used non-pinned context. Throws if none exists.
    void evict_one_lru() {
        auto victim = loaded.end();
        for (auto it = loaded.begin(); it != loaded.end(); ++it) {
            if (it->second.pinned) continue;
            if (victim == loaded.end() ||
                it->second.last_used < victim->second.last_used)
                victim = it;
        }
        if (victim == loaded.end())
            throw std::runtime_error(
                "npu: all loaded contexts are pinned; cannot free a slot");
        loaded.erase(victim);
    }

    LoadedKernel& ensure_loaded(const std::string& kind, int N, int K, int B, int H = 0, int I = 0, const std::string& act = "") {
        const std::string key = shape_key(kind, N, K, B, H, I, act);
        auto it = loaded.find(key);
        if (it != loaded.end()) {
            it->second.last_used = ++tick;
            return it->second;
        }

        const KernelSpec* spec = find_spec(kind, N, K, B, H, I, act);
        if (!spec) throw std::runtime_error("npu: no kernel for " + key);

        // Load, evicting on context exhaustion (XRT errno 22) and retrying.
        for (int guard = 0; guard <= max_contexts; ++guard) {
            if (static_cast<int>(loaded.size()) >= max_contexts) evict_one_lru();
            try {
                LoadedKernel lk;
                lk.spec = *spec;
                lk.xclbin = xrt::xclbin(manifest_dir + '/' + spec->xclbin);
                device.register_xclbin(lk.xclbin);
                lk.ctx = xrt::hw_context(device, lk.xclbin.get_uuid());
                lk.kernel = xrt::kernel(lk.ctx, kernel_name);

                const auto instrs = load_instrs(spec->insts);
                lk.ninstr = static_cast<uint32_t>(instrs.size());
                lk.instr = xrt::bo(device, instrs.size() * sizeof(uint32_t),
                                   XCL_BO_FLAGS_CACHEABLE, lk.kernel.group_id(1));
                std::memcpy(lk.instr.map<void*>(), instrs.data(),
                            instrs.size() * sizeof(uint32_t));
                lk.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                size_t x_bytes = 0, y_bytes = 0;
                if (kind == "ffn_fused") {
                    x_bytes = size_t(H) * sizeof(uint16_t);
                    y_bytes = size_t(spec->n_cores) * size_t(H) * sizeof(uint16_t);
                } else {
                    const size_t rows = (kind == "gemm") ? size_t(B) : 1;
                    x_bytes = rows * size_t(K) * sizeof(uint16_t);
                    y_bytes = rows * size_t(N) * sizeof(uint16_t);
                }
                lk.x_bo = xrt::bo(device, x_bytes, XRT_BO_FLAGS_HOST_ONLY,
                                  lk.kernel.group_id(4));
                lk.y_bo = xrt::bo(device, y_bytes, XRT_BO_FLAGS_HOST_ONLY,
                                  lk.kernel.group_id(5));
                lk.last_used = ++tick;

                auto res = loaded.emplace(key, std::move(lk));
                return res.first->second;
            } catch (const std::exception&) {
                // If nothing is loaded, this is not a capacity failure -> rethrow.
                if (loaded.empty() || guard == max_contexts) throw;
                evict_one_lru();
            }
        }
        throw std::runtime_error("npu: failed to load context for " + key);
    }
};

NpuRegistry::NpuRegistry(const std::string& manifest_path, unsigned device_index,
                         int max_contexts)
    : impl_(new Impl) {
    impl_->device = xrt::device(device_index);
    impl_->manifest_dir = dirname_of(manifest_path);
    impl_->max_contexts = max_contexts;

    std::ifstream f(manifest_path);
    if (!f) throw std::runtime_error("npu: cannot open manifest " + manifest_path);
    nlohmann::json j;
    f >> j;

    impl_->model_type = j.value("model_type", std::string{});
    impl_->kernel_name = j.value("kernel_name", std::string{"MLIR_AIE"});
    impl_->opcode = j.value("opcode", 3u);

    for (const auto& e : j.at("kernels")) {
        KernelSpec s;
        s.kind = e.value("kind", std::string{});
        s.N = e.value("N", 0);
        s.K = e.value("K", 0);
        s.B = e.value("B", 0);
        s.H = e.value("H", 0);
        s.I = e.value("I", 0);
        s.activation = e.value("activation", std::string{});
        s.m = e.value("m", 0);
        s.k_tile = e.value("k_tile", 0);
        s.n_cores = e.value("n_cores", 0);
        s.xclbin = e.value("xclbin", std::string{});
        s.insts = e.value("insts", std::string{});
        impl_->specs.push_back(std::move(s));
    }
}

NpuRegistry::~NpuRegistry() = default;

const std::vector<KernelSpec>& NpuRegistry::kernels() const {
    return impl_->specs;
}

const std::string& NpuRegistry::model_type() const { return impl_->model_type; }

bool NpuRegistry::has_gemv(int N, int K) const {
    return impl_->find_spec("gemv", N, K, 0) != nullptr;
}

bool NpuRegistry::has_gemm(int B, int N, int K) const {
    return impl_->find_spec("gemm", N, K, B) != nullptr;
}

bool NpuRegistry::has_ffn_fused(int H, int I, const std::string& activation) const {
    return impl_->find_spec("ffn_fused", 0, 0, 0, H, I, activation) != nullptr;
}

WeightHandle NpuRegistry::create_gemv_weight(int N, int K, const void* packed,
                                             size_t nbytes) {
    LoadedKernel& lk = impl_->ensure_loaded("gemv", N, K, 0);
    ResidentWeight rw;
    rw.N = N;
    rw.K = K;
    rw.bo = xrt::bo(impl_->device, nbytes, XRT_BO_FLAGS_HOST_ONLY,
                    lk.kernel.group_id(3));
    std::memcpy(rw.bo.map<void*>(), packed, nbytes);
    rw.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    impl_->weights.push_back(std::move(rw));
    return static_cast<WeightHandle>(impl_->weights.size() - 1);
}

WeightHandle NpuRegistry::create_ffn_fused_weight(int H, int I, const std::string& activation,
                                                  const void* packed, size_t nbytes) {
    LoadedKernel& lk = impl_->ensure_loaded("ffn_fused", 0, 0, 0, H, I, activation);
    ResidentWeight rw;
    rw.N = H; // Abuse N,K for H,I to avoid changing struct
    rw.K = I;
    rw.bo = xrt::bo(impl_->device, nbytes, XRT_BO_FLAGS_HOST_ONLY,
                    lk.kernel.group_id(3));
    std::memcpy(rw.bo.map<void*>(), packed, nbytes);
    rw.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    impl_->weights.push_back(std::move(rw));
    return static_cast<WeightHandle>(impl_->weights.size() - 1);
}

void NpuRegistry::run_gemv(int N, int K, WeightHandle w, const void* x_bf16,
                           void* y_bf16) {
    prof::ScopedTimer _prof_timer(nullptr);
    if (w >= impl_->weights.size())
        throw std::runtime_error("npu: invalid weight handle");
    const ResidentWeight& rw = impl_->weights[w];
    if (rw.N != N || rw.K != K)
        throw std::runtime_error("npu: weight/shape mismatch in run_gemv: expected (" + std::to_string(rw.N) + ", " + std::to_string(rw.K) + "), got (" + std::to_string(N) + ", " + std::to_string(K) + ")");

    LoadedKernel& lk = impl_->ensure_loaded("gemv", N, K, 0);

    std::memcpy(lk.x_bo.map<void*>(), x_bf16, size_t(K) * sizeof(uint16_t));
    lk.x_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = lk.kernel(impl_->opcode, lk.instr, lk.ninstr, rw.bo, lk.x_bo,
                         lk.y_bo);
    run.wait();

    lk.y_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(y_bf16, lk.y_bo.map<void*>(), size_t(N) * sizeof(uint16_t));
}

void NpuRegistry::run_ffn_fused(int H, int I, const std::string& activation, WeightHandle w,
                                const void* x_bf16, void* y_bf16) {
    prof::ScopedTimer _prof_timer(&prof::ffn_seconds);
    if (w >= impl_->weights.size())
        throw std::runtime_error("npu: invalid weight handle");
    const ResidentWeight& rw = impl_->weights[w];
    if (rw.N != H || rw.K != I)
        throw std::runtime_error("npu: weight/shape mismatch in run_ffn_fused");

    LoadedKernel& lk = impl_->ensure_loaded("ffn_fused", 0, 0, 0, H, I, activation);

    std::memcpy(lk.x_bo.map<void*>(), x_bf16, size_t(H) * sizeof(uint16_t));
    lk.x_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = lk.kernel(impl_->opcode, lk.instr, lk.ninstr, rw.bo, lk.x_bo,
                         lk.y_bo);
    run.wait();

    lk.y_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    // CPU reduction: sum the n_cores partial results into y_bf16
    const uint16_t* y_partial = lk.y_bo.map<const uint16_t*>();
    uint16_t* y_out = static_cast<uint16_t*>(y_bf16);
    int n_cores = lk.spec.n_cores;
    
    // Convert to fp32, sum, convert back to bf16
    std::vector<float> y_fp32(H, 0.0f);
    for (int c = 0; c < n_cores; ++c) {
        for (int h = 0; h < H; ++h) {
            bf16 val;
            val.v = y_partial[c * H + h];
            y_fp32[h] += val.to_float();
        }
    }
    for (int h = 0; h < H; ++h) {
        bf16 val(y_fp32[h]);
        y_out[h] = val.v;
    }
}

void NpuRegistry::pin_gemv(int N, int K) {
    impl_->ensure_loaded("gemv", N, K, 0).pinned = true;
}

void NpuRegistry::pin_ffn_fused(int H, int I, const std::string& activation) {
    impl_->ensure_loaded("ffn_fused", 0, 0, 0, H, I, activation).pinned = true;
}

int NpuRegistry::loaded_contexts() const {
    return static_cast<int>(impl_->loaded.size());
}

} // namespace alveare
