#include "alveare/npu.h"

#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

#include "nlohmann/json.hpp"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

namespace alveare {

namespace {

std::string shape_key(const std::string& kind, int N, int K, int B) {
    return kind + ':' + std::to_string(N) + ':' + std::to_string(K) + ':' +
           std::to_string(B);
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
                                int B) const {
        for (const auto& s : specs)
            if (s.kind == kind && s.N == N && s.K == K && s.B == B) return &s;
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

    LoadedKernel& ensure_loaded(const std::string& kind, int N, int K, int B) {
        const std::string key = shape_key(kind, N, K, B);
        auto it = loaded.find(key);
        if (it != loaded.end()) {
            it->second.last_used = ++tick;
            return it->second;
        }

        const KernelSpec* spec = find_spec(kind, N, K, B);
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

                const size_t rows = (kind == "gemm") ? size_t(B) : 1;
                const size_t x_bytes = rows * size_t(K) * sizeof(uint16_t);
                const size_t y_bytes = rows * size_t(N) * sizeof(uint16_t);
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

void NpuRegistry::run_gemv(int N, int K, WeightHandle w, const void* x_bf16,
                           void* y_bf16) {
    if (w >= impl_->weights.size())
        throw std::runtime_error("npu: invalid weight handle");
    const ResidentWeight& rw = impl_->weights[w];
    if (rw.N != N || rw.K != K)
        throw std::runtime_error("npu: weight/shape mismatch in run_gemv");

    LoadedKernel& lk = impl_->ensure_loaded("gemv", N, K, 0);

    std::memcpy(lk.x_bo.map<void*>(), x_bf16, size_t(K) * sizeof(uint16_t));
    lk.x_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = lk.kernel(impl_->opcode, lk.instr, lk.ninstr, rw.bo, lk.x_bo,
                         lk.y_bo);
    run.wait();

    lk.y_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(y_bf16, lk.y_bo.map<void*>(), size_t(N) * sizeof(uint16_t));
}

void NpuRegistry::pin_gemv(int N, int K) {
    impl_->ensure_loaded("gemv", N, K, 0).pinned = true;
}

int NpuRegistry::loaded_contexts() const {
    return static_cast<int>(impl_->loaded.size());
}

} // namespace alveare
