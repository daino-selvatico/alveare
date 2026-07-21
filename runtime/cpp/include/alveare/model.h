#pragma once
#include <vector>
#include <cstdint>
#include "alveare/config.h"
#include "alveare/weights.h"
#include "alveare/npu.h"
#include "alveare/bf16.h"

namespace alveare {

class Model {
public:
    Model(const ModelConfig& config, const ModelWeights& weights, NpuRegistry& reg);

    // Returns the model config
    const ModelConfig& get_config() const { return config_; }

    // Access to the NPU registry (e.g. for the LM head gemv driven by Generator).
    NpuRegistry& registry() { return reg_; }

    // Run a single layer decode natively.
    // Returns true on success.
    void run_layer(const bf16* x_bf16, int pos, int layer, bf16* out_bf16);

    // Batched prefill of one layer over `nrows` (<= 16) consecutive positions
    // starting at pos_start. x_batch/out_batch are (nrows, hidden_size) row-major.
    // Uses the GEMM kernels (resident attn weights, streamed FFN weights) with
    // CPU RMSNorm/RoPE/causal attention. gemma4 only.
    void run_layer_batch(const bf16* x_batch, int nrows, int pos_start, int layer,
                         bf16* out_batch);

    void reset_caches();

private:
    ModelConfig config_;
    const ModelWeights& weights_;
    NpuRegistry& reg_;

    // KV Caches
    // Stored as [layer][head][pos][dim]
    std::vector<std::vector<bf16>> k_caches_;
    std::vector<std::vector<bf16>> v_caches_;

    // Precomputed RoPE tables
    std::vector<bf16> cos_sin_table_; // Llama
    std::vector<bf16> cos_sin_table_sliding_; // Gemma
    std::vector<bf16> cos_sin_table_full_; // Gemma

    void init_kv_caches();
    void precompute_rope();

    void run_rmsnorm_cpu(const bf16* x, const float* w, bf16* out, int K = 0);
    void run_rope_cpu_llama(const bf16* x, int pos, int num_heads, bf16* out);
    void run_rope_cpu_gemma(const bf16* x, int pos, float base_freq, int num_heads, bf16* out);
    void run_attention_host(const bf16* q_rope, int pos, int layer, bf16* out);
};

} // namespace alveare
