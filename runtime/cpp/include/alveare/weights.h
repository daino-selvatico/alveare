#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "alveare/config.h"
#include "alveare/npu.h"

namespace alveare {

struct LayerWeights {
    WeightHandle w_q;
    WeightHandle w_k;
    WeightHandle w_v;
    WeightHandle w_o;
    WeightHandle w_ffn_fused;

    // Fused Q/K/V projection (gemma4): w_q, w_k, w_v concatenated along the output
    // dimension into one resident weight so the three projections run as a SINGLE
    // gemv — one NPU launch and one kernel-shape context instead of three, which
    // avoids ~2.6 ms of per-shape context-switch overhead each. kInvalidWeight
    // when not built (non-gemma4). n_qkv is the fused output width; the slices are
    // q[0:n_q], k[n_q:n_q+n_kv], v[n_q+n_kv:] (global layers reuse k for v).
    WeightHandle w_qkv = kInvalidWeight;
    int n_qkv = 0;
    int n_q = 0;
    int n_kv = 0;

    std::vector<float> attn_norm;
    std::vector<float> ffn_norm;
    std::vector<float> post_attention_norm;
    std::vector<float> post_ffw_norm;
    std::vector<float> q_norm;
    std::vector<float> k_norm;

    // Gemma-4 only: scalar applied to the whole layer output (residual included)
    // at the end of the block. 1.0 for models without a per-layer output scale.
    float output_scale = 1.0f;

    // Host-resident Q4_0 packed FFN weights, kept for batched prefill (GEMM):
    // gate/up are (I=16384, K=4096), down is (H=4096, I=16384). The decode path
    // uses only the fused device weight above; these feed run_gemm_streamed.
    std::vector<uint8_t> ffn_gate_bytes;
    std::vector<uint8_t> ffn_up_bytes;
    std::vector<uint8_t> ffn_down_bytes;
};

struct ModelWeights {
    std::vector<LayerWeights> layers;
    std::vector<float> token_embd;
    std::vector<float> output_norm;

    // LM head. When a matching NPU gemv kernel exists, the packed weight is
    // uploaded to the device split into `lm_head_chunks` row-tiles of
    // (lm_head_chunk_N, lm_head_K) each, and `lm_head` (raw bytes) is released.
    // Otherwise `lm_head` keeps the packed bytes for the CPU dequant fallback.
    std::vector<uint8_t> lm_head;
    std::vector<WeightHandle> lm_head_chunks;
    int lm_head_vocab = 0;
    int lm_head_K = 0;
    int lm_head_chunk_N = 0;
};

ModelWeights load_weights(const std::string& dir, const ModelConfig& config, NpuRegistry& reg);

} // namespace alveare
