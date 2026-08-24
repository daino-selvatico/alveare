#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "alveare/config.h"
#include "alveare/npu.h"

namespace alveare {

struct LayerWeights {
    WeightHandle w_q = kInvalidWeight;
    WeightHandle w_k = kInvalidWeight;
    WeightHandle w_v = kInvalidWeight;
    WeightHandle w_o = kInvalidWeight;
    WeightHandle w_ffn_fused = kInvalidWeight;

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

    // Output-projection gemv width. For gemma4 sliding layers w_o is zero-padded
    // in N so it shares the SAME (8192, 4096) kernel shape as w_qkv — the two run
    // back-to-back with no kernel context switch between them (~2.6 ms saved per
    // layer). 0 means "use the natural output width" (no padding). The real
    // output slice is the first hidden_size (padded) rows.
    // ALVEARE_ONESHAPE: run the FFN as gemv tiles on the SAME (N,K) as the fused QKV,
    // so a whole layer uses ONE kernel context (a hw-context switch costs a fixed
    // ~2.5 ms, and decode pays two per layer). Empty unless the flag is on.
    std::vector<WeightHandle> os_gateup;  // 7 tiles of (os_n, os_k): gate rows ++ up rows
    std::vector<WeightHandle> os_down;    // 4 K-chunks of (os_n, os_k)
    // QKV/O are SPLIT into tiles of (os_n, os_k) rather than padded up to a bigger
    // shape: with os_n == padded hidden every dim of the 12B is an exact multiple, so
    // there is no padding waste and still no context switch.
    std::vector<WeightHandle> os_qkv_tiles;
    std::vector<WeightHandle> os_o_tiles;
    bool os_o_is_kchunked = false;
    int os_n = 0;                         // tile output rows (== fused-QKV N)
    int os_k = 0;                         // tile input dim  (== fused-QKV K)

    int o_gemv_n = 0;
    // When >0, the O projection was zero-padded in its INPUT dim to this K so it
    // reuses the fused-QKV kernel's (N,K) context (no per-layer context switch).
    int o_gemv_k = 0;

    std::vector<float> attn_norm;
    std::vector<float> ffn_norm;
    std::vector<float> post_attention_norm;
    std::vector<float> post_ffw_norm;
    std::vector<float> q_norm;
    std::vector<float> k_norm;

    // Gemma-4 only: scalar applied to the whole layer output (residual included)
    // at the end of the block. 1.0 for models without a per-layer output scale.
    float output_scale = 1.0f;

    // Gemma-4-E4B PLE per-layer injection tensors
    std::vector<uint8_t> inp_gate_bytes;
    std::vector<uint8_t> proj_bytes;
    std::vector<float> post_norm;

    // Host-resident Q4_0 packed FFN weights, kept for batched prefill (GEMM):
    // gate/up are (I=16384, K=4096), down is (H=4096, I=16384). The decode path
    // uses only the fused device weight above; these feed run_gemm_streamed.
    std::vector<uint8_t> ffn_gate_bytes;
    std::vector<uint8_t> ffn_up_bytes;
    std::vector<uint8_t> ffn_down_bytes;

    std::vector<uint8_t> w_o_bytes;
};

struct ModelWeights {
    std::vector<LayerWeights> layers;
    std::vector<float> token_embd;
    std::vector<float> output_norm;

    // Gemma-4-E4B PLE model-level tensors
    std::vector<uint16_t> per_layer_token_embd_f16;
    std::vector<uint8_t> per_layer_model_proj_packed;
    std::vector<float> per_layer_proj_norm;

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
std::vector<float> load_float_npy(const std::string& path);
std::vector<uint16_t> load_uint16_npy(const std::string& path);

} // namespace alveare
