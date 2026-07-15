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

    std::vector<float> attn_norm;
    std::vector<float> ffn_norm;
    std::vector<float> post_attention_norm;
    std::vector<float> post_ffw_norm;
    std::vector<float> q_norm;
    std::vector<float> k_norm;
};

struct ModelWeights {
    std::vector<LayerWeights> layers;
    std::vector<float> token_embd;
    std::vector<float> output_norm;
    std::vector<uint8_t> lm_head;
};

ModelWeights load_weights(const std::string& dir, const ModelConfig& config, NpuRegistry& reg);

} // namespace alveare
