#pragma once
#include <string>

namespace alveare {

struct ModelConfig {
    std::string model_type; // "llama", "gemma3", "gemma4"
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    int intermediate_size;
    int vocab_size;
    int head_dim;
    float rms_norm_eps;
};

ModelConfig load_config(const std::string& path);

} // namespace alveare
