#pragma once
#include <string>

namespace alveare {

struct ModelConfig {
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    int intermediate_size;
    int vocab_size;
    float rms_norm_eps;
};

ModelConfig load_config(const std::string& path);

} // namespace alveare
