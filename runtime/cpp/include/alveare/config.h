#pragma once
#include <string>

namespace alveare {

struct ModelConfig {
    std::string model_type; // "llama", "gemma3", "gemma4", "gemma4-e4b"
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int num_hidden_layers;
    int intermediate_size;
    int vocab_size;
    int head_dim;
    int head_dim_global = 512;
    int per_layer_input = 0;
    int shared_kv_layers = 0;
    int sliding_window = 512;
    int sliding_pattern_period = 6;
    float rms_norm_eps;

    bool is_gemma4() const {
        return model_type == "gemma4" || model_type == "gemma4-e4b";
    }

    int get_padded_hidden_size() const {
        if (is_gemma4()) {
            return (hidden_size == 3840) ? 4096 : hidden_size;
        }
        return hidden_size;
    }

    int get_padded_intermediate_size() const {
        if (is_gemma4()) {
            return (intermediate_size == 15360) ? 16384 : intermediate_size;
        }
        return intermediate_size;
    }
};

ModelConfig load_config(const std::string& path);

} // namespace alveare
