#include "alveare/config.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace alveare {

ModelConfig load_config(const std::string& path) {
    ModelConfig cfg{};
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: could not open " << path << ", using Llama-3.2 defaults\n";
        cfg.model_type = "llama";
        cfg.hidden_size = 2048;
        cfg.intermediate_size = 8192;
        cfg.num_attention_heads = 32;
        cfg.num_key_value_heads = 8;
        cfg.head_dim = 64;
        cfg.num_hidden_layers = 16;
        cfg.vocab_size = 128256;
        cfg.rms_norm_eps = 1e-5f;
        return cfg;
    }

    json j;
    f >> j;

    cfg.model_type = j.value("model_type", "llama");
    
    if (cfg.model_type == "gemma3") {
        cfg.hidden_size = j.value("hidden_size", 1152);
        cfg.intermediate_size = j.value("intermediate_size", 6912);
        cfg.num_attention_heads = j.value("num_attention_heads", 4);
        cfg.num_key_value_heads = j.value("num_key_value_heads", 1);
        cfg.head_dim = j.value("head_dim", 256);
        cfg.num_hidden_layers = j.value("num_hidden_layers", 26);
        cfg.vocab_size = j.value("vocab_size", 262144);
        cfg.rms_norm_eps = 1e-6f;
    } else if (cfg.model_type == "gemma4") {
        cfg.hidden_size = j.value("hidden_size", 3840);
        cfg.intermediate_size = j.value("intermediate_size", 15360);
        cfg.num_attention_heads = j.value("num_attention_heads", 16);
        cfg.num_key_value_heads = j.value("num_key_value_heads", 8);
        cfg.head_dim = j.value("head_dim", 256);
        cfg.num_hidden_layers = j.value("num_hidden_layers", 48);
        cfg.vocab_size = j.value("vocab_size", 262144);
        cfg.rms_norm_eps = 1e-6f;
    } else {
        cfg.hidden_size = j.value("hidden_size", 2048);
        cfg.intermediate_size = j.value("intermediate_size", 8192);
        cfg.num_attention_heads = j.value("num_attention_heads", 32);
        cfg.num_key_value_heads = j.value("num_key_value_heads", 8);
        cfg.head_dim = j.value("head_dim", 64);
        cfg.num_hidden_layers = j.value("num_hidden_layers", 16);
        cfg.vocab_size = j.value("vocab_size", 128256);
        cfg.rms_norm_eps = 1e-5f;
    }
    
    return cfg;
}

} // namespace alveare
