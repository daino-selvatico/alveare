#include "alveare/weights.h"
#include "alveare/npy.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace alveare {

static std::vector<float> load_float_npy(const std::string& path) {
    NpyArray arr = load_npy(path);
    if (!arr.data) {
        std::cerr << "Warning: Failed to load " << path << "\n";
        return {};
    }
    std::vector<float> vec(arr.data_size / sizeof(float));
    std::memcpy(vec.data(), arr.data, arr.data_size);
    free_npy(arr);
    return vec;
}

static std::vector<uint8_t> load_uint8_npy(const std::string& path) {
    NpyArray arr = load_npy(path);
    if (!arr.data) {
        return {};
    }
    std::vector<uint8_t> vec(arr.data_size);
    std::memcpy(vec.data(), arr.data, arr.data_size);
    free_npy(arr);
    return vec;
}

static std::vector<uint8_t> pack_ffn_fused_weights(
    const std::vector<uint8_t>& w_gate,
    const std::vector<uint8_t>& w_up,
    const std::vector<uint8_t>& w_down,
    int H, int I, int m_I, int k_tile) {

    int m_H = k_tile;
    int n_cores = 1;
    if (I % (8 * m_I) == 0) n_cores = 8;
    else if (I % (4 * m_I) == 0) n_cores = 4;
    else if (I % (2 * m_I) == 0) n_cores = 2;

    int I_div_n_cores = I / n_cores;
    int num_blocks_I = I_div_n_cores / m_I;
    int chunks_per_gate_up = k_tile / 32;

    std::vector<uint8_t> fused;
    // Pre-allocate to avoid reallocations
    size_t total_size = w_gate.size() + w_up.size() + w_down.size();
    fused.reserve(total_size);

    int gate_up_stride = (H / 32) * 20;
    int down_stride = (I / 32) * 20;

    auto append_tile = [&](const std::vector<uint8_t>& w, int r_start, int r_end, int c_start_bytes, int c_end_bytes, int stride) {
        for (int r = r_start; r < r_end; ++r) {
            const uint8_t* ptr = w.data() + r * stride + c_start_bytes;
            fused.insert(fused.end(), ptr, ptr + (c_end_bytes - c_start_bytes));
        }
    };

    for (int c = 0; c < n_cores; ++c) {
        int start_I = c * I_div_n_cores;
        int end_I = (c + 1) * I_div_n_cores;

        for (int b_I = 0; b_I < num_blocks_I; ++b_I) {
            int row_start = start_I + b_I * m_I;
            int row_end = start_I + (b_I + 1) * m_I;

            for (int h_blk = 0; h_blk < H / k_tile; ++h_blk) {
                int col_start_bytes = h_blk * chunks_per_gate_up * 20;
                int col_end_bytes = (h_blk + 1) * chunks_per_gate_up * 20;

                // Gate tile
                append_tile(w_gate, row_start, row_end, col_start_bytes, col_end_bytes, gate_up_stride);
                // Up tile
                append_tile(w_up, row_start, row_end, col_start_bytes, col_end_bytes, gate_up_stride);
            }

            int start_block = start_I / 32;
            int col_start_bytes_down = start_block * 20 + b_I * (m_I / 32) * 20;
            int col_end_bytes_down = start_block * 20 + (b_I + 1) * (m_I / 32) * 20;

            for (int h_blk_down = 0; h_blk_down < H / m_H; ++h_blk_down) {
                int row_start_down = h_blk_down * m_H;
                int row_end_down = (h_blk_down + 1) * m_H;

                append_tile(w_down, row_start_down, row_end_down, col_start_bytes_down, col_end_bytes_down, down_stride);
            }
        }
    }

    return fused;
}

ModelWeights load_weights(const std::string& dir, const ModelConfig& config, NpuRegistry& reg) {
    ModelWeights mw;
    mw.token_embd = load_float_npy(dir + "/token_embd.npy");
    mw.output_norm = load_float_npy(dir + "/output_norm.weight.npy");
    mw.lm_head = load_uint8_npy(dir + "/lm_head_packed.npy");

    int K_attn = config.hidden_size;
    int N_q = config.num_attention_heads * config.head_dim;
    int N_kv = config.num_key_value_heads * config.head_dim;
    int N_out = config.hidden_size;

    for (int l = 0; l < config.num_hidden_layers; ++l) {
        std::cout << "Loading weights for layer " << l << " ...\r" << std::flush;
        LayerWeights lw;
        lw.attn_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".attn_norm.weight.npy");
        lw.ffn_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".ffn_norm.weight.npy");

        if (config.model_type == "gemma3" || config.model_type == "gemma4") {
            lw.post_attention_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".post_attention_norm.weight.npy");
            lw.post_ffw_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".post_ffw_norm.weight.npy");
            lw.q_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".attn_q_norm.weight.npy");
            lw.k_norm = load_float_npy(dir + "/blk." + std::to_string(l) + ".attn_k_norm.weight.npy");
        }

        // QKV, O projections
        bool is_sliding = (config.model_type == "gemma4" && (l + 1) % 6 != 0);
        int l_N_q = is_sliding ? 4096 : N_q;
        int l_N_kv = is_sliding ? 2048 : N_kv;
        int l_N_out = is_sliding ? 4096 : N_out;

        std::string act_type = (config.model_type == "gemma3" || config.model_type == "gemma4") ? "gelu" : "silu";

        std::string q_path = dir + "/blk." + std::to_string(l) + ".attn_q.weight_packed.npy";
        std::string k_path = dir + "/blk." + std::to_string(l) + ".attn_k.weight_packed.npy";
        std::string v_path = dir + "/blk." + std::to_string(l) + ".attn_v.weight_packed.npy";
        std::string o_path = dir + "/blk." + std::to_string(l) + ".attn_output.weight_packed.npy";

        NpyArray q_arr = load_npy(q_path);
        lw.w_q = reg.create_gemv_weight(l_N_q, K_attn, q_arr.data, q_arr.data_size);
        free_npy(q_arr);

        NpyArray k_arr = load_npy(k_path);
        lw.w_k = reg.create_gemv_weight(l_N_kv, K_attn, k_arr.data, k_arr.data_size);
        free_npy(k_arr);

        if (config.model_type != "gemma4" || is_sliding) {
            NpyArray v_arr = load_npy(v_path);
            lw.w_v = reg.create_gemv_weight(l_N_kv, K_attn, v_arr.data, v_arr.data_size);
            free_npy(v_arr);
        } else {
            // For global gemma4 layers, V uses K weight
            lw.w_v = lw.w_k;
        }

        NpyArray o_arr = load_npy(o_path);
        lw.w_o = reg.create_gemv_weight(K_attn, l_N_out, o_arr.data, o_arr.data_size);
        free_npy(o_arr);

        // FFN Fused
        std::string ffn_path = dir + "/blk." + std::to_string(l) + ".ffn_fused.weight_packed.npy";
        NpyArray ffn_arr = load_npy(ffn_path);
        if (ffn_arr.data) {
            lw.w_ffn_fused = reg.create_ffn_fused_weight(config.hidden_size, config.intermediate_size, act_type, ffn_arr.data, ffn_arr.data_size);
            free_npy(ffn_arr);
        } else {
            // Fallback: load gate, up, down and pack in C++
            std::cout << "\nPre-packing FFN fused weights for layer " << l << " ...\r" << std::flush;
            std::string gate_path = dir + "/blk." + std::to_string(l) + ".ffn_gate.weight_packed.npy";
            std::string up_path = dir + "/blk." + std::to_string(l) + ".ffn_up.weight_packed.npy";
            std::string down_path = dir + "/blk." + std::to_string(l) + ".ffn_down.weight_packed.npy";

            auto w_gate = load_uint8_npy(gate_path);
            auto w_up = load_uint8_npy(up_path);
            auto w_down = load_uint8_npy(down_path);

            if (w_gate.empty() || w_up.empty() || w_down.empty()) {
                throw std::runtime_error("Missing FFN weights for layer " + std::to_string(l));
            }

            int k_tile = (config.hidden_size == 1152) ? 128 : 256;
            auto fused = pack_ffn_fused_weights(w_gate, w_up, w_down, config.hidden_size, config.intermediate_size, 32, k_tile);
            
            lw.w_ffn_fused = reg.create_ffn_fused_weight(config.hidden_size, config.intermediate_size, act_type, fused.data(), fused.size());
        }

        mw.layers.push_back(lw);
    }
    std::cout << "\nLoaded all weights.\n";
    return mw;
}

} // namespace alveare
