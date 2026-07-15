#include "alveare/model.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace alveare {

Model::Model(const ModelConfig& config, const ModelWeights& weights, NpuRegistry& reg)
    : config_(config), weights_(weights), reg_(reg) {
    init_kv_caches();
    precompute_rope();
}

void Model::init_kv_caches() {
    int max_seq_len = 2048; // For now hardcoded or passed in config
    int n_layers = config_.num_hidden_layers;

    k_caches_.resize(n_layers);
    v_caches_.resize(n_layers);

    for (int l = 0; l < n_layers; ++l) {
        int n_kv_heads = config_.num_key_value_heads;
        int h_dim = config_.head_dim;
        
        if (config_.model_type == "gemma4") {
            bool is_sliding = ((l + 1) % 6 != 0);
            n_kv_heads = is_sliding ? 8 : 1;
            h_dim = is_sliding ? 256 : 512;
        }

        size_t layer_kv_size = size_t(n_kv_heads) * max_seq_len * h_dim;
        k_caches_[l].resize(layer_kv_size, bf16(0.0f));
        v_caches_[l].resize(layer_kv_size, bf16(0.0f));
    }
}

void Model::reset_caches() {
    for (size_t l = 0; l < k_caches_.size(); ++l) {
        std::fill(k_caches_[l].begin(), k_caches_[l].end(), bf16(0.0f));
        std::fill(v_caches_[l].begin(), v_caches_[l].end(), bf16(0.0f));
    }
}

void Model::precompute_rope() {
    int max_seq_len = 2048; // Hardcoded for now
    if (config_.model_type == "gemma3") {
        cos_sin_table_sliding_.resize(max_seq_len * config_.head_dim);
        cos_sin_table_full_.resize(max_seq_len * config_.head_dim);
        
        auto precompute = [&](float base, std::vector<bf16>& table) {
            for (int pos = 0; pos < max_seq_len; ++pos) {
                for (int i = 0; i < config_.head_dim / 2; ++i) {
                    float inv_freq = 1.0f / std::pow(base, float(i * 2) / config_.head_dim);
                    float freq = pos * inv_freq;
                    table[pos * config_.head_dim + i] = bf16(std::cos(freq));
                    table[pos * config_.head_dim + config_.head_dim / 2 + i] = bf16(std::sin(freq));
                }
            }
        };
        precompute(10000.0f, cos_sin_table_sliding_);
        precompute(1000000.0f, cos_sin_table_full_);
    } else if (config_.model_type == "gemma4") {
        cos_sin_table_sliding_.resize(max_seq_len * 256);
        cos_sin_table_full_.resize(max_seq_len * 512);
        
        auto precompute = [&](float base, int dim, std::vector<bf16>& table) {
            for (int pos = 0; pos < max_seq_len; ++pos) {
                for (int i = 0; i < dim / 2; ++i) {
                    float inv_freq = 0.0f;
                    if (dim == 512) {
                        int rope_angles = static_cast<int>(0.25f * dim / 2.0f);
                        if (i < rope_angles) {
                            inv_freq = 1.0f / std::pow(base, float(i * 2) / dim);
                        }
                    } else {
                        inv_freq = 1.0f / std::pow(base, float(i * 2) / dim);
                    }
                    float freq = pos * inv_freq;
                    table[pos * dim + i] = bf16(std::cos(freq));
                    table[pos * dim + dim / 2 + i] = bf16(std::sin(freq));
                }
            }
        };
        precompute(10000.0f, 256, cos_sin_table_sliding_);
        precompute(1000000.0f, 512, cos_sin_table_full_);
    } else {
        // Llama
        int dim = 64;
        cos_sin_table_.resize(max_seq_len * 128);
        float base = 500000.0f;
        float factor = 32.0f;
        float low_freq_factor = 1.0f;
        float high_freq_factor = 4.0f;
        float old_context_len = 8192.0f;

        for (int pos = 0; pos < max_seq_len; ++pos) {
            for (int i = 0; i < dim / 2; ++i) {
                float inv_freq = 1.0f / std::pow(base, float(i * 2) / dim);
                float wavelen = 2.0f * M_PI / inv_freq;
                float low_freq_wavelen = old_context_len / low_freq_factor;
                float high_freq_wavelen = old_context_len / high_freq_factor;

                float final_inv_freq = inv_freq;
                if (wavelen > low_freq_wavelen) {
                    final_inv_freq = inv_freq / factor;
                } else if (wavelen >= high_freq_wavelen && wavelen <= low_freq_wavelen) {
                    float smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                    smooth_factor = std::max(0.0f, std::min(1.0f, smooth_factor));
                    final_inv_freq = (1.0f - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq;
                }

                float freq = pos * final_inv_freq;
                float c = std::cos(freq);
                float s = std::sin(freq);

                // Llama format: [cos, cos, sin, sin] duplicated
                cos_sin_table_[pos * 128 + i] = bf16(c);
                cos_sin_table_[pos * 128 + dim / 2 + i] = bf16(c);
                cos_sin_table_[pos * 128 + dim + i] = bf16(s);
                cos_sin_table_[pos * 128 + dim + dim / 2 + i] = bf16(s);
            }
        }
    }
}

void Model::run_rmsnorm_cpu(const bf16* x, const float* w, bf16* out) {
    int K = config_.hidden_size;
    float variance = 0.0f;
    for (int i = 0; i < K; ++i) {
        float val = x[i].to_float();
        variance += val * val;
    }
    variance /= K;
    float inv_denom = 1.0f / std::sqrt(variance + config_.rms_norm_eps);

    for (int i = 0; i < K; ++i) {
        float val = x[i].to_float() * inv_denom;
        if (w) val *= w[i];
        out[i] = bf16(val);
    }
}

void Model::run_rope_cpu_llama(const bf16* x, int pos, bf16* out) {
    int K = config_.hidden_size;
    const bf16* cos_sin = &cos_sin_table_[pos * 128];
    const bf16* cos_ptr = cos_sin;
    const bf16* sin_ptr = cos_sin + 64;

    int num_heads = K / 64;
    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < 32; ++i) {
            float x1 = x[h * 64 + i].to_float();
            float x2 = x[h * 64 + 32 + i].to_float();
            float c = cos_ptr[i].to_float();
            float s = sin_ptr[i].to_float();

            out[h * 64 + i] = bf16(x1 * c - x2 * s);
            out[h * 64 + 32 + i] = bf16(x2 * c + x1 * s);
        }
    }
}

void Model::run_rope_cpu_gemma(const bf16* x, int pos, float base_freq, bf16* out) {
    int K = config_.hidden_size;
    int dim = 256;
    if (config_.model_type == "gemma4" && base_freq > 10000.0f) {
        dim = 512;
    }
    int num_heads = K / dim;
    const bf16* cos_sin = nullptr;
    if (base_freq == 10000.0f) {
        cos_sin = &cos_sin_table_sliding_[pos * dim];
    } else {
        cos_sin = &cos_sin_table_full_[pos * dim];
    }

    const bf16* cos_ptr = cos_sin;
    const bf16* sin_ptr = cos_sin + dim / 2;

    for (int h = 0; h < num_heads; ++h) {
        for (int i = 0; i < dim / 2; ++i) {
            float x1 = x[h * dim + i].to_float();
            float x2 = x[h * dim + dim / 2 + i].to_float();
            float c = cos_ptr[i].to_float();
            float s = sin_ptr[i].to_float();

            out[h * dim + i] = bf16(x1 * c - x2 * s);
            out[h * dim + dim / 2 + i] = bf16(x2 * c + x1 * s);
        }
    }
}

void Model::run_attention_host(const bf16* q_rope, int pos, int layer, bf16* out) {
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int dim = config_.head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    int window_size = 512;
    int max_seq_len = 2048; // Must match init_kv_caches
    
    if (config_.model_type == "gemma4") {
        bool is_sliding = ((layer + 1) % 6 != 0);
        num_heads = 16;
        num_kv_heads = is_sliding ? 8 : 1;
        dim = is_sliding ? 256 : 512;
        scale = 1.0f;
        window_size = 1024;
    }

    int seq_len = pos + 1;
    bool is_sliding_layer = (config_.model_type == "gemma3" && (layer + 1) % 6 != 0) || 
                            (config_.model_type == "gemma4" && (layer + 1) % 6 != 0);

    int start_pos = 0;
    if (is_sliding_layer && seq_len > window_size) {
        start_pos = seq_len - window_size;
    }
    int W = seq_len - start_pos;
    int group_ratio = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / group_ratio;

        std::vector<float> scores(W, 0.0f);
        float max_score = -1e9f;

        const bf16* q_ptr = &q_rope[h * dim];

        for (int w = 0; w < W; ++w) {
            int cache_pos = start_pos + w;
            int kv_idx = (kv_h * max_seq_len + cache_pos) * dim;
            
            float dot = 0.0f;
            const bf16* k_ptr = &k_caches_[layer][kv_idx];
            for (int i = 0; i < dim; ++i) {
                dot += q_ptr[i].to_float() * k_ptr[i].to_float();
            }
            dot *= scale;
            scores[w] = dot;
            if (dot > max_score) max_score = dot;
        }

        float sum_exp = 0.0f;
        for (int w = 0; w < W; ++w) {
            scores[w] = std::exp(scores[w] - max_score);
            sum_exp += scores[w];
        }

        std::vector<float> out_f(dim, 0.0f);
        for (int w = 0; w < W; ++w) {
            float prob = scores[w] / sum_exp;
            int cache_pos = start_pos + w;
            int kv_idx = (kv_h * max_seq_len + cache_pos) * dim;
            const bf16* v_ptr = &v_caches_[layer][kv_idx];
            
            for (int i = 0; i < dim; ++i) {
                out_f[i] += prob * v_ptr[i].to_float();
            }
        }

        for (int i = 0; i < dim; ++i) {
            out[h * dim + i] = bf16(out_f[i]);
        }
    }
}

void Model::run_layer(const bf16* x_bf16, int pos, int layer, bf16* out_bf16) {
    int K = config_.hidden_size;
    const LayerWeights& lw = weights_.layers[layer];
    
    // 1. Input RMSNorm
    std::vector<bf16> x_norm(K);
    run_rmsnorm_cpu(x_bf16, lw.attn_norm.empty() ? nullptr : lw.attn_norm.data(), x_norm.data());

    // 2. QKV Projections (NPU)
    bool is_sliding = (config_.model_type == "gemma4" && (layer + 1) % 6 != 0);
    int N_q = config_.num_attention_heads * config_.head_dim;
    int N_kv = config_.num_key_value_heads * config_.head_dim;
    if (config_.model_type == "gemma4") {
        N_q = is_sliding ? 4096 : 8192;
        N_kv = is_sliding ? 2048 : 512;
    }
    std::vector<bf16> q(N_q);
    std::vector<bf16> k(N_kv);
    std::vector<bf16> v(N_kv);

    reg_.run_gemv(N_q, K, lw.w_q, x_norm.data(), q.data());
    reg_.run_gemv(N_kv, K, lw.w_k, x_norm.data(), k.data());
    if (config_.model_type != "gemma4" || is_sliding) {
        reg_.run_gemv(N_kv, K, lw.w_v, x_norm.data(), v.data());
    } else {
        v = k; // Gemma4 global layers use K for V
    }

    // 3. QK-Norm & V-Norm (Gemma only)
    if (config_.model_type == "gemma3" || config_.model_type == "gemma4") {
        for (int h = 0; h < config_.num_attention_heads; ++h) {
            int h_dim = config_.head_dim;
            if (config_.model_type == "gemma4") h_dim = is_sliding ? 256 : 512;
            std::vector<bf16> q_h(h_dim);
            run_rmsnorm_cpu(&q[h * h_dim], lw.q_norm.empty() ? nullptr : lw.q_norm.data(), q_h.data());
            std::memcpy(&q[h * h_dim], q_h.data(), h_dim * sizeof(bf16));
        }
        for (int h = 0; h < config_.num_key_value_heads; ++h) {
            int h_dim = config_.head_dim;
            if (config_.model_type == "gemma4") h_dim = is_sliding ? 256 : 512;
            std::vector<bf16> k_h(h_dim);
            run_rmsnorm_cpu(&k[h * h_dim], lw.k_norm.empty() ? nullptr : lw.k_norm.data(), k_h.data());
            std::memcpy(&k[h * h_dim], k_h.data(), h_dim * sizeof(bf16));
        }
        if (config_.model_type == "gemma4") {
            for (int h = 0; h < config_.num_key_value_heads; ++h) {
                int h_dim = is_sliding ? 256 : 512;
                std::vector<bf16> v_h(h_dim);
                run_rmsnorm_cpu(&v[h * h_dim], nullptr, v_h.data());
                std::memcpy(&v[h * h_dim], v_h.data(), h_dim * sizeof(bf16));
            }
        }
    }

    // 4. RoPE
    std::vector<bf16> q_rope(N_q);
    std::vector<bf16> k_rope(N_kv);
    if (config_.model_type == "gemma3" || config_.model_type == "gemma4") {
        float base_freq = is_sliding ? 10000.0f : 1000000.0f;
        if (config_.model_type == "gemma3") {
            bool g3_sliding = ((layer + 1) % 6 != 0);
            base_freq = g3_sliding ? 10000.0f : 1000000.0f;
        }
        run_rope_cpu_gemma(q.data(), pos, base_freq, q_rope.data());
        run_rope_cpu_gemma(k.data(), pos, base_freq, k_rope.data());
    } else {
        run_rope_cpu_llama(q.data(), pos, q_rope.data());
        run_rope_cpu_llama(k.data(), pos, k_rope.data());
    }

    // 5. Update KV Cache
    int max_seq_len = 2048;
    int n_kv_heads = config_.num_key_value_heads;
    int h_dim = config_.head_dim;
    if (config_.model_type == "gemma4") {
        n_kv_heads = is_sliding ? 8 : 1;
        h_dim = is_sliding ? 256 : 512;
    }
    for (int h = 0; h < n_kv_heads; ++h) {
        int kv_idx = (h * max_seq_len + pos) * h_dim;
        std::memcpy(&k_caches_[layer][kv_idx], &k_rope[h * h_dim], h_dim * sizeof(bf16));
        std::memcpy(&v_caches_[layer][kv_idx], &v[h * h_dim], h_dim * sizeof(bf16));
    }

    // 6. Attention
    std::vector<bf16> attn_out(N_q);
    run_attention_host(q_rope.data(), pos, layer, attn_out.data());

    // 7. Output Projection
    int N_out = K;
    std::vector<bf16> attn_proj(N_out);
    reg_.run_gemv(N_out, N_q, lw.w_o, attn_out.data(), attn_proj.data());

    // 8. Post-attention norm and residual
    std::vector<bf16> x_post_attn(K);
    if (config_.model_type == "gemma3" || config_.model_type == "gemma4") {
        std::vector<bf16> attn_proj_normed(K);
        run_rmsnorm_cpu(attn_proj.data(), lw.post_attention_norm.empty() ? nullptr : lw.post_attention_norm.data(), attn_proj_normed.data());
        for (int i = 0; i < K; ++i) {
            x_post_attn[i] = bf16(x_bf16[i].to_float() + attn_proj_normed[i].to_float());
        }
    } else {
        for (int i = 0; i < K; ++i) {
            x_post_attn[i] = bf16(x_bf16[i].to_float() + attn_proj[i].to_float());
        }
    }

    // 9. Pre-FFN norm
    std::vector<bf16> x_norm2(K);
    run_rmsnorm_cpu(x_post_attn.data(), lw.ffn_norm.empty() ? nullptr : lw.ffn_norm.data(), x_norm2.data());

    // 10. FFN Fused NPU
    std::vector<bf16> down(K);
    std::string act_type = (config_.model_type == "gemma3" || config_.model_type == "gemma4") ? "gelu" : "silu";
    reg_.run_ffn_fused(K, config_.intermediate_size, act_type, lw.w_ffn_fused, x_norm2.data(), down.data());

    // 11. Post-FFN norm and residual
    if (config_.model_type == "gemma3" || config_.model_type == "gemma4") {
        std::vector<bf16> down_normed(K);
        run_rmsnorm_cpu(down.data(), lw.post_ffw_norm.empty() ? nullptr : lw.post_ffw_norm.data(), down_normed.data());
        for (int i = 0; i < K; ++i) {
            out_bf16[i] = bf16(x_post_attn[i].to_float() + down_normed[i].to_float());
        }
    } else {
        for (int i = 0; i < K; ++i) {
            out_bf16[i] = bf16(x_post_attn[i].to_float() + down[i].to_float());
        }
    }
}

} // namespace alveare
