#include "alveare/model.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <iomanip>

// Per-token decode profiler (ALVEARE_PROFILE_DECODE=1). Accumulates wall time by
// phase across all layers of one forward; run_layer prints + resets on the last
// layer. Coarse split: NPU dispatch (qkv/o/ffn gemv) vs host attention vs the
// remaining CPU work (rmsnorm/rope/residual/kv), to locate the 1 tok/s bottleneck.
namespace {
struct DecodeProf { double npu_qkv=0, npu_o=0, npu_ffn=0, attn=0, layer=0; int nl=0; };
static DecodeProf g_prof;
// Batched-verify profiler: gemm(qkv+o resident) vs ffn(gate/up/down streamed:
// upload+repack+compute) vs host attention vs the rest (rmsnorm/rope/geglu/kv).
struct BatchProf { double gemm=0, ffn_up=0, ffn_repack=0, ffn_cmp=0, attn=0, layer=0; int nl=0; };
static BatchProf g_bprof;
static const bool g_prof_on = (std::getenv("ALVEARE_PROFILE_DECODE") != nullptr);
using pclock = std::chrono::steady_clock;
static inline double ms_since(std::chrono::time_point<pclock> t) {
    return std::chrono::duration<double, std::milli>(pclock::now() - t).count();
}

static inline float q4_0_dot_product(const uint8_t* row, const float* x, int K) {
    const int K_blocks = K / 32;
    const int block_bytes = 20;
    float dot = 0.0f;
    for (int bk = 0; bk < K_blocks; ++bk) {
        const uint8_t* blk = row + bk * block_bytes;
        alveare::bf16 sc;
        sc.v = static_cast<uint16_t>(blk[16]) | (static_cast<uint16_t>(blk[17]) << 8);
        const float* xb = &x[bk * 32];
        float bsum = 0.0f;
        for (int j = 0; j < 16; ++j) {
            int lo = blk[j] & 0x0F; if (lo >= 8) lo -= 16;
            int hi = (blk[j] >> 4) & 0x0F; if (hi >= 8) hi -= 16;
            bsum += lo * xb[2 * j] + hi * xb[2 * j + 1];
        }
        dot += bsum * sc.to_float();
    }
    return dot;
}

static inline float q4_0_dot_product(const uint8_t* row, const alveare::bf16* x, int K) {
    const int K_blocks = K / 32;
    const int block_bytes = 20;
    float dot = 0.0f;
    for (int bk = 0; bk < K_blocks; ++bk) {
        const uint8_t* blk = row + bk * block_bytes;
        alveare::bf16 sc;
        sc.v = static_cast<uint16_t>(blk[16]) | (static_cast<uint16_t>(blk[17]) << 8);
        const alveare::bf16* xb = &x[bk * 32];
        float bsum = 0.0f;
        for (int j = 0; j < 16; ++j) {
            int lo = blk[j] & 0x0F; if (lo >= 8) lo -= 16;
            int hi = (blk[j] >> 4) & 0x0F; if (hi >= 8) hi -= 16;
            bsum += lo * xb[2 * j].to_float() + hi * xb[2 * j + 1].to_float();
        }
        dot += bsum * sc.to_float();
    }
    return dot;
}
}

namespace alveare {

Model::Model(const ModelConfig& config, const ModelWeights& weights, NpuRegistry& reg)
    : config_(config), weights_(weights), reg_(reg) {
    init_kv_caches();
    precompute_rope();
}

void Model::compute_per_layer_inputs(int token_id, const float* inpL, std::vector<float>& out_per_layer) {
    int n_embd = config_.hidden_size; // 2560
    int n_embd_per_layer = config_.per_layer_input; // 256
    int n_layer = config_.num_hidden_layers; // 42
    int total_dim = n_layer * n_embd_per_layer; // 10752

    out_per_layer.resize(total_dim);
    if (token_id < 0 || token_id >= config_.vocab_size || weights_.per_layer_token_embd_f16.empty() || weights_.per_layer_model_proj_packed.empty()) {
        return;
    }

    // 1. Projection of main token embedding (per_layer_model_proj: 10752 x 2560)
    std::vector<float> proj_scaled(total_dim);
    const float proj_scale_factor = 1.0f / std::sqrt(static_cast<float>(n_embd));
    const uint8_t* proj_base = weights_.per_layer_model_proj_packed.data();
    const int proj_row_bytes = (n_embd / 32) * 20;

    for (int r = 0; r < total_dim; ++r) {
        const uint8_t* row = proj_base + static_cast<size_t>(r) * proj_row_bytes;
        float dot = q4_0_dot_product(row, inpL, n_embd);
        proj_scaled[r] = dot * proj_scale_factor;
    }

    // RMSNorm per layer (across 256-dim embedding vector) with eps=1e-6
    std::vector<float> proj_normed(total_dim);
    for (int l = 0; l < n_layer; ++l) {
        const float* p_in = &proj_scaled[l * n_embd_per_layer];
        float variance = 0.0f;
        for (int i = 0; i < n_embd_per_layer; ++i) {
            variance += p_in[i] * p_in[i];
        }
        variance /= n_embd_per_layer;
        float inv_denom = 1.0f / std::sqrt(variance + 1e-6f);

        for (int i = 0; i < n_embd_per_layer; ++i) {
            float w = weights_.per_layer_proj_norm.empty() ? 1.0f : weights_.per_layer_proj_norm[i];
            proj_normed[l * n_embd_per_layer + i] = p_in[i] * inv_denom * w;
        }
    }

    // 2. Token Embedding Lookup (per_layer_token_embd: vocab_size x 10752, FP16)
    const float lookup_scale = std::sqrt(static_cast<float>(n_embd_per_layer)); // 16.0f
    const uint16_t* tok_ptr = &weights_.per_layer_token_embd_f16[static_cast<size_t>(token_id) * total_dim];

    // 3. Combine
    const float blend_scale = 1.0f / std::sqrt(2.0f);
    for (int i = 0; i < total_dim; ++i) {
        float emb = half_to_float(tok_ptr[i]) * lookup_scale;
        out_per_layer[i] = (proj_normed[i] + emb) * blend_scale;
    }
}

void Model::init_kv_caches() {
    int max_seq_len = 2048; // For now hardcoded or passed in config
    int n_layers = config_.num_hidden_layers;

    k_caches_.resize(n_layers);
    v_caches_.resize(n_layers);

    for (int l = 0; l < n_layers; ++l) {
        int n_kv_heads = config_.num_key_value_heads;
        int h_dim = config_.head_dim;
        
        if (config_.is_gemma4()) {
            bool is_sliding = ((l + 1) % config_.sliding_pattern_period != 0);
            n_kv_heads = (config_.model_type == "gemma4-e4b") ? config_.num_key_value_heads : (is_sliding ? config_.num_key_value_heads : 1);
            h_dim = is_sliding ? config_.head_dim : config_.head_dim_global;
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
    } else if (config_.is_gemma4()) {
        cos_sin_table_sliding_.resize(max_seq_len * config_.head_dim);
        cos_sin_table_full_.resize(max_seq_len * config_.head_dim_global);
        
        auto precompute = [&](float base, int dim, std::vector<bf16>& table) {
            for (int pos = 0; pos < max_seq_len; ++pos) {
                for (int i = 0; i < dim / 2; ++i) {
                    float inv_freq = 0.0f;
                    if (dim == config_.head_dim_global) {
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
        precompute(10000.0f, config_.head_dim, cos_sin_table_sliding_);
        precompute(1000000.0f, config_.head_dim_global, cos_sin_table_full_);
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

void Model::run_rmsnorm_cpu(const bf16* x, const float* w, bf16* out, int override_K) {
    int K = override_K > 0 ? override_K : config_.hidden_size;
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

void Model::run_rope_cpu_llama(const bf16* x, int pos, int num_heads, bf16* out) {
    int K = config_.hidden_size;
    const bf16* cos_sin = &cos_sin_table_[pos * 128];
    const bf16* cos_ptr = cos_sin;
    const bf16* sin_ptr = cos_sin + 64;

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

void Model::run_rope_cpu_gemma(const bf16* x, int pos, float base_freq, int num_heads, bf16* out) {
    int dim = (config_.is_gemma4() && base_freq > 10000.0f) ? config_.head_dim_global : config_.head_dim;
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
    
    if (config_.is_gemma4()) {
        bool is_sliding = ((layer + 1) % config_.sliding_pattern_period != 0);
        num_heads = config_.num_attention_heads;
        num_kv_heads = (config_.model_type == "gemma4-e4b") ? config_.num_key_value_heads : (is_sliding ? config_.num_key_value_heads : 1);
        dim = is_sliding ? config_.head_dim : config_.head_dim_global;
        scale = 1.0f;
        window_size = config_.sliding_window;
    }

    int seq_len = pos + 1;
    bool is_sliding_layer = (config_.model_type == "gemma3" && (layer + 1) % config_.sliding_pattern_period != 0) || 
                            (config_.is_gemma4() && (layer + 1) % config_.sliding_pattern_period != 0);

    int start_pos = 0;
    if (is_sliding_layer && seq_len > window_size) {
        start_pos = seq_len - window_size;
    }
    int W = seq_len - start_pos;
    int group_ratio = num_heads / num_kv_heads;

    int target_layer = layer;
    if (config_.shared_kv_layers > 0 && layer >= config_.num_hidden_layers - config_.shared_kv_layers) {
        int n_kv_start = config_.num_hidden_layers - config_.shared_kv_layers;
        bool is_sliding = ((layer + 1) % config_.sliding_pattern_period != 0);
        target_layer = is_sliding ? (n_kv_start - 2) : (n_kv_start - 1);
    }

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / group_ratio;

        std::vector<float> scores(W, 0.0f);
        float max_score = -1e9f;

        const bf16* q_ptr = &q_rope[h * dim];

        for (int w = 0; w < W; ++w) {
            int cache_pos = start_pos + w;
            int kv_idx = (kv_h * max_seq_len + cache_pos) * dim;
            
            float dot = 0.0f;
            const bf16* k_ptr = &k_caches_[target_layer][kv_idx];
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
            const bf16* v_ptr = &v_caches_[target_layer][kv_idx];
            
            for (int i = 0; i < dim; ++i) {
                out_f[i] += prob * v_ptr[i].to_float();
            }
        }

        for (int i = 0; i < dim; ++i) {
            out[h * dim + i] = bf16(out_f[i]);
        }
    }
}

void Model::run_layer(const bf16* x_bf16, int pos, int layer, bf16* out_bf16, const float* inp_per_layer) {
    auto t_layer = pclock::now();
    int K = config_.hidden_size;
    const LayerWeights& lw = weights_.layers[layer];
    
    int K_padded = config_.get_padded_hidden_size();

    int n_kv_start = config_.num_hidden_layers - config_.shared_kv_layers;
    bool has_kv = (config_.shared_kv_layers == 0) || (layer < n_kv_start);
    
    // 1. Input RMSNorm
    std::vector<bf16> x_norm(K_padded, bf16(0.0f));
    run_rmsnorm_cpu(x_bf16, lw.attn_norm.empty() ? nullptr : lw.attn_norm.data(), x_norm.data());

    // 2. QKV Projections (NPU)
    bool is_sliding = (config_.is_gemma4() && (layer + 1) % config_.sliding_pattern_period != 0);
    int h_dim = config_.head_dim;
    int n_q_heads = config_.num_attention_heads;
    int n_kv_heads = config_.num_key_value_heads;
    
    if (config_.is_gemma4()) {
        h_dim = is_sliding ? config_.head_dim : config_.head_dim_global;
        n_q_heads = config_.num_attention_heads;
        n_kv_heads = (config_.model_type == "gemma4-e4b") ? config_.num_key_value_heads : (is_sliding ? config_.num_key_value_heads : 1);
    }

    int N_q = n_q_heads * h_dim;
    int N_kv = n_kv_heads * h_dim;

    std::vector<bf16> q(N_q);
    std::vector<bf16> k(N_kv);
    std::vector<bf16> v(N_kv);

    auto t_qkv = pclock::now();
    if (has_kv) {
        if (lw.w_qkv != kInvalidWeight) {
            std::vector<bf16> qkv(lw.n_qkv);
            reg_.run_gemv(lw.n_qkv, K_padded, lw.w_qkv, x_norm.data(), qkv.data());
            std::memcpy(q.data(), qkv.data(), size_t(N_q) * sizeof(bf16));
            std::memcpy(k.data(), qkv.data() + N_q, size_t(N_kv) * sizeof(bf16));
            if (is_sliding)
                std::memcpy(v.data(), qkv.data() + N_q + N_kv, size_t(N_kv) * sizeof(bf16));
            else
                v = k; // Gemma4 global layers use K for V
        } else {
            reg_.run_gemv(N_q, K_padded, lw.w_q, x_norm.data(), q.data());
            reg_.run_gemv(N_kv, K_padded, lw.w_k, x_norm.data(), k.data());
            if (!config_.is_gemma4() || is_sliding) {
                reg_.run_gemv(N_kv, K_padded, lw.w_v, x_norm.data(), v.data());
            } else {
                v = k; // Gemma4 global layers use K for V
            }
        }
    } else {
        // Layers 24-41 (shared KV) only project Q
        if (lw.w_qkv != kInvalidWeight) {
            std::vector<bf16> qkv(lw.n_qkv);
            reg_.run_gemv(lw.n_qkv, K_padded, lw.w_qkv, x_norm.data(), qkv.data());
            std::memcpy(q.data(), qkv.data(), size_t(N_q) * sizeof(bf16));
        } else {
            reg_.run_gemv(N_q, K_padded, lw.w_q, x_norm.data(), q.data());
        }
    }
    g_prof.npu_qkv += ms_since(t_qkv);

    // 3. QK-Norm & V-Norm (Gemma only)
    if (config_.model_type == "gemma3" || config_.is_gemma4()) {
        for (int h = 0; h < n_q_heads; ++h) {
            int h_dim = config_.head_dim;
            if (config_.is_gemma4()) h_dim = is_sliding ? config_.head_dim : config_.head_dim_global;
            std::vector<bf16> q_h(h_dim);
            run_rmsnorm_cpu(&q[h * h_dim], lw.q_norm.empty() ? nullptr : lw.q_norm.data(), q_h.data(), h_dim);
            std::memcpy(&q[h * h_dim], q_h.data(), h_dim * sizeof(bf16));
        }
        if (has_kv) {
            for (int h = 0; h < n_kv_heads; ++h) {
                int h_dim = config_.head_dim;
                if (config_.is_gemma4()) h_dim = is_sliding ? config_.head_dim : config_.head_dim_global;
                std::vector<bf16> k_h(h_dim);
                run_rmsnorm_cpu(&k[h * h_dim], lw.k_norm.empty() ? nullptr : lw.k_norm.data(), k_h.data(), h_dim);
                std::memcpy(&k[h * h_dim], k_h.data(), h_dim * sizeof(bf16));
            }
            if (config_.is_gemma4()) {
                for (int h = 0; h < n_kv_heads; ++h) {
                    int h_dim = is_sliding ? config_.head_dim : config_.head_dim_global;
                    std::vector<bf16> v_h(h_dim);
                    run_rmsnorm_cpu(&v[h * h_dim], nullptr, v_h.data(), h_dim);
                    std::memcpy(&v[h * h_dim], v_h.data(), h_dim * sizeof(bf16));
                }
            }
        }
    }

    // 4. RoPE
    std::vector<bf16> q_rope(N_q);
    std::vector<bf16> k_rope(N_kv);
    if (config_.model_type == "gemma3" || config_.is_gemma4()) {
        float base_freq = is_sliding ? 10000.0f : 1000000.0f;
        if (config_.model_type == "gemma3") {
            bool g3_sliding = ((layer + 1) % config_.sliding_pattern_period != 0);
            base_freq = g3_sliding ? 10000.0f : 1000000.0f;
        }
        run_rope_cpu_gemma(q.data(), pos, base_freq, n_q_heads, q_rope.data());
        if (has_kv) {
            run_rope_cpu_gemma(k.data(), pos, base_freq, n_kv_heads, k_rope.data());
        }
    } else {
        run_rope_cpu_llama(q.data(), pos, n_q_heads, q_rope.data());
        if (has_kv) {
            run_rope_cpu_llama(k.data(), pos, n_kv_heads, k_rope.data());
        }
    }

    // 5. Update KV Cache (only for layers that own KV states)
    if (has_kv) {
        int max_seq_len = 2048;
        h_dim = config_.head_dim;
        if (config_.is_gemma4()) {
            h_dim = is_sliding ? config_.head_dim : config_.head_dim_global;
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            int kv_idx = (h * max_seq_len + pos) * h_dim;
            std::memcpy(&k_caches_[layer][kv_idx], &k_rope[h * h_dim], h_dim * sizeof(bf16));
            std::memcpy(&v_caches_[layer][kv_idx], &v[h * h_dim], h_dim * sizeof(bf16));
        }
    }

    // 6. Attention
    std::vector<bf16> attn_out(N_q);
    auto t_attn = pclock::now();
    run_attention_host(q_rope.data(), pos, layer, attn_out.data());
    g_prof.attn += ms_since(t_attn);

    // 7. Output Projection
    int N_out = K;
    int N_out_padded = config_.get_padded_hidden_size();
    int o_n = lw.o_gemv_n > 0 ? lw.o_gemv_n : N_out_padded;
    std::vector<bf16> attn_proj(o_n, bf16(0.0f));
    auto t_o = pclock::now();
    reg_.run_gemv(o_n, N_q, lw.w_o, attn_out.data(), attn_proj.data());
    g_prof.npu_o += ms_since(t_o);

    // 8. Post-attention norm and residual
    std::vector<bf16> x_post_attn(K);
    if (config_.model_type == "gemma3" || config_.is_gemma4()) {
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
    std::vector<bf16> x_norm2(K_padded, bf16(0.0f));
    run_rmsnorm_cpu(x_post_attn.data(), lw.ffn_norm.empty() ? nullptr : lw.ffn_norm.data(), x_norm2.data());

    // 10. FFN (Fused NPU or CPU/q4_0 fallback)
    int H_padded = config_.get_padded_hidden_size();
    int I_padded = config_.get_padded_intermediate_size();
    std::vector<bf16> down(H_padded, bf16(0.0f));
    std::string act_type = (config_.model_type == "gemma3" || config_.is_gemma4()) ? "gelu" : "silu";
    auto t_ffn = pclock::now();
    if (lw.w_ffn_fused != kInvalidWeight) {
        reg_.run_ffn_fused(H_padded, I_padded, act_type, lw.w_ffn_fused, x_norm2.data(), down.data());
    } else {
        int gate_stride = (K_padded / 32) * 20;
        int down_stride = (I_padded / 32) * 20;
        int intermediate_size = config_.intermediate_size;
        std::vector<float> geglu(I_padded, 0.0f);
        std::vector<float> x_norm2_f(K_padded);
        for (int i = 0; i < K_padded; ++i) x_norm2_f[i] = x_norm2[i].to_float();

        #pragma omp parallel for schedule(static)
        for (int r = 0; r < intermediate_size; ++r) {
            const uint8_t* row_g = lw.ffn_gate_bytes.data() + static_cast<size_t>(r) * gate_stride;
            const uint8_t* row_u = lw.ffn_up_bytes.data() + static_cast<size_t>(r) * gate_stride;
            float g = q4_0_dot_product(row_g, x_norm2_f.data(), K_padded);
            float u = q4_0_dot_product(row_u, x_norm2_f.data(), K_padded);
            float a = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f));
            geglu[r] = a * u;
        }

        #pragma omp parallel for schedule(static)
        for (int r = 0; r < K; ++r) {
            const uint8_t* row_d = lw.ffn_down_bytes.data() + static_cast<size_t>(r) * down_stride;
            float d = q4_0_dot_product(row_d, geglu.data(), I_padded);
            down[r] = bf16(d);
        }
    }
    g_prof.npu_ffn += ms_since(t_ffn);

    // 11. Post-FFN norm and residual
    std::vector<bf16> x_post_ffn(K);
    if (config_.model_type == "gemma3" || config_.is_gemma4()) {
        std::vector<bf16> down_normed(K);
        run_rmsnorm_cpu(down.data(), lw.post_ffw_norm.empty() ? nullptr : lw.post_ffw_norm.data(), down_normed.data());
        for (int i = 0; i < K; ++i) {
            x_post_ffn[i] = bf16(x_post_attn[i].to_float() + down_normed[i].to_float());
        }
    } else {
        for (int i = 0; i < K; ++i) {
            x_post_ffn[i] = bf16(x_post_attn[i].to_float() + down[i].to_float());
        }
    }

    // 12. PLE Injection (Gemma-4-E4B)
    if (config_.per_layer_input > 0 && inp_per_layer && !lw.inp_gate_bytes.empty()) {
        int n_ple = config_.per_layer_input; // 256
        const float* h_l = inp_per_layer + static_cast<size_t>(layer) * n_ple;

        // Gate: GELU(inp_gate * x_post_ffn)
        std::vector<float> z(n_ple);
        std::vector<float> x_post_ffn_f(K);
        for (int i = 0; i < K; ++i) x_post_ffn_f[i] = x_post_ffn[i].to_float();
        const uint8_t* gate_base = lw.inp_gate_bytes.data();
        const int gate_row_bytes = (K / 32) * 20;
        for (int r = 0; r < n_ple; ++r) {
            const uint8_t* row = gate_base + static_cast<size_t>(r) * gate_row_bytes;
            float g = q4_0_dot_product(row, x_post_ffn_f.data(), K);
            float a = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f));
            z[r] = a * h_l[r];
        }

        // Proj: proj * z
        std::vector<bf16> p(K);
        const uint8_t* proj_base = lw.proj_bytes.data();
        const int proj_row_bytes = (n_ple / 32) * 20;
        for (int r = 0; r < K; ++r) {
            const uint8_t* row = proj_base + static_cast<size_t>(r) * proj_row_bytes;
            float dot = q4_0_dot_product(row, z.data(), n_ple);
            p[r] = bf16(dot);
        }

        // Post norm: RMSNorm(p, lw.post_norm)
        std::vector<bf16> y(K);
        run_rmsnorm_cpu(p.data(), lw.post_norm.empty() ? nullptr : lw.post_norm.data(), y.data(), K);

        // Residual: x_post_ffn += y
        for (int i = 0; i < K; ++i) {
            x_post_ffn[i] = bf16(x_post_ffn[i].to_float() + y[i].to_float());
        }
    }

    // 13. Layer output scale
    float oscale = (config_.is_gemma4()) ? lw.output_scale : 1.0f;
    for (int i = 0; i < K; ++i) {
        out_bf16[i] = bf16(x_post_ffn[i].to_float() * oscale);
    }

    // Profiling: accumulate this layer, print + reset on the last layer.
    if (g_prof_on) {
        g_prof.layer += ms_since(t_layer);
        g_prof.nl++;
        if (layer == config_.num_hidden_layers - 1) {
            double npu = g_prof.npu_qkv + g_prof.npu_o + g_prof.npu_ffn;
            double cpu_rest = g_prof.layer - npu - g_prof.attn;
            std::cerr << std::fixed << std::setprecision(1)
                << "[decode-prof] layers=" << g_prof.nl
                << " total=" << g_prof.layer << "ms"
                << " | NPU=" << npu << " (qkv=" << g_prof.npu_qkv
                << " o=" << g_prof.npu_o << " ffn=" << g_prof.npu_ffn << ")"
                << " | attn(cpu)=" << g_prof.attn
                << " | cpu_rest=" << cpu_rest << "\n" << std::flush;
            g_prof = DecodeProf();
        }
    }
}

void Model::run_layer_batch(const bf16* x_batch, int nrows, int pos_start,
                            int layer, bf16* out_batch, const float* inp_per_layer) {
    auto t_blayer = pclock::now();
    const int K = config_.hidden_size;
    const int K_padded = config_.get_padded_hidden_size();
    const int B = (nrows <= 8) ? 8 : 16;
    const LayerWeights& lw = weights_.layers[layer];

    const bool is_sliding = ((layer + 1) % config_.sliding_pattern_period != 0);
    const int h_dim = is_sliding ? config_.head_dim : config_.head_dim_global;
    const int n_heads = config_.num_attention_heads;
    const int n_kv_heads = (config_.model_type == "gemma4-e4b") ? config_.num_key_value_heads : (is_sliding ? config_.num_key_value_heads : 1);
    const int N_q = n_heads * h_dim;
    const int N_kv = n_kv_heads * h_dim;

    const float* attn_norm_w = lw.attn_norm.empty() ? nullptr : lw.attn_norm.data();
    const float* q_norm_w = lw.q_norm.empty() ? nullptr : lw.q_norm.data();
    const float* k_norm_w = lw.k_norm.empty() ? nullptr : lw.k_norm.data();
    const float* post_attn_w = lw.post_attention_norm.empty() ? nullptr : lw.post_attention_norm.data();
    const float* ffn_norm_w = lw.ffn_norm.empty() ? nullptr : lw.ffn_norm.data();
    const float* post_ffw_w = lw.post_ffw_norm.empty() ? nullptr : lw.post_ffw_norm.data();

    const int n_kv_start = config_.num_hidden_layers - config_.shared_kv_layers;
    const bool has_kv = (config_.shared_kv_layers == 0) || (layer < n_kv_start);

    // 1. Input RMSNorm, padded to K_padded (unused rows/cols stay zero).
    std::vector<bf16> x_norm_pad(size_t(B) * K_padded, bf16(0.0f));
    for (int b = 0; b < nrows; ++b)
        run_rmsnorm_cpu(&x_batch[size_t(b) * K], attn_norm_w, &x_norm_pad[size_t(b) * K_padded], K);

    // 2. Fused Q/K/V projection (one batched GEMM), then slice — matches the
    // decode path's w_qkv (q++k++v sliding, q++k global with v==k).
    std::vector<bf16> q(size_t(B) * N_q), k(size_t(B) * N_kv), v(size_t(B) * N_kv);
    std::vector<bf16> qkv(size_t(B) * lw.n_qkv);
    auto t_bqkv = pclock::now();
    reg_.run_gemm(B, lw.n_qkv, K_padded, lw.w_qkv, x_norm_pad.data(), qkv.data());
    g_bprof.gemm += ms_since(t_bqkv);
    for (int b = 0; b < nrows; ++b) {
        const bf16* row = &qkv[size_t(b) * lw.n_qkv];
        std::memcpy(&q[size_t(b) * N_q], row, size_t(N_q) * sizeof(bf16));
        if (has_kv) {
            std::memcpy(&k[size_t(b) * N_kv], row + N_q, size_t(N_kv) * sizeof(bf16));
            if (is_sliding)
                std::memcpy(&v[size_t(b) * N_kv], row + N_q + N_kv, size_t(N_kv) * sizeof(bf16));
        }
    }
    if (has_kv && !is_sliding)
        v = k;  // gemma4 global layers reuse K for V (before per-head norm)

    // 3. QK-norm and V-norm (per head, per row).
    std::vector<bf16> tmp(h_dim);
    for (int b = 0; b < nrows; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            run_rmsnorm_cpu(&q[size_t(b) * N_q + h * h_dim], q_norm_w, tmp.data(), h_dim);
            std::memcpy(&q[size_t(b) * N_q + h * h_dim], tmp.data(), h_dim * sizeof(bf16));
        }
        if (has_kv) {
            for (int h = 0; h < n_kv_heads; ++h) {
                run_rmsnorm_cpu(&k[size_t(b) * N_kv + h * h_dim], k_norm_w, tmp.data(), h_dim);
                std::memcpy(&k[size_t(b) * N_kv + h * h_dim], tmp.data(), h_dim * sizeof(bf16));
            }
            for (int h = 0; h < n_kv_heads; ++h) {
                run_rmsnorm_cpu(&v[size_t(b) * N_kv + h * h_dim], nullptr, tmp.data(), h_dim);
                std::memcpy(&v[size_t(b) * N_kv + h * h_dim], tmp.data(), h_dim * sizeof(bf16));
            }
        }
    }

    // 4. RoPE (q, k).
    const float base_freq = is_sliding ? 10000.0f : 1000000.0f;
    std::vector<bf16> q_rope(size_t(B) * N_q), k_rope(size_t(B) * N_kv);
    for (int b = 0; b < nrows; ++b) {
        run_rope_cpu_gemma(&q[size_t(b) * N_q], pos_start + b, base_freq, n_heads, &q_rope[size_t(b) * N_q]);
        if (has_kv) {
            run_rope_cpu_gemma(&k[size_t(b) * N_kv], pos_start + b, base_freq, n_kv_heads, &k_rope[size_t(b) * N_kv]);
        }
    }

    // 5. Write KV cache (only for layers that own KV states)
    if (has_kv) {
        const int max_seq_len = 2048;
        for (int b = 0; b < nrows; ++b) {
            for (int h = 0; h < n_kv_heads; ++h) {
                int kv_idx = (h * max_seq_len + pos_start + b) * h_dim;
                std::memcpy(&k_caches_[layer][kv_idx], &k_rope[size_t(b) * N_kv + h * h_dim], h_dim * sizeof(bf16));
                std::memcpy(&v_caches_[layer][kv_idx], &v[size_t(b) * N_kv + h * h_dim], h_dim * sizeof(bf16));
            }
        }
    }

    // 6. Attention (per row, causal against the cache).
    std::vector<bf16> attn_out(size_t(B) * N_q, bf16(0.0f));
    auto t_battn = pclock::now();
    for (int b = 0; b < nrows; ++b)
        run_attention_host(&q_rope[size_t(b) * N_q], pos_start + b, layer, &attn_out[size_t(b) * N_q]);
    g_bprof.attn += ms_since(t_battn);

    // 7. Output projection (batched GEMM). w_o may be output-padded to o_gemv_n
    // (gemma4 sliding shares the w_qkv kernel shape); only the first K columns
    // of each row are the real projection.
    const int o_n = lw.o_gemv_n > 0 ? lw.o_gemv_n : K_padded;
    std::vector<bf16> attn_proj(size_t(B) * o_n);
    auto t_bo = pclock::now();
    reg_.run_gemm(B, o_n, N_q, lw.w_o, attn_out.data(), attn_proj.data());
    g_bprof.gemm += ms_since(t_bo);

    // 8. Post-attention norm + residual.
    std::vector<bf16> x_post_attn(size_t(nrows) * K);
    std::vector<bf16> normed(K);
    for (int b = 0; b < nrows; ++b) {
        run_rmsnorm_cpu(&attn_proj[size_t(b) * o_n], post_attn_w, normed.data(), K);
        for (int i = 0; i < K; ++i)
            x_post_attn[size_t(b) * K + i] = bf16(x_batch[size_t(b) * K + i].to_float() + normed[i].to_float());
    }

    // 9. Pre-FFN norm, padded to K_padded.
    std::vector<bf16> x_norm2_pad(size_t(B) * K_padded, bf16(0.0f));
    for (int b = 0; b < nrows; ++b)
        run_rmsnorm_cpu(&x_post_attn[size_t(b) * K], ffn_norm_w, &x_norm2_pad[size_t(b) * K_padded], K);

    // 10. FFN: gate/up GEMM (streamed) -> GeGLU (CPU) -> down GEMM.
    const int I = config_.get_padded_intermediate_size();
    const int I_real = config_.intermediate_size;
    std::vector<bf16> gate(size_t(B) * I), up(size_t(B) * I);
    auto t_bgu = pclock::now();
    reg_.run_gemm_streamed(B, I, K_padded, lw.ffn_gate_bytes.data(), lw.ffn_gate_bytes.size(),
                           x_norm2_pad.data(), gate.data());
    reg_.run_gemm_streamed(B, I, K_padded, lw.ffn_up_bytes.data(), lw.ffn_up_bytes.size(),
                           x_norm2_pad.data(), up.data());
    g_bprof.ffn_cmp += ms_since(t_bgu);

    std::vector<bf16> geglu(size_t(B) * I, bf16(0.0f));
    const float kGeluC = std::sqrt(2.0f / static_cast<float>(M_PI));
    for (int b = 0; b < nrows; ++b) {
        for (int i = 0; i < I_real; ++i) {
            float g = gate[size_t(b) * I + i].to_float();
            float gelu = 0.5f * g * (1.0f + std::tanh(kGeluC * (g + 0.044715f * g * g * g)));
            geglu[size_t(b) * I + i] = bf16(gelu * up[size_t(b) * I + i].to_float());
        }
    }

    // down: GEMM output is Nd = K_padded. If I > 8192, split K into chunks of 8192.
    const int Nd = K_padded;
    const int chunkK = (I > 8192) ? 8192 : I;
    const int num_chunks = (I + chunkK - 1) / chunkK;
    const int row_bytes = (I / 32) * 20;
    const int chunk_row_bytes = (chunkK / 32) * 20;
    std::vector<float> down_acc(size_t(nrows) * K, 0.0f);
    std::vector<uint8_t> wchunk(size_t(Nd) * chunk_row_bytes);
    std::vector<bf16> xchunk(size_t(B) * chunkK), ychunk(size_t(B) * Nd);
    for (int c = 0; c < num_chunks; ++c) {
        auto t_rep = pclock::now();
        for (int r = 0; r < Nd; ++r)
            std::memcpy(&wchunk[size_t(r) * chunk_row_bytes],
                        &lw.ffn_down_bytes[size_t(r) * row_bytes + size_t(c) * chunk_row_bytes],
                        chunk_row_bytes);
        for (int b = 0; b < B; ++b)
            for (int i = 0; i < chunkK; ++i)
                xchunk[size_t(b) * chunkK + i] = geglu[size_t(b) * I + size_t(c) * chunkK + i];
        g_bprof.ffn_repack += ms_since(t_rep);
        auto t_dn = pclock::now();
        reg_.run_gemm_streamed(B, Nd, chunkK, wchunk.data(), wchunk.size(),
                               xchunk.data(), ychunk.data());
        g_bprof.ffn_cmp += ms_since(t_dn);
        for (int b = 0; b < nrows; ++b)
            for (int i = 0; i < K; ++i)
                down_acc[size_t(b) * K + i] += ychunk[size_t(b) * Nd + i].to_float();
    }

    // 11. Post-FFN norm + residual + per-layer output scale.
    const float oscale = lw.output_scale;
    std::vector<bf16> down_bf(K), down_normed(K);
    for (int b = 0; b < nrows; ++b) {
        std::vector<bf16> x_post_ffn(K);
        for (int i = 0; i < K; ++i) down_bf[i] = bf16(down_acc[size_t(b) * K + i]);
        run_rmsnorm_cpu(down_bf.data(), post_ffw_w, down_normed.data(), K);
        for (int i = 0; i < K; ++i) {
            x_post_ffn[i] = bf16(x_post_attn[size_t(b) * K + i].to_float() + down_normed[i].to_float());
        }

        // PLE Injection per row if enabled
        if (config_.per_layer_input > 0 && inp_per_layer && !lw.inp_gate_bytes.empty()) {
            int n_ple = config_.per_layer_input;
            const float* h_l = inp_per_layer + (static_cast<size_t>(b) * config_.num_hidden_layers + layer) * n_ple;
            std::vector<float> z(n_ple);
            const uint8_t* gate_base = lw.inp_gate_bytes.data();
            const int gate_row_bytes = (K / 32) * 20;
            for (int r = 0; r < n_ple; ++r) {
                const uint8_t* row = gate_base + static_cast<size_t>(r) * gate_row_bytes;
                float g = q4_0_dot_product(row, x_post_ffn.data(), K);
                float a = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f));
                z[r] = a * h_l[r];
            }
            std::vector<bf16> p(K);
            const uint8_t* proj_base = lw.proj_bytes.data();
            const int proj_row_bytes = (n_ple / 32) * 20;
            for (int r = 0; r < K; ++r) {
                const uint8_t* row = proj_base + static_cast<size_t>(r) * proj_row_bytes;
                float dot = q4_0_dot_product(row, z.data(), n_ple);
                p[r] = bf16(dot);
            }
            std::vector<bf16> y(K);
            run_rmsnorm_cpu(p.data(), lw.post_norm.empty() ? nullptr : lw.post_norm.data(), y.data(), K);
            for (int i = 0; i < K; ++i) {
                x_post_ffn[i] = bf16(x_post_ffn[i].to_float() + y[i].to_float());
            }
        }

        for (int i = 0; i < K; ++i) {
            out_batch[size_t(b) * K + i] = bf16(x_post_ffn[i].to_float() * oscale);
        }
    }

    if (g_prof_on) {
        g_bprof.layer += ms_since(t_blayer);
        g_bprof.nl++;
        if (layer == config_.num_hidden_layers - 1) {
            double acc = g_bprof.gemm + g_bprof.ffn_cmp + g_bprof.ffn_repack + g_bprof.attn;
            double rest = g_bprof.layer - acc;
            std::cerr << std::fixed << std::setprecision(1)
                << "[verify-prof] layers=" << g_bprof.nl << " B=" << B
                << " total=" << g_bprof.layer << "ms"
                << " | gemm(qkv+o)=" << g_bprof.gemm
                << " | ffn_cmp=" << g_bprof.ffn_cmp
                << " | ffn_repack=" << g_bprof.ffn_repack
                << " | attn=" << g_bprof.attn
                << " | rest=" << rest << "\n" << std::flush;
            g_bprof = BatchProf();
        }
    }
}

} // namespace alveare
