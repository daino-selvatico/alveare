#include "alveare/model.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <immintrin.h>

// Per-token decode profiler (ALVEARE_PROFILE_DECODE=1). Accumulates wall time by
// phase across all layers of one forward; run_layer prints + resets on the last
// layer. Coarse split: NPU dispatch (qkv/o/ffn gemv) vs host attention vs the
// remaining CPU work (rmsnorm/rope/residual/kv), to locate the 1 tok/s bottleneck.
namespace {
struct DecodeProf { double npu_qkv=0, npu_o=0, npu_ffn=0, attn=0, layer=0, ple=0; int nl=0; };
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

#ifdef __AVX2__
static inline float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128  = _mm_add_ps(vlow, vhigh);
    __m128 shuf  = _mm_movehdup_ps(v128);
    __m128 sums  = _mm_add_ps(v128, shuf);
    shuf         = _mm_movehl_ps(shuf, sums);
    __m128 res   = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(res);
}

static inline float q4_0_dot_product(const uint8_t* row, const float* x, int K) {
    const int K_blocks = K / 32;
    const int block_bytes = 20;
    float dot = 0.0f;
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i val_7   = _mm_set1_epi8(7);
    const __m128i val_16  = _mm_set1_epi8(16);

    for (int bk = 0; bk < K_blocks; ++bk) {
        const uint8_t* blk = row + bk * block_bytes;
        alveare::bf16 sc;
        sc.v = static_cast<uint16_t>(blk[16]) | (static_cast<uint16_t>(blk[17]) << 8);
        const float scale = sc.to_float();

        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk));
        __m128i lo = _mm_and_si128(raw, mask_0f);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

        __m128i lo_hi_0 = _mm_unpacklo_epi8(lo, hi);
        __m128i lo_hi_1 = _mm_unpackhi_epi8(lo, hi);

        __m128i sub16_0 = _mm_and_si128(_mm_cmpgt_epi8(lo_hi_0, val_7), val_16);
        __m128i s8_0 = _mm_sub_epi8(lo_hi_0, sub16_0);

        __m128i sub16_1 = _mm_and_si128(_mm_cmpgt_epi8(lo_hi_1, val_7), val_16);
        __m128i s8_1 = _mm_sub_epi8(lo_hi_1, sub16_1);

        __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s8_0));
        __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(s8_0, 8)));
        __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s8_1));
        __m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(s8_1, 8)));

        const float* xb = &x[bk * 32];
        __m256 x0 = _mm256_loadu_ps(xb + 0);
        __m256 x1 = _mm256_loadu_ps(xb + 8);
        __m256 x2 = _mm256_loadu_ps(xb + 16);
        __m256 x3 = _mm256_loadu_ps(xb + 24);

        __m256 acc = _mm256_mul_ps(q0, x0);
        acc = _mm256_fmadd_ps(q1, x1, acc);
        acc = _mm256_fmadd_ps(q2, x2, acc);
        acc = _mm256_fmadd_ps(q3, x3, acc);

        float bsum = hsum256_ps_avx(acc);
        dot += bsum * scale;
    }
    return dot;
}

static inline float q4_0_dot_product(const uint8_t* row, const alveare::bf16* x, int K) {
    const int K_blocks = K / 32;
    const int block_bytes = 20;
    float dot = 0.0f;
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m128i val_7   = _mm_set1_epi8(7);
    const __m128i val_16  = _mm_set1_epi8(16);

    for (int bk = 0; bk < K_blocks; ++bk) {
        const uint8_t* blk = row + bk * block_bytes;
        alveare::bf16 sc;
        sc.v = static_cast<uint16_t>(blk[16]) | (static_cast<uint16_t>(blk[17]) << 8);
        const float scale = sc.to_float();

        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk));
        __m128i lo = _mm_and_si128(raw, mask_0f);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

        __m128i lo_hi_0 = _mm_unpacklo_epi8(lo, hi);
        __m128i lo_hi_1 = _mm_unpackhi_epi8(lo, hi);

        __m128i sub16_0 = _mm_and_si128(_mm_cmpgt_epi8(lo_hi_0, val_7), val_16);
        __m128i s8_0 = _mm_sub_epi8(lo_hi_0, sub16_0);

        __m128i sub16_1 = _mm_and_si128(_mm_cmpgt_epi8(lo_hi_1, val_7), val_16);
        __m128i s8_1 = _mm_sub_epi8(lo_hi_1, sub16_1);

        __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s8_0));
        __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(s8_0, 8)));
        __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(s8_1));
        __m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(s8_1, 8)));

        const alveare::bf16* xb = &x[bk * 32];
        const __m128i* x128 = reinterpret_cast<const __m128i*>(xb);
        __m256 x0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(x128 + 0)), 16));
        __m256 x1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(x128 + 1)), 16));
        __m256 x2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(x128 + 2)), 16));
        __m256 x3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(x128 + 3)), 16));

        __m256 acc = _mm256_mul_ps(q0, x0);
        acc = _mm256_fmadd_ps(q1, x1, acc);
        acc = _mm256_fmadd_ps(q2, x2, acc);
        acc = _mm256_fmadd_ps(q3, x3, acc);

        float bsum = hsum256_ps_avx(acc);
        dot += bsum * scale;
    }
    return dot;
}
#else
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
#endif
}

namespace alveare {

Model::Model(const ModelConfig& config, const ModelWeights& weights, NpuRegistry& reg)
    : config_(config), weights_(weights), reg_(reg) {
    init_kv_caches();
    init_scratch();
    precompute_rope();
}

void Model::compute_per_layer_inputs(int token_id, const float* inpL, float* out_per_layer) {
    if (!out_per_layer) return;
    int n_embd = config_.hidden_size; // 2560
    int n_embd_per_layer = config_.per_layer_input; // 256
    int n_layer = config_.num_hidden_layers; // 42
    int total_dim = n_layer * n_embd_per_layer; // 10752

    if (weights_.per_layer_model_proj_packed.empty()) {
        std::memset(out_per_layer, 0, size_t(total_dim) * sizeof(float));
        return;
    }

    // 1. Projection of main token embedding (per_layer_model_proj: 10752 x 2560)
    float* proj_scaled = scratch_.proj_scaled.data();
    const float proj_scale_factor = 1.0f / std::sqrt(static_cast<float>(n_embd));
    const uint8_t* proj_base = weights_.per_layer_model_proj_packed.data();
    const int proj_row_bytes = (n_embd / 32) * 20;

    // Parallelize the big per-layer-projection (10752 x 2560 Q4 dots).
    // Uses fast AVX2 dot product across OpenMP threads.
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < total_dim; ++r) {
        const uint8_t* row = proj_base + static_cast<size_t>(r) * proj_row_bytes;
        float dot = q4_0_dot_product(row, inpL, n_embd);
        proj_scaled[r] = dot * proj_scale_factor;
    }

    // RMSNorm per layer (across 256-dim embedding vector) with eps=1e-6
    float* proj_normed = scratch_.proj_normed.data();
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
    // For raw/multimodal embeddings (token_id < 0 or audio/image tokens 258881/259000),
    // token embedding lookup is zero.
    const bool is_custom = (token_id < 0 || token_id == 258881 || token_id == 259000 ||
                            token_id >= config_.vocab_size || weights_.per_layer_token_embd_f16.empty());
    const float lookup_scale = std::sqrt(static_cast<float>(n_embd_per_layer)); // 16.0f
    const uint16_t* tok_ptr = is_custom ? nullptr : &weights_.per_layer_token_embd_f16[static_cast<size_t>(token_id) * total_dim];

    // 3. Combine
    const float blend_scale = 1.0f / std::sqrt(2.0f);
    for (int i = 0; i < total_dim; ++i) {
        float emb = tok_ptr ? (half_to_float(tok_ptr[i]) * lookup_scale) : 0.0f;
        out_per_layer[i] = (proj_normed[i] + emb) * blend_scale;
    }
}

void Model::compute_per_layer_inputs(int token_id, const float* inpL, std::vector<float>& out_per_layer) {
    int total_dim = config_.num_hidden_layers * config_.per_layer_input;
    out_per_layer.resize(total_dim);
    compute_per_layer_inputs(token_id, inpL, out_per_layer.data());
}

void Model::init_kv_caches() {
    int max_seq_len = config_.max_position_embeddings;
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

void Model::init_scratch() {
    int H_padded = config_.get_padded_hidden_size();
    int I_padded = config_.get_padded_intermediate_size();
    int max_N_q = std::max(config_.num_attention_heads * config_.head_dim_global,
                           config_.num_attention_heads * config_.head_dim);
    int max_N_kv = std::max(config_.num_key_value_heads * config_.head_dim_global,
                            config_.num_key_value_heads * config_.head_dim);
    int max_N_qkv = max_N_q + 2 * max_N_kv;
    int max_N = std::max({H_padded, max_N_qkv, 8192});
    int max_I = std::max(I_padded, 16384);

    scratch_.x_norm.assign(max_N, bf16(0.0f));
    scratch_.q.assign(max_N, bf16(0.0f));
    scratch_.k.assign(max_N, bf16(0.0f));
    scratch_.v.assign(max_N, bf16(0.0f));
    scratch_.qkv.assign(max_N * 2, bf16(0.0f));
    scratch_.q_buf.assign(max_N, bf16(0.0f));
    scratch_.k_buf.assign(max_N, bf16(0.0f));
    scratch_.v_buf.assign(max_N, bf16(0.0f));
    int max_h_dim = std::max(config_.head_dim_global, config_.head_dim);
    if (max_h_dim == 0) max_h_dim = 256;
    scratch_.q_h.assign(max_h_dim, bf16(0.0f));
    scratch_.k_h.assign(max_h_dim, bf16(0.0f));
    scratch_.v_h.assign(max_h_dim, bf16(0.0f));
    scratch_.q_rope.assign(max_N, bf16(0.0f));
    scratch_.k_rope.assign(max_N, bf16(0.0f));
    scratch_.attn_out.assign(max_N, bf16(0.0f));
    scratch_.attn_out_padded.assign(max_N, bf16(0.0f));
    scratch_.attn_proj.assign(max_N * 2, bf16(0.0f));
    scratch_.attn_proj_normed.assign(max_N, bf16(0.0f));
    scratch_.x_post_attn.assign(max_N, bf16(0.0f));
    scratch_.x_norm2.assign(max_N, bf16(0.0f));
    scratch_.down.assign(max_N, bf16(0.0f));
    scratch_.gu.assign(max_I * 4, bf16(0.0f));
    scratch_.act.assign(max_I * 2, bf16(0.0f));
    scratch_.acc_f.assign(max_N, 0.0f);
    scratch_.part_bf16.assign(max_N, bf16(0.0f));
    scratch_.chunk_in.assign(max_N, bf16(0.0f));
    scratch_.geglu.assign(max_I * 2, 0.0f);
    scratch_.x_norm2_f.assign(max_N, 0.0f);
    scratch_.down_normed.assign(max_N, bf16(0.0f));
    scratch_.x_post_ffn.assign(max_N, bf16(0.0f));
    scratch_.ple_z.assign(std::max(config_.per_layer_input, 256), 0.0f);
    scratch_.ple_x_f.assign(max_N, 0.0f);
    scratch_.ple_p.assign(max_N, bf16(0.0f));
    scratch_.ple_y.assign(max_N, bf16(0.0f));
    int total_ple_dim = config_.num_hidden_layers * config_.per_layer_input;
    if (total_ple_dim > 0) {
        scratch_.proj_scaled.assign(total_ple_dim, 0.0f);
        scratch_.proj_normed.assign(total_ple_dim, 0.0f);
    }
}

void Model::reset_caches() {
    for (size_t l = 0; l < k_caches_.size(); ++l) {
        std::fill(k_caches_[l].begin(), k_caches_[l].end(), bf16(0.0f));
        std::fill(v_caches_[l].begin(), v_caches_[l].end(), bf16(0.0f));
    }
}

void Model::precompute_rope() {
    int max_seq_len = config_.max_position_embeddings;
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
        
        // Gemma-4 applies FULL RoPE to every head dim (n_rot == head_dim): llama.cpp
        // sets n_rot_full = rope.dimension_count = 512 and ASSERTS it equals the head
        // dim (llama-model.cpp ~L816-822). FULL RoPE is required for all Gemma-4 models.
        auto precompute = [&](float base, int dim, std::vector<bf16>& table) {
            for (int pos = 0; pos < max_seq_len; ++pos) {
                for (int i = 0; i < dim / 2; ++i) {
                    float inv_freq = 0.0f;
                    if (dim == config_.head_dim_global && config_.model_type != "gemma4-e4b") {
                        // Gemma-4 12B global layers rotate only 25% of dimensions (64 angles = 128 elements)
                        int rope_angles = 64;
                        if (i < rope_angles) inv_freq = 1.0f / std::pow(base, float(i * 2) / dim);
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
    int half_dim = dim / 2;

#ifdef __AVX2__
    for (int h = 0; h < num_heads; ++h) {
        const bf16* x_h1 = x + h * dim;
        const bf16* x_h2 = x + h * dim + half_dim;
        bf16* out_h1 = out + h * dim;
        bf16* out_h2 = out + h * dim + half_dim;

        int i = 0;
        for (; i + 7 < half_dim; i += 8) {
            __m128i in_x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_h1 + i));
            __m128i in_x2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_h2 + i));
            __m128i in_c  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cos_ptr + i));
            __m128i in_s  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sin_ptr + i));

            __m256 x1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(in_x1), 16));
            __m256 x2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(in_x2), 16));
            __m256 c  = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(in_c), 16));
            __m256 s  = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(in_s), 16));

            __m256 o1 = _mm256_fmsub_ps(x1, c, _mm256_mul_ps(x2, s));
            __m256 o2 = _mm256_fmadd_ps(x2, c, _mm256_mul_ps(x1, s));

            __m256i sh1 = _mm256_srli_epi32(_mm256_castps_si256(o1), 16);
            __m128i p1 = _mm_packus_epi32(_mm256_castsi256_si128(sh1), _mm256_extractf128_si256(sh1, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_h1 + i), p1);

            __m256i sh2 = _mm256_srli_epi32(_mm256_castps_si256(o2), 16);
            __m128i p2 = _mm_packus_epi32(_mm256_castsi256_si128(sh2), _mm256_extractf128_si256(sh2, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_h2 + i), p2);
        }
        for (; i < half_dim; ++i) {
            float x1_s = x_h1[i].to_float();
            float x2_s = x_h2[i].to_float();
            float c_s  = cos_ptr[i].to_float();
            float s_s  = sin_ptr[i].to_float();
            out_h1[i] = bf16(x1_s * c_s - x2_s * s_s);
            out_h2[i] = bf16(x2_s * c_s + x1_s * s_s);
        }
    }
#else
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
#endif
}

#if defined(__AVX2__) && defined(__FMA__)
static inline float bf16_dot_product_avx2(const bf16* a, const bf16* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m128i raw_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i raw_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m256i wide_a = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_a), 16);
        __m256i wide_b = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_b), 16);
        __m256 fa = _mm256_castsi256_ps(wide_a);
        __m256 fb = _mm256_castsi256_ps(wide_b);
        sum = _mm256_fmadd_ps(fa, fb, sum);
    }
    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    hsum = _mm_add_ps(hsum, _mm_movehl_ps(hsum, hsum));
    hsum = _mm_add_ss(hsum, _mm_shuffle_ps(hsum, hsum, 1));
    float total = _mm_cvtss_f32(hsum);
    for (; i < n; ++i) {
        total += a[i].to_float() * b[i].to_float();
    }
    return total;
}

static inline void bf16_axpy_avx2(float alpha, const bf16* v, float* out, int n) {
    __m256 valpha = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m128i raw_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(v + i));
        __m256i wide_v = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_v), 16);
        __m256 fv = _mm256_castsi256_ps(wide_v);
        __m256 fo = _mm256_loadu_ps(out + i);
        fo = _mm256_fmadd_ps(valpha, fv, fo);
        _mm256_storeu_ps(out + i, fo);
    }
    for (; i < n; ++i) {
        out[i] += alpha * v[i].to_float();
    }
}
#endif

void Model::run_attention_host(const bf16* q_rope, int pos, int layer, bf16* out) {
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int dim = config_.head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    int window_size = 512;
    int max_seq_len = config_.max_position_embeddings;
    
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

    float scores_buf[2048];
    float out_f[512];

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / group_ratio;
        float max_score = -1e9f;
        const bf16* q_ptr = &q_rope[h * dim];

        for (int w = 0; w < W; ++w) {
            int cache_pos = start_pos + w;
            int kv_idx = (kv_h * max_seq_len + cache_pos) * dim;
            const bf16* k_ptr = &k_caches_[target_layer][kv_idx];
#if defined(__AVX2__) && defined(__FMA__)
            float dot = bf16_dot_product_avx2(q_ptr, k_ptr, dim) * scale;
#else
            float dot = 0.0f;
            for (int i = 0; i < dim; ++i) {
                dot += q_ptr[i].to_float() * k_ptr[i].to_float();
            }
            dot *= scale;
#endif
            scores_buf[w] = dot;
            if (dot > max_score) max_score = dot;
        }

        float sum_exp = 0.0f;
        for (int w = 0; w < W; ++w) {
            scores_buf[w] = std::exp(scores_buf[w] - max_score);
            sum_exp += scores_buf[w];
        }

        float inv_sum = 1.0f / sum_exp;
        std::memset(out_f, 0, size_t(dim) * sizeof(float));
        for (int w = 0; w < W; ++w) {
            float prob = scores_buf[w] * inv_sum;
            int cache_pos = start_pos + w;
            int kv_idx = (kv_h * max_seq_len + cache_pos) * dim;
            const bf16* v_ptr = &v_caches_[target_layer][kv_idx];
#if defined(__AVX2__) && defined(__FMA__)
            bf16_axpy_avx2(prob, v_ptr, out_f, dim);
#else
            for (int i = 0; i < dim; ++i) {
                out_f[i] += prob * v_ptr[i].to_float();
            }
#endif
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
    run_rmsnorm_cpu(x_bf16, lw.attn_norm.empty() ? nullptr : lw.attn_norm.data(), scratch_.x_norm.data());

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

    auto t_qkv = pclock::now();
    if (has_kv) {
        if (lw.w_qkv != kInvalidWeight) {
            // ALVEARE_ONESHAPE: run QKV as tiles of the shared shape (same hw context
            // as O and the FFN tiles). q/k/v are sliced from the start of the
            // concatenated output, so running the tiles in order is equivalent.
            const bool os_q = (!lw.os_qkv_tiles.empty() && lw.os_n > 0);
            bf16* qkv_ptr = scratch_.qkv.data();
            if (os_q) {
                for (size_t t = 0; t < lw.os_qkv_tiles.size(); ++t)
                    reg_.run_gemv(lw.os_n, lw.os_k, lw.os_qkv_tiles[t],
                                  scratch_.x_norm.data(), qkv_ptr + t * lw.os_n);
            } else {
                reg_.run_gemv(lw.n_qkv, K_padded, lw.w_qkv, scratch_.x_norm.data(), qkv_ptr);
            }
            std::memcpy(scratch_.q.data(), qkv_ptr, size_t(N_q) * sizeof(bf16));
            std::memcpy(scratch_.k.data(), qkv_ptr + N_q, size_t(N_kv) * sizeof(bf16));
            if (is_sliding || config_.model_type == "gemma4-e4b")
                std::memcpy(scratch_.v.data(), qkv_ptr + N_q + N_kv, size_t(N_kv) * sizeof(bf16));
            else
                std::memcpy(scratch_.v.data(), scratch_.k.data(), size_t(N_kv) * sizeof(bf16)); // 12B gemma4 global layers are MQA: reuse K for V
        } else {
            int N_q_padded = config_.model_type == "gemma3" ? 2048 : N_q;
            int N_kv_padded = config_.model_type == "gemma3" ? 2048 : N_kv;

            reg_.run_gemv(N_q_padded, K_padded, lw.w_q, scratch_.x_norm.data(), scratch_.q_buf.data());
            std::memcpy(scratch_.q.data(), scratch_.q_buf.data(), size_t(N_q) * sizeof(bf16));

            reg_.run_gemv(N_kv_padded, K_padded, lw.w_k, scratch_.x_norm.data(), scratch_.k_buf.data());
            std::memcpy(scratch_.k.data(), scratch_.k_buf.data(), size_t(N_kv) * sizeof(bf16));

            if (!config_.is_gemma4() || is_sliding || config_.model_type == "gemma4-e4b") {
                reg_.run_gemv(N_kv_padded, K_padded, lw.w_v, scratch_.x_norm.data(), scratch_.v_buf.data());
                std::memcpy(scratch_.v.data(), scratch_.v_buf.data(), size_t(N_kv) * sizeof(bf16));
            } else {
                std::memcpy(scratch_.v.data(), scratch_.k.data(), size_t(N_kv) * sizeof(bf16)); // 12B gemma4 global layers are MQA: reuse K for V
            }
        }
    } else {
        // Layers 24-41 (shared KV) only project Q
        if (lw.w_qkv != kInvalidWeight) {
            // ALVEARE_ONESHAPE: run QKV as tiles of the shared shape (same hw context
            // as O and the FFN tiles). q/k/v are sliced from the start of the
            // concatenated output, so running the tiles in order is equivalent.
            const bool os_q = (!lw.os_qkv_tiles.empty() && lw.os_n > 0);
            bf16* qkv_ptr = scratch_.qkv.data();
            if (os_q) {
                for (size_t t = 0; t < lw.os_qkv_tiles.size(); ++t)
                    reg_.run_gemv(lw.os_n, lw.os_k, lw.os_qkv_tiles[t],
                                  scratch_.x_norm.data(), qkv_ptr + t * lw.os_n);
            } else {
                reg_.run_gemv(lw.n_qkv, K_padded, lw.w_qkv, scratch_.x_norm.data(), qkv_ptr);
            }
            std::memcpy(scratch_.q.data(), qkv_ptr, size_t(N_q) * sizeof(bf16));
        } else {
            int N_q_padded = config_.model_type == "gemma3" ? 2048 : N_q;
            reg_.run_gemv(N_q_padded, K_padded, lw.w_q, scratch_.x_norm.data(), scratch_.q_buf.data());
            std::memcpy(scratch_.q.data(), scratch_.q_buf.data(), size_t(N_q) * sizeof(bf16));
        }
    }
    g_prof.npu_qkv += ms_since(t_qkv);

    // 3. QK-Norm & V-Norm (Gemma only)
    if (config_.model_type == "gemma3" || config_.is_gemma4()) {
        for (int h = 0; h < n_q_heads; ++h) {
            int head_dim_cur = config_.head_dim;
            if (config_.is_gemma4()) head_dim_cur = is_sliding ? config_.head_dim : config_.head_dim_global;
            run_rmsnorm_cpu(&scratch_.q[h * head_dim_cur], lw.q_norm.empty() ? nullptr : lw.q_norm.data(), scratch_.q_h.data(), head_dim_cur);
            std::memcpy(&scratch_.q[h * head_dim_cur], scratch_.q_h.data(), head_dim_cur * sizeof(bf16));
        }
        if (has_kv) {
            for (int h = 0; h < n_kv_heads; ++h) {
                int head_dim_cur = config_.head_dim;
                if (config_.is_gemma4()) head_dim_cur = is_sliding ? config_.head_dim : config_.head_dim_global;
                run_rmsnorm_cpu(&scratch_.k[h * head_dim_cur], lw.k_norm.empty() ? nullptr : lw.k_norm.data(), scratch_.k_h.data(), head_dim_cur);
                std::memcpy(&scratch_.k[h * head_dim_cur], scratch_.k_h.data(), head_dim_cur * sizeof(bf16));
            }
            if (config_.is_gemma4()) {
                for (int h = 0; h < n_kv_heads; ++h) {
                    int head_dim_cur = is_sliding ? config_.head_dim : config_.head_dim_global;
                    run_rmsnorm_cpu(&scratch_.v[h * head_dim_cur], nullptr, scratch_.v_h.data(), head_dim_cur);
                    std::memcpy(&scratch_.v[h * head_dim_cur], scratch_.v_h.data(), head_dim_cur * sizeof(bf16));
                }
            }
        }
    }

    // 4. RoPE
    if (config_.model_type == "gemma3" || config_.is_gemma4()) {
        float base_freq = is_sliding ? 10000.0f : 1000000.0f;
        if (config_.model_type == "gemma3") {
            bool g3_sliding = ((layer + 1) % config_.sliding_pattern_period != 0);
            base_freq = g3_sliding ? 10000.0f : 1000000.0f;
        }
        run_rope_cpu_gemma(scratch_.q.data(), pos, base_freq, n_q_heads, scratch_.q_rope.data());
        if (has_kv) {
            run_rope_cpu_gemma(scratch_.k.data(), pos, base_freq, n_kv_heads, scratch_.k_rope.data());
        }
    } else {
        run_rope_cpu_llama(scratch_.q.data(), pos, n_q_heads, scratch_.q_rope.data());
        if (has_kv) {
            run_rope_cpu_llama(scratch_.k.data(), pos, n_kv_heads, scratch_.k_rope.data());
        }
    }

    // 5. Update KV Cache (only for layers that own KV states)
    if (has_kv) {
        int max_seq_len = config_.max_position_embeddings;
        h_dim = config_.head_dim;
        if (config_.is_gemma4()) {
            h_dim = is_sliding ? config_.head_dim : config_.head_dim_global;
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            int kv_idx = (h * max_seq_len + pos) * h_dim;
            std::memcpy(&k_caches_[layer][kv_idx], &scratch_.k_rope[h * h_dim], h_dim * sizeof(bf16));
            std::memcpy(&v_caches_[layer][kv_idx], &scratch_.v[h * h_dim], h_dim * sizeof(bf16));
        }
    }

    // 6. Attention. (run_attention_host maps shared-KV layers to their source
    // cache internally via target_layer, so pass the real layer index.)
    auto t_attn = pclock::now();
    run_attention_host(scratch_.q_rope.data(), pos, layer, scratch_.attn_out.data());
    g_prof.attn += ms_since(t_attn);

    // 7. Output Projection
    int N_out_padded = config_.get_padded_hidden_size();
    int o_n = lw.o_gemv_n > 0 ? lw.o_gemv_n : N_out_padded;
    // o_gemv_k > 0 means O was zero-padded to share the fused-QKV kernel context.
    int N_q_padded = lw.o_gemv_k > 0 ? lw.o_gemv_k
                                     : ((config_.model_type == "gemma3") ? 2048 : N_q);

    const bool os_o_on = (!lw.os_o_tiles.empty() && lw.os_n > 0);
    auto t_o = pclock::now();
    if (os_o_on) {
        if (lw.os_o_is_kchunked) {
            std::fill(scratch_.acc_f.begin(), scratch_.acc_f.begin() + K, 0.0f);
            for (size_t c = 0; c < lw.os_o_tiles.size(); ++c) {
                int in_start = static_cast<int>(c) * lw.os_k;
                int in_len = std::min(lw.os_k, N_q - in_start);
                std::fill(scratch_.chunk_in.begin(), scratch_.chunk_in.begin() + lw.os_k, bf16(0.0f));
                if (in_len > 0) {
                    std::memcpy(scratch_.chunk_in.data(), scratch_.attn_out.data() + in_start,
                                size_t(in_len) * sizeof(bf16));
                }
                reg_.run_gemv(lw.os_n, lw.os_k, lw.os_o_tiles[c],
                              scratch_.chunk_in.data(), scratch_.part_bf16.data());
                for (int i = 0; i < K; ++i) {
                    scratch_.acc_f[i] += scratch_.part_bf16[i].to_float();
                }
            }
            for (int i = 0; i < K; ++i) {
                scratch_.attn_proj[i] = bf16(scratch_.acc_f[i]);
            }
        } else {
            std::fill(scratch_.attn_out_padded.begin(), scratch_.attn_out_padded.begin() + lw.os_k, bf16(0.0f));
            std::memcpy(scratch_.attn_out_padded.data(), scratch_.attn_out.data(), size_t(N_q) * sizeof(bf16));
            for (size_t t = 0; t < lw.os_o_tiles.size(); ++t) {
                reg_.run_gemv(lw.os_n, lw.os_k, lw.os_o_tiles[t],
                              scratch_.attn_out_padded.data(), scratch_.attn_proj.data() + t * lw.os_n);
            }
        }
    } else {
        std::fill(scratch_.attn_out_padded.begin(), scratch_.attn_out_padded.begin() + N_q_padded, bf16(0.0f));
        std::memcpy(scratch_.attn_out_padded.data(), scratch_.attn_out.data(), size_t(N_q) * sizeof(bf16));
        reg_.run_gemv(o_n, N_q_padded, lw.w_o, scratch_.attn_out_padded.data(), scratch_.attn_proj.data());
    }
    g_prof.npu_o += ms_since(t_o);

    // 8. Post-attention norm and residual
    if (config_.model_type == "gemma3" || config_.is_gemma4()) {
        run_rmsnorm_cpu(scratch_.attn_proj.data(), lw.post_attention_norm.empty() ? nullptr : lw.post_attention_norm.data(), scratch_.attn_proj_normed.data());
        for (int i = 0; i < K; ++i) {
            scratch_.x_post_attn[i] = bf16(x_bf16[i].to_float() + scratch_.attn_proj_normed[i].to_float());
        }
    } else {
        for (int i = 0; i < K; ++i) {
            scratch_.x_post_attn[i] = bf16(x_bf16[i].to_float() + scratch_.attn_proj[i].to_float());
        }
    }

    // 9. Pre-FFN norm
    run_rmsnorm_cpu(scratch_.x_post_attn.data(), lw.ffn_norm.empty() ? nullptr : lw.ffn_norm.data(), scratch_.x_norm2.data());

    // 10. FFN (Fused NPU or CPU/q4_0 fallback)
    int H_padded = config_.get_padded_hidden_size();
    int I_padded = config_.get_padded_intermediate_size();
    std::string act_type = (config_.model_type == "gemma3" || config_.is_gemma4()) ? "gelu" : "silu";
    auto t_ffn = pclock::now();
    if (!lw.os_gateup.empty() && !lw.os_down.empty()) {
        // ALVEARE_ONESHAPE: FFN as gemv tiles on the fused-QKV kernel shape, so the
        // layer never switches hw context. gate++up in tiles, GELU*up on the host,
        // then the down projection as K-chunks whose partials the host sums.
        const int TN = lw.os_n, TK = lw.os_k;
        const int I_rows = int(lw.os_down.size()) * TK;  // == I_padded (matches the tiles)
        for (size_t t = 0; t < lw.os_gateup.size(); ++t)
            reg_.run_gemv(TN, TK, lw.os_gateup[t], scratch_.x_norm2.data(), scratch_.gu.data() + t * TN);

        for (int i = 0; i < I_rows; ++i) {
            float g = scratch_.gu[i].to_float();
            float u = scratch_.gu[size_t(I_rows) + i].to_float();
            scratch_.act[i] = bf16(0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f)) * u);
        }

        std::fill(scratch_.acc_f.begin(), scratch_.acc_f.begin() + K, 0.0f);
        for (size_t c = 0; c < lw.os_down.size(); ++c) {
            reg_.run_gemv(TN, TK, lw.os_down[c], scratch_.act.data() + c * TK, scratch_.part_bf16.data());
            for (int i = 0; i < K; ++i) scratch_.acc_f[i] += scratch_.part_bf16[i].to_float();
        }
        for (int i = 0; i < K; ++i) scratch_.down[i] = bf16(scratch_.acc_f[i]);
    } else if (lw.w_ffn_fused != kInvalidWeight) {
        reg_.run_ffn_fused(H_padded, I_padded, act_type, lw.w_ffn_fused, scratch_.x_norm2.data(), scratch_.down.data());
    } else {
        int gate_stride = (K_padded / 32) * 20;
        int down_stride = (I_padded / 32) * 20;
        int intermediate_size = config_.intermediate_size;
        for (int i = 0; i < K_padded; ++i) scratch_.x_norm2_f[i] = scratch_.x_norm2[i].to_float();

        #pragma omp parallel for schedule(static)
        for (int r = 0; r < intermediate_size; ++r) {
            const uint8_t* row_g = lw.ffn_gate_bytes.data() + static_cast<size_t>(r) * gate_stride;
            const uint8_t* row_u = lw.ffn_up_bytes.data() + static_cast<size_t>(r) * gate_stride;
            float g = q4_0_dot_product(row_g, scratch_.x_norm2_f.data(), K_padded);
            float u = q4_0_dot_product(row_u, scratch_.x_norm2_f.data(), K_padded);
            float a = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f));
            scratch_.geglu[r] = a * u;
        }

        #pragma omp parallel for schedule(static)
        for (int r = 0; r < K; ++r) {
            const uint8_t* row_d = lw.ffn_down_bytes.data() + static_cast<size_t>(r) * down_stride;
            float d = q4_0_dot_product(row_d, scratch_.geglu.data(), I_padded);
            scratch_.down[r] = bf16(d);
        }
    }
    g_prof.npu_ffn += ms_since(t_ffn);

    // 11. Post-FFN norm and residual
    if (config_.model_type == "gemma3" || config_.is_gemma4()) {
        run_rmsnorm_cpu(scratch_.down.data(), lw.post_ffw_norm.empty() ? nullptr : lw.post_ffw_norm.data(), scratch_.down_normed.data());
        for (int i = 0; i < K; ++i) {
            scratch_.x_post_ffn[i] = bf16(scratch_.x_post_attn[i].to_float() + scratch_.down_normed[i].to_float());
        }
    } else {
        for (int i = 0; i < K; ++i) {
            scratch_.x_post_ffn[i] = bf16(scratch_.x_post_attn[i].to_float() + scratch_.down[i].to_float());
        }
    }

    // 12. PLE Injection (Gemma-4-E4B)
    auto t_ple = pclock::now();
    if (config_.per_layer_input > 0 && inp_per_layer && !lw.inp_gate_bytes.empty()) {
        int n_ple = config_.per_layer_input; // 256
        const float* h_l = inp_per_layer + static_cast<size_t>(layer) * n_ple;

        // Gate: GELU(inp_gate * x_post_ffn)
        for (int i = 0; i < K; ++i) scratch_.ple_x_f[i] = scratch_.x_post_ffn[i].to_float();
        const uint8_t* gate_base = lw.inp_gate_bytes.data();
        const int gate_row_bytes = (K / 32) * 20;
        for (int r = 0; r < n_ple; ++r) {
            const uint8_t* row = gate_base + static_cast<size_t>(r) * gate_row_bytes;
            float g = q4_0_dot_product(row, scratch_.ple_x_f.data(), K);
            float a = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f));
            scratch_.ple_z[r] = a * h_l[r];
        }

        // Proj: proj * z
        const uint8_t* proj_base = lw.proj_bytes.data();
        const int proj_row_bytes = (n_ple / 32) * 20;
        for (int r = 0; r < K; ++r) {
            const uint8_t* row = proj_base + static_cast<size_t>(r) * proj_row_bytes;
            float dot = q4_0_dot_product(row, scratch_.ple_z.data(), n_ple);
            scratch_.ple_p[r] = bf16(dot);
        }

        // Post norm: RMSNorm(p, lw.post_norm)
        run_rmsnorm_cpu(scratch_.ple_p.data(), lw.post_norm.empty() ? nullptr : lw.post_norm.data(), scratch_.ple_y.data(), K);

        // Residual: x_post_ffn += y
        for (int i = 0; i < K; ++i) {
            scratch_.x_post_ffn[i] = bf16(scratch_.x_post_ffn[i].to_float() + scratch_.ple_y[i].to_float());
        }
    }

    g_prof.ple += ms_since(t_ple);

    // 13. Layer output scale
    float oscale = (config_.is_gemma4()) ? lw.output_scale : 1.0f;
    for (int i = 0; i < K; ++i) {
        out_bf16[i] = bf16(scratch_.x_post_ffn[i].to_float() * oscale);
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
                << " | ple(cpu)=" << g_prof.ple
                << " | cpu_rest=" << (cpu_rest - g_prof.ple) << "\n" << std::flush;
            g_prof = DecodeProf();
            npu_gemv_prof_report();
        }
    }
}

void Model::run_layer_batch(const bf16* x_batch, int nrows, int pos_start,
                            int layer, bf16* out_batch, const float* inp_per_layer) {
    auto t_blayer = pclock::now();
    const int K = config_.hidden_size;
    const int K_padded = config_.get_padded_hidden_size();
    const int B = 16;
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

    // 2. Fused Q/K/V projection (resident tiles or single resident GEMM).
    std::vector<bf16> q(size_t(B) * N_q, bf16(0.0f)), k(size_t(B) * N_kv, bf16(0.0f)), v(size_t(B) * N_kv, bf16(0.0f));
    auto t_bqkv = pclock::now();
    const bool os_q = (!lw.os_qkv_tiles.empty() && lw.os_n > 0);
    if (os_q) {
        const int TN = lw.os_n, TK = lw.os_k;
        const int total_out_rows = int(lw.os_qkv_tiles.size()) * TN;
        std::vector<bf16> qkv(size_t(B) * total_out_rows);
        std::vector<bf16> tile_out(size_t(B) * TN);
        for (size_t t = 0; t < lw.os_qkv_tiles.size(); ++t) {
            reg_.run_gemm(B, TN, TK, lw.os_qkv_tiles[t], x_norm_pad.data(), tile_out.data());
            for (int b = 0; b < B; ++b) {
                std::memcpy(&qkv[size_t(b) * total_out_rows + t * TN],
                            &tile_out[size_t(b) * TN], size_t(TN) * sizeof(bf16));
            }
        }
        for (int b = 0; b < nrows; ++b) {
            const bf16* row = &qkv[size_t(b) * total_out_rows];
            std::memcpy(&q[size_t(b) * N_q], row, size_t(N_q) * sizeof(bf16));
            if (has_kv) {
                std::memcpy(&k[size_t(b) * N_kv], row + N_q, size_t(N_kv) * sizeof(bf16));
                if (is_sliding || config_.model_type == "gemma4-e4b")
                    std::memcpy(&v[size_t(b) * N_kv], row + N_q + N_kv, size_t(N_kv) * sizeof(bf16));
            }
        }
    } else {
        std::vector<bf16> qkv(size_t(B) * lw.n_qkv);
        reg_.run_gemm(B, lw.n_qkv, K_padded, lw.w_qkv, x_norm_pad.data(), qkv.data());
        for (int b = 0; b < nrows; ++b) {
            const bf16* row = &qkv[size_t(b) * lw.n_qkv];
            std::memcpy(&q[size_t(b) * N_q], row, size_t(N_q) * sizeof(bf16));
            if (has_kv) {
                std::memcpy(&k[size_t(b) * N_kv], row + N_q, size_t(N_kv) * sizeof(bf16));
                if (is_sliding || config_.model_type == "gemma4-e4b")
                    std::memcpy(&v[size_t(b) * N_kv], row + N_q + N_kv, size_t(N_kv) * sizeof(bf16));
            }
        }
    }
    g_bprof.gemm += ms_since(t_bqkv);
    if (has_kv && !is_sliding && config_.model_type != "gemma4-e4b")
        v = k;  // gemma4 12B global layers reuse K for V

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
        const int max_seq_len = config_.max_position_embeddings;
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

    // 7. Output projection (batched GEMM with resident ONESHAPE tiles or padded shape).
    const bool os_o_on = (!lw.os_o_tiles.empty() && lw.os_n > 0);
    std::vector<bf16> attn_proj(size_t(B) * K, bf16(0.0f));
    auto t_bo = pclock::now();
    if (os_o_on) {
        const int TN = lw.os_n, TK = lw.os_k;
        if (lw.os_o_is_kchunked) {
            std::vector<float> acc(size_t(B) * K, 0.0f);
            std::vector<bf16> chunk_in(size_t(B) * TK, bf16(0.0f));
            std::vector<bf16> part(size_t(B) * TN);
            for (size_t c = 0; c < lw.os_o_tiles.size(); ++c) {
                int in_start = static_cast<int>(c) * TK;
                int in_len = std::min(TK, N_q - in_start);
                std::fill(chunk_in.begin(), chunk_in.end(), bf16(0.0f));
                if (in_len > 0) {
                    for (int b = 0; b < nrows; ++b) {
                        std::memcpy(&chunk_in[size_t(b) * TK],
                                    &attn_out[size_t(b) * N_q + in_start],
                                    size_t(in_len) * sizeof(bf16));
                    }
                }
                reg_.run_gemm(B, TN, TK, lw.os_o_tiles[c], chunk_in.data(), part.data());
                for (int b = 0; b < nrows; ++b) {
                    for (int i = 0; i < K; ++i) {
                        acc[size_t(b) * K + i] += part[size_t(b) * TN + i].to_float();
                    }
                }
            }
            for (int b = 0; b < nrows; ++b) {
                for (int i = 0; i < K; ++i) {
                    attn_proj[size_t(b) * K + i] = bf16(acc[size_t(b) * K + i]);
                }
            }
        } else {
            std::vector<bf16> in_pad(size_t(B) * TK, bf16(0.0f));
            for (int b = 0; b < nrows; ++b) {
                std::memcpy(&in_pad[size_t(b) * TK], &attn_out[size_t(b) * N_q], size_t(N_q) * sizeof(bf16));
            }
            std::vector<bf16> o_tile(size_t(B) * TN);
            for (size_t t = 0; t < lw.os_o_tiles.size(); ++t) {
                reg_.run_gemm(B, TN, TK, lw.os_o_tiles[t], in_pad.data(), o_tile.data());
                for (int b = 0; b < nrows; ++b) {
                    int rows_to_copy = std::min(K - int(t * TN), TN);
                    if (rows_to_copy > 0) {
                        std::memcpy(&attn_proj[size_t(b) * K + t * TN],
                                    &o_tile[size_t(b) * TN],
                                    size_t(rows_to_copy) * sizeof(bf16));
                    }
                }
            }
        }
    } else {
        const int o_n = lw.o_gemv_n > 0 ? lw.o_gemv_n : K_padded;
        std::vector<bf16> full_proj(size_t(B) * o_n);
        reg_.run_gemm(B, o_n, N_q, lw.w_o, attn_out.data(), full_proj.data());
        for (int b = 0; b < nrows; ++b) {
            std::memcpy(&attn_proj[size_t(b) * K], &full_proj[size_t(b) * o_n], size_t(K) * sizeof(bf16));
        }
    }
    g_bprof.gemm += ms_since(t_bo);

    // 8. Post-attention norm + residual.
    std::vector<bf16> x_post_attn(size_t(nrows) * K);
    std::vector<bf16> normed(K);
    for (int b = 0; b < nrows; ++b) {
        run_rmsnorm_cpu(&attn_proj[size_t(b) * K], post_attn_w, normed.data(), K);
        for (int i = 0; i < K; ++i)
            x_post_attn[size_t(b) * K + i] = bf16(x_batch[size_t(b) * K + i].to_float() + normed[i].to_float());
    }

    // 9. Pre-FFN norm, padded to K_padded.
    std::vector<bf16> x_norm2_pad(size_t(B) * K_padded, bf16(0.0f));
    for (int b = 0; b < nrows; ++b)
        run_rmsnorm_cpu(&x_post_attn[size_t(b) * K], ffn_norm_w, &x_norm2_pad[size_t(b) * K_padded], K);

    // 10. FFN: resident ONESHAPE tiles or streamed GEMM fallback
    std::vector<float> down_acc(size_t(nrows) * K, 0.0f);
    if (!lw.os_gateup.empty() && !lw.os_down.empty()) {
        const int TN = lw.os_n, TK = lw.os_k;
        const int I_rows = int(lw.os_down.size()) * TK;
        const int gu_rows = int(lw.os_gateup.size()) * TN;
        std::vector<bf16> gu_batch(size_t(B) * gu_rows, bf16(0.0f));
        std::vector<bf16> gu_tile(size_t(B) * TN);
        auto t_bgu = pclock::now();
        for (size_t t = 0; t < lw.os_gateup.size(); ++t) {
            reg_.run_gemm(B, TN, TK, lw.os_gateup[t], x_norm2_pad.data(), gu_tile.data());
            for (int b = 0; b < B; ++b) {
                std::memcpy(&gu_batch[size_t(b) * gu_rows + size_t(t) * TN],
                            &gu_tile[size_t(b) * TN], size_t(TN) * sizeof(bf16));
            }
        }
        g_bprof.ffn_cmp += ms_since(t_bgu);

        std::vector<bf16> act(size_t(B) * I_rows, bf16(0.0f));
        for (int b = 0; b < nrows; ++b) {
            for (int i = 0; i < I_rows; ++i) {
                float g = gu_batch[size_t(b) * gu_rows + i].to_float();
                float u = gu_batch[size_t(b) * gu_rows + size_t(I_rows) + i].to_float();
                act[size_t(b) * I_rows + i] = bf16(0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f)) * u);
            }
        }

        std::vector<bf16> chunk_in(size_t(B) * TK, bf16(0.0f));
        std::vector<bf16> down_tile(size_t(B) * TN);
        auto t_bdn = pclock::now();
        for (size_t c = 0; c < lw.os_down.size(); ++c) {
            for (int b = 0; b < B; ++b) {
                std::memcpy(&chunk_in[size_t(b) * TK],
                            &act[size_t(b) * I_rows + size_t(c) * TK],
                            size_t(TK) * sizeof(bf16));
            }
            reg_.run_gemm(B, TN, TK, lw.os_down[c], chunk_in.data(), down_tile.data());
            for (int b = 0; b < nrows; ++b) {
                for (int i = 0; i < K; ++i) {
                    down_acc[size_t(b) * K + i] += down_tile[size_t(b) * TN + i].to_float();
                }
            }
        }
        g_bprof.ffn_cmp += ms_since(t_bdn);
    } else {
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
        for (int b = 0; b < nrows; ++b) {
            for (int i = 0; i < I_real; ++i) {
                float g = gate[size_t(b) * I + i].to_float();
                float gelu = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f));
                geglu[size_t(b) * I + i] = bf16(gelu * up[size_t(b) * I + i].to_float());
            }
        }

        const int Nd = K_padded;
        const int chunkK = (I > 8192) ? 8192 : I;
        const int num_chunks = (I + chunkK - 1) / chunkK;
        const int row_bytes = (I / 32) * 20;
        const int chunk_row_bytes = (chunkK / 32) * 20;
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
