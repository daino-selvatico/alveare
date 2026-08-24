#include "alveare/vision_embedder.h"
#include "alveare/weights.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <dlfcn.h>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "alveare/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "alveare/stb_image_resize2.h"

namespace alveare {

namespace {

// Base64 decoding helper
std::vector<uint8_t> base64_decode(const std::string& in) {
    std::string clean = in;
    size_t comma = clean.find(',');
    if (comma != std::string::npos) {
        clean = clean.substr(comma + 1);
    }
    // Remove whitespace/newlines
    clean.erase(std::remove_if(clean.begin(), clean.end(), [](unsigned char c) {
        return std::isspace(c);
    }), clean.end());

    static const std::string b64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::vector<uint8_t> out;
    int val = 0, valb = -8;
    for (unsigned char c : clean) {
        if (c == '=') break;
        size_t idx = b64_chars.find(c);
        if (idx == std::string::npos) continue;
        val = (val << 6) + idx;
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// AVX2 dot product
inline float avx2_dot_product(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    float buf[8];
    _mm256_storeu_ps(buf, acc0);
    float sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// LayerNorm in place with scale and bias
void layernorm(float* x, int dim, const float* weight, const float* bias) {
    float mean = 0.0f;
    for (int i = 0; i < dim; ++i) mean += x[i];
    mean /= dim;

    float var = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= dim;
    float inv_std = 1.0f / std::sqrt(var + 1e-6f);

    int i = 0;
    __m256 v_mean = _mm256_set1_ps(mean);
    __m256 v_inv_std = _mm256_set1_ps(inv_std);
    for (; i + 7 < dim; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 vb = bias ? _mm256_loadu_ps(bias + i) : _mm256_setzero_ps();
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(vx, v_mean), v_inv_std);
        __m256 res = _mm256_fmadd_ps(norm, vw, vb);
        _mm256_storeu_ps(x + i, res);
    }
    for (; i < dim; ++i) {
        x[i] = ((x[i] - mean) * inv_std) * weight[i] + (bias ? bias[i] : 0.0f);
    }
}

// RMSNorm in place with optional scale
void rmsnorm(float* x, int dim, const float* weight = nullptr, float eps = 1e-6f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / dim + eps);

    int i = 0;
    __m256 v_inv_rms = _mm256_set1_ps(inv_rms);
    if (weight) {
        for (; i + 7 < dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, v_inv_rms), vw);
            _mm256_storeu_ps(x + i, res);
        }
        for (; i < dim; ++i) {
            x[i] = (x[i] * inv_rms) * weight[i];
        }
    } else {
        for (; i + 7 < dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 res = _mm256_mul_ps(vx, v_inv_rms);
            _mm256_storeu_ps(x + i, res);
        }
        for (; i < dim; ++i) {
            x[i] = x[i] * inv_rms;
        }
    }
}

// Matmul transposed: out (rows x out_dim) = x (rows x in_dim) * w^T (in_dim x out_dim)
void matmul_transposed(const float* x, const float* w, float* out, int rows, int out_dim, int in_dim) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out_dim; ++o) {
            const float* w_row = &w[o * in_dim];
            const float* x_row = &x[r * in_dim];
            out[r * out_dim + o] = avx2_dot_product(x_row, w_row, in_dim);
        }
    }
}

static uint8_t* decode_webp_rgba(const uint8_t* data, size_t len, int* w, int* h) {
    static void* handle = nullptr;
    static uint8_t* (*p_WebPDecodeRGBA)(const uint8_t*, size_t, int*, int*) = nullptr;
    if (!handle) {
        handle = dlopen("libwebp.so.7", RTLD_LAZY);
        if (!handle) handle = dlopen("libwebp.so", RTLD_LAZY);
        if (handle) {
            p_WebPDecodeRGBA = (uint8_t* (*)(const uint8_t*, size_t, int*, int*))dlsym(handle, "WebPDecodeRGBA");
        }
    }
    if (p_WebPDecodeRGBA) {
        return p_WebPDecodeRGBA(data, len, w, h);
    }
    return nullptr;
}

// -------------------------------------------------------------
// Gemma4UvVisionEncoder (Gemma-4 12B)
// -------------------------------------------------------------
class Gemma4UvVisionEncoder : public IVisionEncoder {
public:
    Gemma4UvVisionEncoder() = default;

    bool load(const std::string& vision_dir) override {
        try {
            patch_norm1_w_ = load_float_npy(vision_dir + "/v_patch_norm_1_weight.npy");
            patch_norm1_b_ = load_float_npy(vision_dir + "/v_patch_norm_1_bias.npy");
            patch_embd_w_  = load_float_npy(vision_dir + "/v_patch_embd_weight.npy");
            patch_embd_b_  = load_float_npy(vision_dir + "/v_patch_embd_bias.npy");
            patch_norm2_w_ = load_float_npy(vision_dir + "/v_patch_norm_2_weight.npy");
            patch_norm2_b_ = load_float_npy(vision_dir + "/v_patch_norm_2_bias.npy");
            position_embd_ = load_float_npy(vision_dir + "/v_position_embd_weight.npy");
            patch_norm3_w_ = load_float_npy(vision_dir + "/v_patch_norm_3_weight.npy");
            patch_norm3_b_ = load_float_npy(vision_dir + "/v_patch_norm_3_bias.npy");
            mm_proj_w_     = load_float_npy(vision_dir + "/mm_input_projection_weight.npy");

            if (patch_embd_w_.empty() || mm_proj_w_.empty() || position_embd_.empty()) {
                return false;
            }
            hidden_size_ = static_cast<int>(patch_embd_b_.size());
            std::cout << "[Gemma4UvVisionEncoder] Loaded 35M linear patch embedder (hidden=" << hidden_size_ << ")" << std::endl;
            return true;
        } catch (...) {
            return false;
        }
    }

    int output_dim() const override { return hidden_size_; }
    VisionArch arch() const override { return VisionArch::GEMMA4_UV; }

    std::vector<std::vector<float>> encode_rgb(const uint8_t* rgb_data, int width, int height) override {
        if (!rgb_data || width <= 0 || height <= 0) return {};

        int target_w = ((width + kPatchSize - 1) / kPatchSize) * kPatchSize;
        int target_h = ((height + kPatchSize - 1) / kPatchSize) * kPatchSize;

        if (target_w > 480 || target_h > 480) {
            float ratio = std::min(480.0f / width, 480.0f / height);
            target_w = std::max(kPatchSize, static_cast<int>(std::round((width * ratio) / kPatchSize) * kPatchSize));
            target_h = std::max(kPatchSize, static_cast<int>(std::round((height * ratio) / kPatchSize) * kPatchSize));
        }

        std::vector<uint8_t> resized_rgb(target_w * target_h * 3);
        if (target_w == width && target_h == height) {
            std::memcpy(resized_rgb.data(), rgb_data, width * height * 3);
        } else {
            stbir_resize_uint8_linear(rgb_data, width, height, 0,
                                     resized_rgb.data(), target_w, target_h, 0,
                                     STBIR_RGB);
        }

        int num_patches_x = target_w / kPatchSize;
        int num_patches_y = target_h / kPatchSize;
        int total_patches = num_patches_x * num_patches_y;

        std::vector<std::vector<float>> output_tokens(total_patches, std::vector<float>(hidden_size_));
        std::vector<float> patch_buf(kPatchDim);
        std::vector<float> emb_buf(hidden_size_);

        int patch_idx = 0;
        for (int py = 0; py < num_patches_y; ++py) {
            for (int px = 0; px < num_patches_x; ++px) {
                for (int c = 0; c < 3; ++c) {
                    int c_offset = c * (kPatchSize * kPatchSize);
                    for (int dy = 0; dy < kPatchSize; ++dy) {
                        int img_y = py * kPatchSize + dy;
                        for (int dx = 0; dx < kPatchSize; ++dx) {
                            int img_x = px * kPatchSize + dx;
                            int img_idx = (img_y * target_w + img_x) * 3 + c;
                            float pixel_val = static_cast<float>(resized_rgb[img_idx]) / 255.0f;
                            patch_buf[c_offset + dy * kPatchSize + dx] = pixel_val * 2.0f - 1.0f;
                        }
                    }
                }

                layernorm(patch_buf.data(), kPatchDim, patch_norm1_w_.data(), patch_norm1_b_.data());

                for (int h = 0; h < hidden_size_; ++h) {
                    const float* w_row = &patch_embd_w_[h * kPatchDim];
                    emb_buf[h] = avx2_dot_product(w_row, patch_buf.data(), kPatchDim) + patch_embd_b_[h];
                }

                layernorm(emb_buf.data(), hidden_size_, patch_norm2_w_.data(), patch_norm2_b_.data());

                const float* pos_x = &position_embd_[(0 * 1120 + px) * hidden_size_];
                const float* pos_y = &position_embd_[(1 * 1120 + py) * hidden_size_];
                for (int h = 0; h < hidden_size_; ++h) {
                    emb_buf[h] += pos_x[h] + pos_y[h];
                }

                layernorm(emb_buf.data(), hidden_size_, patch_norm3_w_.data(), patch_norm3_b_.data());

                float sum_sq = 0.0f;
                for (int h = 0; h < hidden_size_; ++h) {
                    const float* w_row = &mm_proj_w_[h * hidden_size_];
                    float val = avx2_dot_product(w_row, emb_buf.data(), hidden_size_);
                    output_tokens[patch_idx][h] = val;
                    sum_sq += val * val;
                }

                float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_size_ + 1e-6f);
                for (int h = 0; h < hidden_size_; ++h) {
                    output_tokens[patch_idx][h] *= inv_rms;
                }

                patch_idx++;
            }
        }
        return output_tokens;
    }

private:
    int hidden_size_ = 3840;
    static constexpr int kPatchSize = 48;
    static constexpr int kPatchDim = 48 * 48 * 3;

    std::vector<float> patch_norm1_w_, patch_norm1_b_;
    std::vector<float> patch_embd_w_, patch_embd_b_;
    std::vector<float> patch_norm2_w_, patch_norm2_b_;
    std::vector<float> position_embd_;
    std::vector<float> patch_norm3_w_, patch_norm3_b_;
    std::vector<float> mm_proj_w_;
};

// -------------------------------------------------------------
// VitVisionEncoder (Gemma-4 E4B, Qwen2/3.5-VL)
// -------------------------------------------------------------
class VitVisionEncoder : public IVisionEncoder {
public:
    struct VitLayer {
        std::vector<float> ln1_w;
        std::vector<float> q_w, k_w, v_w, o_w;
        std::vector<float> q_norm_w, k_norm_w, attn_post_w;
        std::vector<float> ln2_w;
        std::vector<float> gate_w, up_w, down_w, ffn_post_w;
    };

    VitVisionEncoder() = default;

    bool load(const std::string& vision_dir) override {
        try {
            patch_embd_w_ = load_float_npy(vision_dir + "/v_patch_embd_weight.npy");
            position_embd_ = load_float_npy(vision_dir + "/v_position_embd_weight.npy");
            mm_proj_w_ = load_float_npy(vision_dir + "/mm_input_projection_weight.npy");

            if (patch_embd_w_.empty() || position_embd_.empty() || mm_proj_w_.empty()) {
                return false;
            }

            // Identify dimensions
            vit_dim_ = 768; // default E4B
            output_dim_ = static_cast<int>(mm_proj_w_.size() / vit_dim_);
            if (output_dim_ == 0) output_dim_ = 2560;

            layers_.clear();
            for (int l = 0; l < 32; ++l) {
                std::string prefix = vision_dir + "/v_blk_" + std::to_string(l) + "_";
                auto ln1 = load_float_npy(prefix + "ln1_weight.npy");
                if (ln1.empty()) break;

                VitLayer layer;
                layer.ln1_w = std::move(ln1);
                layer.q_w = load_float_npy(prefix + "attn_q_weight.npy");
                layer.k_w = load_float_npy(prefix + "attn_k_weight.npy");
                layer.v_w = load_float_npy(prefix + "attn_v_weight.npy");
                layer.o_w = load_float_npy(prefix + "attn_out_weight.npy");
                layer.q_norm_w = load_float_npy(prefix + "attn_q_norm_weight.npy");
                layer.k_norm_w = load_float_npy(prefix + "attn_k_norm_weight.npy");
                layer.attn_post_w = load_float_npy(prefix + "attn_post_norm_weight.npy");

                layer.ln2_w = load_float_npy(prefix + "ln2_weight.npy");
                layer.gate_w = load_float_npy(prefix + "ffn_gate_weight.npy");
                layer.up_w = load_float_npy(prefix + "ffn_up_weight.npy");
                layer.down_w = load_float_npy(prefix + "ffn_down_weight.npy");
                layer.ffn_post_w = load_float_npy(prefix + "ffn_post_norm_weight.npy");

                layers_.push_back(std::move(layer));
            }

            num_layers_ = static_cast<int>(layers_.size());
            if (num_layers_ == 0) return false;
            std::cout << "[VitVisionEncoder] Successfully loaded " << num_layers_ 
                      << "-layer Vision Transformer (vit_dim=" << vit_dim_ 
                      << ", output_dim=" << output_dim_ << ")" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[VitVisionEncoder] Error loading weights: " << e.what() << std::endl;
            return false;
        }
    }

    int output_dim() const override { return output_dim_; }
    VisionArch arch() const override { return VisionArch::GEMMA4_V; }

    std::vector<std::vector<float>> encode_rgb(const uint8_t* rgb_data, int width, int height) override {
        if (!rgb_data || width <= 0 || height <= 0 || layers_.empty()) return {};

        int target_w = ((width + kPatchSize - 1) / kPatchSize) * kPatchSize;
        int target_h = ((height + kPatchSize - 1) / kPatchSize) * kPatchSize;

        // Ensure reasonable dimensions for ViT (e.g. 224 to 448)
        if (target_w > 448 || target_h > 448) {
            float ratio = std::min(448.0f / width, 448.0f / height);
            target_w = std::max(kPatchSize, static_cast<int>(std::round((width * ratio) / kPatchSize) * kPatchSize));
            target_h = std::max(kPatchSize, static_cast<int>(std::round((height * ratio) / kPatchSize) * kPatchSize));
        }

        std::vector<uint8_t> resized_rgb(target_w * target_h * 3);
        if (target_w == width && target_h == height) {
            std::memcpy(resized_rgb.data(), rgb_data, width * height * 3);
        } else {
            stbir_resize_uint8_linear(rgb_data, width, height, 0,
                                     resized_rgb.data(), target_w, target_h, 0,
                                     STBIR_RGB);
        }

        int patches_x = target_w / kPatchSize;
        int patches_y = target_h / kPatchSize;
        int n_patches = patches_x * patches_y;

        // 1. Patch Embedding + Positional Embedding
        std::vector<float> x(static_cast<size_t>(n_patches) * vit_dim_, 0.0f);
        std::vector<int> pos_x(n_patches);
        std::vector<int> pos_y(n_patches);

        for (int py = 0; py < patches_y; ++py) {
            for (int px = 0; px < patches_x; ++px) {
                int p_idx = py * patches_x + px;
                pos_x[p_idx] = px;
                pos_y[p_idx] = py;

                float* dst = &x[static_cast<size_t>(p_idx) * vit_dim_];
                for (int h = 0; h < vit_dim_; ++h) {
                    float sum = 0.0f;
                    for (int c = 0; c < 3; ++c) {
                        for (int dy = 0; dy < kPatchSize; ++dy) {
                            int img_y = py * kPatchSize + dy;
                            for (int dx = 0; dx < kPatchSize; ++dx) {
                                int img_x = px * kPatchSize + dx;
                                int img_idx = (img_y * target_w + img_x) * 3 + c;
                                float pixel_val = (static_cast<float>(resized_rgb[img_idx]) / 255.0f) * 2.0f - 1.0f;
                                size_t w_idx = ((size_t(h) * 3 + c) * kPatchSize + dy) * kPatchSize + dx;
                                sum += patch_embd_w_[w_idx] * pixel_val;
                            }
                        }
                    }
                    // Positional embedding lookup (table_x and table_y)
                    const float* tbl_x = &position_embd_[(0 * 10240 + px) * vit_dim_];
                    const float* tbl_y = &position_embd_[(1 * 10240 + py) * vit_dim_];
                    dst[h] = sum + tbl_x[h] + tbl_y[h];
                }
            }
        }

        // 2. Forward through ViT Layers
        std::vector<float> x_norm(static_cast<size_t>(n_patches) * vit_dim_);
        std::vector<float> q(static_cast<size_t>(n_patches) * vit_dim_);
        std::vector<float> k(static_cast<size_t>(n_patches) * vit_dim_);
        std::vector<float> v(static_cast<size_t>(n_patches) * vit_dim_);
        std::vector<float> q_rope(static_cast<size_t>(n_patches) * vit_dim_);
        std::vector<float> k_rope(static_cast<size_t>(n_patches) * vit_dim_);
        std::vector<float> attn_out(static_cast<size_t>(n_patches) * vit_dim_);
        std::vector<float> attn_proj(static_cast<size_t>(n_patches) * vit_dim_);
        std::vector<float> ffn_gate(static_cast<size_t>(n_patches) * kIntermediateDim);
        std::vector<float> ffn_up(static_cast<size_t>(n_patches) * kIntermediateDim);
        std::vector<float> ffn_geglu(static_cast<size_t>(n_patches) * kIntermediateDim);
        std::vector<float> ffn_proj(static_cast<size_t>(n_patches) * vit_dim_);

        for (int l = 0; l < num_layers_; ++l) {
            const auto& layer = layers_[l];

            // Pre-norm
            for (int p = 0; p < n_patches; ++p) {
                std::memcpy(&x_norm[p * vit_dim_], &x[p * vit_dim_], vit_dim_ * sizeof(float));
                rmsnorm(&x_norm[p * vit_dim_], vit_dim_, layer.ln1_w.data());
            }

            // QKV Matmul
            matmul_transposed(x_norm.data(), layer.q_w.data(), q.data(), n_patches, vit_dim_, vit_dim_);
            matmul_transposed(x_norm.data(), layer.k_w.data(), k.data(), n_patches, vit_dim_, vit_dim_);
            matmul_transposed(x_norm.data(), layer.v_w.data(), v.data(), n_patches, vit_dim_, vit_dim_);

            // Q/K head norms & V norm
            for (int p = 0; p < n_patches; ++p) {
                for (int h = 0; h < kNumHeads; ++h) {
                    rmsnorm(&q[(p * kNumHeads + h) * kHeadDim], kHeadDim, layer.q_norm_w.data());
                    rmsnorm(&k[(p * kNumHeads + h) * kHeadDim], kHeadDim, layer.k_norm_w.data());
                }
                rmsnorm(&v[p * vit_dim_], vit_dim_);
            }

            // 2D RoPE (Neox ordering with pos_x on first 32 dims, pos_y on second 32 dims)
            const int half_dim = kHeadDim / 2; // 32
            const float theta = 10000.0f;
            std::vector<float> freqs(half_dim / 2);
            for (int i = 0; i < half_dim / 2; ++i) {
                freqs[i] = 1.0f / std::pow(theta, (2.0f * i) / half_dim);
            }

            for (int p = 0; p < n_patches; ++p) {
                int px = pos_x[p], py = pos_y[p];
                for (int h = 0; h < kNumHeads; ++h) {
                    const float* q_src = &q[(p * kNumHeads + h) * kHeadDim];
                    const float* k_src = &k[(p * kNumHeads + h) * kHeadDim];
                    float* q_dst = &q_rope[(p * kNumHeads + h) * kHeadDim];
                    float* k_dst = &k_rope[(p * kNumHeads + h) * kHeadDim];

                    // First half (0..31): pos_x
                    for (int i = 0; i < 16; ++i) {
                        float ang = px * freqs[i];
                        float c = std::cos(ang), s = std::sin(ang);
                        q_dst[i]      = q_src[i] * c - q_src[i + 16] * s;
                        q_dst[i + 16] = q_src[i] * s + q_src[i + 16] * c;
                        k_dst[i]      = k_src[i] * c - k_src[i + 16] * s;
                        k_dst[i + 16] = k_src[i] * s + k_src[i + 16] * c;
                    }
                    // Second half (32..63): pos_y
                    for (int i = 0; i < 16; ++i) {
                        float ang = py * freqs[i];
                        float c = std::cos(ang), s = std::sin(ang);
                        q_dst[32 + i]      = q_src[32 + i] * c - q_src[32 + i + 16] * s;
                        q_dst[32 + i + 16] = q_src[32 + i] * s + q_src[32 + i + 16] * c;
                        k_dst[32 + i]      = k_src[32 + i] * c - k_src[32 + i + 16] * s;
                        k_dst[32 + i + 16] = k_src[32 + i] * s + k_src[32 + i + 16] * c;
                    }
                }
            }

            // Bidirectional Multi-Head Attention
            #pragma omp parallel for collapse(2) schedule(static)
            for (int p = 0; p < n_patches; ++p) {
                for (int h = 0; h < kNumHeads; ++h) {
                    const float* q_vec = &q_rope[(p * kNumHeads + h) * kHeadDim];
                    std::vector<float> scores(n_patches);
                    float max_sc = -1e9f;
                    for (int j = 0; j < n_patches; ++j) {
                        const float* k_vec = &k_rope[(j * kNumHeads + h) * kHeadDim];
                        float dot = avx2_dot_product(q_vec, k_vec, kHeadDim);
                        scores[j] = dot;
                        if (dot > max_sc) max_sc = dot;
                    }
                    float sum_exp = 0.0f;
                    for (int j = 0; j < n_patches; ++j) {
                        scores[j] = std::exp(scores[j] - max_sc);
                        sum_exp += scores[j];
                    }
                    float inv_sum = 1.0f / sum_exp;
                    for (int j = 0; j < n_patches; ++j) scores[j] *= inv_sum;

                    float* out_vec = &attn_out[(p * kNumHeads + h) * kHeadDim];
                    std::fill(out_vec, out_vec + kHeadDim, 0.0f);
                    for (int j = 0; j < n_patches; ++j) {
                        float w = scores[j];
                        const float* v_vec = &v[(j * kNumHeads + h) * kHeadDim];
                        for (int d = 0; d < kHeadDim; ++d) {
                            out_vec[d] += w * v_vec[d];
                        }
                    }
                }
            }

            // Attn Out Proj
            matmul_transposed(attn_out.data(), layer.o_w.data(), attn_proj.data(), n_patches, vit_dim_, vit_dim_);
            for (int p = 0; p < n_patches; ++p) {
                rmsnorm(&attn_proj[p * vit_dim_], vit_dim_, layer.attn_post_w.data());
                for (int d = 0; d < vit_dim_; ++d) {
                    x[p * vit_dim_ + d] += attn_proj[p * vit_dim_ + d];
                }
            }

            // Pre-FFN norm
            for (int p = 0; p < n_patches; ++p) {
                std::memcpy(&x_norm[p * vit_dim_], &x[p * vit_dim_], vit_dim_ * sizeof(float));
                rmsnorm(&x_norm[p * vit_dim_], vit_dim_, layer.ln2_w.data());
            }

            // GeGLU FFN
            matmul_transposed(x_norm.data(), layer.gate_w.data(), ffn_gate.data(), n_patches, kIntermediateDim, vit_dim_);
            matmul_transposed(x_norm.data(), layer.up_w.data(), ffn_up.data(), n_patches, kIntermediateDim, vit_dim_);

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < size_t(n_patches) * kIntermediateDim; ++i) {
                float g = ffn_gate[i];
                float gelu = 0.5f * g * (1.0f + std::erf(g * 0.7071067811865475f));
                ffn_geglu[i] = gelu * ffn_up[i];
            }

            matmul_transposed(ffn_geglu.data(), layer.down_w.data(), ffn_proj.data(), n_patches, vit_dim_, kIntermediateDim);

            for (int p = 0; p < n_patches; ++p) {
                rmsnorm(&ffn_proj[p * vit_dim_], vit_dim_, layer.ffn_post_w.data());
                for (int d = 0; d < vit_dim_; ++d) {
                    x[p * vit_dim_ + d] += ffn_proj[p * vit_dim_ + d];
                }
            }
        }

        // 3. Gemma4VisionPooler (2x2 Average Pool + scale sqrt(vit_dim))
        int pooled_y = patches_y / 2;
        int pooled_x = patches_x / 2;
        int n_tokens = pooled_y * pooled_x;
        const float pool_scale = std::sqrt(static_cast<float>(vit_dim_));

        std::vector<float> pooled(static_cast<size_t>(n_tokens) * vit_dim_, 0.0f);
        for (int py = 0; py < pooled_y; ++py) {
            for (int px = 0; px < pooled_x; ++px) {
                int tok_idx = py * pooled_x + px;
                float* dst = &pooled[static_cast<size_t>(tok_idx) * vit_dim_];
                for (int d = 0; d < vit_dim_; ++d) {
                    float sum = 0.0f;
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            int src_idx = (py * 2 + dy) * patches_x + (px * 2 + dx);
                            sum += x[static_cast<size_t>(src_idx) * vit_dim_ + d];
                        }
                    }
                    dst[d] = (sum * 0.25f) * pool_scale;
                }
            }
        }

        // 4. Multimodal Projection to output_dim (e.g. 2560)
        std::vector<std::vector<float>> output_tokens(n_tokens, std::vector<float>(output_dim_));
        for (int t = 0; t < n_tokens; ++t) {
            const float* src = &pooled[static_cast<size_t>(t) * vit_dim_];
            float* dst = output_tokens[t].data();
            for (int o = 0; o < output_dim_; ++o) {
                const float* w_row = &mm_proj_w_[o * vit_dim_];
                dst[o] = avx2_dot_product(src, w_row, vit_dim_);
            }
            // Embedding post projection norm (RMSNorm)
            rmsnorm(dst, output_dim_);
        }

        return output_tokens;
    }

private:
    int vit_dim_ = 768;
    int output_dim_ = 2560;
    int num_layers_ = 16;
    static constexpr int kPatchSize = 16;
    static constexpr int kNumHeads = 12;
    static constexpr int kHeadDim = 64;
    static constexpr int kIntermediateDim = 3072;

    std::vector<float> patch_embd_w_;
    std::vector<float> position_embd_;
    std::vector<VitLayer> layers_;
    std::vector<float> mm_proj_w_;
};

} // namespace

bool VisionEmbedder::load(const std::string& vision_dir) {
    // 1. Try ViT (Gemma-4 E4B, Qwen)
    auto vit = std::make_unique<VitVisionEncoder>();
    if (vit->load(vision_dir)) {
        backend_ = std::move(vit);
        return true;
    }

    // 2. Try Gemma-4 12B linear embedder
    auto gemma12b = std::make_unique<Gemma4UvVisionEncoder>();
    if (gemma12b->load(vision_dir)) {
        backend_ = std::move(gemma12b);
        return true;
    }

    std::cerr << "[VisionEmbedder] Failed to load any supported vision architecture from " << vision_dir << std::endl;
    backend_ = nullptr;
    return false;
}

std::vector<std::vector<float>> VisionEmbedder::encode_image_base64(const std::string& b64_str) {
    if (!backend_) return {};
    std::vector<uint8_t> bytes = base64_decode(b64_str);
    if (bytes.empty()) return {};
    return encode_image_bytes(bytes.data(), bytes.size());
}

std::vector<std::vector<float>> VisionEmbedder::encode_image_bytes(const uint8_t* image_bytes, size_t len) {
    if (!backend_ || !image_bytes || len == 0) return {};

    int width = 0, height = 0, channels = 0;
    bool is_webp = false;
    uint8_t* rgba_data = stbi_load_from_memory(image_bytes, static_cast<int>(len), &width, &height, &channels, 4);
    if (!rgba_data) {
        rgba_data = decode_webp_rgba(image_bytes, len, &width, &height);
        if (rgba_data) {
            is_webp = true;
        } else {
            std::cerr << "[VisionEmbedder] Failed to decode image format" << std::endl;
            return {};
        }
    }

    // Composite RGBA onto solid white background into RGB
    std::vector<uint8_t> rgb_data(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        float a = rgba_data[i * 4 + 3] / 255.0f;
        rgb_data[i * 3 + 0] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, std::round(rgba_data[i * 4 + 0] * a + 255.0f * (1.0f - a)))));
        rgb_data[i * 3 + 1] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, std::round(rgba_data[i * 4 + 1] * a + 255.0f * (1.0f - a)))));
        rgb_data[i * 3 + 2] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, std::round(rgba_data[i * 4 + 2] * a + 255.0f * (1.0f - a)))));
    }

    if (is_webp) {
        free(rgba_data);
    } else {
        stbi_image_free(rgba_data);
    }

    return encode_rgb(rgb_data.data(), width, height);
}

std::vector<std::vector<float>> VisionEmbedder::encode_rgb(const uint8_t* rgb_data, int width, int height) {
    if (!backend_) return {};
    return backend_->encode_rgb(rgb_data, width, height);
}

} // namespace alveare
