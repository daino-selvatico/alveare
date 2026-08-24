#include "alveare/vision_embedder.h"
#include "alveare/weights.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

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
        __m256 vb = _mm256_loadu_ps(bias + i);
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(vx, v_mean), v_inv_std);
        __m256 res = _mm256_fmadd_ps(norm, vw, vb);
        _mm256_storeu_ps(x + i, res);
    }
    for (; i < dim; ++i) {
        x[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
    }
}

} // namespace

bool VisionEmbedder::load(const std::string& vision_dir) {
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
            std::cerr << "[VisionEmbedder] Failed to load some weights from " << vision_dir << std::endl;
            loaded_ = false;
            return false;
        }

        hidden_size_ = static_cast<int>(patch_embd_b_.size());
        loaded_ = true;
        std::cout << "[VisionEmbedder] Successfully loaded 35M vision embedder from " << vision_dir 
                  << " (hidden_size=" << hidden_size_ << ")" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[VisionEmbedder] Error loading weights: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<std::vector<float>> VisionEmbedder::encode_image_base64(const std::string& b64_str) {
    if (!loaded_) return {};
    std::vector<uint8_t> bytes = base64_decode(b64_str);
    if (bytes.empty()) return {};
    return encode_image_bytes(bytes.data(), bytes.size());
}

#include <dlfcn.h>

static uint8_t* decode_webp_rgb(const uint8_t* data, size_t len, int* w, int* h) {
    static void* handle = nullptr;
    static uint8_t* (*p_WebPDecodeRGB)(const uint8_t*, size_t, int*, int*) = nullptr;
    if (!handle) {
        handle = dlopen("libwebp.so.7", RTLD_LAZY);
        if (!handle) handle = dlopen("libwebp.so", RTLD_LAZY);
        if (handle) {
            p_WebPDecodeRGB = (uint8_t* (*)(const uint8_t*, size_t, int*, int*))dlsym(handle, "WebPDecodeRGB");
        }
    }
    if (p_WebPDecodeRGB) {
        return p_WebPDecodeRGB(data, len, w, h);
    }
    return nullptr;
}

std::vector<std::vector<float>> VisionEmbedder::encode_image_bytes(const uint8_t* image_bytes, size_t len) {
    if (!loaded_ || !image_bytes || len == 0) return {};

    int width = 0, height = 0, channels = 0;
    bool is_webp = false;
    uint8_t* rgb_data = stbi_load_from_memory(image_bytes, static_cast<int>(len), &width, &height, &channels, 3);
    if (!rgb_data) {
        // Fallback to WebP decoder
        rgb_data = decode_webp_rgb(image_bytes, len, &width, &height);
        if (rgb_data) {
            is_webp = true;
        } else {
            std::cerr << "[VisionEmbedder] Failed to decode image format (STB and WebP)" << std::endl;
            return {};
        }
    }

    auto result = encode_rgb(rgb_data, width, height);
    if (is_webp) {
        free(rgb_data);
    } else {
        stbi_image_free(rgb_data);
    }
    return result;
}

std::vector<std::vector<float>> VisionEmbedder::encode_rgb(const uint8_t* rgb_data, int width, int height) {
    if (!loaded_ || !rgb_data || width <= 0 || height <= 0) return {};

    // 1. Determine target dimensions (multiples of 48, max dimension 480)
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

    // Working buffer for one patch
    std::vector<float> patch_buf(kPatchDim);
    std::vector<float> emb_buf(hidden_size_);

    int patch_idx = 0;
    for (int py = 0; py < num_patches_y; ++py) {
        for (int px = 0; px < num_patches_x; ++px) {
            // Extract planar 3x48x48 patch normalized to [-1.0, 1.0]
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

            // LayerNorm 1 on raw patch
            layernorm(patch_buf.data(), kPatchDim, patch_norm1_w_.data(), patch_norm1_b_.data());

            // Linear projection: patch_embd_w (3840 x 6912) * patch_buf (6912) + patch_embd_b (3840)
            for (int h = 0; h < hidden_size_; ++h) {
                const float* w_row = &patch_embd_w_[h * kPatchDim];
                emb_buf[h] = avx2_dot_product(w_row, patch_buf.data(), kPatchDim) + patch_embd_b_[h];
            }

            // LayerNorm 2
            layernorm(emb_buf.data(), hidden_size_, patch_norm2_w_.data(), patch_norm2_b_.data());

            // Add 2D factorized positional embedding: pos[0, px, :] + pos[1, py, :]
            // position_embd_ shape is [2, 1120, 3840]
            const float* pos_x = &position_embd_[(0 * 1120 + px) * hidden_size_];
            const float* pos_y = &position_embd_[(1 * 1120 + py) * hidden_size_];
            for (int h = 0; h < hidden_size_; ++h) {
                emb_buf[h] += pos_x[h] + pos_y[h];
            }

            // LayerNorm 3
            layernorm(emb_buf.data(), hidden_size_, patch_norm3_w_.data(), patch_norm3_b_.data());

            // Multimodal projection: mm_proj_w (3840 x 3840) * emb_buf (3840)
            for (int h = 0; h < hidden_size_; ++h) {
                const float* w_row = &mm_proj_w_[h * hidden_size_];
                output_tokens[patch_idx][h] = avx2_dot_product(w_row, emb_buf.data(), hidden_size_);
            }

            patch_idx++;
        }
    }

    return output_tokens;
}

} // namespace alveare
