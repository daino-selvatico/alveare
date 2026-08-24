#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace alveare {

class VisionEmbedder {
public:
    VisionEmbedder() = default;
    ~VisionEmbedder() = default;

    // Load vision weights from directory (e.g. quantized_weights_gemma4/vision)
    bool load(const std::string& vision_dir);
    bool is_loaded() const { return loaded_; }

    // Encode an image given raw file bytes (PNG, JPG, WebP, etc.)
    // Returns N embeddings of size hidden_size (3840)
    std::vector<std::vector<float>> encode_image_bytes(const uint8_t* image_bytes, size_t len);

    // Encode an image given base64 string (supports data:image/...;base64, prefix)
    std::vector<std::vector<float>> encode_image_base64(const std::string& b64_str);

    // Encode RGB uint8 image [height x width x 3]
    std::vector<std::vector<float>> encode_rgb(const uint8_t* rgb_data, int width, int height);

private:
    bool loaded_ = false;
    int hidden_size_ = 3840;
    static constexpr int kPatchSize = 48;
    static constexpr int kPatchDim = 48 * 48 * 3; // 6912

    std::vector<float> patch_norm1_w_;
    std::vector<float> patch_norm1_b_;
    std::vector<float> patch_embd_w_; // 3840 x 6912
    std::vector<float> patch_embd_b_; // 3840
    std::vector<float> patch_norm2_w_;
    std::vector<float> patch_norm2_b_;
    std::vector<float> position_embd_; // 2 x 1120 x 3840
    std::vector<float> patch_norm3_w_;
    std::vector<float> patch_norm3_b_;
    std::vector<float> mm_proj_w_;     // 3840 x 3840
};

} // namespace alveare
