#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace alveare {

enum class VisionArch {
    GEMMA4_UV, // Linear patch embedder (Gemma-4 12B)
    GEMMA4_V,  // Multi-layer ViT (Gemma-4 E4B)
    QWEN_VL,   // Multi-layer ViT with 2D merger + MLP (Qwen2-VL, Qwen3.5-VL)
    UNKNOWN
};

class IVisionEncoder {
public:
    virtual ~IVisionEncoder() = default;
    virtual bool load(const std::string& vision_dir) = 0;
    virtual std::vector<std::vector<float>> encode_rgb(const uint8_t* rgb_data, int width, int height) = 0;
    virtual int output_dim() const = 0;
    virtual VisionArch arch() const = 0;
};

class VisionEmbedder {
public:
    VisionEmbedder() = default;
    ~VisionEmbedder() = default;

    // Load vision weights from directory (auto-detects architecture: Gemma-4 12B, Gemma-4 E4B, Qwen)
    bool load(const std::string& vision_dir);
    bool is_loaded() const { return backend_ != nullptr; }

    // Encode an image given raw file bytes (PNG, JPG, WebP, etc.)
    std::vector<std::vector<float>> encode_image_bytes(const uint8_t* image_bytes, size_t len);

    // Encode an image given base64 string (supports data:image/...;base64, prefix)
    std::vector<std::vector<float>> encode_image_base64(const std::string& b64_str);

    // Encode RGB uint8 image [height x width x 3]
    std::vector<std::vector<float>> encode_rgb(const uint8_t* rgb_data, int width, int height);

    int output_dim() const { return backend_ ? backend_->output_dim() : 0; }
    VisionArch arch() const { return backend_ ? backend_->arch() : VisionArch::UNKNOWN; }

private:
    std::unique_ptr<IVisionEncoder> backend_;
};

} // namespace alveare
