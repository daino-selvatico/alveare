#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace alveare {

enum class AudioArch {
    NONE,
    GEMMA4_UA,  // 12B Unified Audio (linear projection of 5x128 mel = 640 to 3840)
    GEMMA4_A,   // E4B Conformer Audio Transformer (12 layers, 1024 dim to 2560)
    WHISPER,    // Whisper / Qwen-Audio style
    CUSTOM
};

class IAudioEncoder {
public:
    virtual ~IAudioEncoder() = default;
    virtual bool load(const std::string& weights_dir) = 0;
    virtual int output_dim() const = 0;
    virtual AudioArch arch() const = 0;
    virtual std::vector<std::vector<float>> encode_audio_samples(const float* pcm_16k, size_t num_samples) = 0;
};

class AudioEmbedder {
public:
    AudioEmbedder();
    ~AudioEmbedder();

    bool load(const std::string& weights_dir);
    bool is_loaded() const;
    int output_dim() const;
    AudioArch arch() const;

    // High-level API: accepts raw audio file bytes (WAV, PCM, etc.) or base64 / file path
    std::vector<std::vector<float>> encode_audio_bytes(const uint8_t* audio_bytes, size_t len);
    std::vector<std::vector<float>> encode_audio_base64(const std::string& b64_or_path);
    std::vector<std::vector<float>> encode_pcm_16k(const float* pcm_16k, size_t num_samples);

private:
    std::unique_ptr<IAudioEncoder> backend_;
};

} // namespace alveare
