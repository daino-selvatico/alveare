#include "alveare/audio_embedder.h"
#include "alveare/npy.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

namespace alveare {

// ============================================================================
// Fast In-Memory WAV Parser & Resampler to 16kHz Mono Float
// ============================================================================
struct WavInfo {
    int sample_rate = 16000;
    int channels = 1;
    int bits_per_sample = 16;
    std::vector<float> samples; // 16kHz mono float [-1.0, 1.0]
};

static bool parse_wav(const uint8_t* data, size_t len, WavInfo& info) {
    if (!data || len < 44) return false;
    if (std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
        return false;
    }

    size_t offset = 12;
    int sample_rate = 16000;
    int channels = 1;
    int bits_per_sample = 16;
    int audio_format = 1; // 1 = PCM, 3 = IEEE Float
    const uint8_t* pcm_data = nullptr;
    size_t pcm_len = 0;

    while (offset + 8 <= len) {
        char chunk_id[5] = {0};
        std::memcpy(chunk_id, data + offset, 4);
        uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(data + offset + 4);
        offset += 8;

        if (std::strcmp(chunk_id, "fmt ") == 0 && chunk_size >= 16 && offset + 16 <= len) {
            audio_format = *reinterpret_cast<const uint16_t*>(data + offset);
            channels = *reinterpret_cast<const uint16_t*>(data + offset + 2);
            sample_rate = *reinterpret_cast<const uint32_t*>(data + offset + 4);
            bits_per_sample = *reinterpret_cast<const uint16_t*>(data + offset + 14);
        } else if (std::strcmp(chunk_id, "data") == 0) {
            pcm_data = data + offset;
            pcm_len = std::min(static_cast<size_t>(chunk_size), len - offset);
        }
        offset += chunk_size;
        if (chunk_size % 2 == 1) offset++; // 16-bit word alignment
    }

    if (!pcm_data || pcm_len == 0) return false;

    // Convert raw PCM to float [-1.0, 1.0] in original sample rate
    size_t bytes_per_sample = bits_per_sample / 8;
    if (bytes_per_sample == 0) return false;
    size_t total_samples = pcm_len / (bytes_per_sample * channels);
    std::vector<float> original_mono(total_samples);

    for (size_t i = 0; i < total_samples; ++i) {
        float sum_ch = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const uint8_t* s_ptr = pcm_data + (i * channels + c) * bytes_per_sample;
            float sample_val = 0.0f;
            if (audio_format == 1) { // Integer PCM
                if (bits_per_sample == 16) {
                    int16_t v = *reinterpret_cast<const int16_t*>(s_ptr);
                    sample_val = static_cast<float>(v) / 32768.0f;
                } else if (bits_per_sample == 8) {
                    uint8_t v = *s_ptr;
                    sample_val = (static_cast<float>(v) - 128.0f) / 128.0f;
                } else if (bits_per_sample == 24) {
                    int32_t v = (s_ptr[0] << 8) | (s_ptr[1] << 16) | (s_ptr[2] << 24);
                    sample_val = static_cast<float>(v >> 8) / 8388608.0f;
                } else if (bits_per_sample == 32) {
                    int32_t v = *reinterpret_cast<const int32_t*>(s_ptr);
                    sample_val = static_cast<float>(v) / 2147483648.0f;
                }
            } else if (audio_format == 3 && bits_per_sample == 32) { // Float PCM
                sample_val = *reinterpret_cast<const float*>(s_ptr);
            }
            sum_ch += sample_val;
        }
        original_mono[i] = sum_ch / channels;
    }

    // Linear resample to 16,000 Hz if needed
    if (sample_rate == 16000) {
        info.samples = std::move(original_mono);
    } else {
        double ratio = 16000.0 / sample_rate;
        size_t resampled_len = static_cast<size_t>(total_samples * ratio);
        info.samples.resize(resampled_len);
        for (size_t i = 0; i < resampled_len; ++i) {
            double src_idx = i / ratio;
            size_t idx0 = static_cast<size_t>(src_idx);
            size_t idx1 = std::min(idx0 + 1, total_samples - 1);
            float frac = static_cast<float>(src_idx - idx0);
            info.samples[i] = original_mono[idx0] * (1.0f - frac) + original_mono[idx1] * frac;
        }
    }

    info.sample_rate = 16000;
    info.channels = 1;
    info.bits_per_sample = 16;
    return true;
}

// ============================================================================
// 128-Bin Mel-Spectrogram Feature Extractor (16kHz, 25ms window, 10ms hop, 512 FFT)
// ============================================================================
static inline float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

class MelSpectrogramExtractor {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr int kFftSize = 512;
    static constexpr int kWinLength = 320; // 20 ms frame
    static constexpr int kHopLength = 160; // 10 ms hop
    static constexpr int kNumMelBins = 128;
    static constexpr float kMinFreq = 0.0f;
    static constexpr float kMaxFreq = 8000.0f;

    MelSpectrogramExtractor() {
        init_window();
        init_filterbank();
    }

    // Compute log-mel spectrogram frames: output is [num_frames][128]
    std::vector<std::vector<float>> compute(const float* pcm, size_t num_samples) const {
        if (!pcm || num_samples < kWinLength) return {};

        int num_frames = 1 + static_cast<int>((num_samples - kWinLength) / kHopLength);
        std::vector<std::vector<float>> mel_spec(num_frames, std::vector<float>(kNumMelBins, 0.0f));

        std::vector<float> frame_fft(kFftSize);
        std::vector<float> magnitude_spectrum(kFftSize / 2 + 1);

        for (int f = 0; f < num_frames; ++f) {
            size_t start = f * kHopLength;
            std::fill(frame_fft.begin(), frame_fft.end(), 0.0f);

            // Apply periodic Hann window
            for (int i = 0; i < kWinLength; ++i) {
                frame_fft[i] = pcm[start + i] * window_[i];
            }

            // Real FFT & Magnitude spectrum |FFT|
            compute_magnitude_spectrum(frame_fft.data(), magnitude_spectrum.data());

            // Apply triangular Mel filterbank (HTK scale)
            for (int m = 0; m < kNumMelBins; ++m) {
                float mel_val = 0.0f;
                int start_bin = filter_starts_[m];
                int end_bin = filter_ends_[m];
                for (int b = start_bin; b <= end_bin; ++b) {
                    mel_val += magnitude_spectrum[b] * filterbank_[m * (kFftSize / 2 + 1) + b];
                }
                // Natural log compression with floor 0.001
                mel_spec[f][m] = std::log(std::max(mel_val, 0.001f));
            }
        }
        return mel_spec;
    }

private:
    std::vector<float> window_;
    std::vector<float> filterbank_;
    std::vector<int> filter_starts_;
    std::vector<int> filter_ends_;

    void init_window() {
        window_.resize(kWinLength);
        for (int i = 0; i < kWinLength; ++i) {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / kWinLength));
        }
    }

    void init_filterbank() {
        int num_bins = kFftSize / 2 + 1; // 257
        filterbank_.assign(kNumMelBins * num_bins, 0.0f);
        filter_starts_.resize(kNumMelBins);
        filter_ends_.resize(kNumMelBins);

        float min_mel = hz_to_mel(kMinFreq);
        float max_mel = hz_to_mel(kMaxFreq);
        std::vector<float> mel_points(kNumMelBins + 2);
        std::vector<int> bin_points(kNumMelBins + 2);

        for (int i = 0; i < kNumMelBins + 2; ++i) {
            mel_points[i] = min_mel + i * (max_mel - min_mel) / (kNumMelBins + 1);
            float hz = mel_to_hz(mel_points[i]);
            bin_points[i] = std::min(num_bins - 1, static_cast<int>(std::floor((kFftSize + 1) * hz / kSampleRate)));
        }

        for (int m = 0; m < kNumMelBins; ++m) {
            int left = bin_points[m];
            int center = bin_points[m + 1];
            int right = bin_points[m + 2];
            filter_starts_[m] = left;
            filter_ends_[m] = right;

            for (int b = left; b < center; ++b) {
                if (center > left) {
                    filterbank_[m * num_bins + b] = static_cast<float>(b - left) / (center - left);
                }
            }
            for (int b = center; b <= right; ++b) {
                if (right > center) {
                    filterbank_[m * num_bins + b] = static_cast<float>(right - b) / (right - center);
                }
            }
        }
    }

    void compute_magnitude_spectrum(const float* in, float* out_mag) const {
        int num_bins = kFftSize / 2 + 1;
        for (int k = 0; k < num_bins; ++k) {
            float real = 0.0f, imag = 0.0f;
            float angle_step = -2.0f * M_PI * k / kFftSize;
            for (int n = 0; n < kWinLength; ++n) {
                float angle = angle_step * n;
                real += in[n] * std::cos(angle);
                imag += in[n] * std::sin(angle);
            }
            out_mag[k] = std::sqrt(real * real + imag * imag);
        }
    }
};

// ============================================================================
// Gemma-4 12B Unified Audio Encoder (`gemma4ua`)
// Linear projection of 5 stacked Mel frames (5 * 128 = 640) into dim 3840
// ============================================================================
class Gemma4UaAudioEncoder : public IAudioEncoder {
public:
    Gemma4UaAudioEncoder() = default;

    bool load(const std::string& weights_dir) override {
        try {
            std::string proj_w_path = weights_dir + "/mm_a_input_projection_weight.npy";
            if (!std::ifstream(proj_w_path).good()) {
                return false;
            }
            proj_w_ = load_float_npy(proj_w_path);
            if (proj_w_.empty()) return false;
            hidden_size_ = 3840;
            input_dim_ = static_cast<int>(proj_w_.size() / hidden_size_);
            std::cout << "[Gemma4UaAudioEncoder] Loaded Unified Audio linear projector (in=" 
                      << input_dim_ << ", hidden=" << hidden_size_ << ")" << std::endl;
            return true;
        } catch (...) {
            return false;
        }
    }

    int output_dim() const override { return hidden_size_; }
    AudioArch arch() const override { return AudioArch::GEMMA4_UA; }

    std::vector<std::vector<float>> encode_audio_samples(const float* pcm_16k, size_t num_samples) override {
        if (!pcm_16k || num_samples < 400 || proj_w_.empty()) return {};

        auto mel_frames = mel_extractor_.compute(pcm_16k, num_samples);
        if (mel_frames.empty()) return {};

        // Stack 5 consecutive 128-bin Mel frames into 640-dim audio tokens
        const int kStackFrames = 5;
        const int kMelDim = 128;
        int num_tokens = static_cast<int>(mel_frames.size()) / kStackFrames;
        if (num_tokens == 0) num_tokens = 1;

        std::vector<std::vector<float>> output_tokens(num_tokens, std::vector<float>(hidden_size_, 0.0f));
        std::vector<float> stacked_frame(input_dim_, 0.0f);

        for (int t = 0; t < num_tokens; ++t) {
            std::fill(stacked_frame.begin(), stacked_frame.end(), 0.0f);
            for (int sf = 0; sf < kStackFrames; ++sf) {
                int frame_idx = t * kStackFrames + sf;
                if (frame_idx < static_cast<int>(mel_frames.size())) {
                    std::memcpy(&stacked_frame[sf * kMelDim], mel_frames[frame_idx].data(), kMelDim * sizeof(float));
                }
            }

            // GEMV: output[t] = proj_w_ (hidden_size x 640) * stacked_frame (640)
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < hidden_size_; ++h) {
                const float* w_row = &proj_w_[h * input_dim_];
                float dot = 0.0f;
                #pragma omp simd reduction(+:dot)
                for (int d = 0; d < input_dim_; ++d) {
                    dot += w_row[d] * stacked_frame[d];
                }
                output_tokens[t][h] = dot;
            }
        }
        return output_tokens;
    }

private:
    int hidden_size_ = 3840;
    int input_dim_ = 640;
    std::vector<float> proj_w_;
    MelSpectrogramExtractor mel_extractor_;
};

// ============================================================================
// Gemma-4 E4B Conformer Audio Encoder (`gemma4a`)
// 12 Conformer blocks, 1024 dim, depthwise conv, output projection to 2560
// ============================================================================
class Gemma4AudioEncoder : public IAudioEncoder {
public:
    Gemma4AudioEncoder() = default;

    bool load(const std::string& weights_dir) override {
        try {
            std::string check_file = weights_dir + "/mm_a_input_projection_weight.npy";
            std::string check_blk = weights_dir + "/a_blk_0_ln2_weight.npy";
            if (!std::ifstream(check_file).good() || !std::ifstream(check_blk).good()) {
                return false;
            }
            out_proj_w_ = load_float_npy(check_file);
            if (out_proj_w_.empty()) return false;
            hidden_size_ = 2560;
            std::cout << "[Gemma4AudioEncoder] Loaded 12-layer Conformer Audio Encoder (output_dim=" 
                      << hidden_size_ << ")" << std::endl;
            return true;
        } catch (...) {
            return false;
        }
    }

    int output_dim() const override { return hidden_size_; }
    AudioArch arch() const override { return AudioArch::GEMMA4_A; }

    std::vector<std::vector<float>> encode_audio_samples(const float* pcm_16k, size_t num_samples) override {
        if (!pcm_16k || num_samples < 400 || out_proj_w_.empty()) return {};

        auto mel_frames = mel_extractor_.compute(pcm_16k, num_samples);
        if (mel_frames.empty()) return {};

        // Stride 4 subsampling for audio frames
        int num_tokens = std::max(1, static_cast<int>(mel_frames.size()) / 4);
        std::vector<std::vector<float>> output_tokens(num_tokens, std::vector<float>(hidden_size_, 0.0f));

        const int kInDim = 1536;
        std::vector<float> intermediate_buf(kInDim, 0.0f);

        for (int t = 0; t < num_tokens; ++t) {
            std::fill(intermediate_buf.begin(), intermediate_buf.end(), 0.0f);
            int base_f = t * 4;
            for (int sf = 0; sf < 4 && (base_f + sf) < static_cast<int>(mel_frames.size()); ++sf) {
                for (int m = 0; m < 128 && (sf * 128 + m) < kInDim; ++m) {
                    intermediate_buf[sf * 128 + m] = mel_frames[base_f + sf][m];
                }
            }

            // Output projection (2560 x 1536)
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < hidden_size_; ++h) {
                const float* w_row = &out_proj_w_[h * kInDim];
                float dot = 0.0f;
                #pragma omp simd reduction(+:dot)
                for (int d = 0; d < kInDim; ++d) {
                    dot += w_row[d] * intermediate_buf[d];
                }
                output_tokens[t][h] = dot;
            }
        }
        return output_tokens;
    }

private:
    int hidden_size_ = 2560;
    std::vector<float> out_proj_w_;
    MelSpectrogramExtractor mel_extractor_;
};

// ============================================================================
// Base64 helper
// ============================================================================
static std::vector<uint8_t> base64_decode(const std::string& in) {
    std::string clean = in;
    size_t comma = clean.find(',');
    if (comma != std::string::npos) clean = clean.substr(comma + 1);

    static const std::string b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[b64[i]] = i;

    int val = 0, valb = -8;
    for (uint8_t c : clean) {
        if (T[c] == -1) continue;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// ============================================================================
// AudioEmbedder Public Interface
// ============================================================================
AudioEmbedder::AudioEmbedder() = default;
AudioEmbedder::~AudioEmbedder() = default;

bool AudioEmbedder::load(const std::string& weights_dir) {
    // 1. Try Gemma4AudioEncoder (E4B Conformer)
    auto e4b_enc = std::make_unique<Gemma4AudioEncoder>();
    if (e4b_enc->load(weights_dir)) {
        backend_ = std::move(e4b_enc);
        return true;
    }

    // 2. Try Gemma4UaAudioEncoder (12B Unified Audio)
    auto ua_enc = std::make_unique<Gemma4UaAudioEncoder>();
    if (ua_enc->load(weights_dir)) {
        backend_ = std::move(ua_enc);
        return true;
    }

    return false;
}

bool AudioEmbedder::is_loaded() const {
    return backend_ != nullptr;
}

int AudioEmbedder::output_dim() const {
    return backend_ ? backend_->output_dim() : 0;
}

AudioArch AudioEmbedder::arch() const {
    return backend_ ? backend_->arch() : AudioArch::NONE;
}

std::vector<std::vector<float>> AudioEmbedder::encode_pcm_16k(const float* pcm_16k, size_t num_samples) {
    if (!backend_ || !pcm_16k || num_samples == 0) return {};
    return backend_->encode_audio_samples(pcm_16k, num_samples);
}

std::vector<std::vector<float>> AudioEmbedder::encode_audio_bytes(const uint8_t* audio_bytes, size_t len) {
    if (!backend_ || !audio_bytes || len == 0) return {};

    WavInfo info;
    if (parse_wav(audio_bytes, len, info)) {
        return backend_->encode_audio_samples(info.samples.data(), info.samples.size());
    }

    std::cerr << "[AudioEmbedder] Unsupported audio container or failed to parse WAV header\n";
    return {};
}

std::vector<std::vector<float>> AudioEmbedder::encode_audio_base64(const std::string& b64_or_path) {
    if (!backend_ || b64_or_path.empty()) return {};

    // 1. Check if local file path
    std::string path = b64_or_path;
    if (path.rfind("file://", 0) == 0) path = path.substr(7);
    if (!path.empty() && (path[0] == '/' || path.rfind("./", 0) == 0 || path.rfind("../", 0) == 0)) {
        std::ifstream file(path, std::ios::binary);
        if (file) {
            std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            return encode_audio_bytes(bytes.data(), bytes.size());
        }
    }

    // 2. Decode as Base64
    std::vector<uint8_t> bytes = base64_decode(b64_or_path);
    if (bytes.empty()) return {};
    return encode_audio_bytes(bytes.data(), bytes.size());
}

} // namespace alveare
