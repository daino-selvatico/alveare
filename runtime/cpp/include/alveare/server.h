#pragma once
#include "alveare/generator.h"
#include <string>
#include <mutex>

namespace alveare {

class VisionEmbedder;
class AudioEmbedder;

class ApiServer {
public:
    ApiServer(Generator& generator, VisionEmbedder* vision_embedder = nullptr, AudioEmbedder* audio_embedder = nullptr);
    void set_vision_embedder(VisionEmbedder* ve) { vision_embedder_ = ve; }
    void set_audio_embedder(AudioEmbedder* ae) { audio_embedder_ = ae; }

    // Starts the HTTP server on the specified port.
    void start(int port);
    
    // Stops the HTTP server.
    void stop();

private:
    Generator& generator_;
    VisionEmbedder* vision_embedder_{nullptr};
    AudioEmbedder* audio_embedder_{nullptr};
    std::mutex generate_mutex_;
    void* svr_ptr_{nullptr};
};

} // namespace alveare
