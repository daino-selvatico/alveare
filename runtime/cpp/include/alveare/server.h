#pragma once
#include "alveare/generator.h"
#include <string>

namespace alveare {

class VisionEmbedder;

class ApiServer {
public:
    ApiServer(Generator& generator, VisionEmbedder* vision_embedder = nullptr);
    void set_vision_embedder(VisionEmbedder* ve) { vision_embedder_ = ve; }

    // Starts the HTTP server on the specified port.
    void start(int port);
    
    // Stops the HTTP server.
    void stop();

private:
    Generator& generator_;
    VisionEmbedder* vision_embedder_{nullptr};
    void* svr_ptr_{nullptr};
};

} // namespace alveare
