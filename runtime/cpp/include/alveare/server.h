#pragma once
#include "alveare/generator.h"
#include <string>

namespace alveare {

class ApiServer {
public:
    ApiServer(Generator& generator);

    // Starts the HTTP server on the specified port.
    void start(int port);
    
    // Stops the HTTP server.
    void stop();

private:
    Generator& generator_;
    void* svr_ptr_{nullptr};
};

} // namespace alveare
