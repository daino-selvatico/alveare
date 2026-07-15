#pragma once
#include "alveare/generator.h"
#include <string>

namespace alveare {

class ApiServer {
public:
    ApiServer(Generator& generator);

    // Starts the HTTP server on the specified port.
    void start(int port);

private:
    Generator& generator_;
};

} // namespace alveare
