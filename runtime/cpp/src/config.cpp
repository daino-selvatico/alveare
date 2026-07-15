#include "alveare/config.h"

namespace alveare {

ModelConfig load_config(const std::string& path) {
    // Scaffold: We'll use nlohmann/json or similar later.
    ModelConfig cfg{};
    cfg.hidden_size = 2048;
    return cfg;
}

} // namespace alveare
