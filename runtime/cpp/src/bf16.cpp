#include "alveare/bf16.h"
#include <cstring>
#include <cmath>

namespace alveare {

bf16::bf16(float f) {
    if (std::isnan(f)) {
        v = 0x7FC0; // quiet NaN
        return;
    }
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    // Round to nearest even
    uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1);
    u += rounding_bias;
    v = static_cast<uint16_t>(u >> 16);
}

float bf16::to_float() const {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

} // namespace alveare
