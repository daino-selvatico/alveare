#pragma once

#include <cstdint>

namespace alveare {

struct bf16 {
    uint16_t v;

    bf16() = default;
    explicit bf16(float f);
    float to_float() const;
};

} // namespace alveare
