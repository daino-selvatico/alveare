#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_M
#define DIM_M 32
#endif

void zero_kernel_bf16(bfloat16 *restrict cOut) {
    for (int i = 0; i < DIM_M; ++i) {
        cOut[i] = (bfloat16)0.0f;
    }
}

}
