#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_B
#define DIM_B 16
#endif

// Zero the fp32 output accumulator tile (DIM_B x DIM_M).
void zero_kernel_bf16(float *restrict cOut) {
    aie::vector<float, 16> zeros = aie::broadcast<float, 16>(0.0f);
    for (int i = 0; i < DIM_M * DIM_B; i += 16) {
        aie::store_v(cOut + i, zeros);
    }
}

}
