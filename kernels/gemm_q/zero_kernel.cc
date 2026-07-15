#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_B
#define DIM_B 16
#endif

void zero_kernel_bf16(bfloat16 *restrict cOut) {
    aie::vector<bfloat16, 32> zeros = aie::broadcast<bfloat16, 32>((bfloat16)0.0f);
    for (int i = 0; i < DIM_M * DIM_B; i += 32) {
        aie::store_v(cOut + i, zeros);
    }
}

}
