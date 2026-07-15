#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 32
#endif

extern bfloat16 gate_accum[DIM_M];
extern bfloat16 up_accum[DIM_M];

extern "C" {

void ffn_init_gate_up() {
    for (int i = 0; i < DIM_M; ++i) {
        gate_accum[i] = (bfloat16)0.0f;
        up_accum[i] = (bfloat16)0.0f;
    }
}

}
