#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 256
#endif

#ifndef DIM_H
#define DIM_H 2048
#endif

// Ensure all global static buffers are aligned to 64-byte vector boundaries
alignas(64) bfloat16 y_accum[DIM_H];
alignas(64) bfloat16 gate_accum[DIM_M];
alignas(64) bfloat16 up_accum[DIM_M];
alignas(64) bfloat16 act[DIM_M];

extern "C" {

void ffn_init() {
    for (int i = 0; i < DIM_H; ++i) {
        y_accum[i] = (bfloat16)0.0f;
    }
}

}
