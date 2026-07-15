#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef DIM_K
#define DIM_K 256
#endif

#ifndef DIM_H
#define DIM_H 2048
#endif

extern bfloat16 y_accum[DIM_H];

extern "C" {

void ffn_finalize(
    bfloat16 *restrict y_out,
    int h_offset
) {
    for (int i = 0; i < DIM_K; ++i) {
        y_out[i] = y_accum[h_offset + i];
    }
}

}
