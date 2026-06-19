#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_K
#define DIM_K 2048
#endif

#ifndef DIM_H
#define DIM_H 64
#endif

void rope(
    const bfloat16 *restrict x,
    const bfloat16 *restrict cos_sin,
    bfloat16 *restrict y
) {
    int num_heads = DIM_K / DIM_H;
    int half_dim = DIM_H / 2;
    
    const bfloat16 *cos_val = cos_sin;
    const bfloat16 *sin_val = &cos_sin[DIM_H];
    
    for (int h = 0; h < num_heads; ++h) {
        const bfloat16 *x_head = &x[h * DIM_H];
        bfloat16 *y_head = &y[h * DIM_H];
        
        for (int i = 0; i < half_dim; ++i) {
            float x1 = (float)x_head[i];
            float x2 = (float)x_head[i + half_dim];
            
            float c = (float)cos_val[i];
            float s = (float)sin_val[i];
            
            y_head[i] = (bfloat16)(x1 * c - x2 * s);
            y_head[i + half_dim] = (bfloat16)(x2 * c + x1 * s);
        }
    }
}

}
