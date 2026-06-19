#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_K
#define DIM_K 2048
#endif

void rmsnorm(
    const bfloat16 *restrict x,
    const float *restrict w,
    bfloat16 *restrict y
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < DIM_K; ++i) {
        float val = (float)x[i];
        sum_sq += val * val;
    }
    
    float mean_sq = sum_sq / (float)DIM_K;
    // epsilon is 1e-5
    float inv_std = aie::invsqrt(mean_sq + 1e-5f);
    
    for (int i = 0; i < DIM_K; ++i) {
        y[i] = (bfloat16)((float)x[i] * inv_std * w[i]);
    }
}

}
