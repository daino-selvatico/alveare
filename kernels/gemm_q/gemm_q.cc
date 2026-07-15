#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 256
#endif

#ifndef DIM_B
#define DIM_B 16
#endif

void gemm_q(
    const uint8_t *restrict w_combined,
    const bfloat16 *restrict x,
    bfloat16 *restrict y
) {
    // Loop over rows of the weight matrix tile (DIM_M)
    for (int r = 0; r < DIM_M; ++r) {
        const uint8_t *row_ptr = &w_combined[r * (DIM_K / 32) * 20];
        
        for (int batch = 0; batch < DIM_B; ++batch) {
            float sum = 0.0f;
            
            // Loop over blocks of 32 columns along the K dimension (DIM_K // 32)
            for (int b = 0; b < DIM_K / 32; ++b) {
                const uint8_t *block_ptr = &row_ptr[b * 20];
                bfloat16 scale = *(const bfloat16 *)&block_ptr[16];
                aie::vector<bfloat16, 16> scale_v = aie::broadcast<bfloat16, 16>(scale);
                aie::vector<int8_t, 16> packed_w = aie::load_unaligned_v<16>((const int8_t*)&block_ptr[0]);
                aie::vector<int16_t, 16> unpacked_w = packed_w.unpack();
                using namespace aie::operators;
                aie::vector<int16_t, 16> q0_i16 = (unpacked_w << 12) >> 12;
                aie::vector<int16_t, 16> q1_i16 = (unpacked_w << 8) >> 12;
                aie::vector<bfloat16, 16> q0_bf16 = aie::to_float<bfloat16>(q0_i16);
                aie::vector<bfloat16, 16> q1_bf16 = aie::to_float<bfloat16>(q1_i16);
                aie::vector<bfloat16, 16> w0_bf16 = aie::mul(q0_bf16, scale_v).to_vector<bfloat16>();
                aie::vector<bfloat16, 16> w1_bf16 = aie::mul(q1_bf16, scale_v).to_vector<bfloat16>();
                aie::vector<bfloat16, 32> x_bf16 = aie::load_v<32>(&x[batch * DIM_K + b * 32]);
                aie::vector<bfloat16, 16> x0_bf16 = aie::filter_even(x_bf16, 1);
                aie::vector<bfloat16, 16> x1_bf16 = aie::filter_odd(x_bf16, 1);
                aie::accum<accfloat, 16> prod0 = aie::mul(w0_bf16, x0_bf16);
                aie::accum<accfloat, 16> prod1 = aie::mac(prod0, w1_bf16, x1_bf16);
                sum += aie::reduce_add(prod1.to_vector<float>());
            }
            y[batch * DIM_M + r] = (bfloat16)((float)y[batch * DIM_M + r] + sum);
        }
    }
}

}
