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

#ifndef DIM_HOUT
#define DIM_HOUT DIM_H
#endif

extern float y_accum[DIM_HOUT];
extern bfloat16 act[DIM_M];

extern "C" {

void ffn_accumulate_down(
    const uint8_t *restrict w_down,
    int h_offset
) {
    for (int r = 0; r < DIM_K; ++r) {
        float sum = 0.0f;
        const uint8_t *row_down_ptr = &w_down[r * (DIM_M / 32) * 20];
        
        for (int b = 0; b < DIM_M / 32; ++b) {
            const uint8_t *blk_down = &row_down_ptr[b * 20];
            
            bfloat16 scale_down = *(const bfloat16 *)&blk_down[16];
            aie::vector<bfloat16, 16> scale_down_v = aie::broadcast<bfloat16, 16>(scale_down);
            
            aie::vector<int8_t, 16> packed_down = aie::load_unaligned_v<16>((const int8_t*)&blk_down[0]);
            aie::vector<int16_t, 16> unpacked_down = packed_down.unpack();
            using namespace aie::operators;
            aie::vector<int16_t, 16> q0_down = (unpacked_down << 12) >> 12;
            aie::vector<int16_t, 16> q1_down = (unpacked_down << 8) >> 12;
            aie::vector<bfloat16, 16> q0_down_bf16 = aie::to_float<bfloat16>(q0_down);
            aie::vector<bfloat16, 16> q1_down_bf16 = aie::to_float<bfloat16>(q1_down);
            aie::vector<bfloat16, 16> w0_down_bf16 = aie::mul(q0_down_bf16, scale_down_v).to_vector<bfloat16>();
            aie::vector<bfloat16, 16> w1_down_bf16 = aie::mul(q1_down_bf16, scale_down_v).to_vector<bfloat16>();
            
            aie::vector<bfloat16, 32> act_bf16 = aie::load_v<32>(&act[b * 32]);
            aie::vector<bfloat16, 16> act0_bf16 = aie::filter_even(act_bf16, 1);
            aie::vector<bfloat16, 16> act1_bf16 = aie::filter_odd(act_bf16, 1);
            
            aie::accum<accfloat, 16> prod0 = aie::mul(w0_down_bf16, act0_bf16);
            aie::accum<accfloat, 16> prod1 = aie::mac(prod0, w1_down_bf16, act1_bf16);
            sum += aie::reduce_add(prod1.to_vector<float>());
        }
        
        y_accum[h_offset + r] = y_accum[h_offset + r] + sum;
    }
}

}
