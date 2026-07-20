#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 256
#endif

extern float gate_accum[DIM_M];
extern float up_accum[DIM_M];

extern "C" {

void ffn_compute_gate_up(
    const uint8_t *restrict w_gate,
    const uint8_t *restrict w_up,
    const bfloat16 *restrict x,
    int h_offset
) {
    for (int r = 0; r < DIM_M; ++r) {
        float sum_gate = 0.0f;
        float sum_up = 0.0f;
        
        const uint8_t *row_gate_ptr = &w_gate[r * (DIM_K / 32) * 20];
        const uint8_t *row_up_ptr = &w_up[r * (DIM_K / 32) * 20];
        
        for (int b = 0; b < DIM_K / 32; ++b) {
            const uint8_t *blk_gate = &row_gate_ptr[b * 20];
            const uint8_t *blk_up = &row_up_ptr[b * 20];
            
            bfloat16 scale_gate = *(const bfloat16 *)&blk_gate[16];
            bfloat16 scale_up = *(const bfloat16 *)&blk_up[16];
            
            aie::vector<bfloat16, 16> scale_gate_v = aie::broadcast<bfloat16, 16>(scale_gate);
            aie::vector<bfloat16, 16> scale_up_v = aie::broadcast<bfloat16, 16>(scale_up);
            
            aie::vector<int8_t, 16> packed_gate = aie::load_unaligned_v<16>((const int8_t*)&blk_gate[0]);
            aie::vector<int16_t, 16> unpacked_gate = packed_gate.unpack();
            using namespace aie::operators;
            aie::vector<int16_t, 16> q0_gate = (unpacked_gate << 12) >> 12;
            aie::vector<int16_t, 16> q1_gate = (unpacked_gate << 8) >> 12;
            aie::vector<bfloat16, 16> q0_gate_bf16 = aie::to_float<bfloat16>(q0_gate);
            aie::vector<bfloat16, 16> q1_gate_bf16 = aie::to_float<bfloat16>(q1_gate);
            aie::vector<bfloat16, 16> w0_gate_bf16 = aie::mul(q0_gate_bf16, scale_gate_v).to_vector<bfloat16>();
            aie::vector<bfloat16, 16> w1_gate_bf16 = aie::mul(q1_gate_bf16, scale_gate_v).to_vector<bfloat16>();
            
            aie::vector<int8_t, 16> packed_up = aie::load_unaligned_v<16>((const int8_t*)&blk_up[0]);
            aie::vector<int16_t, 16> unpacked_up = packed_up.unpack();
            aie::vector<int16_t, 16> q0_up = (unpacked_up << 12) >> 12;
            aie::vector<int16_t, 16> q1_up = (unpacked_up << 8) >> 12;
            aie::vector<bfloat16, 16> q0_up_bf16 = aie::to_float<bfloat16>(q0_up);
            aie::vector<bfloat16, 16> q1_up_bf16 = aie::to_float<bfloat16>(q1_up);
            aie::vector<bfloat16, 16> w0_up_bf16 = aie::mul(q0_up_bf16, scale_up_v).to_vector<bfloat16>();
            aie::vector<bfloat16, 16> w1_up_bf16 = aie::mul(q1_up_bf16, scale_up_v).to_vector<bfloat16>();
            
            // Use load_unaligned_v for safety against unaligned L1 activation buffer pointers
            aie::vector<bfloat16, 32> x_bf16 = aie::load_unaligned_v<32>(&x[h_offset + b * 32]);
            aie::vector<bfloat16, 16> x0_bf16 = aie::filter_even(x_bf16, 1);
            aie::vector<bfloat16, 16> x1_bf16 = aie::filter_odd(x_bf16, 1);
            
            aie::accum<accfloat, 16> prod0_gate = aie::mul(w0_gate_bf16, x0_bf16);
            aie::accum<accfloat, 16> prod1_gate = aie::mac(prod0_gate, w1_gate_bf16, x1_bf16);
            sum_gate += aie::reduce_add(prod1_gate.to_vector<float>());
            
            aie::accum<accfloat, 16> prod0_up = aie::mul(w0_up_bf16, x0_bf16);
            aie::accum<accfloat, 16> prod1_up = aie::mac(prod0_up, w1_up_bf16, x1_bf16);
            sum_up += aie::reduce_add(prod1_up.to_vector<float>());
        }
        
        gate_accum[r] = gate_accum[r] + sum_gate;
        up_accum[r] = up_accum[r] + sum_up;
    }
}

}
