#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 256
#endif

void gemv_q(
    const uint8_t *restrict w_combined,
    const bfloat16 *restrict x,
    bfloat16 *restrict y
) {
    // Loop over rows of the weight matrix tile (DIM_M)
    for (int r = 0; r < DIM_M; ++r) {
        float sum = 0.0f;
        
        // Pointer to the start of the row in the combined weight buffer.
        // Each block of 32 elements takes 20 bytes.
        // There are (DIM_K / 32) blocks per row.
        const uint8_t *row_ptr = &w_combined[r * (DIM_K / 32) * 20];
        
        // Loop over blocks of 32 columns along the K dimension (DIM_K // 32)
        for (int b = 0; b < DIM_K / 32; ++b) {
            const uint8_t *block_ptr = &row_ptr[b * 20];
            
            // Extract scale from bytes 16 and 17 of the block
            bfloat16 scale = *(const bfloat16 *)&block_ptr[16];
            
            // Broadcast the scale to a vector
            aie::vector<bfloat16, 16> scale_v = aie::broadcast<bfloat16, 16>(scale);
            
            // Load 16 bytes of packed weights using unaligned load
            aie::vector<int8_t, 16> packed_w = aie::load_unaligned_v<16>((const int8_t*)&block_ptr[0]);
            
            // Unpack to int16 first
            aie::vector<int16_t, 16> unpacked_w = packed_w.unpack();
            
            // Extract q0 and q1 using bitwise shifts on the int16 vector
            using namespace aie::operators;
            aie::vector<int16_t, 16> q0_i16 = (unpacked_w << 12) >> 12;
            aie::vector<int16_t, 16> q1_i16 = (unpacked_w << 8) >> 12;
            
            // Convert to bfloat16
            aie::vector<bfloat16, 16> q0_bf16 = aie::to_float<bfloat16>(q0_i16);
            aie::vector<bfloat16, 16> q1_bf16 = aie::to_float<bfloat16>(q1_i16);
            
            // Dequantize weights: weight * scale
            aie::vector<bfloat16, 16> w0_bf16 = aie::mul(q0_bf16, scale_v).to_vector<bfloat16>();
            aie::vector<bfloat16, 16> w1_bf16 = aie::mul(q1_bf16, scale_v).to_vector<bfloat16>();
            
            // Load activations
            aie::vector<bfloat16, 32> x_bf16 = aie::load_v<32>(&x[b * 32]);
            aie::vector<bfloat16, 16> x0_bf16 = aie::filter_even(x_bf16, 1);
            aie::vector<bfloat16, 16> x1_bf16 = aie::filter_odd(x_bf16, 1);
            
            // Multiply and accumulate block dot product in FP32
            aie::accum<accfloat, 16> prod0 = aie::mul(w0_bf16, x0_bf16);
            aie::accum<accfloat, 16> prod1 = aie::mac(prod0, w1_bf16, x1_bf16);
            
            // Sum all elements of prod1
            sum += aie::reduce_add(prod1.to_vector<float>());
        }
        
        // Accumulate into the output tile (across multiple column blocks of the full matrix K)
        y[r] = (bfloat16)((float)y[r] + sum);
    }
}

}
