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
            
            // Dequantize and multiply-accumulate 32 elements
            for (int i = 0; i < 16; ++i) {
                uint8_t byte = block_ptr[i];
                
                // Unpack lower 4 bits (q0) and upper 4 bits (q1) using sign extension
                int8_t q0 = (int8_t)(byte << 4) >> 4;
                int8_t q1 = (int8_t)byte >> 4;
                
                // Load corresponding activation elements
                float x0 = (float)x[b * 32 + 2 * i];
                float x1 = (float)x[b * 32 + 2 * i + 1];
                
                // Dequantize and accumulate
                sum += (float)scale * (float)q0 * x0;
                sum += (float)scale * (float)q1 * x1;
            }
        }
        
        // Accumulate into the output tile (across multiple column blocks of the full matrix K)
        y[r] = (bfloat16)((float)y[r] + sum);
    }
}

}
