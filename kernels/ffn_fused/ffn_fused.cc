#include <stdint.h>
#include <string.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_M
#define DIM_M 32 // Tile size for intermediate dimension (m_I)
#endif

#ifndef DIM_K
#define DIM_K 256 // Tile size for hidden dimension (k_tile, which also equals m_H)
#endif

#ifndef DIM_H
#define DIM_H 2048 // Full hidden dimension (H)
#endif

// Static L1 buffers
static bfloat16 y_accum[DIM_H];
static bfloat16 gate_accum[DIM_M];
static bfloat16 up_accum[DIM_M];
static bfloat16 act[DIM_M];

// Stable fast exponential approximation
inline float exp_approx(float z) {
    int32_t ix = (int32_t)(z * 1.442695040888963f);
    float fx = z * 1.442695040888963f - ix;
    ix = (ix + 127) << 23;
    float pow2_ix;
    memcpy(&pow2_ix, &ix, sizeof(float));
    float pow2_fx = 1.0f + 0.6931471805599453f * fx + 0.2401598148889220f * fx * fx;
    return pow2_ix * pow2_fx;
}

// Stable tanh approximation
inline float tanh_approx(float y) {
    if (y > 9.0f) return 1.0f;
    if (y < -9.0f) return -1.0f;
    float z = 2.0f * y;
    float exp_z = exp_approx(z);
    return (exp_z - 1.0f) / (exp_z + 1.0f);
}

// GELU approximation matching PyTorch: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu_approx(float x) {
    float y = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh_approx(y));
}

// 1. Initialize static y_accum to zero
inline void ffn_init() {
    for (int i = 0; i < DIM_H; ++i) {
        y_accum[i] = (bfloat16)0.0f;
    }
}

// 2. Initialize static gate_accum and up_accum to zero
inline void ffn_init_gate_up() {
    for (int i = 0; i < DIM_M; ++i) {
        gate_accum[i] = (bfloat16)0.0f;
        up_accum[i] = (bfloat16)0.0f;
    }
}

// 3. Compute gate and up tiles of size m_I for a single tile of H (size k_tile)
inline void ffn_compute_gate_up(
    const uint8_t *restrict w_gate,
    const uint8_t *restrict w_up,
    const bfloat16 *restrict x,
    int h_offset
) {
    // Loop over rows of the gate/up weight matrix tile (DIM_M = m_I)
    for (int r = 0; r < DIM_M; ++r) {
        float sum_gate = 0.0f;
        float sum_up = 0.0f;
        
        // Quantized rows are packed: 20 bytes per 32 elements
        const uint8_t *row_gate_ptr = &w_gate[r * (DIM_K / 32) * 20];
        const uint8_t *row_up_ptr = &w_up[r * (DIM_K / 32) * 20];
        
        for (int b = 0; b < DIM_K / 32; ++b) {
            const uint8_t *blk_gate = &row_gate_ptr[b * 20];
            const uint8_t *blk_up = &row_up_ptr[b * 20];
            
            // Extract scales
            bfloat16 scale_gate = *(const bfloat16 *)&blk_gate[16];
            bfloat16 scale_up = *(const bfloat16 *)&blk_up[16];
            
            aie::vector<bfloat16, 16> scale_gate_v = aie::broadcast<bfloat16, 16>(scale_gate);
            aie::vector<bfloat16, 16> scale_up_v = aie::broadcast<bfloat16, 16>(scale_up);
            
            // Load and unpack gate weights
            aie::vector<int8_t, 16> packed_gate = aie::load_unaligned_v<16>((const int8_t*)&blk_gate[0]);
            aie::vector<int16_t, 16> unpacked_gate = packed_gate.unpack();
            using namespace aie::operators;
            aie::vector<int16_t, 16> q0_gate = (unpacked_gate << 12) >> 12;
            aie::vector<int16_t, 16> q1_gate = (unpacked_gate << 8) >> 12;
            aie::vector<bfloat16, 16> q0_gate_bf16 = aie::to_float<bfloat16>(q0_gate);
            aie::vector<bfloat16, 16> q1_gate_bf16 = aie::to_float<bfloat16>(q1_gate);
            aie::vector<bfloat16, 16> w0_gate_bf16 = aie::mul(q0_gate_bf16, scale_gate_v).to_vector<bfloat16>();
            aie::vector<bfloat16, 16> w1_gate_bf16 = aie::mul(q1_gate_bf16, scale_gate_v).to_vector<bfloat16>();
            
            // Load and unpack up weights
            aie::vector<int8_t, 16> packed_up = aie::load_unaligned_v<16>((const int8_t*)&blk_up[0]);
            aie::vector<int16_t, 16> unpacked_up = packed_up.unpack();
            aie::vector<int16_t, 16> q0_up = (unpacked_up << 12) >> 12;
            aie::vector<int16_t, 16> q1_up = (unpacked_up << 8) >> 12;
            aie::vector<bfloat16, 16> q0_up_bf16 = aie::to_float<bfloat16>(q0_up);
            aie::vector<bfloat16, 16> q1_up_bf16 = aie::to_float<bfloat16>(q1_up);
            aie::vector<bfloat16, 16> w0_up_bf16 = aie::mul(q0_up_bf16, scale_up_v).to_vector<bfloat16>();
            aie::vector<bfloat16, 16> w1_up_bf16 = aie::mul(q1_up_bf16, scale_up_v).to_vector<bfloat16>();
            
            // Load input activation slice x
            aie::vector<bfloat16, 32> x_bf16 = aie::load_v<32>(&x[h_offset + b * 32]);
            aie::vector<bfloat16, 16> x0_bf16 = aie::filter_even(x_bf16, 1);
            aie::vector<bfloat16, 16> x1_bf16 = aie::filter_odd(x_bf16, 1);
            
            // Accumulate gate dot product
            aie::accum<accfloat, 16> prod0_gate = aie::mul(w0_gate_bf16, x0_bf16);
            aie::accum<accfloat, 16> prod1_gate = aie::mac(prod0_gate, w1_gate_bf16, x1_bf16);
            sum_gate += aie::reduce_add(prod1_gate.to_vector<float>());
            
            // Accumulate up dot product
            aie::accum<accfloat, 16> prod0_up = aie::mul(w0_up_bf16, x0_bf16);
            aie::accum<accfloat, 16> prod1_up = aie::mac(prod0_up, w1_up_bf16, x1_bf16);
            sum_up += aie::reduce_add(prod1_up.to_vector<float>());
        }
        
        gate_accum[r] = (bfloat16)((float)gate_accum[r] + sum_gate);
        up_accum[r] = (bfloat16)((float)up_accum[r] + sum_up);
    }
}

// 4. Compute activation (GELU(gate) * up) of size m_I
inline void ffn_compute_activation() {
    for (int i = 0; i < DIM_M; ++i) {
        float g_val = (float)gate_accum[i];
        float u_val = (float)up_accum[i];
        act[i] = (bfloat16)(gelu_approx(g_val) * u_val);
    }
}

// 5. Multiply act (size m_I) by W_down tile (shape m_H x m_I, where m_H = DIM_K) and accumulate to y_accum
inline void ffn_accumulate_down(
    const uint8_t *restrict w_down,
    int h_offset
) {
    // Loop over rows of the W_down tile (size DIM_K)
    for (int r = 0; r < DIM_K; ++r) {
        float sum = 0.0f;
        
        // W_down row slice for this r (since DIM_M is m_I, each row has DIM_M columns = DIM_M/32 blocks)
        const uint8_t *row_down_ptr = &w_down[r * (DIM_M / 32) * 20];
        
        for (int b = 0; b < DIM_M / 32; ++b) {
            const uint8_t *blk_down = &row_down_ptr[b * 20];
            
            // Extract scale
            bfloat16 scale_down = *(const bfloat16 *)&blk_down[16];
            aie::vector<bfloat16, 16> scale_down_v = aie::broadcast<bfloat16, 16>(scale_down);
            
            // Load and unpack down weights
            aie::vector<int8_t, 16> packed_down = aie::load_unaligned_v<16>((const int8_t*)&blk_down[0]);
            aie::vector<int16_t, 16> unpacked_down = packed_down.unpack();
            using namespace aie::operators;
            aie::vector<int16_t, 16> q0_down = (unpacked_down << 12) >> 12;
            aie::vector<int16_t, 16> q1_down = (unpacked_down << 8) >> 12;
            aie::vector<bfloat16, 16> q0_down_bf16 = aie::to_float<bfloat16>(q0_down);
            aie::vector<bfloat16, 16> q1_down_bf16 = aie::to_float<bfloat16>(q1_down);
            aie::vector<bfloat16, 16> w0_down_bf16 = aie::mul(q0_down_bf16, scale_down_v).to_vector<bfloat16>();
            aie::vector<bfloat16, 16> w1_down_bf16 = aie::mul(q1_down_bf16, scale_down_v).to_vector<bfloat16>();
            
            // Load act values
            aie::vector<bfloat16, 32> act_bf16 = aie::load_v<32>(&act[b * 32]);
            aie::vector<bfloat16, 16> act0_bf16 = aie::filter_even(act_bf16, 1);
            aie::vector<bfloat16, 16> act1_bf16 = aie::filter_odd(act_bf16, 1);
            
            // Multiply-accumulate
            aie::accum<accfloat, 16> prod0 = aie::mul(w0_down_bf16, act0_bf16);
            aie::accum<accfloat, 16> prod1 = aie::mac(prod0, w1_down_bf16, act1_bf16);
            sum += aie::reduce_add(prod1.to_vector<float>());
        }
        
        y_accum[h_offset + r] = (bfloat16)((float)y_accum[h_offset + r] + sum);
    }
}

// 6. Finalize: copy a slice of y_accum of size DIM_K starting at h_offset to y_out
inline void ffn_finalize(
    bfloat16 *restrict y_out,
    int h_offset
) {
    for (int i = 0; i < DIM_K; ++i) {
        y_out[i] = y_accum[h_offset + i];
    }
}

// Unified step entry point to prevent linker duplicate symbols
void ffn_fused_step(
    int step,
    const uint8_t *restrict w0,
    const uint8_t *restrict w1,
    const bfloat16 *restrict x,
    bfloat16 *restrict y,
    int h_offset
) {
    if (step == 0) {
        ffn_init();
    } else if (step == 1) {
        ffn_init_gate_up();
    } else if (step == 2) {
        ffn_compute_gate_up(w0, w1, x, h_offset);
    } else if (step == 3) {
        ffn_compute_activation();
    } else if (step == 4) {
        ffn_accumulate_down(w0, h_offset);
    } else if (step == 5) {
        ffn_finalize(y, h_offset);
    }
}

}
