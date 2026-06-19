#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

#ifndef DIM_H
#define DIM_H 64
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 32
#endif

void attention(
    const bfloat16 *restrict q_group,
    const bfloat16 *restrict kv_group,
    bfloat16 *restrict o_group
) {
    float scale = 1.0f / 8.0f; // 1 / sqrt(64)
    
    for (int q = 0; q < 4; ++q) {
        const bfloat16 *q_head = &q_group[q * DIM_H];
        
        float scores[MAX_SEQ_LEN];
        float max_score = -1e9f;
        
        // 1. Compute attention scores: Q @ K.T / sqrt(64)
        for (int t = 0; t < MAX_SEQ_LEN; ++t) {
            const bfloat16 *k_ptr = &kv_group[t * DIM_H * 2];
            float dot = 0.0f;
            for (int d = 0; d < DIM_H; ++d) {
                dot += (float)q_head[d] * (float)k_ptr[d];
            }
            float score = dot * scale;
            scores[t] = score;
            if (score > max_score) {
                max_score = score;
            }
        }
        
        // 2. Softmax exponentials
        float sum_exp = 0.0f;
        for (int t = 0; t < MAX_SEQ_LEN; ++t) {
            float x = scores[t] - max_score;
            
            int32_t ix = (int32_t)(x * 1.442695040888963f);
            float fx = x * 1.442695040888963f - ix;
            ix = (ix + 127) << 23;
            float pow2_ix;
            memcpy(&pow2_ix, &ix, sizeof(float));
            float pow2_fx = 1.0f + 0.6931471805599453f * fx + 0.2401598148889220f * fx * fx;
            float exp_val = pow2_ix * pow2_fx;
            
            scores[t] = exp_val;
            sum_exp += exp_val;
        }
        
        // 3. Normalization
        float inv_sum = 1.0f / (sum_exp + 1e-7f);
        for (int t = 0; t < MAX_SEQ_LEN; ++t) {
            scores[t] *= inv_sum;
        }
        
        // 4. Value aggregation: scores @ V
        bfloat16 *o_head = &o_group[q * DIM_H];
        for (int d = 0; d < DIM_H; ++d) {
            float sum = 0.0f;
            for (int t = 0; t < MAX_SEQ_LEN; ++t) {
                const bfloat16 *v_ptr = &kv_group[t * DIM_H * 2 + DIM_H];
                sum += scores[t] * (float)v_ptr[d];
            }
            o_head[d] = (bfloat16)sum;
        }
    }
}

}
