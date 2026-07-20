#include <stdint.h>
#include <string.h>
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_IC
#define DIM_IC 2048
#endif

extern float gate_accum[DIM_M];
extern float up_accum[DIM_M];
extern bfloat16 act_all[DIM_IC];

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

#ifdef ACTIVATION_SILU
// SiLU/Swish: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
inline float silu_approx(float x) {
    if (x < -9.0f) return 0.0f;
    if (x > 9.0f) return x;
    return x / (1.0f + exp_approx(-x));
}
#endif

extern "C" {

void ffn_compute_activation(int ic_offset) {
    for (int i = 0; i < DIM_M; ++i) {
        float g_val = (float)gate_accum[i];
        float u_val = (float)up_accum[i];
#ifdef ACTIVATION_SILU
        act_all[ic_offset + i] = (bfloat16)(silu_approx(g_val) * u_val);
#else
        act_all[ic_offset + i] = (bfloat16)(gelu_approx(g_val) * u_val);
#endif
    }
}

}
