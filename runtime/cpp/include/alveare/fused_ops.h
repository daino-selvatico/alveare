#pragma once

#include "alveare/bf16.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace alveare {

// Convert 8 consecutive bf16 elements to __m256 float vector
static inline __m256 bf16_to_m256(const bf16* ptr) {
    __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    __m256i wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
    return _mm256_castsi256_ps(wide);
}

// Convert __m256 float vector and store 8 consecutive bf16 elements
static inline void m256_to_bf16(__m256 val, bf16* ptr) {
    __m256i sh = _mm256_srli_epi32(_mm256_castps_si256(val), 16);
    __m128i p = _mm_packus_epi32(_mm256_castsi256_si128(sh), _mm256_extractf128_si256(sh, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), p);
}

// Fast AVX2 Vectorized RMSNorm
void run_rmsnorm_avx2(const bf16* x, const float* w, bf16* out, int K, float eps = 1e-6f);

// Multihead RMSNorm (Q, K, V heads) in cache
void run_multihead_rmsnorm_avx2(bf16* qkv, const float* w, int num_heads, int head_dim, float eps = 1e-6f);

// Fast Vectorized GeGLU for FFN: act = 0.5 * g * (1 + erf(g / sqrt(2))) * u
void run_fast_geglu_avx2(const bf16* gu, bf16* act, int I_rows);

// Fused Residual + RMSNorm:
// x_res = x_in + RMSNorm(proj_in, post_norm_w)
// x_next_norm = RMSNorm(x_res, next_norm_w)
void run_fused_residual_and_norm_avx2(
    const bf16* x_in,
    const bf16* proj_in,
    const float* post_norm_w,
    const float* next_norm_w,
    bf16* x_res_out,
    bf16* x_next_norm_out,
    int K,
    float eps = 1e-6f
);

// Fused accumulation for Down projection tiles
void run_fused_down_accumulate_avx2(const bf16* part, float* acc, int K);

} // namespace alveare
