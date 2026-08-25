#include "alveare/fused_ops.h"

namespace alveare {

void run_rmsnorm_avx2(const bf16* x, const float* w, bf16* out, int K, float eps) {
    __m256 sum_sq = _mm256_setzero_ps();
    int i = 0;
    for (; i <= K - 8; i += 8) {
        __m256 v = bf16_to_m256(x + i);
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }
    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(sum_sq), _mm256_extractf128_ps(sum_sq, 1));
    hsum = _mm_add_ps(hsum, _mm_movehl_ps(hsum, hsum));
    hsum = _mm_add_ss(hsum, _mm_shuffle_ps(hsum, hsum, 1));
    float variance = _mm_cvtss_f32(hsum);
    for (; i < K; ++i) {
        float f = x[i].to_float();
        variance += f * f;
    }
    variance /= K;
    float inv_scale = 1.0f / std::sqrt(variance + eps);
    __m256 vinv = _mm256_set1_ps(inv_scale);

    i = 0;
    if (w) {
        for (; i <= K - 8; i += 8) {
            __m256 v = bf16_to_m256(x + i);
            __m256 vw = _mm256_loadu_ps(w + i);
            __m256 res = _mm256_mul_ps(_mm256_mul_ps(v, vinv), vw);
            m256_to_bf16(res, out + i);
        }
    } else {
        for (; i <= K - 8; i += 8) {
            __m256 v = bf16_to_m256(x + i);
            __m256 res = _mm256_mul_ps(v, vinv);
            m256_to_bf16(res, out + i);
        }
    }
    for (; i < K; ++i) {
        float f = x[i].to_float() * inv_scale;
        if (w) f *= w[i];
        out[i] = bf16(f);
    }
}

void run_multihead_rmsnorm_avx2(bf16* qkv, const float* w, int num_heads, int head_dim, float eps) {
    for (int h = 0; h < num_heads; ++h) {
        bf16* head_ptr = qkv + h * head_dim;
        run_rmsnorm_avx2(head_ptr, w, head_ptr, head_dim, eps);
    }
}

// Fast Vectorized GeGLU using standard numerical approximation for erf
// GELU(g) * u where GELU(g) = 0.5 * g * (1 + erf(g * 0.7071067811865475f))
void run_fast_geglu_avx2(const bf16* gu, bf16* act, int I_rows) {
    const bf16* g_ptr = gu;
    const bf16* u_ptr = gu + I_rows;
    const float k_inv_sqrt2 = 0.7071067811865475f;
    const __m256 v_inv_sqrt2 = _mm256_set1_ps(k_inv_sqrt2);
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_one = _mm256_set1_ps(1.0f);

    // Polynomial constants for erf approximation (Abramowitz & Stegun 7.1.26)
    const __m256 p_const  = _mm256_set1_ps(0.3275911f);
    const __m256 a1_const = _mm256_set1_ps(0.254829592f);
    const __m256 a2_const = _mm256_set1_ps(-0.284496736f);
    const __m256 a3_const = _mm256_set1_ps(1.421413741f);
    const __m256 a4_const = _mm256_set1_ps(-1.453152027f);
    const __m256 a5_const = _mm256_set1_ps(1.061405429f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f); // for sign bit

    int i = 0;
    for (; i <= I_rows - 8; i += 8) {
        __m256 vg = bf16_to_m256(g_ptr + i);
        __m256 vu = bf16_to_m256(u_ptr + i);

        __m256 x = _mm256_mul_ps(vg, v_inv_sqrt2);
        __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
        __m256 sign_x = _mm256_and_ps(sign_mask, x);

        // t = 1.0 / (1.0 + p * |x|)
        __m256 denom = _mm256_fmadd_ps(p_const, abs_x, v_one);
        __m256 t = _mm256_div_ps(v_one, denom);

        // poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
        __m256 poly = _mm256_fmadd_ps(a5_const, t, a4_const);
        poly = _mm256_fmadd_ps(poly, t, a3_const);
        poly = _mm256_fmadd_ps(poly, t, a2_const);
        poly = _mm256_fmadd_ps(poly, t, a1_const);
        poly = _mm256_mul_ps(poly, t);

        // exp(-x^2)
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 neg_x2 = _mm256_sub_ps(_mm256_setzero_ps(), x2);

        // Approximate exp for neg_x2: using standard expf scalar / vectorized or fast Taylor
        // Since x2 >= 0, neg_x2 <= 0.
        // For precision and safety:
        alignas(32) float exp_arr[8];
        alignas(32) float neg_x2_arr[8];
        _mm256_store_ps(neg_x2_arr, neg_x2);
        for (int k = 0; k < 8; ++k) exp_arr[k] = std::exp(neg_x2_arr[k]);
        __m256 vexp = _mm256_load_ps(exp_arr);

        // erf = 1.0 - poly * exp(-x^2)
        __m256 erf_pos = _mm256_fnmadd_ps(poly, vexp, v_one);
        // apply sign of x: erf = sign(x) * erf_pos
        __m256 erf_val = _mm256_xor_ps(erf_pos, sign_x);

        // GELU(g) = 0.5 * g * (1 + erf)
        __m256 one_plus_erf = _mm256_add_ps(v_one, erf_val);
        __m256 gelu_g = _mm256_mul_ps(_mm256_mul_ps(v_half, vg), one_plus_erf);
        __m256 res = _mm256_mul_ps(gelu_g, vu);

        m256_to_bf16(res, act + i);
    }

    for (; i < I_rows; ++i) {
        float g = g_ptr[i].to_float();
        float u = u_ptr[i].to_float();
        float a = 0.5f * g * (1.0f + std::erf(g * k_inv_sqrt2));
        act[i] = bf16(a * u);
    }
}

void run_fused_residual_and_norm_avx2(
    const bf16* x_in,
    const bf16* proj_in,
    const float* post_norm_w,
    const float* next_norm_w,
    bf16* x_res_out,
    bf16* x_next_norm_out,
    int K,
    float eps
) {
    // 1. RMSNorm on proj_in
    __m256 sum_sq_p = _mm256_setzero_ps();
    int i = 0;
    for (; i <= K - 8; i += 8) {
        __m256 vp = bf16_to_m256(proj_in + i);
        sum_sq_p = _mm256_fmadd_ps(vp, vp, sum_sq_p);
    }
    __m128 hsum_p = _mm_add_ps(_mm256_castps256_ps128(sum_sq_p), _mm256_extractf128_ps(sum_sq_p, 1));
    hsum_p = _mm_add_ps(hsum_p, _mm_movehl_ps(hsum_p, hsum_p));
    hsum_p = _mm_add_ss(hsum_p, _mm_shuffle_ps(hsum_p, hsum_p, 1));
    float var_p = _mm_cvtss_f32(hsum_p);
    for (; i < K; ++i) {
        float f = proj_in[i].to_float();
        var_p += f * f;
    }
    var_p /= K;
    float inv_scale_p = 1.0f / std::sqrt(var_p + eps);
    __m256 vinv_p = _mm256_set1_ps(inv_scale_p);

    // 2. Compute x_res = x_in + Norm(proj_in, post_norm_w) and accumulate sum_sq of x_res
    __m256 sum_sq_res = _mm256_setzero_ps();
    i = 0;
    for (; i <= K - 8; i += 8) {
        __m256 vx = bf16_to_m256(x_in + i);
        __m256 vp = bf16_to_m256(proj_in + i);
        __m256 vp_norm = _mm256_mul_ps(vp, vinv_p);
        if (post_norm_w) {
            __m256 vpw = _mm256_loadu_ps(post_norm_w + i);
            vp_norm = _mm256_mul_ps(vp_norm, vpw);
        }
        __m256 vres = _mm256_add_ps(vx, vp_norm);
        m256_to_bf16(vres, x_res_out + i);
        sum_sq_res = _mm256_fmadd_ps(vres, vres, sum_sq_res);
    }
    __m128 hsum_res = _mm_add_ps(_mm256_castps256_ps128(sum_sq_res), _mm256_extractf128_ps(sum_sq_res, 1));
    hsum_res = _mm_add_ps(hsum_res, _mm_movehl_ps(hsum_res, hsum_res));
    hsum_res = _mm_add_ss(hsum_res, _mm_shuffle_ps(hsum_res, hsum_res, 1));
    float var_res = _mm_cvtss_f32(hsum_res);
    for (; i < K; ++i) {
        float f_p = proj_in[i].to_float() * inv_scale_p;
        if (post_norm_w) f_p *= post_norm_w[i];
        float f_res = x_in[i].to_float() + f_p;
        x_res_out[i] = bf16(f_res);
        var_res += f_res * f_res;
    }
    var_res /= K;
    float inv_scale_res = 1.0f / std::sqrt(var_res + eps);
    __m256 vinv_res = _mm256_set1_ps(inv_scale_res);

    // 3. Compute x_next_norm = Norm(x_res, next_norm_w)
    i = 0;
    for (; i <= K - 8; i += 8) {
        __m256 vres = bf16_to_m256(x_res_out + i);
        __m256 vnext = _mm256_mul_ps(vres, vinv_res);
        if (next_norm_w) {
            __m256 vnw = _mm256_loadu_ps(next_norm_w + i);
            vnext = _mm256_mul_ps(vnext, vnw);
        }
        m256_to_bf16(vnext, x_next_norm_out + i);
    }
    for (; i < K; ++i) {
        float f = x_res_out[i].to_float() * inv_scale_res;
        if (next_norm_w) f *= next_norm_w[i];
        x_next_norm_out[i] = bf16(f);
    }
}

void run_fused_down_accumulate_avx2(const bf16* part, float* acc, int K) {
    int i = 0;
    for (; i <= K - 8; i += 8) {
        __m256 vp = bf16_to_m256(part + i);
        __m256 va = _mm256_loadu_ps(acc + i);
        _mm256_storeu_ps(acc + i, _mm256_add_ps(va, vp));
    }
    for (; i < K; ++i) {
        acc[i] += part[i].to_float();
    }
}

} // namespace alveare
