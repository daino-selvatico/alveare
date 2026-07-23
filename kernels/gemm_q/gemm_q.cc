#include <stdint.h>
#include <aie_api/aie.hpp>

// Batched Q4_0 GEMM using the AIE2P systolic aie::mmul intrinsic.
//   y[DIM_B, DIM_M] += x[DIM_B, DIM_K] @ dequant(w)[DIM_M, DIM_K]^T
// The weight tile is dequantized ONCE to bf16 (amortized over the DIM_B batch
// rows), then fed to mmul. This is the batched/prefill path; a batch makes the
// systolic array efficient (vs the element-wise per-row dot product, which is
// ~1 output/instr and can't fill the array).

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

// mmul tile: C[r,t] += A[r,s] @ B[s,t]
static constexpr int R = 4;
static constexpr int S = 8;
static constexpr int T = 8;

void gemm_q(
    const uint8_t *restrict w_combined,
    const bfloat16 *restrict x,
    bfloat16 *restrict y
) {
    using MMUL = aie::mmul<R, S, T, bfloat16, bfloat16, accfloat>;
    using namespace aie::operators;

    // --- Phase 1: dequantize the whole weight tile to bf16, natural [m][k]. ---
    alignas(32) bfloat16 wbf[DIM_M * DIM_K];
    for (int r = 0; r < DIM_M; ++r) {
        const uint8_t *row_ptr = &w_combined[r * (DIM_K / 32) * 20];
        for (int b = 0; b < DIM_K / 32; ++b) {
            const uint8_t *blk = &row_ptr[b * 20];
            bfloat16 scale = *(const bfloat16 *)&blk[16];
            aie::vector<bfloat16, 16> sv = aie::broadcast<bfloat16, 16>(scale);
            aie::vector<int8_t, 16> pk = aie::load_unaligned_v<16>((const int8_t *)&blk[0]);
            aie::vector<int16_t, 16> up = pk.unpack();
            aie::vector<int16_t, 16> q0 = (up << 12) >> 12;   // low nibble
            aie::vector<int16_t, 16> q1 = (up << 8) >> 12;    // high nibble
            aie::vector<bfloat16, 16> w0 = aie::mul(aie::to_float<bfloat16>(q0), sv).to_vector<bfloat16>();
            aie::vector<bfloat16, 16> w1 = aie::mul(aie::to_float<bfloat16>(q1), sv).to_vector<bfloat16>();
            // Natural k order: element 2i = w0[i], 2i+1 = w1[i].
            auto zipped = aie::interleave_zip(w0, w1, 1);
            aie::vector<bfloat16, 32> w01 = aie::concat(zipped.first, zipped.second);
            aie::store_v(&wbf[r * DIM_K + b * 32], w01);
        }
    }

    // --- Phase 2: mmul. C[bi,mi] = sum_ki A[bi,ki] @ B[ki,mi], B = W^T. ---
    // wbf is [m][k] = column-major B (B[k][m] stored as [m][k]); transpose each
    // [T,S] weight block to the [S,T] mmul B operand.
    for (int bi = 0; bi < DIM_B / R; ++bi) {
        for (int mi = 0; mi < DIM_M / T; ++mi) {
            // Load current partial y[bi*R:+R][mi*T:+T] into the accumulator.
            aie::vector<bfloat16, R * T> c_prev;
            for (int rr = 0; rr < R; ++rr)
                for (int tt = 0; tt < T; ++tt)
                    c_prev[rr * T + tt] = y[(bi * R + rr) * DIM_M + mi * T + tt];
            MMUL C(c_prev);

            for (int ki = 0; ki < DIM_K / S; ++ki) {
                // A tile [R,S] from x[bi*R:+R][ki*S:+S], row-major.
                aie::vector<bfloat16, R * S> a;
                for (int rr = 0; rr < R; ++rr)
                    for (int ss = 0; ss < S; ++ss)
                        a[rr * S + ss] = x[(bi * R + rr) * DIM_K + ki * S + ss];
                // B tile [S,T] = W^T[ki*S:+S][mi*T:+T] = wbf[mi*T:+T][ki*S:+S]^T.
                aie::vector<bfloat16, S * T> bt;
                for (int ss = 0; ss < S; ++ss)
                    for (int tt = 0; tt < T; ++tt)
                        bt[ss * T + tt] = wbf[(mi * T + tt) * DIM_K + ki * S + ss];
                C.mac(a, bt);
            }

            aie::vector<bfloat16, R * T> c_out = C.to_vector<bfloat16>();
            for (int rr = 0; rr < R; ++rr)
                for (int tt = 0; tt < T; ++tt)
                    y[(bi * R + rr) * DIM_M + mi * T + tt] = c_out[rr * T + tt];
        }
    }
}

}
