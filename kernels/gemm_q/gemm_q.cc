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

// Dequantized weights for ONE output-row tile (T rows x DIM_K), kept in tile
// data memory (.bss). Only T rows at a time keeps this small (T*DIM_K*2 bytes);
// dequantizing the whole DIM_M tile at once overflows tile memory.
alignas(32) static bfloat16 wtile[T * DIM_K];

// The same tile pre-transposed to the mmul B operand layout, one [S,T] block per
// ki: btile[ki*S*T + s*T + t] = wtile[t*DIM_K + ki*S + s]. Building this ONCE per
// mi tile (instead of transposing inside the batch loop, where the weight was
// re-loaded and re-transposed for every bi) removes ~3/4 of the transpose+load
// work — the weight does not depend on the batch index.
alignas(32) static bfloat16 btile[(DIM_K / S) * S * T];

void gemm_q(
    const uint8_t *restrict w_combined,
    const bfloat16 *restrict x,
    float *restrict y
) {
    using MMUL = aie::mmul<R, S, T, bfloat16, bfloat16, accfloat>;
    using namespace aie::operators;

    // Process the DIM_M output rows T at a time (one mmul output-column tile).
    for (int mi = 0; mi < DIM_M / T; ++mi) {
        // Dequantize this tile's T weight rows to bf16 (natural [t][k]).
        for (int t = 0; t < T; ++t) {
            const uint8_t *row_ptr = &w_combined[(mi * T + t) * (DIM_K / 32) * 20];
            for (int b = 0; b < DIM_K / 32; ++b) {
                const uint8_t *blk = &row_ptr[b * 20];
                bfloat16 scale = *(const bfloat16 *)&blk[16];
                aie::vector<int8_t, 16> pk = aie::load_unaligned_v<16>((const int8_t *)&blk[0]);
                aie::vector<int16_t, 16> up = pk.unpack();
                aie::vector<int16_t, 16> q0 = (up << 12) >> 12;   // low nibble  -> even
                aie::vector<int16_t, 16> q1 = (up << 8) >> 12;    // high nibble -> odd
                // Interleave the integer nibbles FIRST (2i=q0[i], 2i+1=q1[i]), then do
                // ONE 32-wide to_float and ONE 32-wide scale multiply (was 2 of each).
                auto zipped = aie::interleave_zip(q0, q1, 1);
                aie::vector<int16_t, 32> qi = aie::concat(zipped.first, zipped.second);
                aie::vector<bfloat16, 32> sv = aie::broadcast<bfloat16, 32>(scale);
                aie::vector<bfloat16, 32> w01 = aie::mul(aie::to_float<bfloat16>(qi), sv).to_vector<bfloat16>();
                aie::store_v(&wtile[t * DIM_K + b * 32], w01);
            }
        }

        // Pre-transpose the whole tile once: wtile[T,DIM_K] -> btile[ki][S,T].
        for (int ki = 0; ki < DIM_K / S; ++ki) {
            aie::vector<bfloat16, S * T> bts = aie::concat(
                aie::load_v<S>(&wtile[0 * DIM_K + ki * S]),
                aie::load_v<S>(&wtile[1 * DIM_K + ki * S]),
                aie::load_v<S>(&wtile[2 * DIM_K + ki * S]),
                aie::load_v<S>(&wtile[3 * DIM_K + ki * S]),
                aie::load_v<S>(&wtile[4 * DIM_K + ki * S]),
                aie::load_v<S>(&wtile[5 * DIM_K + ki * S]),
                aie::load_v<S>(&wtile[6 * DIM_K + ki * S]),
                aie::load_v<S>(&wtile[7 * DIM_K + ki * S]));
            aie::store_v(&btile[ki * S * T], aie::transpose(bts, T, S));
        }

        // Run all NB batch accumulators concurrently so the systolic pipeline
        // stays full: each C.mac depends on its own C, so a single accumulator
        // serializes on mmul latency; NB independent chains hide it. The B operand
        // is loaded ONCE per ki and shared across all NB (it is batch-independent).
        // NB = DIM_B / R: 4 for DIM_B=16 (prefill), 2 for DIM_B=8 (spec verify).
        constexpr int NB = DIM_B / R;
        static_assert(NB == 2 || NB == 4, "supported DIM_B: 8 or 16");
        auto load_c = [&](int bi) {
            return aie::concat(
                aie::load_v<T>(&y[(bi * R + 0) * DIM_M + mi * T]),
                aie::load_v<T>(&y[(bi * R + 1) * DIM_M + mi * T]),
                aie::load_v<T>(&y[(bi * R + 2) * DIM_M + mi * T]),
                aie::load_v<T>(&y[(bi * R + 3) * DIM_M + mi * T]));
        };
        // C2/C3 are unused (and never mac'd/stored) when NB==2; init them from a
        // valid row so the load stays in bounds for the DIM_B=8 y buffer.
        MMUL C0(load_c(0)), C1(load_c(1)),
             C2(load_c(NB > 2 ? 2 : 0)), C3(load_c(NB > 2 ? 3 : 0));

        auto load_a = [&](int bi, int ki) {
            return aie::concat(
                aie::load_v<S>(&x[(bi * R + 0) * DIM_K + ki * S]),
                aie::load_v<S>(&x[(bi * R + 1) * DIM_K + ki * S]),
                aie::load_v<S>(&x[(bi * R + 2) * DIM_K + ki * S]),
                aie::load_v<S>(&x[(bi * R + 3) * DIM_K + ki * S]));
        };

        for (int ki = 0; ki < DIM_K / S; ++ki) {
            aie::vector<bfloat16, S * T> b = aie::load_v<S * T>(&btile[ki * S * T]);
            C0.mac(load_a(0, ki), b);
            C1.mac(load_a(1, ki), b);
            if constexpr (NB > 2) {
                C2.mac(load_a(2, ki), b);
                C3.mac(load_a(3, ki), b);
            }
        }

        auto store_c = [&](int bi, MMUL& C) {
            aie::vector<float, R * T> co = C.template to_vector<float>();
            aie::store_v(&y[(bi * R + 0) * DIM_M + mi * T], co.extract<T>(0));
            aie::store_v(&y[(bi * R + 1) * DIM_M + mi * T], co.extract<T>(1));
            aie::store_v(&y[(bi * R + 2) * DIM_M + mi * T], co.extract<T>(2));
            aie::store_v(&y[(bi * R + 3) * DIM_M + mi * T], co.extract<T>(3));
        };
        store_c(0, C0); store_c(1, C1);
        if constexpr (NB > 2) { store_c(2, C2); store_c(3, C3); }
    }
}

}
