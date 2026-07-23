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
                aie::vector<bfloat16, 16> sv = aie::broadcast<bfloat16, 16>(scale);
                aie::vector<int8_t, 16> pk = aie::load_unaligned_v<16>((const int8_t *)&blk[0]);
                aie::vector<int16_t, 16> up = pk.unpack();
                aie::vector<int16_t, 16> q0 = (up << 12) >> 12;   // low nibble
                aie::vector<int16_t, 16> q1 = (up << 8) >> 12;    // high nibble
                aie::vector<bfloat16, 16> w0 = aie::mul(aie::to_float<bfloat16>(q0), sv).to_vector<bfloat16>();
                aie::vector<bfloat16, 16> w1 = aie::mul(aie::to_float<bfloat16>(q1), sv).to_vector<bfloat16>();
                auto zipped = aie::interleave_zip(w0, w1, 1);  // element 2i=w0[i], 2i+1=w1[i]
                aie::vector<bfloat16, 32> w01 = aie::concat(zipped.first, zipped.second);
                aie::store_v(&wtile[t * DIM_K + b * 32], w01);
            }
        }

        // mmul: for each batch tile, C[bi,mi] = sum_ki A[bi,ki] @ B[ki,mi].
        for (int bi = 0; bi < DIM_B / R; ++bi) {
            // C init from the fp32 partial y[bi*R:+R][mi*T:+T] (rows strided DIM_M).
            aie::vector<float, R * T> c0 = aie::concat(
                aie::load_v<T>(&y[(bi * R + 0) * DIM_M + mi * T]),
                aie::load_v<T>(&y[(bi * R + 1) * DIM_M + mi * T]),
                aie::load_v<T>(&y[(bi * R + 2) * DIM_M + mi * T]),
                aie::load_v<T>(&y[(bi * R + 3) * DIM_M + mi * T]));
            MMUL C(c0);

            for (int ki = 0; ki < DIM_K / S; ++ki) {
                aie::vector<bfloat16, R * S> a = aie::concat(
                    aie::load_v<S>(&x[(bi * R + 0) * DIM_K + ki * S]),
                    aie::load_v<S>(&x[(bi * R + 1) * DIM_K + ki * S]),
                    aie::load_v<S>(&x[(bi * R + 2) * DIM_K + ki * S]),
                    aie::load_v<S>(&x[(bi * R + 3) * DIM_K + ki * S]));
                // wtile block [T,S] (rows strided DIM_K) -> transpose to [S,T] (W^T).
                aie::vector<bfloat16, S * T> bts = aie::concat(
                    aie::load_v<S>(&wtile[0 * DIM_K + ki * S]),
                    aie::load_v<S>(&wtile[1 * DIM_K + ki * S]),
                    aie::load_v<S>(&wtile[2 * DIM_K + ki * S]),
                    aie::load_v<S>(&wtile[3 * DIM_K + ki * S]),
                    aie::load_v<S>(&wtile[4 * DIM_K + ki * S]),
                    aie::load_v<S>(&wtile[5 * DIM_K + ki * S]),
                    aie::load_v<S>(&wtile[6 * DIM_K + ki * S]),
                    aie::load_v<S>(&wtile[7 * DIM_K + ki * S]));
                aie::vector<bfloat16, S * T> b = aie::transpose(bts, T, S);
                C.mac(a, b);
            }

            aie::vector<float, R * T> co = C.to_vector<float>();
            aie::store_v(&y[(bi * R + 0) * DIM_M + mi * T], co.extract<T>(0));
            aie::store_v(&y[(bi * R + 1) * DIM_M + mi * T], co.extract<T>(1));
            aie::store_v(&y[(bi * R + 2) * DIM_M + mi * T], co.extract<T>(2));
            aie::store_v(&y[(bi * R + 3) * DIM_M + mi * T], co.extract<T>(3));
        }
    }
}

}
