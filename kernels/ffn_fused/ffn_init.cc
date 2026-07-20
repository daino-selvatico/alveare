#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 256
#endif

#ifndef DIM_H
#define DIM_H 2048
#endif

// DIM_HOUT = output-H processed per pass. The host runs the block twice (two
// H-halves) so the fp32 output accumulator y_accum only needs DIM_HOUT elements
// and fits the tile's .bss region (a full fp32 y_accum[DIM_H] overflows it).
#ifndef DIM_HOUT
#define DIM_HOUT DIM_H
#endif

// All running accumulators are fp32 so the down projection (summing hundreds of
// intermediate blocks) and gate/up (H//k_tile chunks feeding the nonlinear GELU)
// don't lose ~13% to bf16 rounding. `act` is a matmul operand, stays bf16.
alignas(64) float y_accum[DIM_HOUT];
alignas(64) float gate_accum[DIM_M];
alignas(64) float up_accum[DIM_M];
alignas(64) bfloat16 act[DIM_M];

extern "C" {

void ffn_init() {
    for (int i = 0; i < DIM_HOUT; ++i) {
        y_accum[i] = 0.0f;
    }
}

}
