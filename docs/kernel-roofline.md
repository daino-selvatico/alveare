# Kernel roofline analysis

_Baseline: commit `93e073d`, gemma-4-12B on XDNA2 (npu2/AIE2P, 8 columns).
Numbers from `alveare bench` (see `benchmarks/`)._

## The finding

Every NPU kernel runs at **~5 GMAC/s**, independent of shape or batch:

| kernel | shape (N×K) | ms | GMAC/s |
|---|---|---:|---:|
| gemv q_sliding | 4096×4096 | 3.32 | 5.0 |
| gemv kv | 2048×4096 | 1.70 | 4.9 |
| gemv q_global | 8192×4096 | 6.58 | 5.1 |
| gemv o_global | 4096×8192 | 6.69 | 5.0 |
| gemv lm_head | 16384×4096 | 13.04 | 5.1 |
| ffn_fused | 4096×16384 | 34.44 | 5.8 |
| gemm(B=16) gate | 16384×4096 | 205.25 | 5.2 |

5 GMAC/s ≈ 10 GOP/s across 8 cores ≈ **1.25 GOP/s/core**, orders of magnitude
below the AIE2P vector unit's capability. So the kernels are **far from the
roofline** — the ceiling is in the kernel code, not the hardware and not memory
bandwidth (a `gemm(B=16)`, which reuses each weight 16×, is the *same* 5 GMAC/s
as `gemv`, so we are **not** weight-bandwidth bound — we are compute-bound on
the kernel's inner loop).

## Root cause

All matmul kernels (`gemv_q.cc`, `gemm_q.cc`, and the gate/up/down loops in
`ffn_fused/*.cc`) share one naive dot-product structure:

```c
for (r in DIM_M rows)                 // output row
  for (b in DIM_K/32 blocks)          // K blocks of 32
     // --- dequant Q4_0: unpack, 2 shifts, 2 to_float, 2 mul (~9 vector ops) ---
     prod = mul(w0, x0); prod = mac(prod, w1, x1);
     sum += aie::reduce_add(prod);    // horizontal reduction EVERY block
```

Two problems:

1. **`reduce_add` per K-block.** The vector lanes hold *K positions*, so each
   block needs a cross-lane horizontal reduction to a scalar — a serializing
   operation done `K/32` times per row. The throughput-oriented structure maps
   lanes to *output rows* (or batch) and accumulates over K **in-lane**, doing a
   single reduction (or none) at the very end.

2. **`gemm_q` re-dequantizes weights per batch element.** The `for batch` loop
   sits *inside* the K loop, so the ~9-op Q4_0 dequant runs 16× for the same
   weight block instead of once. This is why `gemm(B=16)` is 16× a `gemv` and
   batched prefill showed no speedup — the weight reuse is thrown away.

## Optimizations (in increasing effort / payoff)

**A. Hoist the dequant out of `gemm_q`'s batch loop.** The dequantized weight
block depends only on `(r, b)`, not on `batch`. Compute `w0_bf16/w1_bf16` once,
then loop batch doing only load-x + MAC. Amortizes the ~9-op dequant over 16
tokens. Expected: `gemm(16)` drops well below 16× `gemv`, which **resurrects
batched prefill** (proven futile only because of this bug). Prefill-only;
contained; measurable immediately via `alveare bench`.

**B. Kill the per-block `reduce_add` (all kernels, incl. `gemv`).** Restructure
the accumulation so vector lanes carry independent outputs and K is accumulated
in-lane, reducing once at the end. This lifts the ~5 GMAC/s ceiling for the
decode path (gemv + fused FFN), which is the dominant real-world cost
(2.6 s/token). Bigger rewrite; needs care with the Q4_0 layout and revalidation.

## Method

Prototype A first (contained, and it validates that the Q4_0 dequant is the
bottleneck), measure with `alveare bench`, then tackle B. Every change is
guarded by the bench trend table (`benchmarks/README.md`) and by output parity
(the `ALVEARE_SELFTEST` greeting must stay coherent and greedy-identical).
