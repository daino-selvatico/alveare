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

## Experiments run (2026-07-21) — both null

Both micro-optimizations were implemented, compiled (16384×4096 shape) and
measured. **Neither changed the timing**, while output stayed bit-exact
(`GEMM-vs-GEMV correctness: row0_maxdiff=0 row15_maxdiff=0`):

- **A — hoist dequant out of gemm_q's batch loop**: `gemm(16)` = 205.1 ms
  (was 205.3 ms). No change → the `-O3` AIE compiler already hoists the
  loop-invariant dequant; it was never redundant work in the emitted code.
- **B(partial) — one `reduce_add` per row instead of per K-block in gemv_q**:
  `gemv` = 13.12 ms (was 13.08 ms). No change.

So the ~5 GMAC/s ceiling is **not** a source-level compute inefficiency the
compiler hasn't already handled. Reading the numbers instead:

- **`gemv` is DMA-bound.** It reads 33.5 MB of Q4_0 weights in 13 ms =
  **2.58 GB/s** — far below the fabric's capability, but that is the ceiling the
  kernel hits, which is why touching the `.cc` compute does nothing. The lever
  is the **dataflow** (ObjectFifo depths, DMA channels, weight tile layout /
  burst) in `gemv_q.py` and `ffn_fused.py`, not the compute `.cc`.
- **`gemm` is compute-bound** at 5 GMAC/s on the element-wise `aie::mul`/`mac`
  formulation. The high-throughput path is the systolic matmul intrinsic
  (`aie::mmul`); moving Q4_0 dequant → `mmul` operands is a **full kernel
  rewrite**.

Both remaining levers are substantial (dataflow tuning; or an `mmul` rewrite),
with uncertain payoff and slow (~compile + 4-min load) iteration — a kernel
research project, not a quick win. Every change must stay guarded by the bench
trend table (`benchmarks/README.md`) and output parity (the `ALVEARE_SELFTEST`
greeting must stay coherent and greedy-identical).

## Core-count experiment (2026-07-21): 4× tiles idle, but shim-DMA capped

npu2 (Strix) is **8 columns × 6 rows = 32 compute tiles** (rows 0/1 are
shim/memtile, rows 2–5 are the 4 compute tiles per column). Every kernel uses
`n_cores = 8` — **one compute tile per column**, leaving **3/4 of the array
idle**. Since decode is compute-bound, using all 32 could give up to ~4×.

Bumping the `gemv_q.py` heuristic to `n_cores = 16` and `32` **fails placement**:

```
no ShimNOCTile has sufficient DMA capacity for 0 input/1 output channels
```

The current dataflow gives every core its **own** shim DMA for weight-in
(`rt.fill(memW_fifos[i]…)`) and y-out (`rt.drain(outY_fifos[i]…)`). With 8 cores
that is ~2 DMA/column (fits); beyond 8 the per-column shim NOC tile runs out of
DMA channels. So the ceiling is the **output/weight gathering topology**, not the
tiles.

**The fix (scoped future work):** adopt the 3-layer ObjectFifo dataflow from
`mlir-aie/programming_examples/basic/matrix_multiplication/whole_array` — DRAM
(L3) → **memtile (L2)** → compute (L1). Per column, one memtile `.split()`s the
weight rows to its 4 cores and `.join()`s their outputs back, and `x` is
broadcast; the shim then does one weight-in + one y-out DMA **per column**
(≤2/shim) regardless of core count. This unlocks 16–32 cores (and likely more
effective bandwidth). It is a substantial rewrite of `gemv_q.py` **and**
`ffn_fused.py` (the FFN is 64% of decode, so both are needed for real impact),
with Q4_0-packed tiling through the split/join and slow iteration — a
multi-session kernel task, not a quick win. Reference: `whole_array.py`
(`Worker.grid`, `.split`/`.join`/`.forward`, per-column memtile fifos).
