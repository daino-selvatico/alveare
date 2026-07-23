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

> **Resolved (2026-07-22): both GEMV and FFN now run on all 32 cores.** See the
> "32-core" update at the end of this section. The account below is the original
> diagnosis.

Bumping the `gemv_q.py` heuristic to `n_cores = 16` and `32` **fails placement**:

```
no ShimNOCTile has sufficient DMA capacity for 0 input/1 output channels
```

The current dataflow gives every core its **own** shim DMA for weight-in
(`rt.fill(memW_fifos[i]…)`) and y-out (`rt.drain(outY_fifos[i]…)`). With 8 cores
that is ~2 DMA/column (fits); beyond 8 the per-column shim NOC tile runs out of
DMA channels. So the ceiling is the **output/weight gathering topology**, not the
tiles.

### Update (2026-07-21): GEMV done at 16 cores; FFN blocked at the DMA layer

Implemented the per-column memtile funnel for **gemv** (`gemv_q.py`): N split
across **16 cores** (2 rows × 8 cols), with the 2 rows' output tiles interleaved
so each round is a contiguous `(2, tile)` DRAM block funneled through the column
memtile (`ObjectFifo.split`/`join`, `x` `forward`-broadcast). Validated bit-exact
on the full model:
- gemv 16384×4096: 13.6 → 7.0 ms (~1.9×); lm_head 213 → 110 ms (2×); decode
  ~2580 → ~2360 ms (~8.5% — the attention GEMVs are small and parallelize less).
- **32 cores (4 rows) place but fail routing** ("Unable to find a legal
  routing"); 16 (2 rows) is the sweet spot.

The **fused FFN (64% of decode) is now also 16-core — DONE.** The first attempt
(strided 2-row weight read) exhausted the memtile DMA descriptors ("Allocator
exhausted available BD IDs, max 24/channel"). The fix: pack the weights
**interleaved per column** so a column's two cores' tiles alternate in DRAM
(`col = [c0.t0, c1.t0, c0.t1, c1.t1, …]`), making the per-column weight fill a
single **contiguous** stream (one trivial BD) that the memtile splits to its 2
cores. This is a coordinated change across `pack_ffn_fused_weights` (weights.cpp,
16-core interleave branch), `ffn_fused.py` (memtile split/join dataflow), the
manifest `n_cores`→16 (read by `run_ffn_fused`, which sums the partials
order-independently), and `build_kernels.py`. Validated bit-exact on the full
model: **FFN 1650 → 899 ms/token (~1.84×)**, decode **~2580 → ~1560 ms/token
(~40%)**, output token-identical.

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

## 32-core: all compute tiles (2026-07-22) — decode ~2580 → ~1236 ms/token

Both GEMV and the fused FFN now run on **all 32 compute tiles** (4 rows × 8
columns). Two problems had to be solved past the 16-core (2-row) design:

1. **Memtile DMA channel exhaustion at 4 rows.** A per-column memtile driving 4
   cores needs W-split (4 out) + x-forward (1 out) + y-to-shim (1 out) = 6 out,
   and W-in (1) + y-join (4 in) + x-in (1) = 6 in — exactly the memtile's DMA
   channel limit, and routing fails. **Fix: broadcast x through a single global
   ObjectFifo** (not a per-column memtile forward), keeping x off the memtiles.
   That drops each column memtile to 5 out / 5 in and it routes. (`--alloc-scheme=
   basic-sequential` is also required for the 4-row routing.)
2. **Weight-fill descriptor blow-up** — already solved for the FFN by the
   per-column interleaved packing (now generalized to N rows/column:
   `col = [c0.t0, c1.t0, c2.t0, c3.t0, c0.t1, …]`).

Measured, bit-exact on gemma-4-12B (greedy tokens identical, rerun matches):

| kernel | 8-core | 16-core | 32-core |
|---|---:|---:|---:|
| GEMV 16384×4096 | 13.6 ms | 7.0 ms | **3.35 ms (~4×)** |
| fused FFN | ~1650 ms | ~900 ms | **~570 ms** |
| lm_head (in decode) | 213 ms | 110 ms | **~59 ms** |
| **decode / token** | ~2580 ms | ~1560 ms | **~1236 ms (2.09×)** |

Now at **~0.81 tok/s**. The kernels are no longer the whole story: at 32 cores the
per-token attention GEMVs (~527 ms) are dominated by **host dispatch overhead**
(192 gemv launches/token × fixed per-call cost), not compute — the next lever is
fewer/fused launches (e.g. one fused Q/K/V GEMV) rather than more cores. 32 is
also the physical tile ceiling; beyond it needs a faster per-core kernel
(systolic `aie::mmul` instead of the element-wise dequant+MAC path).

## Launch / context-switch overhead (2026-07-22)

At 32 cores the attention GEMVs stopped being compute-bound and became
**launch-bound**. A micro-benchmark (`ALVEARE_TEST_LAUNCH`) settles why: 200
same-shape gemv calls average **0.93 ms/call** (≈ the kernel time, ~0 overhead),
but alternating between two shapes averages **3.57 ms/call** — a **~2.6 ms
kernel-context-switch penalty** every time the active kernel shape changes. The
decode loop cycles Q/K/V/O/FFN shapes each layer, so almost every launch pays it.

**Fix applied — fused Q/K/V.** The three input projections share the input
`x_norm`, so their weights are concatenated (`w_qkv`) and run as one gemv:
248 → 160 launches/token, decode ~1236 → ~1123 ms/token (~0.89 tok/s). Needed a
`10240×4096` gemv (global q++k) besides the existing `8192×4096` (sliding
q++k++v).

**Remaining floor.** With separate per-shape kernels each layer still touches 3
shapes (QKV, O, FFN) → 3 switches/layer × 48 × ~2.6 ms ≈ 374 ms/token of pure
switching. Pure kernel time is only ~640 ms, so the switch tax is the largest
remaining cost. Crossing 1 tok/s (<1000 ms) needs to **remove switches**, not add
cores: a single **runtime-shape** gemv context that serves all N/K without
reconfiguring the array (so QKV, O and lm_head stop switching), leaving only the
gemv↔FFN boundary. That is a kernel/runtime change (runtime-configured DMA taps),
the clear next lever.

## Real-time diagnosis (2026-07-23): why decode is at ~1 tok/s, and what it takes to go faster

Goal: real-time (~5-10 tok/s). Decode is ~1006 ms/token, split (isolated `alveare
bench` kernel times + model breakdown):

- **~683 ms real kernel compute** — 48 × (qkv 1.73 + o 1.73 + ffn 9.65) + lm_head 54.
- **~250 ms kernel context switches** (per forward pass, not per token).
- **~70 ms CPU** (RMSNorm/RoPE/attention/sampling).

### The kernel is NOT micro-optimizable at the source level
Removing the per-block `aie::reduce_add` (keep a live 16-lane accumulator, reduce
once per row) is a **real 1.47× device speedup** (16384×4096: 3393 → 2300 µs in the
standalone `run_iters` harness) **but gives ZERO end-to-end improvement** in the C++
runtime — three such null results now. Beware: the standalone harness reports
*device* time; the C++ `run_gemv` wall time is dominated by per-launch overhead
(context switch + host memcpy/sync), so a faster kernel doesn't move it. Also
beware the AOT compile cache (`~/.npu/cache`): `.compile()` can cache-hit and leave
`build/` stale — clear it to force a real recompile.

### Ablation: where the gemv device time goes (16384×4096, 3393 µs baseline)
Replacing the whole Q4_0 dequant (unpack + 2 shifts + 2 `to_float` + 2 scale-mul)
**and** the weight load with a constant weight vector:

- **dequant + weight-load ≈ 1302 µs (38%)** — optimizable (e.g. an int4→bf16 LUT).
- **MAC + reduce + x-load + loop/DMA floor ≈ 2091 µs (62%)** — the hard floor.

So even a *perfect* dequant caps gemv at ~1.6×. The 62% floor is the **batch=1
matrix-vector structure itself** (~20 GMAC/s), which the element-wise
`aie::mul`/`aie::mac` path can't beat.

### The real lever: batch + systolic mmul
Breaking the ~20 GMAC/s floor needs the AIE2P **`aie::mmul`** systolic intrinsic
(4×8×8 bf16 tiles) instead of element-wise MACs — but mmul needs a **batch** (≥4-8
rows of A). Single-stream decode is **batch=1**, which fundamentally underutilizes
the array. (The existing `gemm_q` is *also* element-wise, so batched prefill was
16× a gemv — no win; it would need the mmul rewrite too.)

**Path to real-time (multi-session):**
1. **Speculative decoding** — a small draft model (e.g. Gemma-3-1B) proposes K
   tokens; the 12B verifies all K in one **batch=K** forward. This both amortizes
   the ~250 ms/forward switch cost over K tokens **and** makes the matmuls GEMM.
2. **mmul-based GEMM kernels** (rewrite `gemm_q` + the FFN with `aie::mmul`) so the
   batch=K verify is actually fast per token (not K× a gemv).
3. Optional: int4→bf16 LUT dequant (~1.6× on the gemv/ffn matmuls regardless).

Together these are the credible route from ~1 tok/s to interactive speeds; each is
a substantial kernel/architecture effort, not a micro-optimization.

## mmul validation (2026-07-23): the systolic path is ~44× our kernels

Measured the mlir-aie reference bf16 matmul (`programming_examples/.../whole_array`,
`aie::mmul<4,8,8>`) on npu2, 32 cores, M=512 K=4096 N=4096:

- **1746 GFLOPS = ~873 GMAC/s** (vs our element-wise gemv/gemm at ~20 GMAC/s).

So the systolic `aie::mmul` intrinsic is **~44× faster** than our element-wise
`aie::mul`/`aie::mac` kernels at peak (large batch M). This validates the real-time
architecture: the compute floor is not the hardware — it's the batch=1 element-wise
kernel structure. With a batch (speculative decoding) + an mmul-based Q4_0 kernel,
the matmuls can run vastly faster per token. Even at modest batch and with Q4
dequant overhead (amortized over the batch), there is large headroom over ~20
GMAC/s. Next: build a Q4_0 mmul GEMM kernel (dequant the weight tile to bf16 into
the mmul B operand), wire it into the batched path, and add speculative decoding.
