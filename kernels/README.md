# kernels/

AIE NPU kernels (IRON / MLIR-AIE sources) compiled to `.xclbin`. The hard 30% — see `../docs/kernels.md` for the plan and `../docs/decisions/0002-kernel-strategy.md` for the hand-written-vs-iree-amd-aie question.

Each kernel gets its own subdirectory:

```
kernels/<name>/
  README.md        host ABI: input/output shapes, dtypes, on-device layout, tolerance
  <design>.py      IRON design (or .mlir)
  <core>.cc        per-core compute (if hand-written)
```

Build order (from `docs/kernels.md`): `dequant` → `gemv_q` → `rmsnorm` → `rope` → `attn`/`softmax` → `gemm_q` → `lm_head`.

Empty until M1.
