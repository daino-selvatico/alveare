#!/usr/bin/env bash
set +e
source ~/miniconda3/etc/profile.d/conda.sh; conda activate alveare-aie
source /home/daino/progetti/alveare/mlir-aie/utils/env_setup.sh >/dev/null 2>&1
cd /home/daino/progetti/alveare
BUILD=kernels/build
for shape in "4096 4096" "2048 4096" "8192 4096" "4096 8192" "16384 4096" "10240 4096"; do
  set -- $shape; N=$1; K=$2
  GC=/tmp/gc_${N}x${K}; rm -rf "$GC"; mkdir -p "$GC"
  echo "=== compiling gemm ${N}x${K} via run_iters ==="
  NPU_CACHE_HOME="$GC" timeout 250 python kernels/gemm_q/gemm_q.py -N $N -K $K -B 16 -m 32 -k 256 --iters 3 --warmup 1 2>&1 | grep -iE "CHECK" | head -1
  X=$(find "$GC" -name final.xclbin | head -1)
  I=$(find "$GC" -name insts.bin | head -1)
  if [ -n "$X" ] && [ -n "$I" ]; then
    cp "$X" "$BUILD/gemm_${N}x${K}_b16.xclbin"
    cp "$I" "$BUILD/gemm_${N}x${K}_b16.insts"
    echo "  copied -> gemm_${N}x${K}_b16 (xclbin $(stat -c%s "$X") insts $(stat -c%s "$I"))"
  else
    echo "  FAILED: no xclbin/insts for ${N}x${K}"
  fi
done
echo "DONE"
