#!/usr/bin/env bash
# Profile every spec model with op-level breakdown under qemu RV64GCV.
# Outputs to .cache/profile/<model>.txt for offline analysis.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
OUT_DIR="$REPO_ROOT/.cache/profile"
mkdir -p "$OUT_DIR"

MODELS=(
  mobilenet_v1_1.0_224.tflite
  mobilenet_v1_1.0_224_quant.tflite
  mobilenet_v2_1.0_224.tflite
  mobilenet_v2_1.0_224_quant.tflite
  efficientdet_lite0.tflite
)

for m in "${MODELS[@]}"; do
  echo "=== $m ==="
  out="$OUT_DIR/${m%.tflite}.txt"
  qemu-riscv64-static -cpu rv64,v=true,vlen=256,elen=64 \
    -L /usr/riscv64-linux-gnu \
    "$REPO_ROOT/build-rv64/tools/benchmark/benchmark_model" \
      --graph="$REPO_ROOT/.cache/models/$m" \
      --num_threads=1 --num_runs=3 --warmup_runs=1 \
      --enable_op_profiling=true 2>&1 | tee "$out" \
      | grep -E "^(Timings|	[A-Z]|Number of nodes)" | tail -25
  echo
done

echo "=== Per-model totals (avg ms) ==="
for m in "${MODELS[@]}"; do
  out="$OUT_DIR/${m%.tflite}.txt"
  avg=$(grep -E "^Timings " "$out" | head -1 | grep -oE "avg=[0-9.e+]+" | head -1)
  echo "  $m  $avg"
done
