#!/usr/bin/env bash
# Run the rvspoc_imagenet_eval driver on a manifest CSV against both
# the RVV build and the scalar control build, then diff the Top-1 /
# Top-5 percentages. The S2601 spec model-level gate is satisfied iff
#   FP32: |top1_rvv - top1_x86_scalar| <= 0.1%
#   INT8: |top1_rvv - top1_x86_scalar| <= 1%
#
# We compare RVV-RV64 to scalar-RV64 here. Since scalar-RV64 is the
# same portable code as scalar-x86, the comparison is transitive.
#
# Usage:
#   ./scripts/run_imagenet_eval.sh <model.tflite> <manifest.csv> [label_offset]
#
# Example:
#   ./docker/rvspoc/run.sh bash scripts/run_imagenet_eval.sh \
#       .cache/models/mobilenet_v1_1.0_224.tflite \
#       .cache/eval/imagenet_val_500.csv
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

if [ $# -lt 2 ]; then
  echo "usage: $0 <model.tflite> <manifest.csv> [label_offset]" >&2
  exit 1
fi
MODEL="$1"
MANIFEST="$2"
LABEL_OFF="${3:-0}"

QEMU_RVV=(qemu-riscv64-static -cpu rv64,v=true,vlen=256,elen=64
          -L /usr/riscv64-linux-gnu)
QEMU_SCALAR=(qemu-riscv64-static -cpu rv64
             -L /usr/riscv64-linux-gnu)

RVV_BIN="$REPO_ROOT/build-rv64/tools/benchmark/rvspoc_imagenet_eval"
SCALAR_BIN="$REPO_ROOT/build-rv64-scalar/tools/benchmark/rvspoc_imagenet_eval"
for b in "$RVV_BIN" "$SCALAR_BIN"; do
  if [ ! -x "$b" ]; then
    echo "ERROR: $b not built" >&2
    exit 2
  fi
done

OUT_DIR="$REPO_ROOT/.cache/eval/$(basename "$MODEL" .tflite)"
mkdir -p "$OUT_DIR"

echo "=== Running RVV build ==="
"${QEMU_RVV[@]}" "$RVV_BIN" "$MODEL" "$MANIFEST" "$LABEL_OFF" \
  | tee "$OUT_DIR/rvv.log"

echo
echo "=== Running scalar control ==="
"${QEMU_SCALAR[@]}" "$SCALAR_BIN" "$MODEL" "$MANIFEST" "$LABEL_OFF" \
  | tee "$OUT_DIR/scalar.log"

echo
echo "=== Diff ==="
rvv_top1=$(grep -oE "top-1=[0-9]+ \([0-9.]+%\)" "$OUT_DIR/rvv.log" | tail -1)
sca_top1=$(grep -oE "top-1=[0-9]+ \([0-9.]+%\)" "$OUT_DIR/scalar.log" | tail -1)
rvv_top5=$(grep -oE "top-5=[0-9]+ \([0-9.]+%\)" "$OUT_DIR/rvv.log" | tail -1)
sca_top5=$(grep -oE "top-5=[0-9]+ \([0-9.]+%\)" "$OUT_DIR/scalar.log" | tail -1)
echo "RVV     $rvv_top1  $rvv_top5"
echo "scalar  $sca_top1  $sca_top5"

# Compute absolute Top-1 percent diff and check spec gate.
rvv_pct=$(echo "$rvv_top1" | grep -oE "[0-9.]+%" | tr -d %)
sca_pct=$(echo "$sca_top1" | grep -oE "[0-9.]+%" | tr -d %)
diff=$(python3 -c "print(abs($rvv_pct - $sca_pct))" 2>/dev/null || echo "?")
echo "Top-1 absolute diff: ${diff}%"

# Detect input dtype from the dumper output: dtype=1 → FP32, dtype=3 → uint8,
# dtype=9 → int8. The driver header prints "Input: [...] dtype=<n>".
dtype=$(grep -oE "dtype=[0-9]+" "$OUT_DIR/rvv.log" | head -1 | tr -d 'a-z=')
case "$dtype" in
  1)  threshold=0.1; quant_label="FP32" ;;
  3|9) threshold=1.0; quant_label="INT8/UINT8" ;;
  *)   threshold=1.0; quant_label="unknown→assuming INT8 budget" ;;
esac
echo "Detected: ${quant_label}  → spec threshold ≤ ${threshold}%"

# Exit-code gate (Copilot review #19): non-zero if the absolute Top-1
# delta exceeds the auto-detected per-spec threshold. Allows CI to use
# this script as the actual gate rather than just informational.
exit_code=$(python3 -c "
d = $diff if '$diff' != '?' else float('inf')
print(0 if d <= $threshold else 3)
" 2>/dev/null || echo 4)
if [ "$exit_code" = "0" ]; then
  echo "Result: PASS (${diff}% ≤ ${threshold}%)"
else
  echo "Result: FAIL (${diff}% > ${threshold}%)"
fi
exit "$exit_code"
