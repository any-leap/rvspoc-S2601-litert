#!/usr/bin/env bash
# Build & run RVSPOC RVV unit tests (RV64GCV target, executed via qemu).
# Must run inside the rvspoc-s2601 container.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build-rvv-tests"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$REPO_ROOT/cmake/toolchain-riscv64-rvv.cmake" \
  "$REPO_ROOT/test/rvv"
cmake --build . -j4

echo
echo "=== Running tests under qemu (vlen=128, 256, 512) ==="
for vlen in 128 256 512; do
  echo "--- vlen=$vlen ---"
  for t in depthwise_float_accuracy_test depthwise_int8_accuracy_test depthwise_uint8_accuracy_test gemm_float_accuracy_test gemm_int8_accuracy_test; do
    echo "  > $t"
    qemu-riscv64-static -cpu "rv64,v=true,vlen=$vlen,elen=64" \
      -L /usr/riscv64-linux-gnu \
      "./$t"
  done
done
