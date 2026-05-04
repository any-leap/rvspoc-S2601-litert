#!/usr/bin/env bash
# Cross-compile LiteRT benchmark_model for RV64GCV (riscv64-linux-gnu + RVV 1.0).
# Must run inside the rvspoc-s2601 container. Requires that build_host.sh has
# already produced flatc in build-host/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
HOST_BUILD="$REPO_ROOT/build-host"
BUILD_DIR="$REPO_ROOT/build-rv64"

if [ ! -x "$HOST_BUILD/flatbuffers-flatc/bin/flatc" ] && [ ! -x "$HOST_BUILD/_deps/flatbuffers-build/flatc" ]; then
  echo "ERROR: host flatc not found under $HOST_BUILD." >&2
  echo "       Run scripts/build_host.sh first." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$REPO_ROOT/cmake/toolchain-riscv64-rvv.cmake" \
  -DTFLITE_HOST_TOOLS_DIR="$HOST_BUILD" \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  "$REPO_ROOT/tflite"

# See GOT-002: LiteRT cc1plus is RAM-hungry; cap parallelism for Docker VM.
BUILD_JOBS="${BUILD_JOBS:-4}"
echo "Building with -j${BUILD_JOBS} (override with BUILD_JOBS=N)"
cmake --build . -j"${BUILD_JOBS}" --target benchmark_model
