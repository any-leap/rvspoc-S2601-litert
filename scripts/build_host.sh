#!/usr/bin/env bash
# Native host build of LiteRT (inside the rvspoc-s2601 container, target = host arch).
# Produces:
#   - flatc (needed as -DTFLITE_HOST_TOOLS_DIR= for the riscv64 cross build)
#   - benchmark_model (sanity-check that the CMake config works)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build-host"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  "$REPO_ROOT/tflite"

cmake --build . -j"$(nproc)" --target benchmark_model
