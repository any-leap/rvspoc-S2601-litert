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

# Use pre-extracted TF source if present (avoids docker-network flakiness on
# the multi-GB tensorflow.git fetch — see GOT-001).
TF_SRC_FLAG=()
if [ -d "$REPO_ROOT/.cache/tensorflow-src/tensorflow/lite" ]; then
  TF_SRC_FLAG=(-DTENSORFLOW_SOURCE_DIR="$REPO_ROOT/.cache/tensorflow-src")
  echo "Using pre-extracted TF source at $REPO_ROOT/.cache/tensorflow-src"
fi

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  "${TF_SRC_FLAG[@]}" \
  "$REPO_ROOT/tflite"

# LiteRT compilation (Eigen/ruy/gemmlowp templates) can need 2-4 GB per cc1plus
# process. Docker Desktop on macOS defaults to ~8 GB VM memory; -j$(nproc)=14
# OOMs cc1plus mid-build (see GOT-002). Default to -j4 unless caller sets BUILD_JOBS.
BUILD_JOBS="${BUILD_JOBS:-4}"
echo "Building with -j${BUILD_JOBS} (override with BUILD_JOBS=N)"
# benchmark_model: native sanity check.
# flatc: needed by the riscv64 cross-build via TFLITE_HOST_TOOLS_DIR.
cmake --build . -j"${BUILD_JOBS}" --target benchmark_model flatc
