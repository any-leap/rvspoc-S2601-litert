#!/usr/bin/env bash
# Cross-compile LiteRT benchmark_model for RV64GCV (riscv64-linux-gnu + RVV 1.0).
# Must run inside the rvspoc-s2601 container. Requires that build_host.sh has
# already produced flatc in build-host/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
HOST_BUILD="$REPO_ROOT/build-host"
BUILD_DIR="$REPO_ROOT/build-rv64"

# LiteRT's CMakeLists.txt only searches three layouts under TFLITE_HOST_TOOLS_DIR
# (DIR, DIR/bin, DIR/flatbuffers-flatc/bin). Our host build leaves flatc at
# build-host/_deps/flatbuffers-build/flatc, so point at that subdir directly.
HOST_FLATC_DIR="$HOST_BUILD/_deps/flatbuffers-build"
if [ ! -x "$HOST_FLATC_DIR/flatc" ]; then
  echo "ERROR: host flatc not found at $HOST_FLATC_DIR/flatc" >&2
  echo "       Run scripts/build_host.sh first." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Reuse the pre-extracted TF source (see GOT-001).
TF_SRC_FLAG=()
if [ -d "$REPO_ROOT/.cache/tensorflow-src/tensorflow/lite" ]; then
  TF_SRC_FLAG=(-DTENSORFLOW_SOURCE_DIR="$REPO_ROOT/.cache/tensorflow-src")
fi

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$REPO_ROOT/cmake/toolchain-riscv64-rvv.cmake" \
  -DTFLITE_HOST_TOOLS_DIR="$HOST_FLATC_DIR" \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  "${TF_SRC_FLAG[@]}" \
  "$REPO_ROOT/tflite"

# See GOT-002: LiteRT cc1plus is RAM-hungry; cap parallelism for Docker VM.
BUILD_JOBS="${BUILD_JOBS:-4}"
echo "Building with -j${BUILD_JOBS} (override with BUILD_JOBS=N)"
# benchmark_model + the two RVSPOC verification utilities. SUBMISSION.md /
# verify_model_outputs.sh / run_imagenet_eval.sh all expect these to be
# built; building them in one shot avoids surprising follow-up failures
# (Copilot review #8 / #16).
cmake --build . -j"${BUILD_JOBS}" --target \
    benchmark_model rvspoc_model_output_dumper rvspoc_imagenet_eval
