#!/usr/bin/env bash
# Build a scalar-only (RV64GC, no V) variant of LiteRT into build-rv64-scalar/.
# Used as the accuracy baseline for the model output dumper — running the
# same model on this build vs the RVV build (build-rv64/) and diffing
# outputs verifies we haven't introduced numerical regressions.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
HOST_BUILD="$REPO_ROOT/build-host"
HOST_FLATC_DIR="$HOST_BUILD/_deps/flatbuffers-build"
BUILD_DIR="$REPO_ROOT/build-rv64-scalar"

if [ ! -x "$HOST_FLATC_DIR/flatc" ]; then
  echo "ERROR: host flatc not found at $HOST_FLATC_DIR/flatc" >&2
  echo "       Run scripts/build_host.sh first." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

TF_SRC_FLAG=()
if [ -d "$REPO_ROOT/.cache/tensorflow-src/tensorflow/lite" ]; then
  TF_SRC_FLAG=(-DTENSORFLOW_SOURCE_DIR="$REPO_ROOT/.cache/tensorflow-src")
fi

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$REPO_ROOT/cmake/toolchain-riscv64-scalar.cmake" \
  -DTFLITE_HOST_TOOLS_DIR="$HOST_FLATC_DIR" \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF \
  -DTFLITE_ENABLE_RUY=ON \
  "${TF_SRC_FLAG[@]}" \
  "$REPO_ROOT/tflite"

BUILD_JOBS="${BUILD_JOBS:-4}"
cmake --build . -j"${BUILD_JOBS}" --target rvspoc_model_output_dumper benchmark_model
