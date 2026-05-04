# SPDX-License-Identifier: Apache-2.0
# RVSPOC S2601 — RVV-targeted standalone numerical accuracy tests.
# These compile against the LiteRT in-tree headers but link nothing else
# beyond stdlib, keeping the build hermetic and quick.

cmake_minimum_required(VERSION 3.16)
project(rvspoc_rvv_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Resolve repo root (..) so we can include both tflite/ and the
# pre-extracted TF source (for ruy/profiler/instrumentation.h, etc.).
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# Headers we need:
#  - tflite/...                  : LiteRT itself (in-repo)
#  - .cache/tensorflow-src/...   : (none needed for this test, but keeps
#                                   things consistent if we add more)
#  - ruy/                        : profiler/instrumentation.h is included
#                                   by depthwiseconv_float.h
include_directories(
    "${REPO_ROOT}"
    "${REPO_ROOT}/.cache/tensorflow-src"
)

# Pull in ruy headers from the host build (it ships header-only profiler
# instrumentation that depthwiseconv_float.h transitively needs).
if(EXISTS "${REPO_ROOT}/build-host/ruy")
  include_directories("${REPO_ROOT}/build-host/ruy")
endif()
if(EXISTS "${REPO_ROOT}/build-rv64/ruy")
  include_directories("${REPO_ROOT}/build-rv64/ruy")
endif()
# integer_ops/depthwise_conv.h transitively pulls in cpu_backend_context.h →
# public/gemmlowp.h. That ships in the rv64 build dir.
if(EXISTS "${REPO_ROOT}/build-rv64/gemmlowp")
  include_directories("${REPO_ROOT}/build-rv64/gemmlowp")
endif()
if(EXISTS "${REPO_ROOT}/build-rv64/abseil-cpp")
  include_directories("${REPO_ROOT}/build-rv64/abseil-cpp")
endif()
if(EXISTS "${REPO_ROOT}/build-rv64/eigen")
  include_directories("${REPO_ROOT}/build-rv64/eigen")
endif()
if(EXISTS "${REPO_ROOT}/build-rv64/flatbuffers/include")
  include_directories("${REPO_ROOT}/build-rv64/flatbuffers/include")
endif()

add_executable(depthwise_float_accuracy_test depthwise_float_accuracy_test.cc)
target_compile_options(depthwise_float_accuracy_test PRIVATE -O3 -Wall)

add_executable(depthwise_int8_accuracy_test depthwise_int8_accuracy_test.cc)
target_compile_options(depthwise_int8_accuracy_test PRIVATE -O3 -Wall)

add_executable(depthwise_uint8_accuracy_test depthwise_uint8_accuracy_test.cc)
target_compile_options(depthwise_uint8_accuracy_test PRIVATE -O3 -Wall)
