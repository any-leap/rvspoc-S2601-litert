# SPDX-License-Identifier: Apache-2.0
# CMake toolchain file for cross-compiling to RV64GCV (RISC-V 64-bit + RVV 1.0)
# using the gcc-14-riscv64-linux-gnu toolchain shipped in the rvspoc-s2601
# Docker image.
#
# Use:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-rvv.cmake \
#         -DTFLITE_HOST_TOOLS_DIR=/work/build-host \
#         -G Ninja /work/tflite

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# RV64GC + V (vector) extension. -mabi=lp64d matches the gcc-14 default
# double-float ABI; do not change without updating qemu launch flags.
set(_rvspoc_arch_flags "-march=rv64gcv -mabi=lp64d")
set(CMAKE_C_FLAGS_INIT   "${_rvspoc_arch_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_rvspoc_arch_flags}")

# Search target libs in the cross-sysroot only; host paths are off-limits.
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# CMake try_compile() must run on host (we are cross-compiling), so build
# static for the linker tests instead of shared.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
