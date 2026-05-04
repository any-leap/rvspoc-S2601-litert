# SPDX-License-Identifier: Apache-2.0
# CMake toolchain file for cross-compiling to RV64GC (no V extension).
# Used to produce a scalar-only "control" build for accuracy comparisons:
# our RVV kernels are gated on __riscv_vector, which is undefined when -march
# excludes the V bit, so all RVV paths fall through to LiteRT's scalar code.
#
# Same gcc-14-riscv64-linux-gnu toolchain as toolchain-riscv64-rvv.cmake;
# only the -march flag changes.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# RV64GC, no V — everything else identical to the RVV toolchain.
set(_rvspoc_arch_flags "-march=rv64gc -mabi=lp64d")
set(CMAKE_C_FLAGS_INIT   "${_rvspoc_arch_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_rvspoc_arch_flags}")

set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
