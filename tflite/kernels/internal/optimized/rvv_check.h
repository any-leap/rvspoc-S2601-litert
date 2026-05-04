/* SPDX-License-Identifier: Apache-2.0
 *
 * RVSPOC S2601: RISC-V Vector Extension (RVV 1.0) gating header.
 * Mirrors the role of neon_check.h for the RVV target.
 */
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_CHECK_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_CHECK_H_

// __riscv_vector is auto-defined by GCC/Clang when -march includes the V
// extension (e.g. -march=rv64gcv). Gate all RVV intrinsic code on USE_RVV
// so the file stays portable; without -march=...v, we fall through to the
// generic scalar template, matching the behavior on x86 builds.
#if defined(__riscv_vector)
#define USE_RVV
#include <riscv_vector.h>  // IWYU pragma: export
#endif

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_CHECK_H_
