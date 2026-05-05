/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_CHECK_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_CHECK_H_

// RVSPOC S2601: also pull in the RVV gating header here so that any
// translation unit that already includes neon_check.h (without going
// through cpu_check.h) gets USE_RVV correctly defined on RV64GCV
// targets. Several .cc files in tflite/kernels/ (reduce.cc, pooling.cc,
// quantize.cc, maximum_minimum.cc, ...) include neon_check.h directly
// — without this line the RVV blocks in optimized_ops.h / reduce.h /
// integer_ops/{mean,lut}.h would silently compile out for those files
// on RVV targets (Copilot review #1 / #2 / #3 of round 2).
// The header is a no-op when __riscv_vector is undefined, so this is
// safe on ARM / x86 builds.
#include "tflite/kernels/internal/optimized/rvv_check.h"

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#include <arm_neon.h>  // IWYU pragma: export
#endif

#if defined __GNUC__ && defined __SSE4_1__ && !defined TF_LITE_DISABLE_X86_NEON
#define USE_NEON
#include "NEON_2_SSE.h"  // IWYU pragma: export
#endif

// NEON_OR_PORTABLE(SomeFunc, args) calls NeonSomeFunc(args) if USE_NEON is
// defined, PortableSomeFunc(args) otherwise.
#ifdef USE_NEON
// Always use Neon code
#define NEON_OR_PORTABLE(funcname, ...) Neon##funcname(__VA_ARGS__)

#else
// No NEON available: Use Portable code
#define NEON_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)

#endif  // defined(USE_NEON)

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_CHECK_H_
