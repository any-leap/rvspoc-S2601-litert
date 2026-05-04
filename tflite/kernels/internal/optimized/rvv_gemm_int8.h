/* SPDX-License-Identifier: Apache-2.0
 *
 * RVSPOC S2601: standalone RVV 1.0 INT8 GEMM with per-channel
 * requantization.
 *
 * Math (matches optimized_integer_ops::ConvPerChannel emit):
 *   acc[n,m] = sum_k (lhs[n,k] - 0) * (rhs[k,m] - rhs_zp)
 *            = sum_k lhs[n,k] * rhs[k,m]  -  rhs_zp * sum_k lhs[n,k]
 *   acc += bias[n]
 *   acc = MultiplyByQuantizedMultiplier(acc,
 *           multiplier_fixedpoint[n], multiplier_exponent[n])
 *   acc += dst_zero_point
 *   dst[n,m] = clamp_int8(acc, clamp_min, clamp_max)
 *
 * Layout convention (matches optimized_integer_ops::ConvPerChannel):
 *   lhs: n × k row-major   → lhs[n_idx*k + k_idx]      filter, zp=0
 *   rhs: k × m col-major   → rhs[m_idx*k + k_idx]      activations, zp != 0
 *   dst: n × m col-major   → dst[m_idx*n + n_idx]      i8 output
 *
 * Vector strategy: same dot-product reduction as the FP32 helper but
 * with widening i8→i16→i32 via vsext_vf2 and vwmacc_vv. Per-channel
 * requant + clamp are done scalar — they're 1 op per output element vs
 * k MACs, so vectorising them is low-priority.
 *
 * Header-only, no link dependencies, so the unit test compiles with
 * just the kernel sources.
 */
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_INT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_INT8_H_

#if defined(__riscv_vector)

#include <riscv_vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tflite {
namespace optimized_rvv {

// gemmlowp-style fixed-point requantization (double-rounded variant).
inline int32_t SaturatingDoublingHighMulI32(int32_t a, int32_t b) {
  const int64_t prod = static_cast<int64_t>(a) * b;
  const int32_t nudge = (prod >= 0) ? (1 << 30) : (1 - (1 << 30));
  return static_cast<int32_t>((prod + nudge) / (1LL << 31));
}

inline int32_t RoundingDivideByPOTI32(int32_t x, int exponent) {
  const int32_t mask = (static_cast<int32_t>(1) << exponent) - 1;
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
  return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
}

inline int32_t MultiplyByQuantizedMultiplierRvv(int32_t x, int32_t mul,
                                                int shift) {
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOTI32(
      SaturatingDoublingHighMulI32(x * (static_cast<int32_t>(1) << left_shift),
                                   mul),
      right_shift);
}

// Vectorised dot product over k (signed-widening MAC).
// Returns sum_k (lhs[k] * rhs[k]).
inline int32_t RvvDotProductI8(const int8_t* lhs, const int8_t* rhs, int k) {
  size_t remaining = static_cast<size_t>(k);
  const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
  vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);
  while (remaining > 0) {
    const size_t vl = __riscv_vsetvl_e32m4(remaining);
    vint8m1_t v_lhs_i8 = __riscv_vle8_v_i8m1(lhs, vl);
    vint8m1_t v_rhs_i8 = __riscv_vle8_v_i8m1(rhs, vl);
    vint16m2_t v_lhs_i16 = __riscv_vsext_vf2_i16m2(v_lhs_i8, vl);
    vint16m2_t v_rhs_i16 = __riscv_vsext_vf2_i16m2(v_rhs_i8, vl);
    v_acc = __riscv_vwmacc_vv_i32m4(v_acc, v_lhs_i16, v_rhs_i16, vl);
    lhs += vl;
    rhs += vl;
    remaining -= vl;
  }
  vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
  vint32m1_t v_red =
      __riscv_vredsum_vs_i32m4_i32m1(v_acc, v_zero, vl0);
  return __riscv_vmv_x_s_i32m1_i32(v_red);
}

// Same but returns sum_k lhs[k]  (used for the rhs_zp correction term).
inline int32_t RvvSumI8(const int8_t* lhs, int k) {
  size_t remaining = static_cast<size_t>(k);
  const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
  vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);
  while (remaining > 0) {
    const size_t vl = __riscv_vsetvl_e32m4(remaining);
    vint8m1_t v_lhs_i8 = __riscv_vle8_v_i8m1(lhs, vl);
    vint16m2_t v_lhs_i16 = __riscv_vsext_vf2_i16m2(v_lhs_i8, vl);
    // i32 += i16 widened. Using vwadd.wv with v_acc + v_lhs_i16:
    v_acc = __riscv_vwadd_wv_i32m4(v_acc, v_lhs_i16, vl);
    lhs += vl;
    remaining -= vl;
  }
  vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
  vint32m1_t v_red =
      __riscv_vredsum_vs_i32m4_i32m1(v_acc, v_zero, vl0);
  return __riscv_vmv_x_s_i32m1_i32(v_red);
}

// Full INT8 GEMM with per-channel requantization. Assumes lhs zp == 0
// (symmetric INT8 filter), as set by optimized_integer_ops::ConvPerChannel.
inline void RvvGemmInt8PerChannel(int m, int n, int k,
                                  const int8_t* lhs_data,
                                  const int8_t* rhs_data, int rhs_zp,
                                  int dst_zp, const int32_t* bias_data,
                                  const int32_t* multiplier_per_channel,
                                  const int* shift_per_channel,
                                  int32_t clamp_min, int32_t clamp_max,
                                  int8_t* dst_data) {
  // Precompute per-row sum of lhs (used for the zp correction term).
  // n rows × constant for each (m_idx).
  // Allocated on the stack for typical sizes; LiteRT pointwise convs have
  // n ≤ 1024, so 1024 * 4 = 4 KiB is safe.
  int32_t lhs_row_sum_storage[2048];
  int32_t* lhs_row_sum = lhs_row_sum_storage;
  // Fall back to heap if larger (rare for our spec models).
  int32_t* heap_lhs_sum = nullptr;
  if (n > 2048) {
    heap_lhs_sum = new int32_t[n];
    lhs_row_sum = heap_lhs_sum;
  }
  for (int n_idx = 0; n_idx < n; ++n_idx) {
    lhs_row_sum[n_idx] =
        RvvSumI8(lhs_data + static_cast<size_t>(n_idx) * k, k);
  }

  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    int8_t* dst_col = dst_data + static_cast<size_t>(m_idx) * n;
    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const int8_t* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;
      int32_t acc = RvvDotProductI8(lhs_row, rhs_col, k);
      acc -= rhs_zp * lhs_row_sum[n_idx];  // zp correction
      if (bias_data != nullptr) acc += bias_data[n_idx];
      acc = MultiplyByQuantizedMultiplierRvv(
          acc, multiplier_per_channel[n_idx], shift_per_channel[n_idx]);
      acc += dst_zp;
      if (acc < clamp_min) acc = clamp_min;
      if (acc > clamp_max) acc = clamp_max;
      dst_col[n_idx] = static_cast<int8_t>(acc);
    }
  }

  delete[] heap_lhs_sum;
}

}  // namespace optimized_rvv
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_INT8_H_
