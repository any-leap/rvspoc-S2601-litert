/* SPDX-License-Identifier: Apache-2.0
 *
 * RVSPOC S2601: standalone RVV 1.0 uint8 GEMM with uniform-multiplier
 * (per-tensor) requantization.
 *
 * Math (matches optimized_ops::Conv<uint8>):
 *   acc[n,m] = sum_k (lhs[n,k] - lhs_zp) * (rhs[k,m] - rhs_zp)
 *   acc += bias[n]
 *   acc = MultiplyByQuantizedMultiplier(acc, multiplier_fp, shift)
 *                                       (single multiplier for all output channels)
 *   acc += dst_zp
 *   dst[n,m] = clamp_uint8(acc, clamp_min, clamp_max)
 *
 * Layout convention (matches the legacy uint8 path):
 *   lhs: n × k row-major   filter, zero_point != 0
 *   rhs: k × m col-major   activations, zero_point != 0
 *   dst: n × m col-major
 *
 * Per element, subtract the zero-point in place after vzext (we work in
 * signed i16 once values are widened, since lhs/rhs after subtraction
 * can be negative). Then vwmacc_vv multiplies them and accumulates into
 * the i32 dot-product register. Same dot-product structure as the FP32
 * and INT8 helpers — only the zp-subtract step changes.
 */
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_UINT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_UINT8_H_

#if defined(__riscv_vector)

#include <riscv_vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "tflite/kernels/internal/optimized/rvv_gemm_int8.h"
// (For MultiplyByQuantizedMultiplierRvv & friends.)

namespace tflite {
namespace optimized_rvv {

// Vectorised dot product over k for uint8 inputs with both zero-points
// subtracted in place.
//   returns sum_k (lhs[k] - lhs_zp) * (rhs[k] - rhs_zp)
inline int32_t RvvDotProductU8WithZp(const uint8_t* lhs, int lhs_zp,
                                     const uint8_t* rhs, int rhs_zp, int k) {
  size_t remaining = static_cast<size_t>(k);
  const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
  vint32m4_t v_acc = __riscv_vmv_v_x_i32m4(0, vl0);
  const int16_t neg_lhs_zp = static_cast<int16_t>(-lhs_zp);
  const int16_t neg_rhs_zp = static_cast<int16_t>(-rhs_zp);
  while (remaining > 0) {
    const size_t vl = __riscv_vsetvl_e32m4(remaining);
    vuint8m1_t v_lhs_u8 = __riscv_vle8_v_u8m1(lhs, vl);
    vuint8m1_t v_rhs_u8 = __riscv_vle8_v_u8m1(rhs, vl);
    // u8 -> u16 -> reinterpret i16 -> add (-zp) (signed result)
    vint16m2_t v_lhs_i16 = __riscv_vreinterpret_v_u16m2_i16m2(
        __riscv_vzext_vf2_u16m2(v_lhs_u8, vl));
    v_lhs_i16 = __riscv_vadd_vx_i16m2(v_lhs_i16, neg_lhs_zp, vl);
    vint16m2_t v_rhs_i16 = __riscv_vreinterpret_v_u16m2_i16m2(
        __riscv_vzext_vf2_u16m2(v_rhs_u8, vl));
    v_rhs_i16 = __riscv_vadd_vx_i16m2(v_rhs_i16, neg_rhs_zp, vl);
    v_acc = __riscv_vwmacc_vv_i32m4(v_acc, v_lhs_i16, v_rhs_i16, vl);
    lhs += vl;
    rhs += vl;
    remaining -= vl;
  }
  vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
  vint32m1_t v_red = __riscv_vredsum_vs_i32m4_i32m1(v_acc, v_zero, vl0);
  return __riscv_vmv_x_s_i32m1_i32(v_red);
}

inline void RvvGemmUint8Uniform(int m, int n, int k,
                                const uint8_t* lhs_data, int lhs_zp,
                                const uint8_t* rhs_data, int rhs_zp,
                                int dst_zp, const int32_t* bias_data,
                                int32_t multiplier_fp, int shift,
                                int32_t clamp_min, int32_t clamp_max,
                                uint8_t* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const uint8_t* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    uint8_t* dst_col = dst_data + static_cast<size_t>(m_idx) * n;
    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const uint8_t* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;
      int32_t acc = RvvDotProductU8WithZp(lhs_row, lhs_zp, rhs_col, rhs_zp, k);
      if (bias_data != nullptr) acc += bias_data[n_idx];
      acc = MultiplyByQuantizedMultiplierRvv(acc, multiplier_fp, shift);
      acc += dst_zp;
      if (acc < clamp_min) acc = clamp_min;
      if (acc > clamp_max) acc = clamp_max;
      dst_col[n_idx] = static_cast<uint8_t>(acc);
    }
  }
}

}  // namespace optimized_rvv
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_UINT8_H_
