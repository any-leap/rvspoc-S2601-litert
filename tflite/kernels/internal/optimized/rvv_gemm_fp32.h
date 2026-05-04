/* SPDX-License-Identifier: Apache-2.0
 *
 * RVSPOC S2601: standalone RVV 1.0 FP32 GEMM kernel.
 *
 * Math: dst = clamp(lhs * rhs + bias, clamp_min, clamp_max)
 *
 * Layout convention (matches optimized_ops::Conv FP32 emit):
 *   lhs: n × k row-major   → lhs[n_idx*k + k_idx]
 *   rhs: k × m col-major   → rhs[m_idx*k + k_idx]
 *   dst: n × m col-major   → dst[m_idx*n + n_idx]
 *   bias: optional, length n; added once per output column
 *
 * Both inner-loop loads stream sequentially in this orientation, so
 * the dot-product reduction over k vectorises cleanly. LMUL=4 keeps
 * trip count low at small VLEN.
 *
 * Header-only by design — kept independent of cpu_backend_gemm /
 * cpu_backend_context so the kernel can be unit-tested without linking
 * ruy or the rest of TFLite.
 */
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_FP32_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_FP32_H_

#if defined(__riscv_vector)
#include <riscv_vector.h>

#include <cstddef>

namespace tflite {
namespace optimized_rvv {

inline void RvvGemmFp32(int m, int n, int k, const float* lhs_data,
                        const float* rhs_data, const float* bias_data,
                        float clamp_min, float clamp_max, float* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const float* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    float* dst_col = dst_data + static_cast<size_t>(m_idx) * n;
    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const float* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;

      size_t remaining = static_cast<size_t>(k);
      const size_t vl0 = __riscv_vsetvl_e32m4(remaining);
      vfloat32m4_t v_acc = __riscv_vfmv_v_f_f32m4(0.0f, vl0);
      const float* p_lhs = lhs_row;
      const float* p_rhs = rhs_col;
      while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t v_lhs = __riscv_vle32_v_f32m4(p_lhs, vl);
        vfloat32m4_t v_rhs = __riscv_vle32_v_f32m4(p_rhs, vl);
        v_acc = __riscv_vfmacc_vv_f32m4(v_acc, v_lhs, v_rhs, vl);
        p_lhs += vl;
        p_rhs += vl;
        remaining -= vl;
      }
      vfloat32m1_t v_zero =
          __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
      vfloat32m1_t v_red =
          __riscv_vfredusum_vs_f32m4_f32m1(v_acc, v_zero, vl0);
      float acc = __riscv_vfmv_f_s_f32m1_f32(v_red);
      if (bias_data != nullptr) acc += bias_data[n_idx];
      if (acc < clamp_min) acc = clamp_min;
      if (acc > clamp_max) acc = clamp_max;
      dst_col[n_idx] = acc;
    }
  }
}

}  // namespace optimized_rvv
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_GEMM_FP32_H_
