/* SPDX-License-Identifier: Apache-2.0
 *
 * RVSPOC S2601: RVV 1.0 FP32 GEMM kernel that bypasses ruy's StandardCpp
 * fallback on RV64GCV targets.
 *
 * Why bypass ruy instead of porting it: ruy is a separate FetchContented
 * library (google/ruy) with no upstream RISC-V support. Adding a Path::kRvv
 * to ruy + kernel/pack code is a multi-day effort touching a vendored repo.
 * cpu_backend_gemm already supports partial-specialization overrides
 * (used by gemmlowp / x86), so we hook in a header-only RVV impl that takes
 * the FP32 case and lets ruy keep handling everything else (INT8 GEMM
 * etc., pending follow-up).
 *
 * Algorithm: dot-product form GEMM.
 *   For each (m_idx, n_idx):
 *     acc = bias[n_idx]
 *     For each k_chunk:
 *       v_lhs = lhs[n_idx, k_chunk:k_chunk+vl]   (row-major n×k, sequential)
 *       v_rhs = rhs[k_chunk:k_chunk+vl, m_idx]   (col-major k×m, sequential)
 *       v_acc = vfmacc_vv(v_acc, v_lhs, v_rhs)
 *     acc = vfredusum(v_acc) + bias
 *     dst[n_idx, m_idx] = clamp(acc, clamp_min, clamp_max)
 *
 * Both lhs row-stride and rhs col-stride point at sequential memory in this
 * matrix orientation (set by optimized_ops::Conv FP32), so the inner reduction
 * loop streams cleanly. Cache reuse is not optimal (lhs row reread for each
 * m_idx), but the speedup over scalar StandardCpp is still substantial because
 * the FMA throughput per cycle scales with VLEN.
 *
 * Limitations of this pilot:
 *   - FP32 only (kFloatingPoint quantization flavor).
 *   - Assumes lhs row-major + rhs col-major + dst col-major (matches the
 *     orientation set by optimized_ops::Conv). Other orientations fall back
 *     to ruy via SFINAE conditions in the dispatch.
 *   - No outer blocking; for very large matrices a panel-blocked implementation
 *     would help cache. Future work.
 */
#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_

#if defined(__riscv_vector)

#include <riscv_vector.h>

#include <algorithm>
#include <cstddef>

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/cpu_backend_gemm_ruy.h"
#include "tflite/kernels/internal/optimized/rvv_gemm_fp32.h"
#include "tflite/kernels/internal/optimized/rvv_gemm_int8.h"
#include "tflite/kernels/internal/optimized/rvv_gemm_uint8.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

// FP32 GEMM specialization. Falls back to ruy when matrix orientations
// don't match the layouts emitted by optimized_ops::Conv (the only caller
// that this pilot targets).
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingRvv : GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar,
                                           DstScalar, quantization_flavor> {};

template <>
struct GemmImplUsingRvv<float, float, float, float,
                        QuantizationFlavor::kFloatingPoint> {
  static void Run(
      const MatrixParams<float>& lhs_params, const float* lhs_data,
      const MatrixParams<float>& rhs_params, const float* rhs_data,
      const MatrixParams<float>& dst_params, float* dst_data,
      const GemmParams<float, float, QuantizationFlavor::kFloatingPoint>&
          params,
      CpuBackendContext* context) {
    // The optimized_ops::Conv FP32 caller sets:
    //   lhs (filter):    n×k row-major
    //   rhs (im2col):    k×m col-major
    //   dst (output):    n×m col-major
    // Sanity-check; otherwise defer to ruy so we never silently produce
    // wrong results for an unexpected layout.
    const bool layout_ok = lhs_params.order == Order::kRowMajor &&
                           rhs_params.order == Order::kColMajor &&
                           dst_params.order == Order::kColMajor &&
                           lhs_params.rows == dst_params.rows &&
                           lhs_params.cols == rhs_params.rows &&
                           rhs_params.cols == dst_params.cols;
    if (!layout_ok) {
      GemmImplUsingRuy<float, float, float, float,
                       QuantizationFlavor::kFloatingPoint>::Run(lhs_params,
                                                                lhs_data,
                                                                rhs_params,
                                                                rhs_data,
                                                                dst_params,
                                                                dst_data,
                                                                params,
                                                                context);
      return;
    }

    optimized_rvv::RvvGemmFp32(
        /*m=*/rhs_params.cols, /*n=*/lhs_params.rows,
        /*k=*/lhs_params.cols, lhs_data, rhs_data, params.bias,
        params.clamp_min, params.clamp_max, dst_data);
  }
};

// INT8 (per-channel) GEMM specialization: lhs/rhs/dst all int8, accumulator
// int32, with kIntegerWithPerRowMultiplier requantization. Matches the
// shape produced by optimized_integer_ops::ConvPerChannel (the modern
// per-channel INT8 path used by EfficientDet & similar models).
template <>
struct GemmImplUsingRvv<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                        QuantizationFlavor::kIntegerWithPerRowMultiplier> {
  static void Run(
      const MatrixParams<std::int8_t>& lhs_params,
      const std::int8_t* lhs_data,
      const MatrixParams<std::int8_t>& rhs_params,
      const std::int8_t* rhs_data,
      const MatrixParams<std::int8_t>& dst_params, std::int8_t* dst_data,
      const GemmParams<std::int32_t, std::int8_t,
                       QuantizationFlavor::kIntegerWithPerRowMultiplier>&
          params,
      CpuBackendContext* context) {
    const bool layout_ok = lhs_params.order == Order::kRowMajor &&
                           rhs_params.order == Order::kColMajor &&
                           dst_params.order == Order::kColMajor &&
                           lhs_params.rows == dst_params.rows &&
                           lhs_params.cols == rhs_params.rows &&
                           rhs_params.cols == dst_params.cols &&
                           lhs_params.zero_point == 0 &&
                           params.multiplier_fixedpoint_perchannel != nullptr &&
                           params.multiplier_exponent_perchannel != nullptr;
    if (!layout_ok) {
      GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                       QuantizationFlavor::kIntegerWithPerRowMultiplier>::
          Run(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data,
              params, context);
      return;
    }
    optimized_rvv::RvvGemmInt8PerChannel(
        /*m=*/rhs_params.cols, /*n=*/lhs_params.rows,
        /*k=*/lhs_params.cols, lhs_data, rhs_data,
        /*rhs_zp=*/rhs_params.zero_point,
        /*dst_zp=*/dst_params.zero_point, params.bias,
        params.multiplier_fixedpoint_perchannel,
        params.multiplier_exponent_perchannel,
        /*clamp_min=*/static_cast<int32_t>(params.clamp_min),
        /*clamp_max=*/static_cast<int32_t>(params.clamp_max), dst_data);
  }
};

// uint8 (per-tensor uniform multiplier) GEMM specialization. Targets the
// 2018-era per-tensor uint8 quantization used by older MobileNet INT8
// .tflite models, which dispatches to a different GemmImpl from EfficientDet.
template <>
struct GemmImplUsingRvv<std::uint8_t, std::uint8_t, std::int32_t,
                        std::uint8_t,
                        QuantizationFlavor::kIntegerWithUniformMultiplier> {
  static void Run(
      const MatrixParams<std::uint8_t>& lhs_params,
      const std::uint8_t* lhs_data,
      const MatrixParams<std::uint8_t>& rhs_params,
      const std::uint8_t* rhs_data,
      const MatrixParams<std::uint8_t>& dst_params, std::uint8_t* dst_data,
      const GemmParams<std::int32_t, std::uint8_t,
                       QuantizationFlavor::kIntegerWithUniformMultiplier>&
          params,
      CpuBackendContext* context) {
    const bool layout_ok = lhs_params.order == Order::kRowMajor &&
                           rhs_params.order == Order::kColMajor &&
                           dst_params.order == Order::kColMajor &&
                           lhs_params.rows == dst_params.rows &&
                           lhs_params.cols == rhs_params.rows &&
                           rhs_params.cols == dst_params.cols;
    if (!layout_ok) {
      GemmImplUsingRuy<std::uint8_t, std::uint8_t, std::int32_t,
                       std::uint8_t,
                       QuantizationFlavor::kIntegerWithUniformMultiplier>::
          Run(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data,
              params, context);
      return;
    }
    optimized_rvv::RvvGemmUint8Uniform(
        /*m=*/rhs_params.cols, /*n=*/lhs_params.rows,
        /*k=*/lhs_params.cols, lhs_data,
        /*lhs_zp=*/lhs_params.zero_point, rhs_data,
        /*rhs_zp=*/rhs_params.zero_point,
        /*dst_zp=*/dst_params.zero_point, params.bias,
        params.multiplier_fixedpoint, params.multiplier_exponent,
        /*clamp_min=*/static_cast<int32_t>(params.clamp_min),
        /*clamp_max=*/static_cast<int32_t>(params.clamp_max), dst_data);
  }
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // __riscv_vector

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_RVV_H_
