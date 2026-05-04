// SPDX-License-Identifier: Apache-2.0
//
// RVSPOC S2601: numerical accuracy test for the RVV FP32 GEMM kernel
// (cpu_backend_gemm_rvv.h). Drives our GemmImplUsingRvv against a hand-
// written scalar reference at the same matrix orientations
// optimized_ops::Conv emits (lhs n×k row-major, rhs k×m col-major,
// dst n×m col-major).
//
// Spec: FP32 algo-level relative error <= 1e-5.
// Note: GEMM accumulation order differs between scalar reference and the
// RVV vfmacc reduction tree, so bit-exactness isn't expected — we check
// that errors stay well below the 1e-5 threshold across realistic shapes.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tflite/kernels/internal/optimized/rvv_gemm_fp32.h"

namespace {

// Scalar GEMM reference matching optimized_ops::Conv FP32 layouts.
//   lhs: n×k row-major   → lhs[n_idx, k_idx] = lhs_data[n_idx*k + k_idx]
//   rhs: k×m col-major   → rhs[k_idx, m_idx] = rhs_data[m_idx*k + k_idx]
//   dst: n×m col-major   → dst[n_idx, m_idx] = dst_data[m_idx*n + n_idx]
void ScalarReference(int m, int n, int k, const float* lhs_data,
                     const float* rhs_data, const float* bias,
                     float clamp_min, float clamp_max, float* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const float* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    float* dst_col = dst_data + static_cast<size_t>(m_idx) * n;
    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const float* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;
      double acc = 0.0;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        acc += static_cast<double>(lhs_row[k_idx]) *
               static_cast<double>(rhs_col[k_idx]);
      }
      if (bias != nullptr) acc += static_cast<double>(bias[n_idx]);
      if (acc < clamp_min) acc = clamp_min;
      if (acc > clamp_max) acc = clamp_max;
      dst_col[n_idx] = static_cast<float>(acc);
    }
  }
}

struct Case {
  const char* name;
  int m, n, k;
  bool use_bias;
  float clamp_min, clamp_max;
};

// Cases mirror MobileNetV1 pointwise CONV_2D layer dimensions plus
// odd shapes for vsetvl tail coverage.
constexpr Case kCases[] = {
    {"mn_v1_pw3",   28*28, 128, 64,   true, 0.0f, 6.0f},   // ReLU6
    {"mn_v1_pw5",   28*28, 256, 128,  true, 0.0f, 6.0f},
    {"mn_v1_pw7",   14*14, 512, 256,  true, 0.0f, 6.0f},
    {"mn_v1_pw11",   7*7,  1024, 512, true, 0.0f, 6.0f},
    {"mn_v1_pw13",   7*7,  1024, 1024,true, 0.0f, 6.0f},
    {"unactivated", 64,    32,   16,  true, -3.4e38f, 3.4e38f},
    {"no_bias",     32,    16,   8,   false, -3.4e38f, 3.4e38f},
    {"odd_k",       64,    32,   17,  true, -3.4e38f, 3.4e38f},
    {"tiny_k1",     16,    16,   1,   true, 0.0f, 6.0f},
};

bool RunCase(const Case& c, double tol_rel, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> lhs(static_cast<size_t>(c.n) * c.k);
  std::vector<float> rhs(static_cast<size_t>(c.k) * c.m);
  std::vector<float> bias(c.n);
  std::vector<float> dst_rvv(static_cast<size_t>(c.n) * c.m);
  std::vector<float> dst_ref(static_cast<size_t>(c.n) * c.m);
  for (auto& v : lhs) v = dist(rng);
  for (auto& v : rhs) v = dist(rng);
  for (auto& v : bias) v = dist(rng);

  // Test the standalone RVV GEMM kernel directly (it's the same code the
  // cpu_backend_gemm partial specialization forwards to once the layout
  // check passes — keeping the test off cpu_backend_gemm.h means we don't
  // have to link ruy to compile it).
  tflite::optimized_rvv::RvvGemmFp32(
      c.m, c.n, c.k, lhs.data(), rhs.data(),
      c.use_bias ? bias.data() : nullptr, c.clamp_min, c.clamp_max,
      dst_rvv.data());

  ScalarReference(c.m, c.n, c.k, lhs.data(), rhs.data(),
                  c.use_bias ? bias.data() : nullptr, c.clamp_min, c.clamp_max,
                  dst_ref.data());

  // Mixed tolerance: a result passes if abs(a-b) <= tol_rel * max(|b|, 1).
  // This is the standard convention for IEEE FP comparisons — pure relative
  // error blows up when the reference is exactly zero (e.g. ReLU clipping
  // a negative pre-activation to 0). max_abs is what the spec actually
  // cares about for those cases.
  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < dst_rvv.size(); ++i) {
    const double a = dst_rvv[i], b = dst_ref[i];
    const double abs_err = std::fabs(a - b);
    const double denom = std::max(std::fabs(b), 1.0);
    const double rel_err = abs_err / denom;
    if (abs_err > max_abs) max_abs = abs_err;
    if (rel_err > max_rel) max_rel = rel_err;
  }
  const bool ok = max_rel <= tol_rel;
  std::printf(
      "  %-12s m=%-5d n=%-5d k=%-4d  max_abs=%.3e  max_rel=%.3e  %s\n",
      c.name, c.m, c.n, c.k, max_abs, max_rel, ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  // S2601 algo-level FP32 spec: relative error ≤ 1e-5.
  const double tol_rel = 1e-5;
  const uint32_t seed = (argc > 1) ? std::atoi(argv[1]) : 0xC0DEC0DE;
  std::mt19937 rng(seed);

  std::printf(
      "RVSPOC S2601 — RVV FP32 GEMM (cpu_backend_gemm_rvv.h) accuracy\n"
      "  seed=0x%x  tol_rel=%.1e\n",
      seed, tol_rel);

  int n_pass = 0, n_fail = 0;
  for (const auto& c : kCases) {
    if (RunCase(c, tol_rel, rng)) ++n_pass;
    else ++n_fail;
  }
  std::printf("Result: %d passed, %d failed (%lu cases)\n", n_pass, n_fail,
              sizeof(kCases) / sizeof(kCases[0]));
  return n_fail == 0 ? 0 : 1;
}
