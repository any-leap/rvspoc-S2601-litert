// SPDX-License-Identifier: Apache-2.0
//
// RVSPOC S2601: numerical accuracy test for the RVV uint8 GEMM
// (RvvGemmUint8Uniform) — covers the legacy per-tensor uint8 quant
// path used by older MobileNet INT8 .tflite models.
//
// Spec: INT8/UINT8 algo-level diff <= 1 LSB. We aim for integer-exact (==0).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tflite/kernels/internal/optimized/rvv_gemm_int8.h"  // requantization helpers
#include "tflite/kernels/internal/optimized/rvv_gemm_uint8.h"

namespace {

void ScalarReference(int m, int n, int k, const uint8_t* lhs_data,
                     int lhs_zp, const uint8_t* rhs_data, int rhs_zp,
                     int dst_zp, const int32_t* bias_data,
                     int32_t multiplier_fp, int shift, int32_t clamp_min,
                     int32_t clamp_max, uint8_t* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const uint8_t* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    uint8_t* dst_col = dst_data + static_cast<size_t>(m_idx) * n;
    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const uint8_t* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;
      int32_t acc = 0;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        acc += (static_cast<int32_t>(lhs_row[k_idx]) - lhs_zp) *
               (static_cast<int32_t>(rhs_col[k_idx]) - rhs_zp);
      }
      if (bias_data != nullptr) acc += bias_data[n_idx];
      acc = tflite::optimized_rvv::MultiplyByQuantizedMultiplierRvv(
          acc, multiplier_fp, shift);
      acc += dst_zp;
      if (acc < clamp_min) acc = clamp_min;
      if (acc > clamp_max) acc = clamp_max;
      dst_col[n_idx] = static_cast<uint8_t>(acc);
    }
  }
}

struct Case {
  const char* name;
  int m, n, k;
};

constexpr Case kCases[] = {
    {"mn_v1_pw3",  28*28, 128, 64},
    {"mn_v1_pw5",  28*28, 256, 128},
    {"mn_v1_pw7",  14*14, 512, 256},
    {"mn_v1_pw13",  7*7,  1024, 1024},
    {"odd_k",      64,    32,   17},
    {"tiny",       8,     4,    8},
};

bool RunCase(const Case& c, std::mt19937& rng) {
  std::uniform_int_distribution<int> dist8(0, 255);
  std::uniform_int_distribution<int> dist_zp(0, 255);
  std::uniform_int_distribution<int32_t> dist_mul(1 << 29, (1LL << 31) - 1);
  std::uniform_int_distribution<int> dist_shift(-15, 0);
  std::uniform_int_distribution<int32_t> dist_bias(-1 << 16, 1 << 16);

  const int lhs_zp = dist_zp(rng);
  const int rhs_zp = dist_zp(rng);
  const int dst_zp = dist_zp(rng);
  const int32_t mul = dist_mul(rng);
  const int shift = dist_shift(rng);

  std::vector<uint8_t> lhs(static_cast<size_t>(c.n) * c.k);
  std::vector<uint8_t> rhs(static_cast<size_t>(c.k) * c.m);
  std::vector<int32_t> bias(c.n);
  std::vector<uint8_t> dst_rvv(static_cast<size_t>(c.n) * c.m);
  std::vector<uint8_t> dst_ref(static_cast<size_t>(c.n) * c.m);
  for (auto& v : lhs) v = static_cast<uint8_t>(dist8(rng));
  for (auto& v : rhs) v = static_cast<uint8_t>(dist8(rng));
  for (auto& v : bias) v = dist_bias(rng);

  tflite::optimized_rvv::RvvGemmUint8Uniform(
      c.m, c.n, c.k, lhs.data(), lhs_zp, rhs.data(), rhs_zp, dst_zp,
      bias.data(), mul, shift, 0, 255, dst_rvv.data());
  ScalarReference(c.m, c.n, c.k, lhs.data(), lhs_zp, rhs.data(), rhs_zp,
                  dst_zp, bias.data(), mul, shift, 0, 255, dst_ref.data());

  int max_abs_diff = 0;
  for (size_t i = 0; i < dst_rvv.size(); ++i) {
    const int d = std::abs(static_cast<int>(dst_rvv[i]) -
                           static_cast<int>(dst_ref[i]));
    if (d > max_abs_diff) max_abs_diff = d;
  }
  const bool ok = max_abs_diff == 0;
  std::printf(
      "  %-12s m=%-5d n=%-5d k=%-5d lhs_zp=%3d rhs_zp=%3d dst_zp=%3d  "
      "max_abs_diff=%d  %s\n",
      c.name, c.m, c.n, c.k, lhs_zp, rhs_zp, dst_zp, max_abs_diff,
      ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const uint32_t seed = (argc > 1) ? std::atoi(argv[1]) : 0xC0DEC0DE;
  std::mt19937 rng(seed);

  std::printf(
      "RVSPOC S2601 — RVV uint8 per-tensor GEMM accuracy\n"
      "  seed=0x%x  (require integer-exact match)\n",
      seed);

  int n_pass = 0, n_fail = 0;
  for (const auto& c : kCases) {
    if (RunCase(c, rng)) ++n_pass;
    else ++n_fail;
  }
  std::printf("Result: %d passed, %d failed (%lu cases)\n", n_pass, n_fail,
              sizeof(kCases) / sizeof(kCases[0]));
  return n_fail == 0 ? 0 : 1;
}
