// SPDX-License-Identifier: Apache-2.0
//
// RVSPOC S2601: numerical accuracy test for the RVV INT8 GEMM
// (RvvGemmInt8PerChannel) — the kernel hooked into cpu_backend_gemm
// for int8/int8/int32/int8 + kIntegerWithPerRowMultiplier.
//
// Spec: INT8 algo-level diff <= 1 LSB. We aim for integer-exact match
// (== 0) against a scalar reference using the same gemmlowp-style
// requantisation.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tflite/kernels/internal/optimized/rvv_gemm_int8.h"

namespace {

// Scalar reference using the same MultiplyByQuantizedMultiplier helpers.
void ScalarReference(int m, int n, int k, const int8_t* lhs_data,
                     const int8_t* rhs_data, int rhs_zp, int dst_zp,
                     const int32_t* bias_data,
                     const int32_t* multiplier_per_channel,
                     const int* shift_per_channel, int32_t clamp_min,
                     int32_t clamp_max, int8_t* dst_data) {
  for (int m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* rhs_col = rhs_data + static_cast<size_t>(m_idx) * k;
    int8_t* dst_col = dst_data + static_cast<size_t>(m_idx) * n;
    for (int n_idx = 0; n_idx < n; ++n_idx) {
      const int8_t* lhs_row = lhs_data + static_cast<size_t>(n_idx) * k;
      int32_t acc = 0;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        acc += static_cast<int32_t>(lhs_row[k_idx]) *
               static_cast<int32_t>(rhs_col[k_idx]);
      }
      int32_t lhs_row_sum = 0;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        lhs_row_sum += static_cast<int32_t>(lhs_row[k_idx]);
      }
      acc -= rhs_zp * lhs_row_sum;
      if (bias_data != nullptr) acc += bias_data[n_idx];
      acc = tflite::optimized_rvv::MultiplyByQuantizedMultiplierRvv(
          acc, multiplier_per_channel[n_idx], shift_per_channel[n_idx]);
      acc += dst_zp;
      if (acc < clamp_min) acc = clamp_min;
      if (acc > clamp_max) acc = clamp_max;
      dst_col[n_idx] = static_cast<int8_t>(acc);
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
    {"tiny_k1",    16,    16,   1},
    {"tiny",       8,     4,    8},
};

bool RunCase(const Case& c, std::mt19937& rng) {
  std::uniform_int_distribution<int> dist8(-128, 127);
  std::uniform_int_distribution<int> dist_zp(-128, 127);
  // Per-channel multipliers: a typical TFLite per-channel multiplier is
  // in [1<<29, 1<<31), with shifts roughly in [-15, 0]. Use that range.
  std::uniform_int_distribution<int32_t> dist_mul(1 << 29, (1LL << 31) - 1);
  std::uniform_int_distribution<int> dist_shift(-15, 0);
  std::uniform_int_distribution<int32_t> dist_bias(-1 << 16, 1 << 16);

  const int rhs_zp = dist_zp(rng);
  const int dst_zp = dist_zp(rng);
  std::vector<int8_t> lhs(static_cast<size_t>(c.n) * c.k);
  std::vector<int8_t> rhs(static_cast<size_t>(c.k) * c.m);
  std::vector<int32_t> bias(c.n);
  std::vector<int32_t> mul(c.n);
  std::vector<int> shift(c.n);
  std::vector<int8_t> dst_rvv(static_cast<size_t>(c.n) * c.m);
  std::vector<int8_t> dst_ref(static_cast<size_t>(c.n) * c.m);
  for (auto& v : lhs) v = static_cast<int8_t>(dist8(rng));
  for (auto& v : rhs) v = static_cast<int8_t>(dist8(rng));
  for (auto& v : bias) v = dist_bias(rng);
  for (auto& v : mul) v = dist_mul(rng);
  for (auto& v : shift) v = dist_shift(rng);

  tflite::optimized_rvv::RvvGemmInt8PerChannel(
      c.m, c.n, c.k, lhs.data(), rhs.data(), rhs_zp, dst_zp, bias.data(),
      mul.data(), shift.data(), -128, 127, dst_rvv.data());

  ScalarReference(c.m, c.n, c.k, lhs.data(), rhs.data(), rhs_zp, dst_zp,
                  bias.data(), mul.data(), shift.data(), -128, 127,
                  dst_ref.data());

  int max_abs_diff = 0;
  for (size_t i = 0; i < dst_rvv.size(); ++i) {
    const int d = std::abs(static_cast<int>(dst_rvv[i]) -
                           static_cast<int>(dst_ref[i]));
    if (d > max_abs_diff) max_abs_diff = d;
  }
  const bool ok = max_abs_diff == 0;
  std::printf(
      "  %-12s m=%-5d n=%-5d k=%-5d rhs_zp=%+4d dst_zp=%+4d  max_abs_diff=%d  %s\n",
      c.name, c.m, c.n, c.k, rhs_zp, dst_zp, max_abs_diff,
      ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const uint32_t seed = (argc > 1) ? std::atoi(argv[1]) : 0xC0DEC0DE;
  std::mt19937 rng(seed);

  std::printf(
      "RVSPOC S2601 — RVV INT8 per-channel GEMM accuracy\n"
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
