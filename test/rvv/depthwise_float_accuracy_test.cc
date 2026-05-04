// SPDX-License-Identifier: Apache-2.0
//
// RVSPOC S2601: numerical accuracy test for the RVV
// FloatDepthwiseConvKernel<true, 0, 1> specialization.
//
// Strategy: drive the RVV-enabled kernel and a hand-written scalar
// reference (which mirrors the scalar tail of every Neon/RVV variant)
// over the same randomized inputs, then assert max relative error.
// We don't compare against `FloatDepthwiseConvAccumRowGeneric` because
// it has a different control-flow structure (operates row-wise with
// padding logic), whereas FloatDepthwiseConvKernel::Run is the inner
// per-output-pixel kernel — easier to reason about a 1:1 reference.
//
// Spec gate (FP32 algo-level): relative error <= 1e-5.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Pull in the kernel under test. Note that USE_RVV is auto-defined when
// compiling with -march=rv64gcv, so the RVV specialization will be visible.
#include "tflite/kernels/internal/optimized/depthwiseconv_float.h"

namespace {

// Bit-exact scalar reference matching the math the Neon/RVV kernel does:
//   for each output pixel:
//     for each input channel c:
//       acc[c] += input[c] * filter[c]
//     input += input_ptr_increment   // (per-pixel stride)
void ScalarReference(int num_output_pixels, int input_depth,
                     const float* input_ptr, int input_ptr_increment,
                     const float* filter_ptr, float* acc_buffer_ptr) {
  for (int outp = 0; outp < num_output_pixels; outp++) {
    const float* local_filter = filter_ptr;
    const float* local_input = input_ptr;
    for (int ic = 0; ic < input_depth; ic++) {
      acc_buffer_ptr[ic] += (*local_input++) * (*local_filter++);
    }
    acc_buffer_ptr += input_depth;
    input_ptr += input_ptr_increment;
  }
}

struct Case {
  const char* name;
  int num_output_pixels;
  int input_depth;
};

// Cases covering the input_depth values MobileNetV1 actually uses, plus
// odd sizes (3, 7, 31, 33) to stress the vsetvl tail path.
constexpr Case kCases[] = {
    {"mn_v1_d32",   8, 32},
    {"mn_v1_d64",   8, 64},
    {"mn_v1_d128",  8, 128},
    {"mn_v1_d256",  8, 256},
    {"mn_v1_d512",  8, 512},
    {"mn_v1_d1024", 8, 1024},
    {"odd_d3",      4, 3},
    {"odd_d7",      4, 7},
    {"odd_d31",     4, 31},
    {"odd_d33",     4, 33},
    {"single_pixel_d128", 1, 128},
};

bool RunCase(const Case& c, double tol_rel, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int input_ptr_increment = c.input_depth;  // typical (stride==1)

  // num_output_pixels values of input + input_ptr_increment * (n-1) padding
  std::vector<float> input(static_cast<size_t>(c.num_output_pixels) *
                               c.input_depth +
                           input_ptr_increment);
  std::vector<float> filter(c.input_depth);
  std::vector<float> acc_init(static_cast<size_t>(c.num_output_pixels) *
                              c.input_depth);
  for (auto& v : input) v = dist(rng);
  for (auto& v : filter) v = dist(rng);
  for (auto& v : acc_init) v = dist(rng);

  std::vector<float> acc_rvv = acc_init;
  std::vector<float> acc_ref = acc_init;

  // Kernel under test (RVV when -march=rv64gcv; scalar fallback otherwise).
  tflite::optimized_ops::FloatDepthwiseConvKernel<true, 0, 1>::Run(
      c.num_output_pixels, c.input_depth, /*depth_multiplier=*/1, input.data(),
      input_ptr_increment, filter.data(), acc_rvv.data());

  ScalarReference(c.num_output_pixels, c.input_depth, input.data(),
                  input_ptr_increment, filter.data(), acc_ref.data());

  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < acc_rvv.size(); i++) {
    const double a = acc_rvv[i], b = acc_ref[i];
    const double abs_err = std::fabs(a - b);
    const double denom = std::max(std::fabs(b), 1e-9);
    const double rel_err = abs_err / denom;
    if (abs_err > max_abs) max_abs = abs_err;
    if (rel_err > max_rel) max_rel = rel_err;
  }
  const bool ok = max_rel <= tol_rel;
  std::printf("  %-20s n_out=%2d depth=%4d  max_abs=%.3e  max_rel=%.3e  %s\n",
              c.name, c.num_output_pixels, c.input_depth, max_abs, max_rel,
              ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  // Tolerance: FMA (vfmacc) is one rounding vs scalar's two (mul + add),
  // so RVV is at worst within ~1 ULP. Use 1e-5 to match the algo-level
  // S2601 spec; in practice we expect <1e-7 for these sizes.
  const double tol_rel = 1e-5;
  const uint32_t seed = (argc > 1) ? std::atoi(argv[1]) : 0xC0DEC0DE;
  std::mt19937 rng(seed);

  std::printf(
      "RVSPOC S2601 — RVV depthwise FloatDepthwiseConvKernel<true,0,1> "
      "accuracy\n  seed=0x%x  tol_rel=%.1e\n",
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
