// SPDX-License-Identifier: Apache-2.0
//
// RVSPOC S2601: numerical accuracy test for the RVV
// QuantizedDepthwiseConvKernel<.., 0, *> specializations.
//
// Math: per output pixel, per input channel c:
//   i16 input_val  = sign_extend(input[c]) + input_offset
//   i16 filter_val = sign_extend(filter[c*M + k])
//   acc[c*M + k] += (i32)(input_val * filter_val)
//
// Spec: INT8 algo-level error <= 1 LSB (i.e. integer-exact).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h"

namespace {

void ScalarReference(int num_output_pixels, int input_depth,
                     int depth_multiplier, const int8_t* input_ptr,
                     int16_t input_offset, int input_ptr_increment,
                     const int8_t* filter_ptr, int32_t* acc_buffer_ptr) {
  for (int outp = 0; outp < num_output_pixels; outp++) {
    const int8_t* local_filter = filter_ptr;
    const int8_t* local_input = input_ptr;
    for (int ic = 0; ic < input_depth; ic++) {
      const int16_t input_val =
          static_cast<int16_t>(*local_input++) + input_offset;
      for (int k = 0; k < depth_multiplier; k++) {
        const int16_t filter_val = static_cast<int16_t>(*local_filter++);
        *acc_buffer_ptr++ +=
            static_cast<int32_t>(input_val) * static_cast<int32_t>(filter_val);
      }
    }
    input_ptr += input_ptr_increment;
  }
}

struct Case {
  const char* name;
  int num_output_pixels;
  int input_depth;
  int depth_multiplier;
};

constexpr Case kCases[] = {
    // <true, 0, 1>
    {"m1_d32",    8, 32,   1},
    {"m1_d64",    8, 64,   1},
    {"m1_d128",   8, 128,  1},
    {"m1_d256",   8, 256,  1},
    {"m1_d512",   8, 512,  1},
    {"m1_d1024",  4, 1024, 1},
    {"m1_odd_d3", 4, 3,    1},
    {"m1_odd_d7", 4, 7,    1},
    {"m1_odd_d31",4, 31,   1},
    {"m1_odd_d33",4, 33,   1},
    // <true, 0, 2>
    {"m2_d8",     4, 8,    2},
    {"m2_d32",    4, 32,   2},
    {"m2_odd_d3", 4, 3,    2},
    // <true, 0, 3>
    {"m3_d4",     4, 4,    3},
    {"m3_d8",     4, 8,    3},
    {"m3_d16",    4, 16,   3},
};

void RunKernel(const Case& c, const int8_t* input, int16_t input_offset,
               int input_ptr_increment, const int8_t* filter, int32_t* acc) {
  using tflite::optimized_integer_ops::depthwise_conv::QuantizedDepthwiseConvKernel;
  switch (c.depth_multiplier) {
    case 1:
      QuantizedDepthwiseConvKernel<true, 0, 1>::Run(
          c.num_output_pixels, c.input_depth, 1, input, input_offset,
          input_ptr_increment, filter, acc);
      break;
    case 2:
      QuantizedDepthwiseConvKernel<true, 0, 2>::Run(
          c.num_output_pixels, c.input_depth, 2, input, input_offset,
          input_ptr_increment, filter, acc);
      break;
    case 3:
      QuantizedDepthwiseConvKernel<true, 0, 3>::Run(
          c.num_output_pixels, c.input_depth, 3, input, input_offset,
          input_ptr_increment, filter, acc);
      break;
    default:
      std::fprintf(stderr, "Unsupported depth_multiplier=%d\n",
                   c.depth_multiplier);
      std::abort();
  }
}

bool RunCase(const Case& c, std::mt19937& rng) {
  std::uniform_int_distribution<int> dist8(-128, 127);
  std::uniform_int_distribution<int> dist_off(-128, 127);
  std::uniform_int_distribution<int32_t> dist32(-1 << 20, 1 << 20);
  const int input_ptr_increment = c.input_depth;
  const int output_depth = c.input_depth * c.depth_multiplier;
  const int16_t input_offset = static_cast<int16_t>(dist_off(rng));

  std::vector<int8_t> input(static_cast<size_t>(c.num_output_pixels) *
                                c.input_depth +
                            input_ptr_increment);
  std::vector<int8_t> filter(static_cast<size_t>(c.input_depth) *
                             c.depth_multiplier);
  std::vector<int32_t> acc_init(static_cast<size_t>(c.num_output_pixels) *
                                output_depth);
  for (auto& v : input) v = static_cast<int8_t>(dist8(rng));
  for (auto& v : filter) v = static_cast<int8_t>(dist8(rng));
  for (auto& v : acc_init) v = dist32(rng);

  std::vector<int32_t> acc_rvv = acc_init;
  std::vector<int32_t> acc_ref = acc_init;

  RunKernel(c, input.data(), input_offset, input_ptr_increment, filter.data(),
            acc_rvv.data());
  ScalarReference(c.num_output_pixels, c.input_depth, c.depth_multiplier,
                  input.data(), input_offset, input_ptr_increment,
                  filter.data(), acc_ref.data());

  int64_t max_abs_diff = 0;
  for (size_t i = 0; i < acc_rvv.size(); i++) {
    const int64_t d = std::abs(static_cast<int64_t>(acc_rvv[i]) -
                               static_cast<int64_t>(acc_ref[i]));
    if (d > max_abs_diff) max_abs_diff = d;
  }
  const bool ok = max_abs_diff == 0;  // INT8 spec: <= 1 LSB; we want exact.
  std::printf(
      "  %-12s mult=%-2d n_out=%2d depth=%4d off=%+4d  max_abs_diff=%lld  %s\n",
      c.name, c.depth_multiplier, c.num_output_pixels, c.input_depth,
      input_offset, static_cast<long long>(max_abs_diff),
      ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const uint32_t seed = (argc > 1) ? std::atoi(argv[1]) : 0xC0DEC0DE;
  std::mt19937 rng(seed);

  std::printf(
      "RVSPOC S2601 — RVV INT8 depthwise QuantizedDepthwiseConvKernel<true,0,*> "
      "accuracy\n  seed=0x%x  (require integer-exact match)\n",
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
