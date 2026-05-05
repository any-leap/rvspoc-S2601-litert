// SPDX-License-Identifier: Apache-2.0
//
// RVSPOC S2601: end-to-end model output dumper.
//
// Loads a .tflite model, fills the input tensor with deterministic
// pseudo-random bytes (xorshift32 seed = 1), runs inference, prints:
//   - all output tensor types/shapes/dtypes
//   - first 32 raw bytes of each output as hex (for byte-exact diff)
//   - top-5 indices and values of the largest output tensor
//
// Built twice with different -march flags (rv64gcv vs rv64gc), the
// outputs are diffed to verify our RVV kernels preserve model-level
// numerics. Spec gates:
//   FP32 model Top-1 vs x86 ≤ 0.1%      → expect Top-1 unchanged
//   FP32 algorithm-level rel err ≤ 1e-5 → expect close output values
//   INT8 model Top-1 vs x86 ≤ 1%        → expect Top-1 unchanged
//   INT8 algorithm-level diff ≤ 1 LSB    → expect bit-exact bytes

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "tflite/core/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

#define CHK(x)                                              \
  do {                                                      \
    if (!(x)) {                                             \
      std::fprintf(stderr, "Failed at %s:%d\n", __FILE__,   \
                   __LINE__);                               \
      std::exit(1);                                         \
    }                                                       \
  } while (0)

namespace {

// Deterministic input fill (xorshift32). Pure byte-level so it works
// for any tensor dtype (we cast to bytes in caller).
inline uint32_t Xorshift32(uint32_t* s) {
  uint32_t x = *s;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

void FillBytes(uint8_t* dst, size_t n, uint32_t seed) {
  uint32_t s = seed ? seed : 0x12345678;
  for (size_t i = 0; i < n; i++) dst[i] = static_cast<uint8_t>(Xorshift32(&s));
}

const char* TypeName(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "f32";
    case kTfLiteInt32:   return "i32";
    case kTfLiteUInt8:   return "u8";
    case kTfLiteInt8:    return "i8";
    case kTfLiteInt16:   return "i16";
    case kTfLiteInt64:   return "i64";
    case kTfLiteBool:    return "bool";
    default:             return "?";
  }
}

void PrintShape(const TfLiteIntArray* dims) {
  std::printf("[");
  for (int i = 0; i < dims->size; ++i) {
    std::printf("%s%d", i ? "," : "", dims->data[i]);
  }
  std::printf("]");
}

void PrintHexHead(const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i) std::printf("%02x", p[i]);
}

template <typename T>
void PrintTopK(const T* data, size_t count, int k, const char* label) {
  std::vector<size_t> idx(count);
  for (size_t i = 0; i < count; ++i) idx[i] = i;
  const int top = std::min<int>(k, static_cast<int>(count));
  std::partial_sort(idx.begin(), idx.begin() + top, idx.end(),
                    [&](size_t a, size_t b) { return data[a] > data[b]; });
  std::printf("    %s top-%d:", label, top);
  for (int i = 0; i < top; ++i) {
    if constexpr (std::is_floating_point<T>::value) {
      std::printf(" (%zu, %.6f)", idx[i], static_cast<double>(data[idx[i]]));
    } else {
      std::printf(" (%zu, %d)", idx[i],
                  static_cast<int>(data[idx[i]]));
    }
  }
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.tflite> [seed]\n", argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const uint32_t seed = (argc >= 3) ? std::atoi(argv[2]) : 1u;

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
  CHK(model);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  CHK(interpreter);
  interpreter->SetNumThreads(1);
  CHK(interpreter->AllocateTensors() == kTfLiteOk);

  std::printf("model=%s seed=%u\n", model_path, seed);

  // Fill every input tensor with deterministic data. For FP32 we used to
  // dump xorshift bytes straight into the buffer, which can produce NaN/
  // Inf bit patterns (Copilot review #11). Now generate u8 first then
  // map to a sane numeric range per dtype.
  for (int i : interpreter->inputs()) {
    TfLiteTensor* t = interpreter->tensor(i);
    std::printf("  input  #%d %s ", i, TypeName(t->type));
    PrintShape(t->dims);
    std::printf(" bytes=%zu\n", t->bytes);
    if (t->type == kTfLiteFloat32) {
      std::vector<uint8_t> u8buf(t->bytes / 4);
      FillBytes(u8buf.data(), u8buf.size(), seed + i);
      float* dst = reinterpret_cast<float*>(t->data.raw);
      for (size_t k = 0; k < u8buf.size(); ++k)
        dst[k] = (u8buf[k] / 127.5f) - 1.0f;  // pixel-like [-1, 1]
    } else {
      FillBytes(reinterpret_cast<uint8_t*>(t->data.raw), t->bytes, seed + i);
    }
  }

  CHK(interpreter->Invoke() == kTfLiteOk);

  // Dump every output tensor: type, shape, first 32 raw bytes, top-5.
  for (int i : interpreter->outputs()) {
    const TfLiteTensor* t = interpreter->tensor(i);
    std::printf("  output #%d %s ", i, TypeName(t->type));
    PrintShape(t->dims);
    std::printf(" bytes=%zu\n", t->bytes);
    const auto* p = reinterpret_cast<const uint8_t*>(t->data.raw);
    const size_t head = std::min<size_t>(32, t->bytes);
    std::printf("    raw[0:%zu]=", head);
    PrintHexHead(p, head);
    std::printf("\n");
    const size_t elems = t->bytes / [&] {
      switch (t->type) {
        case kTfLiteFloat32: case kTfLiteInt32: return 4;
        case kTfLiteInt16:                       return 2;
        case kTfLiteUInt8: case kTfLiteInt8:     return 1;
        case kTfLiteInt64:                       return 8;
        default:                                 return 1;
      }
    }();
    if (elems == 0) continue;
    switch (t->type) {
      case kTfLiteFloat32:
        PrintTopK(reinterpret_cast<const float*>(p), elems, 5, "f32");
        break;
      case kTfLiteInt8:
        PrintTopK(reinterpret_cast<const int8_t*>(p), elems, 5, "i8");
        break;
      case kTfLiteUInt8:
        PrintTopK(reinterpret_cast<const uint8_t*>(p), elems, 5, "u8");
        break;
      default:
        std::printf("    (no top-k for type %s)\n", TypeName(t->type));
    }
  }

  return 0;
}
