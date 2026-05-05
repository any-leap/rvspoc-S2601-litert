// SPDX-License-Identifier: Apache-2.0
//
// RVSPOC S2601: ImageNet-style Top-1 / Top-5 accuracy evaluation driver.
//
// Reads a manifest CSV (one entry per line: "image_path,label_index"),
// loads each image via stb_image (JPEG / PNG / BMP supported), pre-
// processes per the model's input dtype + range conventions, runs
// inference, and reports Top-1 / Top-5 hit counts.
//
// Used to satisfy the S2601 spec model-level Top-1 gate
//   FP32: Top-1 vs x86 ≤ 0.1%
//   INT8: Top-1 vs x86 ≤ 1%
// by running the same driver on (a) the RVV-active build and (b) the
// scalar control build, and diffing the resulting Top-1 percentages.
//
// Preprocessing is model-aware:
//   - Float input tensor: assumes input range is [-1, 1] (MobileNet),
//     i.e. (uint8_pixel / 127.5) - 1.
//   - uint8 input tensor: pixel passes through with no rescaling
//     (MobileNet quantized inputs are stored as uint8 with the model's
//     own zero_point/scale handling the scaling).
//   - int8 input tensor: same as uint8 but stored signed
//     (rare for ImageNet-style classifiers).
// Center-crops then resizes to the model's expected (H, W).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#include "third_party/stb_image.h"

#include "tflite/core/interpreter_builder.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"

#define CHK(x)                                                                 \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::fprintf(stderr, "FAIL %s:%d\n", __FILE__, __LINE__);                \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

namespace {

// Center-crop a uint8 image to a square of side min(h,w), in-place pointer
// + new dims (no copy). Implements the standard ImageNet pre-process step
// before resize.
void CenterCropSquare(const uint8_t*& in, int& in_h, int& in_w, int channels,
                     std::vector<uint8_t>& tmp) {
  const int side = std::min(in_h, in_w);
  if (side == in_h && side == in_w) return;
  const int y0 = (in_h - side) / 2;
  const int x0 = (in_w - side) / 2;
  tmp.resize(static_cast<size_t>(side) * side * channels);
  for (int y = 0; y < side; ++y) {
    std::memcpy(&tmp[(y * side) * channels],
                in + ((y0 + y) * in_w + x0) * channels,
                static_cast<size_t>(side) * channels);
  }
  in = tmp.data();
  in_h = side;
  in_w = side;
}

// Bilinear resize uint8 image to (out_h, out_w, channels). Equivalent to
// what tflite's image preprocessing does — keeps the driver self-contained
// (no OpenCV dep).
void ResizeBilinearU8(const uint8_t* in, int in_h, int in_w, int channels,
                      int out_h, int out_w, uint8_t* out) {
  const float fy = static_cast<float>(in_h) / out_h;
  const float fx = static_cast<float>(in_w) / out_w;
  for (int oy = 0; oy < out_h; ++oy) {
    const float src_y = (oy + 0.5f) * fy - 0.5f;
    int y0 = static_cast<int>(std::floor(src_y));
    int y1 = y0 + 1;
    const float dy = src_y - y0;
    y0 = std::clamp(y0, 0, in_h - 1);
    y1 = std::clamp(y1, 0, in_h - 1);
    for (int ox = 0; ox < out_w; ++ox) {
      const float src_x = (ox + 0.5f) * fx - 0.5f;
      int x0 = static_cast<int>(std::floor(src_x));
      int x1 = x0 + 1;
      const float dx = src_x - x0;
      x0 = std::clamp(x0, 0, in_w - 1);
      x1 = std::clamp(x1, 0, in_w - 1);
      for (int c = 0; c < channels; ++c) {
        const float p00 = in[(y0 * in_w + x0) * channels + c];
        const float p01 = in[(y0 * in_w + x1) * channels + c];
        const float p10 = in[(y1 * in_w + x0) * channels + c];
        const float p11 = in[(y1 * in_w + x1) * channels + c];
        const float v = p00 * (1 - dx) * (1 - dy) + p01 * dx * (1 - dy) +
                        p10 * (1 - dx) * dy + p11 * dx * dy;
        out[(oy * out_w + ox) * channels + c] =
            static_cast<uint8_t>(std::clamp(v + 0.5f, 0.0f, 255.0f));
      }
    }
  }
}

// Read manifest: one line per image, "image_path,label_index" (label is
// the class index in [0, num_classes)).
struct Sample {
  std::string path;
  int label;
};

std::vector<Sample> ReadManifest(const char* manifest_path) {
  std::vector<Sample> out;
  std::ifstream f(manifest_path);
  if (!f) {
    std::fprintf(stderr, "Cannot open manifest: %s\n", manifest_path);
    std::exit(2);
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto comma = line.find(',');
    if (comma == std::string::npos) continue;
    Sample s;
    s.path = line.substr(0, comma);
    s.label = std::atoi(line.substr(comma + 1).c_str());
    out.push_back(std::move(s));
  }
  return out;
}

// Pick the predicted class as the argmax of the largest-rank output
// tensor (the classifier head). For MobileNet the output is a single
// 1×1001 (or 1×1000) tensor of float / uint8 / int8.
int Argmax(const TfLiteTensor* t, int* top5_out = nullptr) {
  const auto type = t->type;
  size_t n = 1;
  for (int d = 0; d < t->dims->size; ++d) n *= t->dims->data[d];
  std::vector<float> scores(n);
  switch (type) {
    case kTfLiteFloat32:
      for (size_t i = 0; i < n; ++i) scores[i] = t->data.f[i];
      break;
    case kTfLiteUInt8:
      for (size_t i = 0; i < n; ++i)
        scores[i] = static_cast<float>(t->data.uint8[i]);
      break;
    case kTfLiteInt8:
      for (size_t i = 0; i < n; ++i)
        scores[i] = static_cast<float>(t->data.int8[i]);
      break;
    default:
      std::fprintf(stderr, "Unsupported output type %d\n", type);
      return -1;
  }
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = i;
  const int top_n = top5_out ? 5 : 1;
  std::partial_sort(
      idx.begin(), idx.begin() + std::min<size_t>(top_n, n), idx.end(),
      [&](size_t a, size_t b) { return scores[a] > scores[b]; });
  if (top5_out) {
    for (int k = 0; k < std::min<int>(top_n, n); ++k) top5_out[k] = idx[k];
  }
  return static_cast<int>(idx[0]);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <model.tflite> <manifest.csv> [label_offset]\n"
                 "  manifest.csv: lines of  image_path,label_index\n"
                 "  label_offset: ADD this to manifest labels before comparing\n"
                 "                to the model argmax. Use 1 when the manifest\n"
                 "                holds standard 0-based ImageNet labels and\n"
                 "                the model has a 1001-class output (idx 0 is\n"
                 "                'background'). Default 0 means the manifest\n"
                 "                already encodes the model's index space.\n",
                 argv[0]);
    return 1;
  }
  const char* model_path = argv[1];
  const char* manifest_path = argv[2];
  const int label_offset = (argc >= 4) ? std::atoi(argv[3]) : 0;

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
  CHK(model);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  CHK(interpreter);
  interpreter->SetNumThreads(1);
  CHK(interpreter->AllocateTensors() == kTfLiteOk);

  // Discover input tensor shape.
  CHK(interpreter->inputs().size() == 1);
  const int in_idx = interpreter->inputs()[0];
  const TfLiteTensor* in_t = interpreter->tensor(in_idx);
  CHK(in_t->dims->size == 4);
  const int in_h = in_t->dims->data[1];
  const int in_w = in_t->dims->data[2];
  const int in_c = in_t->dims->data[3];
  CHK(in_c == 3);  // assume RGB classifiers

  std::printf("Model: %s\n", model_path);
  std::printf("Input: [%d,%d,%d] dtype=%d\n", in_h, in_w, in_c, in_t->type);

  auto samples = ReadManifest(manifest_path);
  std::printf("Manifest: %zu samples (label_offset=%d)\n", samples.size(),
              label_offset);

  // Quantization params for int8/uint8 input — the model's calibrated
  // (scale, zero_point) maps stored quantized values back to the float
  // pixel range it was trained on (typically [-1, 1] or [0, 1]). Without
  // using these, INT8 evaluations on models with non-default zp/scale
  // get the wrong input distribution (Copilot review #15).
  const float in_scale =
      (in_t->params.scale != 0.0f) ? in_t->params.scale : 1.0f;
  const int in_zp = in_t->params.zero_point;

  int n_evaluated = 0;
  int n_top1 = 0;
  int n_top5 = 0;
  std::vector<uint8_t> resized(in_h * in_w * in_c);
  std::vector<uint8_t> cropped_tmp;

  for (const auto& s : samples) {
    int img_w = 0, img_h = 0, img_c = 0;
    uint8_t* pixels =
        stbi_load(s.path.c_str(), &img_w, &img_h, &img_c, /*desired=*/3);
    if (!pixels) {
      std::fprintf(stderr, "  skip (decode failed): %s\n", s.path.c_str());
      continue;
    }
    // Standard ImageNet preprocessing: center-crop to square first, then
    // bilinear-resize to model input. Without the crop step, evaluating a
    // non-square image distorts geometry (Copilot review #14).
    const uint8_t* crop_in = pixels;
    int crop_h = img_h, crop_w = img_w;
    CenterCropSquare(crop_in, crop_h, crop_w, 3, cropped_tmp);
    ResizeBilinearU8(crop_in, crop_h, crop_w, 3, in_h, in_w, resized.data());
    stbi_image_free(pixels);

    // Fill the input tensor in the model's expected layout.
    if (in_t->type == kTfLiteFloat32) {
      float* dst = interpreter->typed_input_tensor<float>(0);
      for (int i = 0; i < in_h * in_w * in_c; ++i)
        dst[i] = (resized[i] / 127.5f) - 1.0f;
    } else if (in_t->type == kTfLiteUInt8) {
      // uint8 quant: float_pixel ≈ in_scale * (q - in_zp). Solve for q:
      // q = clamp(round(float_pixel/in_scale) + in_zp). The float_pixel
      // we want to feed is a [0,255] uint8 already scaled to [-1,1] like
      // the float path: (resized[i]/127.5 - 1.0).
      uint8_t* dst = interpreter->typed_input_tensor<uint8_t>(0);
      for (int i = 0; i < in_h * in_w * in_c; ++i) {
        const float f = (resized[i] / 127.5f) - 1.0f;
        const int q = static_cast<int>(std::lround(f / in_scale)) + in_zp;
        dst[i] = static_cast<uint8_t>(std::clamp(q, 0, 255));
      }
    } else if (in_t->type == kTfLiteInt8) {
      int8_t* dst = interpreter->typed_input_tensor<int8_t>(0);
      for (int i = 0; i < in_h * in_w * in_c; ++i) {
        const float f = (resized[i] / 127.5f) - 1.0f;
        const int q = static_cast<int>(std::lround(f / in_scale)) + in_zp;
        dst[i] = static_cast<int8_t>(std::clamp(q, -128, 127));
      }
    } else {
      std::fprintf(stderr, "Unsupported input dtype %d\n", in_t->type);
      return 2;
    }

    CHK(interpreter->Invoke() == kTfLiteOk);
    const TfLiteTensor* out_t = interpreter->tensor(interpreter->outputs()[0]);
    int top5[5] = {-1, -1, -1, -1, -1};
    int pred1 = Argmax(out_t, top5);
    // Add (not subtract) the offset to map manifest's index space to the
    // model's. Convention: label_offset=1 for 0-based ImageNet labels
    // against a 1001-class MobileNet output (Copilot review #18).
    const int label = s.label + label_offset;
    if (pred1 == label) ++n_top1;
    for (int k = 0; k < 5; ++k)
      if (top5[k] == label) {
        ++n_top5;
        break;
      }
    ++n_evaluated;
    if (n_evaluated % 10 == 0) {
      std::fprintf(stderr,
                   "  [%4d/%4zu] top1=%d top5=%d  cur top1=%.2f%%\n",
                   n_evaluated, samples.size(), n_top1, n_top5,
                   100.0 * n_top1 / n_evaluated);
    }
  }

  if (n_evaluated == 0) {
    std::fprintf(stderr, "No images evaluated.\n");
    return 2;
  }
  std::printf("\nResult: evaluated=%d  top-1=%d (%.2f%%)  top-5=%d (%.2f%%)\n",
              n_evaluated, n_top1, 100.0 * n_top1 / n_evaluated, n_top5,
              100.0 * n_top5 / n_evaluated);
  return 0;
}
