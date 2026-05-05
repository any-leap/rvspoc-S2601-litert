<!--
SPDX-License-Identifier: Apache-2.0
RVSPOC S2601 — LiteRT RISC-V Adaptation and RVV Optimization
Submission documentation
-->

# RVSPOC S2601 提交说明 / Submission Notes

**Challenge**: [S2601 — LiteRT RISC-V 架构适配与 RVV 优化](https://rvspoc.org/S2601)
**Submitter fork**: <https://github.com/any-leap/rvspoc-S2601-litert>
**Submission branch**: `rvv-port`
**Target**: PR against [`rv2036/rvspoc-S2601-litert`](https://github.com/rv2036/rvspoc-S2601-litert)
**License**: Apache 2.0 (matches LiteRT upstream)

---

## 1. 实现概览 / Overview

This submission ports LiteRT (TensorFlow Lite) to RISC-V RV64GCV and adds
RVV 1.0 vectorized implementations for the operator paths that dominate
inference time on the spec-listed benchmark models. The implementation
strategy:

- **In-tree** RVV blocks alongside the existing Neon blocks (gated on
  `#ifdef USE_RVV` defined in a new `rvv_check.h`, exactly mirroring
  the existing `neon_check.h` pattern). Scalar fallback preserved.
- **Bypass ruy via `cpu_backend_gemm` partial specialization** for
  the FP32 / int8 / uint8 GEMM paths — ruy itself has no upstream
  RISC-V support, and adding it would require forking & patching
  the FetchContented `google/ruy` repo. The partial-spec hook is a
  smaller surface area and falls back to ruy on any unexpected matrix
  layout.

实现策略：
- 与现有 Neon 块平行，在 `#ifdef USE_RVV` 下加 RVV 路径，scalar fallback 保留
- GEMM 通过 `cpu_backend_gemm` 的 partial specialization 拦截，绕开 ruy 库（ruy 上游无 RISC-V 支持）

---

## 2. 当前覆盖范围 / Coverage

### LiteRT 内核文件覆盖（9 / 26 ≈ 35%）

| File | RVV adds |
|---|---|
| `tflite/kernels/internal/optimized/depthwiseconv_float.h` | 14/14 specs (100%) |
| `tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h` | 22/22 specs (100%) |
| `tflite/kernels/internal/optimized/depthwiseconv_uint8.h` | 22/22 specs (100%) |
| `tflite/kernels/internal/optimized/integer_ops/add.h` | INT8 elementwise add |
| `tflite/kernels/internal/optimized/integer_ops/mul.h` | INT8 elementwise mul |
| `tflite/kernels/internal/optimized/integer_ops/pooling.h` | MaxPool / AvgPool int8 |
| `tflite/kernels/internal/optimized/integer_ops/mean.h` | Mean int8 |
| `tflite/kernels/internal/optimized/integer_ops/lut.h` | LUT u8 (Logistic/Tanh) |
| `tflite/kernels/internal/optimized/reduce.h` | Mean uint8 |

### GEMM coverage（CONV_2D / FULLY_CONNECTED 走的路径）

| Type | RVV impl |
|---|---|
| FP32 GEMM | `tflite/kernels/internal/optimized/rvv_gemm_fp32.h` |
| INT8 per-channel GEMM | `tflite/kernels/internal/optimized/rvv_gemm_int8.h` |
| UINT8 per-tensor GEMM | `tflite/kernels/internal/optimized/rvv_gemm_uint8.h` |

The GEMM hooks are wired into `cpu_backend_gemm.h` via partial
specializations of `GemmImpl`.

### Operator coverage by op type

| Op | Status |
|---|---|
| CONV_2D | ✅ via RVV GEMM (FP32 + int8 + uint8) |
| DEPTHWISE_CONV_2D | ✅ FP32 + int8 + uint8 100% |
| FULLY_CONNECTED | ✅ via RVV GEMM (same hook as CONV_2D) |
| ADD | ✅ int8 |
| MUL | ✅ int8 |
| MAX_POOL_2D | ✅ int8 |
| AVERAGE_POOL_2D | ✅ int8 |
| MEAN | ✅ int8 + uint8 |
| LUT (Logistic, Tanh) | ✅ u8 |

Ops without RVV implementation yet (low priority — FIND-009 measures
each at < 0.5% of inference on the spec models): SOFTMAX, SQUEEZE,
RESHAPE, CONCATENATION, RESIZE_NEAREST_NEIGHBOR, RESIZE_BILINEAR, ADD's
broadcast variant, the legacy `depthwiseconv_uint8_3x3_filter.h` fast
path (13k LOC of hand-written assembly).

---

## 3. 编译步骤 / Build instructions

### 3.1 Prerequisites

- Docker (Apple Silicon / Linux x86_64 / Linux arm64)
- ~10 GB free disk space (build artifacts)
- ~16 GB Docker memory recommended (the LiteRT build is RAM-hungry per
  `cc1plus`; see `docs/gotchas.md` GOT-002)
- The TF v2.21.0-rc0 source tarball (auto-fetched on first run; fallback
  pre-extract documented in `docs/gotchas.md` GOT-001)

### 3.2 One-shot build

```bash
# 1. Build the cross-compile docker image (~2 min, builds once)
./docker/rvspoc/build.sh

# 2. Build host tools (flatc) — needed by the cross-compile step
./docker/rvspoc/run.sh bash scripts/build_host.sh

# 3. Cross-compile benchmark_model + dumper for RV64GCV
./docker/rvspoc/run.sh bash scripts/build_riscv64.sh

# Optional: also build the RV64GC scalar control (for accuracy diffing)
./docker/rvspoc/run.sh bash scripts/build_scalar_baseline.sh
```

Outputs:

- `build-rv64/tools/benchmark/benchmark_model` — TFLite benchmark
  with our RVV path active (`-march=rv64gcv`)
- `build-rv64-scalar/tools/benchmark/benchmark_model` — same source
  with `-march=rv64gc` (no V) for accuracy comparison
- `build-rv64/tools/benchmark/rvspoc_model_output_dumper` — minimal
  inference driver (used for end-to-end output equivalence checking)

### 3.3 Build configuration knobs

- `BUILD_JOBS=N` env var: cap parallel `cc1plus` (default 4; raise if
  Docker has more memory)
- `cmake/toolchain-riscv64-rvv.cmake`: change `-march` if your target
  has additional extensions (e.g. RVV 1.0 + Zvfh half-float)

---

## 4. 运行步骤 / Running

### 4.1 Single benchmark on a model

```bash
# Inside the container; vlen= argument controls the simulated VLEN
./docker/rvspoc/run.sh \
  qemu-riscv64-static -cpu rv64,v=true,vlen=256,elen=64 \
    -L /usr/riscv64-linux-gnu \
    ./build-rv64/tools/benchmark/benchmark_model \
      --graph=.cache/models/mobilenet_v1_1.0_224.tflite \
      --num_threads=1 --num_runs=5 --warmup_runs=1 \
      --enable_op_profiling=true
```

### 4.2 Profile all 5 spec models in one go

```bash
./docker/rvspoc/run.sh bash scripts/profile_all_models.sh
```

Outputs full `--enable_op_profiling` reports per model into
`.cache/profile/`.

### 4.3 Run the unit-test suite (accuracy)

```bash
./docker/rvspoc/run.sh bash scripts/run_rvv_tests.sh
```

Compiles 6 standalone unit tests against the kernel headers, runs each
under qemu at vlen ∈ {128, 256, 512}. Should print `Result: N passed,
0 failed` for every test × every VLEN.

### 4.4 End-to-end model output equivalence (RVV vs scalar)

```bash
./docker/rvspoc/run.sh bash scripts/verify_model_outputs.sh
```

Requires both `build-rv64/` and `build-rv64-scalar/` to exist. Diffs
the dumped output tensors between the two builds for each spec model.

---

## 5. 运行结果 / Measured results

### 5.1 Performance (QEMU vlen=256, single thread, 3-run average)

QEMU emulation overhead makes the absolute numbers far slower than real
silicon (a single MAC takes ≫1 cycle in software emulation). The
*relative* improvement vs the scalar baseline is what matters.

| Model | Scalar baseline (ms) | With RVV (ms) | Speedup |
|---|---|---|---|
| MobileNetV1 1.0 224 FP32  | 15 740 | **6 684** | **2.35×** |
| MobileNetV1 1.0 224 INT8  | 14 969 | **5 217** | **2.87×** |
| MobileNetV2 1.0 224 FP32  |  7 607 | **3 893** | **1.95×** |
| MobileNetV2 1.0 224 INT8  | 11 240 | **3 142** | **3.58×** |
| EfficientDet-Lite0 INT8   | 33 680 | **9 950** | **3.39×** |

CONV_2D operator alone (the dominant 90%+ slice on every model):

| Model | Scalar (ms) | RVV (ms) | Speedup |
|---|---|---|---|
| MobileNetV1 INT8 | 14 060 | **4 905** | 2.87× |
| MobileNetV2 INT8 | 10 174 | **2 748** | 3.70× |
| EfficientDet INT8 | 30 146 | **8 144** | 3.70× |

### 5.2 Accuracy

Algorithm-level (per-kernel) accuracy from the unit tests
(`test/rvv/*_accuracy_test.cc`):

| Test | VLEN | Cases | Spec gate | Achieved |
|---|---|---|---|---|
| FP32 depthwise | 128/256/512 | 22 × 3 = 66 | rel err ≤ 1e-5 | **0** (bit-exact) |
| INT8 depthwise | 128/256/512 | 16 × 3 = 48 | diff ≤ 1 LSB | **0** (integer-exact) |
| UINT8 depthwise | 128/256/512 | 16 × 3 = 48 | diff ≤ 1 LSB | **0** |
| FP32 GEMM | 128/256/512 | 9 × 3 = 27 | rel err ≤ 1e-5 | ≤ 6.7e-6 |
| INT8 GEMM | 128/256/512 | 7 × 3 = 21 | diff ≤ 1 LSB | **0** |
| UINT8 GEMM | 128/256/512 | 6 × 3 = 18 | diff ≤ 1 LSB | **0** |
| **Total** | | **228 / 228 PASS** | | |

Model-level (end-to-end) accuracy
(`scripts/verify_model_outputs.sh`):

| Model | Top-1 vs scalar build (single seeded image) |
|---|---|
| MobileNetV1 FP32 | ✅ AGREE |
| MobileNetV1 INT8 | ✅ AGREE (raw bytes IDENTICAL) |
| MobileNetV2 FP32 | ✅ AGREE |
| MobileNetV2 INT8 | ✅ AGREE |
| EfficientDet-Lite0 INT8 | ⚠ Detection-tensor Top-1 reorders (single image; mAP-over-dataset is the appropriate metric for detection models, not Top-1) |

VLEN-agnostic: every accuracy test runs identically and passes at
vlen=128, vlen=256, and vlen=512, satisfying the spec's "支持不同 VLEN
（128/256/512 bit）的自适应实现" requirement (achieved via per-iteration
`vsetvl` rather than fixed-width unrolls).

### 5.3 Hardware testing

Hardware testing on a real RISC-V development board (SG2044 / A210) is
**pending** — we are awaiting access to the A210 remote environment
the organizing committee will provide. All current numbers are from
QEMU 8.2.2 emulation. The relative speedup ratios are expected to
hold on real silicon, though absolute latency numbers (and the
compute / memory-bandwidth balance) will differ substantially.

---

## 6. 验证步骤 / Verification recipe (for organizers)

```bash
git clone -b rvv-port https://github.com/any-leap/rvspoc-S2601-litert.git
cd rvspoc-S2601-litert

# 1. (Once) build the docker image
./docker/rvspoc/build.sh

# 2. Build host tools
./docker/rvspoc/run.sh bash scripts/build_host.sh

# 3. Build the RVV variant
./docker/rvspoc/run.sh bash scripts/build_riscv64.sh

# 4. Run unit tests — should report 228/228 PASS
./docker/rvspoc/run.sh bash scripts/run_rvv_tests.sh

# 5. Profile spec models (after downloading them into .cache/models/)
./docker/rvspoc/run.sh bash scripts/profile_all_models.sh

# 6. (Optional, slow ~30min) Build scalar control + diff outputs
./docker/rvspoc/run.sh bash scripts/build_scalar_baseline.sh
./docker/rvspoc/run.sh bash scripts/verify_model_outputs.sh
```

Test models (download into `.cache/models/`):

```bash
mkdir -p .cache/models && cd .cache/models
curl -fsSLO https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
tar xzf mobilenet_v1_1.0_224.tgz mobilenet_v1_1.0_224.tflite
curl -fsSLO https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
tar xzf mobilenet_v1_1.0_224_quant.tgz mobilenet_v1_1.0_224_quant.tflite
curl -fsSLO https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
tar xzf mobilenet_v2_1.0_224.tgz mobilenet_v2_1.0_224.tflite
curl -fsSLO https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar xzf mobilenet_v2_1.0_224_quant.tgz mobilenet_v2_1.0_224_quant.tflite
# EfficientDet-Lite0 (Kaggle Models mirror)
curl -fsSL "https://www.kaggle.com/api/v1/models/tensorflow/efficientdet/tfLite/lite0-detection-metadata/1/download" \
  | tar xz && mv 1.tflite efficientdet_lite0.tflite
rm -f *.tgz mobilenet_v1_1.0_224_eval.pbtxt mobilenet_v1_1.0_224_frozen.pb \
      mobilenet_v1_1.0_224_info.txt mobilenet_v1_1.0_224.ckpt.*
```

---

## 7. 限定平台与依赖 / Platform & dependencies

### 限定平台

- **Build host**: any system with Docker (Apple Silicon macOS / Linux
  x86_64 / Linux arm64). Build is hermetic inside the container.
- **Target**: `riscv64-linux-gnu`, `-march=rv64gcv -mabi=lp64d`,
  RVV 1.0 (the V extension is required; pure RV64GC will fall back
  to LiteRT scalar paths and our RVV code is `#ifdef`'d out).
- **Runtime emulator**: QEMU 8.2.2 with `qemu-user-static`. Real
  hardware: Sophon SG2044, RVA22+V profile boards (A210), or
  equivalent — testing on these is pending access.

### 依赖（自动拉取）

- LiteRT mainline (this repo, the rvspoc fork)
- TensorFlow v2.21.0-rc0 (CMake `FetchContent`; fallback: pre-extract
  to `.cache/tensorflow-src/` per `docs/gotchas.md` GOT-001)
- abseil-cpp, ruy, gemmlowp, FlatBuffers, Eigen, protobuf, cpuinfo,
  ml_dtypes (all auto-fetched by LiteRT's CMake)

### 工具链 (in the docker image)

- gcc-14-riscv64-linux-gnu (GCC 14.2.0 — first GCC release with full
  RVV 1.0 intrinsic support)
- qemu-user-static 8.2.2 (RVV 1.0 default vector spec)
- cmake 3.28.3, ninja 1.11.1, protobuf-compiler 3.21.12

---

## 8. AI 辅助说明 / AI-assistance disclosure

Per spec requirement 「若使用 AI 辅助编写代码，需在提交报告中说明使用方式及占比」:

- **Tool**: Anthropic Claude (Claude Opus 4.7 via Claude Code CLI).
- **Usage breakdown**:
  - **Architecture / strategy**: human-driven decisions (which paths
    to attack, how to structure the partial-specialization hooks,
    whether to fork ruy vs bypass it). Documented in
    `docs/decisions.md`.
  - **Kernel implementation**: collaborative — the human directed
    each step (which file, which spec, which RVV intrinsic family);
    AI drafted the C++ then iterated based on build / test feedback.
    Estimated ~80% of the RVV kernel C++ was first-draft generated
    by AI, then refined.
  - **Test infrastructure & CMake / build scripts**: ~70% AI-drafted,
    refined for build issues (notably the proto-path patch documented
    in FIND-004 was AI-found-and-fixed).
  - **Findings / gotchas docs**: collaborative summarization of build
    failures and benchmark observations.
  - **Verification / numerical analysis**: human-driven (interpreting
    accuracy results, deciding tolerance models, choosing test cases).
- **Verification of AI output**: every commit was test-verified
  (228 unit-test cases pass at 3 VLEN values; model-level Top-1
  preserved on classification spec models). No AI-generated code
  was committed without successful build + at least one accuracy
  test pass.

---

## 9. 项目级文档 / Project-internal documentation

For diagnosis and gotcha tracking we maintain three project-level files
in `docs/`:

- `docs/decisions.md` — DEC-NNN: architecture / strategy decisions
- `docs/findings.md` — FIND-NNN: non-obvious discoveries (e.g. the
  rvspoc fork's leftover proto-path bug, the cross-model CONV_2D
  dominance baseline, the ruy-bypass speedup numbers)
- `docs/gotchas.md` — GOT-NNN: build / runtime traps and recovery
  steps

These document the path from cold-start to the current measurements
and are useful both for future maintenance and as a record of what
was tried.

---

## 10. 已知缺口 / Known gaps

| Gap | Why |
|---|---|
| Real hardware data | Awaiting A210 remote access from organizing committee |
| ImageNet 1000-image Top-1 dataset evaluation | Requires download + classification driver; in-progress |
| File coverage 9/26 → 90% target | Ongoing — the remaining 17 files include the 13k+8k LOC hand-tuned 3×3 fast paths and the `optimized_ops.h` catch-all (8k LOC) |
| INT16 GEMM / INT16 ops | Not on the spec model hot-path; deferred |
| `XNNPACK` integration | XNNPACK has no RVV path upstream; we build with `-DTFLITE_ENABLE_XNNPACK=OFF` |
