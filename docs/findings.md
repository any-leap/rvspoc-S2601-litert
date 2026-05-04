# Findings

> 调试 / 探索中的非直觉发现。每条独立编号 `FIND-NNN`，只增不减。

## FIND-001 [代码盘点] LiteRT 现有 Neon 优化文件清单

- 日期：2026-05-04
- 现象：S2601 赛题要求覆盖"所有已有 ARM Neon/SVE 优化"的 ≥ 90%。需要先盘清"所有"到底有多少。
- 根因/机制：在 `tflite/kernels/internal/optimized/` 下 grep `__ARM_NEON | <arm_neon.h> | USE_NEON | TFLITE_USE_NEON`，命中 26 个文件；grep SVE 相关宏（`__ARM_FEATURE_SVE`, `<arm_sve.h>`）**0 命中**。所以"Neon/SVE"实际上只有 Neon。
- 证据/复现：
  ```bash
  grep -rlE "(__ARM_NEON|<arm_neon\.h>|USE_NEON|TFLITE_USE_NEON)" tflite/kernels/internal/optimized
  grep -rlE "(__ARM_FEATURE_SVE|<arm_sve\.h>|USE_SVE)" tflite/kernels/internal/optimized
  ```
- LOC 分布（按文件大小排序，含非 Neon 部分）：

  | 文件 | LOC |
  |---|---|
  | `depthwiseconv_uint8_3x3_filter.h` | 13,442 |
  | `optimized_ops.h` | 8,164 |
  | `depthwiseconv_uint8_transitional.h` | 8,129 |
  | `legacy_optimized_ops.h` | 5,022 |
  | `neon_tensor_utils.cc` | 2,806 |
  | `depthwiseconv_uint8.h` | 2,127 |
  | `integer_ops/depthwise_conv.h` | 2,026 |
  | `resize_bilinear.h` | 1,736 |
  | `depthwiseconv_float.h` | 1,117 |
  | `reduce.h` | 808 |
  | `4bit/neon_fully_connected_aarch64_nosdot.cc` | 731 |
  | `4bit/neon_fully_connected.cc` | 603 |
  | `depthwiseconv_3x3_filter_common.h` | 589 |
  | `integer_ops/add.h` | 513 |
  | `integer_ops/depthwise_conv_hybrid.h` | 511 |
  | `4bit/neon_fully_connected_aarch64_sdot.cc` | 506 |
  | `integer_ops/pooling.h` | 278 |
  | `integer_ops/mul.h` | 266 |
  | `integer_ops/mean.h` | 251 |
  | `4bit/neon_fully_connected_arm32.cc` | 220 |
  | `neon_tensor_utils_impl.h` | 198 |
  | `4bit/neon_fully_connected.h` | 192 |
  | `fully_connected_4bit.h` | 152 |
  | `4bit/neon_fully_connected_impl.h` | 81 |
  | `integer_ops/lut.h` | 72 |
  | `neon_check.h` | 40 |
  | **合计** | **~50,580** |

- 工程含义：
  - 实际 Neon intrinsic 代码量比 50k 小（很多是 scalar + neon 共存），但仍是数千行级别工作量
  - **Hot 4 个文件**（depthwiseconv_uint8_3x3_filter / optimized_ops / depthwiseconv_uint8_transitional / legacy_optimized_ops）占 70% LOC
  - `legacy_optimized_ops.h` 看名字是历史遗留，可能不在 90% 覆盖目标内（待确认）
  - `4bit/` 目录下的 `_aarch64_sdot.cc` / `_aarch64_nosdot.cc` / `_arm32.cc` 是 aarch64/arm32 平台特定汇编，对应到 RISC-V 应统一用 RVV intrinsic 实现（一个 `.cc` 而不是三个）
- #kernels #inventory #neon

## FIND-003 [工具链] Docker + GCC 14.2 + QEMU 8.2 路线打通 RVV 1.0

- 日期：2026-05-05
- 现象：在 macOS Apple Silicon 上需要 cross-compile + run RV64GCV ELF
- 方案：项目自建 Docker 镜像 `rvspoc-s2601:latest`（基于 ubuntu:24.04），装：
  - `gcc-14-riscv64-linux-gnu` / `g++-14-riscv64-linux-gnu`（GCC 14.2.0，full RVV 1.0 intrinsic 支持）
  - `qemu-user-static`（QEMU 8.2.2，默认 vector version v1.0）
  - cmake 3.28.3 + ninja 1.11.1
- 证据/复现：
  ```bash
  ./docker/rvspoc/build.sh
  docker run --rm -v "$(pwd):/work" -w /work rvspoc-s2601:latest \
    bash docker/rvspoc/smoke_test/run_smoke.sh
  # 输出："rvv_hello: ... PASS"，c[0]=16.0 c[15]=16.0
  ```
- ELF 验证：`ELF 64-bit LSB executable, UCB RISC-V, RVC, double-float ABI, statically linked`
- QEMU 调用约定：`qemu-riscv64-static -cpu rv64,v=true,vlen=256,elen=64 ./binary.elf`
- #toolchain #docker #qemu #rvv

## FIND-004 [build] rvspoc fork 的 standalone 拆分残留 — proto 路径不一致

- 日期：2026-05-05
- 现象：host build 跑到 ~600/642 时，protoc 报错：
  ```
  /work/tflite/profiling/proto/profiling_info.proto: File does not reside within
  any path specified using --proto_path (or -I).
  ```
  涉及 3 个 .proto：`tflite/profiling/proto/profiling_info.proto`、`tflite/profiling/proto/model_runtime_info.proto`、`tflite/tools/benchmark/proto/benchmark_result.proto`
- 根因/机制：
  - rvspoc fork 是 LiteRT 拆分后的 standalone 仓库（顶层 `tflite/` 而不是 `tensorflow/lite/`）
  - 但 `tflite/profiling/proto/CMakeLists.txt` 和 `tflite/tools/benchmark/proto/CMakeLists.txt` 没有跟着拆分改干净，仍然写 `--proto_path=${TENSORFLOW_SOURCE_DIR}` 并把 .pb.h 输出到 `${CMAKE_BINARY_DIR}/tensorflow/lite/profiling/proto/`
  - 但 C++ 代码（`tflite/profiling/profile_summary_formatter.h` 等）已经改为 `#include "tflite/profiling/proto/profiling_info.pb.h"`
  - 两边路径对不上 → protoc 找不到 .proto，且就算生成了 C++ 也 include 错路径
- 证据/复现：
  ```bash
  grep -rE 'include.*"(tensorflow/lite|tflite)/profiling/proto/profiling_info' tflite/
  # 全是 "tflite/..." 路径
  cat tflite/profiling/proto/CMakeLists.txt
  # OUTPUT 路径却是 ${CMAKE_BINARY_DIR}/tensorflow/lite/profiling/proto/...
  ```
- 修复：patch 这 3 个 CMakeLists 把 OUTPUT 路径和 protoc 命令改用 `tflite/...` + `--proto_path=${TFLITE_SOURCE_DIR}/..`（即 `/work` 仓库根）。详见 commit。
- 工程含义：
  - 这是 fork 自己的 bug，不是上游问题——可能值得反提 PR 给 rvspoc 组委会
  - 后续 sync 上游时要注意别覆盖这个 patch
- #cmake #proto #fork-bug

## FIND-005 [baseline] MobileNetV1 FP32 在 RV64GCV scalar 上 CONV_2D 占 95.2%

- 日期：2026-05-05
- 测试条件：
  - 模型：mobilenet_v1_1.0_224.tflite (FP32, ~16 MB)
  - 编译：`-march=rv64gcv -O3`，TFLITE_ENABLE_RUY=ON，XNNPACK/GPU OFF
  - 运行：`qemu-riscv64-static -cpu rv64,v=true,vlen=256,elen=64`
  - 命令：`benchmark_model --num_threads=1 --num_runs=5 --warmup_runs=1 --enable_op_profiling=true`
- 总推理延迟：**~15.7 s** avg（QEMU emulation，**真硬件预期数量级低**）
- 算子级分布：

  | 算子 | 次数 | 累计 ms | 占比 |
  |---|---|---|---|
  | CONV_2D | 15 | 14 981 | **95.186%** |
  | DEPTHWISE_CONV_2D | 13 | 757 | 4.807% |
  | AVERAGE_POOL_2D | 1 | 1.0 | 0.006% |
  | SOFTMAX | 1 | 0.14 | 0.001% |
  | SQUEEZE | 1 | 0.03 | 0.000% |

- Top-8 hot kernels（全是 pointwise CONV_2D 1×1，单个 1.36–1.45 s，合计占 71%）：
  - `Conv2d_3_pointwise`、`Conv2d_5_pointwise`、`Conv2d_7_pointwise`、`Conv2d_8_pointwise`、`Conv2d_9_pointwise`、`Conv2d_10_pointwise`、`Conv2d_11_pointwise`、`Conv2d_13_pointwise`
- 工程含义：
  - **优化优先级**: CONV_2D > DEPTHWISE_CONV_2D >> 其它。其它合起来不到 1%，优化它们对赛题 110ms 目标几乎无意义
  - **Pointwise (1×1) conv 占绝大头**——它本质上是 `im2col + GEMM`，所以底层走的应该是 ruy 的 GEMM 路径，不是 `optimized_ops.h` 的卷积手写汇编
  - 因此 RVV 优化 ruy 的 GEMM kernel（`ruy/kernel_*.h` / `pack_*.h`）可能比改 LiteRT optimized_ops 更划算
  - 必须验证：CONV_2D 在 FP32 / INT8 时各走哪条路径——下一步看代码
- 数据稳定性：std=101 057 μs，5 次跑数据 min=15 667 170 μs / max=15 937 766 μs，差异 <2%
- #baseline #profile #conv2d #ruy

## FIND-002 [SVE] LiteRT 当前没有 SVE 优化代码

- 日期：2026-05-04
- 现象：赛题描述写"ARM Neon/SVE 优化的内核"，但实际 grep 上游 LiteRT 源码，SVE 相关宏 0 命中
- 根因/机制：LiteRT 当前优化层只针对 Neon（aarch64 NEON + arm32 NEON），没有 SVE 路径
- 证据/复现：
  ```bash
  grep -rlE "(__ARM_FEATURE_SVE|<arm_sve\.h>|USE_SVE|svfloat32)" tflite/  # 0 命中
  ```
- 工程含义：覆盖目标实际只针对 Neon。这降低了一些工作量。
- #sve #scope
