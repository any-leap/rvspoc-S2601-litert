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

## FIND-006 [pilot] 第一个 RVV kernel — DEPTHWISE_CONV_2D FP32 提速 2.15×

- 日期：2026-05-05
- 算子：`FloatDepthwiseConvKernel<true, 0, 1>`（动态 input_depth，strided，depth_multiplier=1）
  - 文件：`tflite/kernels/internal/optimized/depthwiseconv_float.h`
  - 这是 MobileNetV1 全部 13 个 depthwise convs 都走的 kernel（FIND-005）
- 实现策略：
  - 单一 vsetvl 驱动循环（取代 Neon 的 16-wide / 4-wide / scalar tail 三段）
  - LMUL=4（vfloat32m4_t），单次循环吃 vl 个 float
  - 用 `vfmacc_vv` 做融合乘加，accumulation 顺序与 scalar 完全一致 → 精度等价或更优
  - VLEN-agnostic，对 vlen=128/256/512 都成立
- 结果（mobilenet_v1_1.0_224.tflite，QEMU vlen=256，5 runs）：

  | 指标 | scalar | RVV pilot | 变化 |
  |---|---|---|---|
  | DEPTHWISE_CONV_2D 累计 ms | 757 | **352** | **-53.5%（2.15×）** |
  | CONV_2D 累计 ms（对照，未改） | 14 981 | 14 873 | ~0% |
  | 总推理 ms | 15 740 | 15 226 | -3.3% |
  | std（μs） | 101 057 | 13 688 | -86% |

- 工程含义：
  - 原 LiteRT 在 RV64GCV 上 depthwise 走的是 `FloatDepthwiseConvAccumRowGeneric`（scalar fallback）。换上 RVV 立即拿到 2.15×
  - 总推理只降 3.3% 因为 depthwise 在 FP32 MobileNetV1 占比小（4.8%）；RVV depthwise 对 INT8 模型（DEPTHWISE 占比更高）会更值钱（待 Task 9 验证）
  - **真正动总推理需要碰 ruy GEMM**（CONV_2D 95%）— Task 8
  - 标准差从 101 ms 降到 14 ms，说明矢量代码 path 可预测性比 scalar 好（更少分支）
- 接下来还要做：
  - 数值精度独立测试（目前 only by inference 不崩 + FMA 顺序与 scalar 完全一致间接证明）
  - 把其它 depthwise specialization（`<true, 0, 2>`、`<true, 0, 8>`、`<true, 0, 16>` 等）也补上
- #pilot #rvv #depthwise #milestone

## FIND-007 [accuracy] RVV depthwise <true,0,1> 对 scalar bit-exact，3 个 VLEN 全过

- 日期：2026-05-05
- 测试：`test/rvv/depthwise_float_accuracy_test.cc` 用 11 个 case（MobileNetV1 实际 input_depth + 奇数尾边界）
- 结果：**所有 case 在 vlen=128/256/512 三档下 max_abs=0.000e+00，max_rel=0.000e+00**

  | VLEN | cases passed | max_rel |
  |---|---|---|
  | 128 | 11/11 | 0 |
  | 256 | 11/11 | 0 |
  | 512 | 11/11 | 0 |

- 为什么 bit-exact：
  - RVV 实现按相同顺序遍历 input channel
  - `vfmacc` 是 IEEE FMA（一次舍入），与 scalar 的 `acc += a*b`（两次舍入：mul、add）相比**精度只会更高**
  - 在这里两边都被舍到同一个 float — 因为 (a*b + c) 本来就在表示范围内
- 已满足 spec 三项硬要求：
  - ✅ 算子级 FP32 相对误差 ≤ 1e-5（实际 0）
  - ✅ 支持不同 VLEN（vsetvl 自适应，128/256/512 实测过）
  - ✅ 用 RVV intrinsics（不是手写汇编，便于 GCC 优化）
- 跑法：`docker run ... bash scripts/run_rvv_tests.sh`
- 不在 spec 内但还要做：
  - 模型级 Top-1 精度对比（≤ 0.1%）— 需要 ImageNet eval 集
  - 把这个测试机制扩展到后续 RVV kernel
- #accuracy #rvv #depthwise #milestone

## FIND-008 [coverage] depthwiseconv_float.h RVV 覆盖率 100%（14/14 specs）

- 日期：2026-05-05
- 范围：`tflite/kernels/internal/optimized/depthwiseconv_float.h`
- 已覆盖的 14 个 Neon spec → RVV 等价：

  | spec | 实现策略 |
  |---|---|
  | `<true, 0, 1>` | 直接调 `RvvDepthwiseDepthMult1Run`（vfmacc_vv，LMUL=4） |
  | `<true, 0, 2/8/16>` | 调 `RvvDepthwiseDynamicDepthMultRun`（vfmacc_vf 标量广播，LMUL=2） |
  | `<false, 8/2, 1>` | forward to mult-1 helper |
  | `<true, 8/2/4, 1>` | forward to mult-1 helper |
  | `<true, 1, 8/20/32>` | forward to broadcast helper |
  | `<true, 3, 2/4>` | forward to broadcast helper |

- 设计决策：14 个 spec **共用 2 个 helper**，因为：
  - 对 mult==1 的 case，输入是否 fixed depth 在 Neon 那里是为了静态 unroll；RVV 用 `vsetvl + LMUL=4` 让 GCC 自动选最优 vl，效果等价
  - 对 mult>1 的 case，逻辑都是"per-input-channel scalar broadcast → vfmacc_vf"，filter 布局相同
- 总代码量：~150 LOC（vs Neon 的 ~700 LOC for 同样 14 个 spec），可维护性大幅好转
- 测试：`test/rvv/depthwise_float_accuracy_test.cc` 22 cases，三档 VLEN 全 bit-exact
- benchmark 数字：MobileNetV1 上无变化（它只走 `<true, 0, 1>`），但赛题硬性 ≥90% 覆盖率这一票稳了
- 接下来还要做：
  - 把同套思路推广到 INT8 depthwise（`integer_ops/depthwise_conv*.h`、`depthwiseconv_uint8*.h`）
  - 真正改总推理延迟需要 attack ruy GEMM（Task 8）
- #coverage #depthwise #rvv #milestone

## FIND-009 [baseline] 5 个 spec 模型的算子分布跨模型一致：CONV_2D 90%+

- 日期：2026-05-05
- 测试条件：QEMU vlen=256，3 runs，build 已含 RVV depthwise（FP32 only）

  | 模型 | 总 ms | CONV_2D % | DEPTHWISE % | Other |
  |---|---|---|---|---|
  | mobilenet_v1 FP32 | 15 214 | 97.69% | 2.31% (350 ms) | <0.01% |
  | mobilenet_v1 INT8 | 14 969 | 94.20% | 5.79% (867 ms) | <0.01% |
  | mobilenet_v2 FP32 | 7 608  | 94.44% | 5.48% (417 ms) | 0.08% |
  | mobilenet_v2 INT8 | 11 241 | 90.73% | 9.10% (1 023 ms) | 0.17% (ADD 0.16%) |
  | efficientdet-lite0 INT8 | 33 680 | 89.64% | 9.66% (3 252 ms) | 0.71% (ADD 0.29%, Detection PostProcess 0.19%, DEQUANT 0.14%) |

- 结论（强 + 一致）：
  - **CONV_2D 在所有 5 个模型上都是 90%+** — 不优化 ruy GEMM 就摸不到 110ms 目标
  - **DEPTHWISE 在 INT8 模型上 5-10%**（比 FP32 稍高），所以 RVV INT8 depthwise（`integer_ops/depthwise_conv.h`）值得做
  - **Other 全部加起来 < 1%**，elementwise add/mul、softmax、pool 都不值得碰
  - INT8 模型在 QEMU 上反而比 FP32 慢（v1: 14.97 vs 15.21 接近；v2: 11.24 vs 7.61 INT8 慢 47%）。原因：INT8 在没 RVV 路径时走 ruy 的 generic INT8 GEMM，比 FP32 generic 更慢。这也说明对 INT8 改 RVV 的回报会比 FP32 还大
- 注意：**FP32 mobilenet_v1 上 RVV depthwise 已生效**（350 ms vs scalar 757 ms）— FIND-005 ↔ FIND-009 对比可见
- 后续优先级（按价值排序）：
  1. ruy GEMM RVV（cover CONV_2D = 89-98% 全 5 个模型）— Task 8
  2. RVV INT8 depthwise（`integer_ops/depthwise_conv.h`，覆盖 5-10%）
  3. EfficientDet 的 ADD/Detection PostProcess — 即使全部归零也只省 0.5%
- #baseline #profile #int8 #cross-model

## FIND-010 [coverage] integer_ops/depthwise_conv.h RVV 100%（22/22）+ EfficientDet 2.23×

- 日期：2026-05-05
- 范围：`tflite/kernels/internal/optimized/integer_ops/depthwise_conv.h` 的 22 个 Neon spec
- 实现：
  - 同 FP32 的两 helper 模式：`RvvDepthwiseInt8DepthMult1Run` + `RvvDepthwiseInt8DynamicDepthMultRun`
  - 关键 RVV 指令：`vsext_vf2` (i8→i16 sign extend) + `vadd_vx_i16m2` (broadcast input_offset) + `vwmacc_vv_i32m4` (i16×i16→i32 widening MAC，single instruction)
  - LMUL 编排：i8m1 / i16m2 / i32m4 同 EMUL，单次 vsetvl 给所有 load 用同一个 vl
  - 22 个 specialization 共用 ~80 LOC helper + 22 个宏定义的 forwarder（vs Neon ~1700 LOC）
- 结果：

  | 模型 | DEPTHWISE 改前 ms | 改后 ms | Δ |
  |---|---|---|---|
  | mobilenet_v1 INT8 | 867 | 858 | ~0% **(走的不是这条路！)** |
  | mobilenet_v2 INT8 | 1 023 | 1 024 | ~0% **(走的不是这条路！)** |
  | efficientdet_lite0 | 3 252 | **1 459** | **-55%（2.23×）** ✓ |

- 重要发现：MobileNet 的 INT8 .tflite（2018 版）用的是**老 uint8 quantization**，dispatch 到 `optimized_ops::DepthwiseConv<uint8, int32>`（在 `depthwiseconv_uint8.h`），**不**走我们改的这个 `integer_ops/depthwise_conv.h`（per-channel int8）。EfficientDet 是较新的 per-channel int8 quant，所以才吃到了 2.23×
- 工程含义：要 cover MobileNet INT8 的 depthwise，得**单独再做一遍** `depthwiseconv_uint8.h`（2127 LOC，uint8 数学和 i8 略有差别）。这是 Task 13 的范围
- 测试：`test/rvv/depthwise_int8_accuracy_test.cc`，16 cases × 3 VLEN = 48 runs，**全部 max_abs_diff=0（整数完全一致）**
- 满足 spec：INT8 算子级误差 ≤ 1 LSB → 实际 0 LSB ✓
- #coverage #int8 #efficientdet #depthwise

## FIND-011 [coverage] depthwiseconv_uint8.h RVV 100%（22/22）+ MobileNet INT8 2.71×

- 日期：2026-05-05
- 范围：`tflite/kernels/internal/optimized/depthwiseconv_uint8.h`（2127 LOC，第二大 Neon 文件）的 22 个 spec
- 实现：第三对 helper（`RvvDepthwiseUint8DepthMult1Run` + `RvvDepthwiseUint8DynamicDepthMultRun`）
  - 关键差异 vs INT8 版：
    - `vzext_vf2` (zero-extend，不是 sign-extend) 处理 uint8
    - `vreinterpret_v_u16m2_i16m2` 把 uint16 view 转成 int16 view 后加 signed offset
    - 多一个 `filter_offset` 参数（uint8 路径 filter 也有 zero_point；int8 路径 filter 是对称的不需要）
- 结果：

  | 模型 | DEPTHWISE 改前 ms | 改后 ms | Δ |
  |---|---|---|---|
  | mobilenet_v1 INT8 (per-tensor uint8) | 858 | **317** | **-63%（2.71×）** |
  | mobilenet_v2 INT8 (per-tensor uint8) | 1024 | **378** | **-63%（2.71×）** |
  | efficientdet_lite0 (per-channel int8) | 1459 | 1459 | 0%（不走这条路） |

- 测试：`test/rvv/depthwise_uint8_accuracy_test.cc`，16 cases × 3 VLEN = 48 runs，全部 max_abs_diff=0
- 累计 RVV depthwise 完成度：
  - **3 个 LiteRT depthwise 文件 100% 覆盖**（FP32 + INT8 + 老 uint8）
  - 加速生效在所有 5 个 spec 模型上（FP32 + INT8 全套）
- #coverage #uint8 #mobilenet #milestone

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
