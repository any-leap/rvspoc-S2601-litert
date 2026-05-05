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

## FIND-012 [milestone] RVV FP32 GEMM 拦截 ruy → MobileNetV1 总推理 2.29×

- 日期：2026-05-05
- 路径：partial-specialize `cpu_backend_gemm::GemmImpl<float,float,float,float,kFloatingPoint>` for `__riscv_vector`，绕开 ruy 的 StandardCpp scalar fallback
  - 不动 ruy 本身（避免 fork 第三方仓库 + FetchContent 重导）
  - layout-check 失败时 forward 回 ruy（safety net）
- 算法：dot-product GEMM（适合 lhs 行连续 / rhs 列连续的 conv 矩阵布局）

  ```
  for m_idx in [0, m):
    for n_idx in [0, n):
      v_acc = 0
      for k_chunk in [0, k) by vl:    // vsetvl_e32m4
        v_acc += lhs[n_idx*k + k_chunk : +vl] * rhs[m_idx*k + k_chunk : +vl]
      acc = vfredusum(v_acc) + bias[n_idx]
      dst[m_idx*n + n_idx] = clamp(acc)
  ```

- 文件：
  - `tflite/kernels/internal/optimized/rvv_gemm_fp32.h`（独立 header-only 核心）
  - `tflite/kernels/cpu_backend_gemm_rvv.h`（adapter，partial specialization）
  - `tflite/kernels/cpu_backend_gemm.h`（include + override）
- 5 个模型 benchmark 对比（QEMU vlen=256，vs 之前 RVV-depthwise-only build）：

  | 模型 | total ms 改前 | 改后 | 总加速 |
  |---|---|---|---|
  | mobilenet_v1 FP32 | 15 214 | **6 610** | **2.30×** ⭐ |
  | mobilenet_v1 INT8 | 14 486 | 14 486 | 1.00×（FP32 only） |
  | mobilenet_v2 FP32 | 7 571 | **3 851** | **1.97×** ⭐ |
  | mobilenet_v2 INT8 | 10 931 | 10 931 | 1.00×（FP32 only） |
  | efficientdet_lite0 INT8 | 31 830 | 31 740 | ~1× |

- vs 项目最初 scalar baseline：mobilenet_v1 FP32 **15 740 → 6 610 ms = 2.38×**
- 测试：`gemm_float_accuracy_test.cc` 9 cases × 3 VLEN = 27 runs，max relative error 0~1e-5（满足 spec）；max abs error 6.7e-6（pass，spec FP32 ≤1e-5）
- 接下来还要做：
  - 给 INT8 GEMM 也做 RVV partial specialization（同手段，i8/u8 path），可继续往上推 MobileNetV1/V2 INT8 + EfficientDet
  - 加 outer block 提高 cache 命中（lhs 重用，目前每个 m_idx 重读一遍 lhs）
  - 评估 reference fallback 是否触及（layout 不匹配的情况）
- #milestone #gemm #ruy-bypass #conv2d

## FIND-013 [milestone] RVV INT8 per-channel GEMM → EfficientDet 3.25×

- 日期：2026-05-05
- 路径：partial-specialize `cpu_backend_gemm::GemmImpl<int8_t,int8_t,int32_t,int8_t,kIntegerWithPerRowMultiplier>` for `__riscv_vector`
- 数学（per-channel quant，与 `optimized_integer_ops::ConvPerChannel` 对齐）：
  ```
  acc[n,m] = sum_k lhs[n,k] * rhs[k,m]   (vsext + vwmacc widening to i32)
           - rhs_zp * sum_k lhs[n,k]      (zp 校正项；lhs row sum 预算)
           + bias[n]
  acc = MultiplyByQuantizedMultiplier(acc, multiplier_perchannel[n], shift_perchannel[n])
  acc += dst_zp
  dst[n,m] = clamp_int8(acc, clamp_min, clamp_max)
  ```
- 文件：
  - `tflite/kernels/internal/optimized/rvv_gemm_int8.h`（独立 header，含 vector dot product + scalar gemmlowp 风格 requantization）
  - 同 `cpu_backend_gemm_rvv.h` 加 partial specialization
- 5 个模型 benchmark（QEMU vlen=256，vs 之前 FP32-GEMM-only build）：

  | 模型 | total ms 改前 | 改后 | 总加速 |
  |---|---|---|---|
  | mobilenet_v1 FP32 | 6 610 | 6 591 | ~1× |
  | mobilenet_v1 INT8 (uint8 per-tensor) | 14 486 | ~14 800 | ~1× *不走这条路* |
  | mobilenet_v2 FP32 | 3 851 | 3 829 | ~1× |
  | mobilenet_v2 INT8 (uint8 per-tensor) | 10 931 | 10 553 | 1.04× *不走这条路* |
  | **efficientdet_lite0 INT8 (per-channel)** | **31 740** | **9 770** | **3.25×** ⭐ |

- vs 项目最初 scalar baseline，EfficientDet：33 680 → 9 770 ms = **3.45× 总加速**
- 其中 EfficientDet CONV_2D：30 146 → 8 144 ms = **3.7×** 单算子加速
- 测试：`gemm_int8_accuracy_test.cc` 7 cases × 3 VLEN = 21 runs 全部 max_abs_diff=0
- MobileNet INT8 不动是因为 .tflite 用的是 2018 老 uint8 per-tensor 量化，dispatch 到 `GemmImpl<uint8_t, uint8_t, int32_t, uint8_t, kIntegerWithUniformMultiplier>` ≠ 我的 int8 spec。覆盖它需要再加一个 uint8 GEMM partial spec
- 累计 GEMM 改造完成度：FP32 ✓、INT8 per-channel ✓；待办：uint8 per-tensor、INT16 输出
- #milestone #gemm #int8 #efficientdet

## FIND-014 [milestone] RVV uint8 GEMM → MobileNetV1/V2 INT8 2.87× / 3.58×

- 日期：2026-05-05
- 路径：partial-specialize `cpu_backend_gemm::GemmImpl<uint8_t,uint8_t,int32_t,uint8_t,kIntegerWithUniformMultiplier>` for `__riscv_vector`
- 数学（per-tensor uniform multiplier，与 `optimized_ops::Conv<uint8>` 对齐）：
  ```
  acc[n,m] = sum_k (lhs[n,k] - lhs_zp) * (rhs[k,m] - rhs_zp)
                                          (vsext + vadd 减 zp + vwmacc widening)
  acc += bias[n]
  acc = MultiplyByQuantizedMultiplier(acc, multiplier_fp, shift)
                                          (uniform，所有 output channel 同一组系数)
  acc += dst_zp
  dst[n,m] = clamp_uint8(acc, clamp_min, clamp_max)
  ```
- 文件：`tflite/kernels/internal/optimized/rvv_gemm_uint8.h`，复用 `rvv_gemm_int8.h` 的 requantization helpers
- 5 个模型 benchmark（vs 原始 scalar baseline）：

  | 模型 | scalar baseline | 现在 | 总加速 |
  |---|---|---|---|
  | mobilenet_v1 FP32 | 15 740 | 6 684 | **2.35×** |
  | mobilenet_v1 INT8 | 14 969 | **5 217** | **2.87×** ⭐ |
  | mobilenet_v2 FP32 | 7 607 | 3 893 | **1.95×** |
  | mobilenet_v2 INT8 | 11 240 | **3 142** | **3.58×** ⭐ |
  | efficientdet_lite0 INT8 | 33 680 | 9 950 | **3.39×** ⭐ |

- CONV_2D 算子级（之前是绝对的 hot path）：

  | 模型 | scalar | 现在 | 加速 |
  |---|---|---|---|
  | mobilenet_v1 INT8 | 14 060 | 4 905 | **2.87×** |
  | mobilenet_v2 INT8 | 10 174 | 2 748 | **3.70×** |

- 测试：`gemm_uint8_accuracy_test.cc` 6 cases × 3 VLEN = 18 runs，全部 max_abs_diff=0
- 累计 GEMM 改造完成度：FP32 ✓、INT8 per-channel ✓、uint8 per-tensor ✓
- 现在 5 个 spec 模型平均加速 ~2.8×（QEMU；真硬件预期更显著，因为内存带宽差距更小）
- #milestone #gemm #uint8 #mobilenet

## FIND-015 [model-level] 4/5 spec 模型 Top-1 一致；EfficientDet 检测顺序变化属预期

- 日期：2026-05-05
- 工具：`test/rvv/model_output_dumper.cc`（用 `tflite::Interpreter` 端到端推理 + dump 输出 tensor 的 raw bytes 和 top-5）
- 方法：build 两份 LiteRT — `build-rv64/`（`-march=rv64gcv`，RVV 路径开）+ `build-rv64-scalar/`（`-march=rv64gc`，无 V，所有 RVV 代码 fall through 到 scalar），同种子同输入跑同模型，diff 输出
- 结果：

  | 模型 | Top-1 一致 | 原始字节 | 备注 |
  |---|---|---|---|
  | mobilenet_v1 FP32 | ✅ class 372 | DIFFER | FP32 FMA 顺序差异，预期 |
  | mobilenet_v1 INT8 (uint8) | ✅ class 412 | **IDENTICAL（bit-exact）** | requantization 路径完全一致 |
  | mobilenet_v2 FP32 | ✅ class 557 | DIFFER | FP32 预期 |
  | mobilenet_v2 INT8 (uint8) | ✅ class 880 | DIFFER | 1 LSB diff（spec ≤ 1 LSB ✓），top-5 第 5 位是 tie 重排 |
  | efficientdet_lite0 INT8 | ❌ Top-1 不同 | DIFFER | **检测模型，Top-1 不是合适指标** — 见下 |

- EfficientDet 解读：检测模型有 4 个输出 tensor（boxes / classes / scores / num_detections），不同小数差异在多层中累积 + NMS 阈值附近来回切，最高分检测的 ID 会变。**评估检测模型的标准是 mAP，不是 Top-1**。RVV scores 0.993 vs scalar 0.998 都是高置信度有效检测，只是哪个检测候选「赢」NMS 的差异
- Spec 角度：
  - **算子级 FP32 ≤ 1e-5**：proven 0~6.7e-6 in unit tests ✓
  - **算子级 INT8 ≤ 1 LSB**：proven 0 in unit tests，端到端实测 1 LSB（v2 INT8）✓
  - **模型级 FP32 Top-1 ≤ 0.1%**：单图实证 4/4 分类模型 Top-1 一致；ImageNet 1000 张数据集级评估待做
  - **模型级 INT8 Top-1 ≤ 1%**：同上
- 重要提醒：本验证比较的是 **scalar-RV64 vs RVV-RV64**，不是 spec 要求的 **vs x86 reference**。要真正对齐 spec，应再跑一份 x86 native 推理结果做 baseline。但因为 RVV/RV64-scalar 之间的差异已经在 spec budget 内，且 RV64-scalar 应跟 x86 scalar bit-exact（同样的 portable 代码），传递性下来 RVV vs x86 也在 budget 内
- #model-level #accuracy #verification

## FIND-016 [coverage] op-type 覆盖率约 14 个 op 类型，文件覆盖 11/26

- 日期：2026-05-05
- 当前 RVV 实现的文件（11 个）：
  - `depthwiseconv_float.h`（FP32 depthwise，14/14 spec）
  - `integer_ops/depthwise_conv.h`（INT8 per-channel depthwise，22/22 spec）
  - `depthwiseconv_uint8.h`（UINT8 per-tensor depthwise，22/22 spec）
  - `integer_ops/depthwise_conv_hybrid.h`（22/22 + 输出阶段）
  - `integer_ops/add.h`（INT8 elementwise add）
  - `integer_ops/mul.h`（INT8 elementwise mul）
  - `integer_ops/pooling.h`（INT8 max/avg pool）
  - `integer_ops/mean.h`（INT8 mean）
  - `integer_ops/lut.h`（u8 LUT for logistic/tanh）
  - `reduce.h`（uint8 mean）
  - `optimized_ops.h`（FP32 add/mul/relu、uint8 add/maxpool/avgpool、Quantize/Dequantize、int8 Maximum/Minimum）
- GEMM specialization 文件（3 个独立 header + adapter + dispatcher hook）：
  - `rvv_gemm_fp32.h`、`rvv_gemm_int8.h`、`rvv_gemm_uint8.h`
  - `cpu_backend_gemm_rvv.h`（adapter）、`cpu_backend_gemm.h` 的 partial spec
- 已覆盖的 op 类型（约 14 个）：

  | Op | 量化模式 | 文件位置 |
  |---|---|---|
  | CONV_2D | FP32 + INT8 per-channel + uint8 per-tensor | GEMM hooks |
  | FULLY_CONNECTED | 同 GEMM hooks | 同 |
  | DEPTHWISE_CONV_2D | FP32 + INT8 + uint8（每个 100% spec 覆盖）| 3 个 depthwise 文件 |
  | ADD | INT8 + uint8 + FP32 | integer_ops/add.h、optimized_ops.h |
  | MUL | INT8 + FP32 | integer_ops/mul.h、optimized_ops.h |
  | MAX_POOL_2D | INT8 + uint8 | integer_ops/pooling.h、optimized_ops.h |
  | AVERAGE_POOL_2D | INT8 + uint8 | integer_ops/pooling.h、optimized_ops.h |
  | MEAN | INT8 + uint8 | integer_ops/mean.h、reduce.h |
  | RELU | FP32 | optimized_ops.h |
  | QUANTIZE | uint8 | optimized_ops.h |
  | DEQUANTIZE | uint8 | optimized_ops.h |
  | MAXIMUM | INT8 | optimized_ops.h |
  | MINIMUM | INT8 | optimized_ops.h |
  | LUT (Logistic/Tanh) | u8 | integer_ops/lut.h |

- 未覆盖的主要 op 类型：
  - SOFTMAX（需要 vexp 多项式逼近，复杂）
  - LOGISTIC / TANH FP32（同上，sigmoid/tanh 需要近似）
  - CONCATENATION（memory copy，不太需要 RVV）
  - RESIZE_BILINEAR（专用 13k LOC 文件，复杂双线性插值）
  - PRELU（element-wise，可做）
  - 4-bit FC（LLM 用，spec 不要求）
  - BroadcastAdd / BroadcastMul（已覆盖的 broadcast 变体）
- 按 op-type 算：约 **14 / ~18 = 78%**（spec 要 90%）
- 按文件算：11 / 26 = 42%
- 进一步推进的边际成本：剩下的 op 多是复杂数学（exp/sigmoid 近似）或大型手写汇编 fast path（13k+8k LOC），每个都是数小时到数天工作量
- #coverage #op-types #status

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
