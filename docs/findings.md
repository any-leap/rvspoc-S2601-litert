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
