# CLAUDE.md — RVSPOC S2601 (LiteRT RISC-V + RVV)

> 本文件为本项目特定指令，覆盖于全局 `~/.claude/CLAUDE.md` 之上。

## 项目背景

RVSPOC 2026 挑战赛 S2601 赛题：将 LiteRT（原 TensorFlow Lite）移植到 RISC-V，并对所有已有 ARM Neon/SVE 优化的内核做 RVV 1.0 向量化。

- 赛题详情：见 Obsidian vault `2 Projects/RVSPOC S2601 - LiteRT RISC-V 移植与 RVV 优化.md`
- 提交仓库：https://github.com/rv2036/rvspoc-S2601-litert
- 我方 fork：https://github.com/any-leap/rvspoc-S2601-litert
- 截止：2026-08-31 (AoE)

## 仓库结构

本仓库 = LiteRT 完整源码（566 MB，~7800 文件）。我们的工作以 patch 方式叠加在上面，不动 LiteRT 现有文件结构（除非必要修改 `#ifdef` 分支或 BUILD 配置）。

新增 RVV 内核统一放在原文件相邻位置或 `tflite/kernels/internal/optimized/rvv/` 子目录（实施时再拍板）。

## 开发约束（来自赛题，硬性）

- C++17 或以上
- RVV 实现优先用 intrinsics（`<riscv_vector.h>`），可辅以内联汇编
- **必须**保留 scalar 回退（条件编译 `#ifdef __riscv_vector` 隔离）
- 代码风格：Google C++ Style Guide
- 必须支持 VLEN 128/256/512 自适应（用 `vsetvl`）
- 构建：CMake + `riscv64-unknown-linux-gnu` 交叉编译
- License：Apache 2.0（与 LiteRT 一致）
- **禁止**搬运现有第三方 RISC-V TFLite 移植，自主实现或明确标注引用
- 用 AI 辅助写代码需在最终报告中说明使用方式及占比

## 精度与性能门槛

| 项 | 标准 |
|---|---|
| FP32 模型 Top-1 vs x86 | ≤ 0.1% |
| INT8 模型 Top-1 vs x86 | ≤ 1% |
| FP32 算子相对误差 | ≤ 1e-5 |
| INT8 算子差值 | ≤ 1 LSB |
| 推理延迟 | ≤ 110 ms（avg/p50/p95） |

## 提交规范

- commit message 详细写明改动 + why + 影响面（参考全局 CLAUDE.md 第 4 条）
- 决策类信息记 `docs/decisions.md`，调试发现记 `docs/findings.md`，踩坑记 `docs/gotchas.md`
- 每个新增/修改的 RVV 内核必须有：
  - 配套精度测试（与 scalar 对比）
  - 配套 benchmark（与 scalar 对比）
  - commit message 说明算子、VLEN 假设、测试条件

## 不要做的事

- 不要改动 LiteRT 上游已有 ARM Neon 代码的逻辑（只读不改，作为参考）
- 不要混用 schema 的 push/migrate 模式（本项目无此问题，但保持警觉）
- 不要在 PR 标题/描述/公开文档暴露：内网 IP、个人信息、未脱敏的客户名（参考全局 CLAUDE.md 第 6 条）

## 任务结束收尾格式

照全局 CLAUDE.md 第 8 条，每个任务末尾给：
```
改动：...
验证：...
遗留：...
```


<!-- GLOBAL_AGENT_BASELINE_2026_06_30 -->

## Global Agent Baseline

- Follow the global rules in `/Users/t3st/.codex/AGENTS.md`; nearest project/subdirectory instructions still override for project-specific details, except global safety rules.
- New Web App / TanStack Start / full-stack React / shadcn/ui / Bun work should use the `$start-webapp` Golden Path before scaffolding. UI/UX work should use `frontend-design-ui-ux` and treat `.ulpi/design/` as the design source of truth.
- UI/Web changes require browser acceptance, not just tests/build: inspect the running app for aesthetics, usefulness, responsive viewports, loading/empty/error states, overlap/overflow, and clickable controls.
- For repeatable multi-step work, define the loop before executing: Goal, Context, Actions, Feedback signal, Stop condition, Safety rails, and Handoff. Iterate until verification passes, a blocker is explicit, or a safety boundary is reached.
- Loops should not stay in one context by default: use subagents/parallel agents for independent research, implementation, review, or failure investigation; split long-running or context-heavy loops into fresh threads with explicit handoff notes.
- Before calling work complete, report the standard verification gate as applicable: Type Check, Unit Test, Build, Invariants, and Browser Test. If a gate cannot run, state why and what substitute evidence was used.
- Treat test, lint, typecheck, hook, and CI failures as real signals. Investigate root cause before classifying a failure as pre-existing; do not bypass hooks or delete/suppress rules to land work.
- Prefer root-cause fixes over suppressions: avoid `biome-ignore`, `eslint-disable`, `ts-ignore`, broad excludes, silent catches, and rule downgrades. If a mainline tradeoff requires temporary deferral, record reason, impact, owner, removal trigger, and follow-up task.
- Do not treat placeholder implementations as complete: no fake data paths, TODO stubs, empty handlers, silent fallbacks, temporary hardcode, or unconnected UI/API without explicit owner, reason, risk, and removal trigger.
- Prefer mature, stable, well-maintained packages and framework features over custom implementations for solved problems. For Web UI, prefer shadcn/ui standard components, registry items, and blocks before hand-rolling generic primitives or large layout components.
- New projects must be git repositories: after scaffolding, check for `.git`; run `git init` if missing, and verify `.gitignore` excludes dependencies, build output, env files, local databases, and tool caches.
- Keep project instructions concise and operational. Prefer exact commands, `Always / Ask First / Never` boundaries, known pitfalls, and links to project docs over long duplicated process text.
