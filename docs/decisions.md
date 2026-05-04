# Decisions

> 决策日志。每条独立编号 `DEC-NNN`，只增不减，删除/废弃的保留编号空位。

## DEC-001 [仓库策略] fork rv2036/rvspoc-S2601-litert 直接工作

- 日期：2026-05-04
- 背景：S2601 提交规则要求向 `https://github.com/rv2036/rvspoc-S2601-litert` 提 PR。该仓库本身就是 LiteRT 的完整镜像（566 MB）。
- 选项：
  - A：fork 提交仓库，直接在 fork 上加 RVV patch 并 PR 回去
  - B：fork 提交仓库 + 把 LiteRT upstream 作为 git submodule/subtree 嵌入，patch 单独维护
- 决定：选 A
- 理由：提交仓库就是 LiteRT 完整源码，B 的 submodule 方案纯属无谓复杂度。已 fork 到 `any-leap/rvspoc-S2601-litert` 并 clone 到 `~/developer/rvspoc/rvspoc-S2601-litert/`，添加远端 `upstream` (rv2036) 和 `litert-google` (google-ai-edge) 用于追踪上游变化。
- #repo #setup
