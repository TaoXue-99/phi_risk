# Modeling Plan Layout Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution.

**Goal:** 按用户已确认的目录设计，把声明实现分别归入 model_plan 和 data_plan。
**Architecture:** ModelPlan 的 definition 与 resolver 分离，goal/strategy/契约归入 model_plan。
DataPlan 的 definition、role、feature、split 归入 data_plan；split 拆分 spec、partition、splitter。
**Tech Stack:** Python 3.12+、标准库、pytest、Ruff。
**Spec:** 本任务对话中用户确认的 plan/model_plan、plan/data_plan 设计。

## Global Constraints

- 保持构造参数、校验行为、异常、to_dict、describe 输出不变。
- 保留 modeling 与 plan 的 Plan/Spec 导出；依用户最终要求删除顶层 Goal/Strategy 兼容目录，统一从 plan.model_plan 导入。
- 原实现文件路径随结构迁移，不为所有叶子模块增加镜像兼容文件；不承诺 pickle 兼容。
- experiment 执行行为不变；不新增依赖、训练能力或抽象。
- 保留原有 notebook、pyproject.toml、uv.lock、.idea 用户改动，不提交或覆盖。

## Review Focus

- 各保留的公开入口须返回同一类，避免 isinstance 失效。
- 不同导入顺序须无循环导入。
- 所有声明入口在无 site-packages 环境下可导入，无执行层依赖。
- 迁移前后声明 JSON 与 describe 输出一致。
- 现有 experiment 和自定义 Goal/Strategy 扩展继续工作。

## Task 1: 记录基线与新增入口验收

- [x] 读取所有声明代码以及 experiment 引用，保存代表性序列化基线。
- [x] 新增 tests/modeling/test_plan_imports.py：新入口、类身份、导入边界。
- [x] 运行新增测试，确认仅因新包不存在而失败。

## Task 2: 迁移实现与拆分职责

- [x] goal/strategy 迁入 plan/model_plan；_base.py 改为 base.py。
- [x] ModelPlan 移入 definition.py；原 _resolve 提取为 resolver.resolve_model_contract。
- [x] ObjectiveOptions、RoleRequirements 移入 objective.py、requirements.py。
- [x] DataPlan、RoleSpec、FeatureSpec 迁入 plan/data_plan。
- [x] split.py 拆为 split/split_spec.py、partition.py、splitter.py。
- [x] 实现集中导出，删除顶层 Goal/Strategy 旧包；所有引用使用唯一新路径。
- [x] 运行声明测试、比较基线，确认无行为变化。

## Task 3: 引用、文档与全量验收

- [x] 更新测试、示例、文档为新入口，兼容测试保留旧入口。
- [x] 更新目录说明和迁移边界，记录验证结果。
- [x] 执行完整 pytest -q -W error、Ruff lint/format、git diff --check。


## 验收记录（2026-09-26）

- 新增 11 项入口测试，迁移前全部因新包不存在失败，迁移后全部通过。
- 声明层与入口测试：119 passed；完整项目：485 passed（pytest -q -W error）。
- Ruff lint 与 196 个 Python 文件格式检查通过；git diff --check 通过。
- 迁移前后 18 个代表性声明对象的 to_dict / describe 输出逐字一致。
- 独立只读审查比较 HEAD 与迁移后 AST，确认声明与校验逻辑不变，未发现实质问题。
- 未修改依赖、版本和用户已有 notebook/IDE 配置改动；没有提交或发布。

## 后续清理

依用户明确要求删除两个顶层兼容目录及缓存，测试与文档全部迁移到新入口。
新增两个回归用例验证目录和可导入旧别名均不存在；删除前两项失败。

清理后验收：全项目 487 passed；Ruff lint、194 个 Python 文件格式检查、git diff --check 均通过。
已检索 src/tests/examples（含 notebook）/docs/README/CHANGELOG，无旧导入路径引用。
