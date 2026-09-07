# 变更记录

本文件记录用户可见的 API 与行为变化。未发布内容不代表已经上传 PyPI 或创建 GitHub Release。
当前 `pyproject.toml` 包版本为 `0.1.0`。

## Unreleased

### 新增

- 动态漏斗 `Funnel / Stage / Transition`：任意阶段数，相邻、从首阶段或显式指定转化。
- `Sum / CountWhere / Ratio / Col`：数值求和、向量条件计数及命名指标相除。
- 同时支持 0/1 明细与已汇总数量，复用 Cube 分层、分箱、参考边界和总计。
- Ratio 依赖排序，计划阶段检查缺失引用与循环，指标展示保持声明顺序。
- 漏斗文档、可执行 demo 和回归测试；CI 增加漏斗 demo 运行。

### 修复与文档

- 修复交叉分析 notebook 中 `oot` 使用早于定义的问题。
- 明确自身分箱使用 `fit_compute(oot)`，固定参考边界使用 `fit(train)` 后 `compute(oot)`。
- 补全漏斗缺失数量、零分母、条件计数、权重和行列总计口径。

### 使用与兼容性

- 原有 AUC / KS / Count / Share / EventRate 接口和数值路径保持不变。
- 新增 Sum 默认 `missing="propagate"`：组内任意缺失使数量未知；按零统计需显式 `missing="zero"`。
- Ratio 引用同一 Cube 的指标名，分母为零遵循 `on_invalid`，不平均逐行或逐组比例。
- 漏斗假定阶段单位一致且嵌套，不隐式按用户去重，不自动裁剪转化率。

## 0.1.0 基础能力

此节描述已有源码能力，不声明发行日期或 PyPI 发布状态。

- Cube / Dimension / Measure / Transformer / Engine / Result / Layout 分层。
- 分层 AUC、KS 与双分数交叉分析；显式 fit、compute、fit_compute 生命周期。
- 等频分箱、参考边界复用、区间标签和灵活二维布局。
- `compute(..., totals=True)` 预计算整体与边际指标；layout 展示总计。
- AUC 使用 sklearn `roc_auc_score`，KS 使用 `roc_curve`。

旧调用迁移提示：`cube.explain()` 返回 DataFrame；需要文本时使用 `format="text"`。
仅在 `layout(totals=True)` 请求总计还不够，计算时也需启用 `totals=True`。
