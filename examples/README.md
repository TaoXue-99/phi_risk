# 示例导航

先在仓库根目录执行 `uv sync --group dev`。以下数据均为合成示例。

| 文件 | 适合的场景 | 运行方式 |
|---|---|---|
| [analysis_examples.py](analysis_examples.py) | 分层 AUC/KS、参考分箱交叉、布局和总计 | `uv run python examples/analysis_examples.py` |
| [psi_examples.py](psi_examples.py) | 数值/类别/多字段 PSI、共同分层、固定基准逐日选样 | `uv run python examples/psi_examples.py` |
| [funnel_examples.py](funnel_examples.py) | 动态漏斗、0/1 明细、汇总数量、条件计数、自身和参考分箱 | `uv run python examples/funnel_examples.py` |
| [demo_cube_cross.ipynb](demo_cube_cross.ipynb) | 交互查看日期/样本集/客群表现、OOT 自身交叉与区间总计 | 在 Notebook 编辑器选择项目 `.venv` 内核，重启内核后运行全部单元格 |

analysis、funnel、psi 三个 Python 脚本内含结果断言，并在 CI 中运行。Notebook 的详细逐日计算耗时更长，当前未纳入 CI。
保存 notebook 时使用 `.ipynb` 后缀，例如 `demo_cube_cross.ipynb`。

## 如何选择 fit 和 compute

| 目的 | 写法 |
|---|---|
| 普通字段分层，无需学习边界 | `cube.compute(df)` |
| 在 OOT 自身学习边界并分析 OOT | `cube.fit_compute(oot)` |
| 拆开观察自身边界和计算过程 | `cube.fit(oot)`，查看 `cube.dimensions_`，再 `cube.compute(oot)` |
| train 学习固定边界，用于 OOT | `cube.fit(train)`，再 `cube.compute(oot)` |

fit 学习的是分箱边界，不是训练预测模型。compute 不会自动拟合或修改边界。
相同分数可能使箱数减少、箱人数不完全相等；两个分数的交叉格也不保证等人数。

需要总计时，在 `compute` 或 `fit_compute` 中传 `totals=True`，再用
`result.layout(totals=True, total_label="总计", bin_labels="interval")` 展示。
漏斗总计先合并阶段数量再相除；AUC/KS 总计在合并样本上重新计算。

输入列和完整参数见 [项目 README](../README.md) 与 [漏斗文档](../docs/funnel.md)。

## 数据生命周期

- `uv run python examples/data_quality_basic.py`：reference 与质量报告。
- `uv run python examples/data_prep_basic.py`：解析、填补、RobustScaler 与审计。
- `uv run --extra binning python examples/optbinning_prep.py`：监督分箱与原生分箱表。
- `uv run --extra binning python examples/data_workflow_risk.py`：数据异常处理、save/load 与 OOT。

PSI 的两个样本使用 `compute_comparison(reference, current)`；不要用 `fit_compute(current)`。
详情见 [比较分析](../docs/comparative_analysis.md)。


## 0.3.0 PSI 案例阅读顺序

`psi_examples.py` 分为六步：合成两侧数据 → 整体 PSI → 分层数值/类别批量 PSI →
固定基准逐日比较 → 日期 key 不匹配示例 → 100 字段/3维批量调用。
所有步骤包含断言。数值正向位移、缺失增加和新类别由脚本显式构造，可直接改成业务 DataFrame。

- reference/current 由调用方圈选，所有 dimensions 在两侧按同 key 匹配。
- `result.data_` 查看长表，`layout()` 调整展示，`metadata_` 查看参考边界和缺失策略。
- 已经有占比数组时使用 `metrics.psi_from_proportions`，不再提供旧计数 PSI 函数。
- DataQuality 不提供 PSI；分布比较在 analysis 中独立执行，不嵌入质量检查工作流。
- 数值常量 reference 只有一个箱；比较总计不支持；这些边界不会被示例隐式绕过。
