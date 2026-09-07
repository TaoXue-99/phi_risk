# 示例导航

先在仓库根目录执行 `uv sync --group dev`。以下数据均为合成示例。

| 文件 | 适合的场景 | 运行方式 |
|---|---|---|
| [analysis_examples.py](analysis_examples.py) | 分层 AUC/KS、参考分箱交叉、布局和总计 | `uv run python examples/analysis_examples.py` |
| [funnel_examples.py](funnel_examples.py) | 动态漏斗、0/1 明细、汇总数量、条件计数、自身和参考分箱 | `uv run python examples/funnel_examples.py` |
| [demo_cube_cross.ipynb](demo_cube_cross.ipynb) | 交互查看日期/样本集/客群表现、OOT 自身交叉与区间总计 | 在 Notebook 编辑器选择项目 `.venv` 内核，重启内核后运行全部单元格 |

两个 Python 脚本内含结果断言，并在 CI 中运行。Notebook 的详细逐日计算耗时更长，当前未纳入 CI。
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
