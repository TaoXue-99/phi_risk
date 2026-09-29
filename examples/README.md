# 示例导航

先在仓库根目录执行 `uv sync --group dev`。以下数据均为合成示例。

| 文件 | 适合的场景 | 运行方式 |
|---|---|---|
| [ESMM 500 万～1000 万行指南](modeling/esmm_mmoe_large_scale_guide.ipynb) | 磁盘映射、分块生成、两条训练路径、吞吐量与 TensorBoard | `uv sync --extra torchkeras --extra tensorboard --group dev` 后 Restart & Run All |
| [ESMM 完整使用指南](modeling/esmm_mmoe_complete_guide.ipynb) | 从模型到 Experiment：配置、torchkeras/原生 PyTorch、改参、比较和恢复 | `uv sync --extra torchkeras --extra tensorboard --extra hydra --group dev` 后 Restart & Run All |
| [ESMM 文字指南](modeling/esmm_mmoe_complete_guide.md) | API、数据合同、指标口径、独立 PyTorch 使用及限制 | 配套 [小型脚本](modeling/esmm_mmoe_experiment.py) 与 [大数据脚本](modeling/esmm_mmoe_large_scale.py) |
| [LightGBM 10 万行完整指南](modeling/lightgbm_experiment_complete_guide.ipynb) | 60 特征：分类/回归、自定义 loss、callbacks、YAML/Hydra、RFE、比较与恢复 | 项目内核 Restart & Run All；已保存输出 |
| [LightGBM 小型脚本](modeling/lightgbm_experiment.py) | baseline、RFE、改参和恢复 | `uv run --extra lightgbm --extra hydra python examples/modeling/lightgbm_experiment.py --hydra` |
| [modeling_plan_complete_guide.ipynb](modeling_plan_complete_guide.ipynb) | modeling.plan 使用手册：20 章详解 DataPlan 的角色、特征、四种切分声明、校验、导出、不可变性与扩展；ModelPlan 预留 | 选择项目 Python 内核，重启后运行全部单元格；已保存示例输出 |
| [analysis_complete_guide.ipynb](analysis_complete_guide.ipynb) | analysis 完整教程：20 章，从合成数据、条件统计、分箱交叉、漏斗到 PSI、诊断及自定义扩展 | 选择项目 Python 内核，重启后运行全部单元格 |
| [modeling_plan.py](modeling_plan.py) | 无真实数据的模型与数据声明、联合校验、JSON 导出 | `uv run python examples/modeling_plan.py` |
| [shared_bin_edges.py](shared_bin_edges.py) | score_a 学习边界、两个字段共用分数段，跨样本与自身分析 | `uv run python examples/shared_bin_edges.py` |
| [analysis_examples.py](analysis_examples.py) | 分层 AUC/KS、参考分箱交叉、布局和总计 | `uv run python examples/analysis_examples.py` |
| [psi_examples.py](psi_examples.py) | 数值/类别/多字段 PSI、共同分层、固定基准逐日选样 | `uv run python examples/psi_examples.py` |
| [funnel_examples.py](funnel_examples.py) | 动态漏斗、0/1 明细、汇总数量、条件计数、自身和参考分箱 | `uv run python examples/funnel_examples.py` |
| [demo_cube_cross.ipynb](demo_cube_cross.ipynb) | 交互查看日期/样本集/客群表现、OOT 自身交叉与区间总计 | 在 Notebook 编辑器选择项目 `.venv` 内核，重启内核后运行全部单元格 |

analysis、funnel、psi 三个 Python 脚本内含结果断言，并在 CI 中运行。Notebook 的详细逐日计算耗时更长，当前未纳入 CI。
保存 notebook 时使用 `.ipynb` 后缀，例如 `demo_cube_cross.ipynb`。

## analysis 完整 Notebook

[analysis_complete_guide.ipynb](analysis_complete_guide.ipynb) 将当前 analysis 的公开接口集中在一个可顺序运行的教程中，
使用固定随机种子的合成数据，保留计算输出和五幅图。包含数学对照断言、预期异常和公开接口覆盖清单。

- 入门：架构、数据字典、整体及 N 维分层、Count/Share/Sum/EventRate、where 与全局 filters。
- 指标与分箱：Ratio、AUC/KS、权重、fit/compute 生命周期、区间精度、共享边界和双分数交叉。
- 结果：长表、布局变换、行列总计、动态漏斗、缺失策略、空值原因诊断及当前覆盖限制。
- 比较与扩展：数值/类别/100 字段 PSI、逐日比较、执行计划、引擎和自定义 Dimension/Measure。

在仓库根目录执行 `uv sync --group dev`，再执行
`uv pip install nbformat nbclient nbconvert matplotlib` 安装教程运行及验证工具。
用 Jupyter 或 VS Code 打开文件，选择项目 `.venv` 内核并执行 **Restart & Run All**。
无需下载数据；从仓库根目录或 examples 目录运行时，教程优先加载本仓库 src 下的源码。
预期异常由教程显式捕获并解释；断言失败或未捕获异常则需要检查环境或接口变化。
此 Notebook 已做本地整本执行验证，尚未纳入 CI。

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
- [data_prep_lifecycle.py](data_prep_lifecycle.py)：无状态规则与拟合步骤混合，运行
  `uv run python examples/data_prep_lifecycle.py`，查看每步的 `audit.kind`。
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

## DataPrep 生命周期与迁移

| 入口 | 是否需要 fit | 使用方式 |
|---|---|---|
| ToNumeric、ToDatetime、ValueMapper、Clip、LogTransform、LogitTransform | 否 | `step.transform(df)` 或 `step.run(df)` |
| MissingImputer、KBinsStep、SklearnStep、OptBinningStep | 是 | `step.fit(train)` 后 `step.transform(oot)` |
| DataPrep（包括全无状态组合） | 是 | `prep.fit(train)` 固定配置及 schema，再 `prep.run(oot)` |

独立无状态步骤不再调用 fit，也不产生 mapping_ 或 feature_names_in_ 等训练属性。
自定义有状态步骤继承 FittedPrepStep；固定规则继承 StatelessPrepStep。
旧 DataPrep/包含旧 DataPrep 的 Workflow artifact 需重新 fit/save；详见
[重构及迁移说明](../docs/data_prep_refactor.md)。

- [analysis_diagnostics.py](analysis_diagnostics.py)：空值诊断框架预览，1000 行中一个缺失如何传播到漏斗。运行 `uv run python examples/analysis_diagnostics.py`。

- [conditional_measures.py](conditional_measures.py)：Count/Sum 统一 where，类别与数值组合、缺失筛选和总计；运行 `uv run python examples/conditional_measures.py`。

## LightGBM 原生 Notebook（0.9.4）

安装 `uv sync --extra lightgbm --extra hydra`，并为 Notebook 安装 nbformat、nbclient、ipykernel、matplotlib。
[完整 Guide](modeling/lightgbm_experiment_complete_guide.ipynb) 使用 10 万行、60 特征，
直接展示原生 Dataset/train/predict/cv、分类/回归、自建 loss、callbacks、继续训练、
类别/缺失、稀疏输入、sklearn RFE 与 ranking；Experiment 只负责记录和比较。
GPU/CUDA 有可选分支，本机只实际验证 CPU。重复执行会追加编号，不覆盖旧 Run。
[baseline YAML](modeling/conf/baseline.yaml) 包含 params 与 train；初始化指定 objective/metric，支持指标列表和函数 objective。
Guide 用 Hydra 外部覆盖、独立 YAML 与 start_run(config=...) 自动记录配置；callbacks 仍显式构造。
本轮 Guide 输出位于 experiments/lgb/runs；本次开发已用新结果替换旧示例实验。
Guide 先创建 baseline_cfg、depth3_cfg、depth3_lr_cfg，再选择配置原生训练；展示派生方案另存 YAML。

LightGBM 基础案例直接从 cfg 传入模型与 callback 参数；是否使用 callback 由原生训练代码决定，YAML 不再提供 enabled 开关。

Guide 的比较表默认展示 run 与 name，每次训练后打印二者对应关系；重复执行单元格会追加同名但编号不同的 Run。

## 三种分箱方式

完整 analysis Notebook 第 9.2–9.4 节新增 `EqualWidthBinner` 等距分箱与 `FixedBinner` 自定义边界，
涵盖与等频的对照、参考复用、共享分数段、越界/缺失、行列总计及 PSI。
参数与生命周期见 [分箱说明](../docs/analysis_binning.md)。
