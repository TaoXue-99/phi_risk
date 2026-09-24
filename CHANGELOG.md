# 变更记录

当前源码版本 0.5.0；此记录不代表已发布 PyPI 或 GitHub Release。

## Unreleased — Count / Sum 条件统计

- Count 与 Sum 新增关键字 where，统一复用 Col 类别/数值条件，删除 CountWhere，统一改用 Count(where=...)。
- 筛选先于缺失处理；无匹配返回零，相同条件在一次聚合中共享掩码。
- 更新 explain、缺失诊断说明和可执行条件统计案例。

## Unreleased — Analysis diagnostics 框架预览

- 新增 CubeResult.diagnostics()，保持计算值与异常策略不变。
- 接入 Sum 缺失传播、Ratio 上游缺失/零分母及总计独立诊断。
- 未覆盖计算标记 reason_not_recorded；不诊断 layout 补齐单元格。

## 0.5.0 — LightGBM 二分类实验执行层（2026-09-19）

本版本新增 modeling 的首个实验执行层，保留既有声明 API。版本号已更新为 0.5.0；尚未发布 PyPI 或 GitHub Release。

### 新增功能

- 新增 `phl_risk.modeling.experiment.lightgbm`，以 `LightGBMExperiment` 为用户入口，
  复用既有 ModelPlan/DataPlan，运行时解析 target、weight、features 和 split。
- 正式训练使用 LightGBM 原生 Dataset/train/Booster；每次 Run 重建 Dataset，支持权重、
  类别特征、early stopping、完整 evaluation history 和 gain/split importance。
- 每次训练保存独立 LightGBMRun：UTC 时间戳 + UUID + 名称组成唯一 ID，同名尝试不会覆盖旧 Run。
  原生模型、解析后配置、overrides、特征、各分区 AUC、gap 与训练元数据一并落盘。
- ExperimentStore 提供原子文件写入、失败状态、校验和检查、实验恢复和 Run 独立加载；
  Booster 按需加载，不默认保存原始数据或预测。
- 执行 ColumnSplitter 与确定性 SHA256 HashSplitter；RandomSplitter/TimeSplitter 仍只声明，
  调用执行层时明确报错。
- `exp.compare()` 返回 pandas.DataFrame，以 reference 为基准展示参数/特征变化、AUC delta、
  gap delta 和 best_iteration；支持切换 reference、指定参数列及显式展示独立留出集。
- 可选 Hydra Compose adapter 提供配置合成、插值解析及 override 追踪，不改变 cwd 或管理输出目录。
- sklearn RFE 只使用 train 生成指定数量的特征候选，每个候选再通过原生 exp.run 训练评估；
  tolerance helper 仅提供建议，不自动指定最终模型。

### 数据分区与兼容性

- 默认 `train.validation_partition="valid"`。train 用于拟合/RFE，valid 用于早停和模型选择，
  test 留作独立评估，OOT 用于时间外验证；默认 compare 展示 train/valid，gap 为 train_auc − valid_auc。
- test/OOT 指标仍保存，但仅通过 `include_test=True` / `include_oot=True` 显式展示。
  RFE tolerance 默认跟随候选 Run 的验证分区，不使用独立 test/OOT 指标。
- 原始实现草案曾默认使用 test 作 validation；旧 Run 的完整配置保持原样，显式指定 test 仍可恢复。
  这些旧 Run 的 test 实际承担验证集职责，不能当作独立测试成绩。新配置缺少 valid 会报错，
  不会自动回退到 test；改变既有实验的分区契约时请创建新的 Experiment。
- ModelPlan、DataPlan、Goal、Strategy 公共 API 未修改；声明层继续 backend-free。
  二分类 Plan objective 在执行层映射为 LightGBM binary，评价指标限定 AUC。
- 支持 LightGBM 4.0.0：局部适配旧版本的 NumPy/pandas 与 sklearn 接口，Run 记录兼容项；4.6+ 无需适配。
- 新增可选依赖 `lightgbm`（LightGBM >=4.0,<5、PyYAML >=6,<7）和 `hydra`（hydra-core >=1.3,<2）。
  核心 dict 配置不依赖 Hydra；缺少依赖时抛 OptionalDependencyError。
- 类别 RFE 使用仅由 train 学习的 ordinal codes 排序，正式候选训练恢复原生 categorical 语义。
  RFE 暂不重映射按特征位置绑定的约束或强制分裂/分箱配置，会提前明确报错。
- 不实现 AutoML、自动最佳模型、自动 sweep、其他模型/学习目标、扩展指标或部署服务。

### 文档与验证

- 新增完整 synthetic example、baseline YAML、使用文档和验收记录；README 更新功能总图、入口和训练流程图。
- Python 3.12 全量 `pytest -q -W error`：461 passed；实验模块 69 passed。
- Ruff check 通过；format 检查 180 个 Python 文件通过；四分区示例完成 6 个 Run 并通过恢复比较。
- CI 增加 LightGBM/Hydra 的 Python 3.12/3.13 配置，以及 Python 3.12 的 4.0–4.6 兼容矩阵。
- 兼容性补充验收：LightGBM 4.0.0–4.7.0 八个版本各 71 项实验测试通过；当前环境全量 463 passed，Ruff 通过。以上为本地验证，非远程 CI 状态。
- 详见 [使用与边界](docs/lightgbm_experiment.md)、[验收记录](docs/lightgbm_experiment_review.md)
  和 [可运行示例](examples/modeling/lightgbm_experiment.py)。

## 0.4.0 — 建模声明、数据准备生命周期与分箱扩展（2026-09-17）

本版本汇总 0.3.0 后的改动，包含此前已推送的 DataPrep 重构。

### 跨字段分箱学习

- BinDimension 新增可选 fit_field：从指定参考列学习，对 column 转换，默认行为不变。
- metadata/explain 展示学习来源；补充共享边界示例及跨样本、错误输入、原子 refit 回归测试。

### 分箱区间展示

- QuantileBinner.precision 从默认 3 位有效数字调整为默认 6 位小数；metadata 与 layout 共用区间标签。
- 相邻边界舍入后重合时自动增加显示位数；原始 bin_edges_ 与分箱归属不变。
- 迁移：显式 precision 现在表示小数位数，区间标签文本可能变化。

### Modeling Plan Layer

- 新增四种 ModelingGoal、LightGBM/MLP 声明路线及能力驱动的 ModelPlan resolve。
- 新增并行 DataPlan，组合动态 RoleSpec、FeatureSpec、SplitSpec；支持模型/数据联合校验。
- 新增 Hash/Random/Column/Time 切分声明和动态分区；比例与源字段值分别校验。
- 声明采用不可变容器、JSON 友好导出；加权 objective 要求 weight 角色。
- 不增加依赖、后端导入、训练或数据处理；详见 docs/modeling_plan.md。

### DataPrep 显式生命周期

- BasePrepStep 收敛为转换与审计协议；新增 StatelessPrepStep / FittedPrepStep，删除 Prep requires_fit。
- 无状态转换不创建伪训练属性，ValueMapper.mapping 明确为配置；DataPrep 仍 clone 并固定计划。
- DataPrep 按类型顺序调度，新增逐步骤输出 schema 检查，保留 transactional fit、权重及 backend 能力。
- 新增 Clip、LogTransform、LogitTransform，保留 null、索引和审计，新增可选 PrepAudit.kind。
- 迁移：自定义有状态 Prep 必须继承 FittedPrepStep；stateless 直接 transform，不再调用 fit。
  旧 DataPrep artifact 需重新 fit/save；单个 stateless 的持久化通过 DataPrep 完成。
- 此生命周期重构不改变依赖、DataQuality 或现有第三方算法。

### 本地验证

- Python 3.12：392 项测试通过；Ruff lint/format 全量检查通过。
- 10 个 Python 示例运行通过，包含建模声明、共享分箱及可选 OptBinning。
- 锁文件离线检查、wheel/sdist 构建及 wheel 导入冒烟通过。
- 以上为本地验证；远程 CI 状态以 GitHub Actions 为准，未发布 PyPI。

## 0.3.0 — Comparative analysis 与统一 PSI

- 新增 Cube.compute_comparison、ComparativeMeasure、单/双样本模式校验。
- 新增 PSI，支持参考数值分箱、类别并集、缺失箱、小样本策略、N 维和批量字段。
- 共享一次维度编码，使用 NumPy bincount 与矩阵 PSI，复用原有 CubeResult。
- 删除旧计数 PSI 接口与 DataQuality 的 PSIState/PSIMetric/DriftMetric/DRIFT_METRICS，统一使用比较框架。
- 移除 DataQuality 的 DistributionDriftCheck，质量检查保留11类，不再依赖 analysis 或 metrics。
- 默认 epsilon 统一为 1e-8；类别使用两侧并集，常量数值参考遵循 QuantileBinner 单箱语义。
- **迁移**：旧计数函数调用改为 Cube.compute_comparison，或计数归一化后调用 psi_from_proportions。
  含旧漂移检查的 artifact 需移除该检查后重新 fit 和保存；PSI 改为独立 analysis 调用。
- 更新 README 功能架构图、入口导航、使用案例及包版本/锁文件；项目定位为通用数据分析与模型评估框架。
- 新增比较示例、独立数学对照、非 PSI 扩展测试与 10 万/100 万行性能脚本。


## 0.2.0 — Data lifecycle

- 新增独立 DataQuality、DataPrep、DataWorkflow，保留 analysis 的现有 API 与异常兼容性。
- 12 类质量检查、固定 reference PSI、不可变结果与有界数据 profile。
- 7 类 Prep：pandas 转换、通用 sklearn adapter、SimpleImputer、KBinsDiscretizer、OptBinning。
- 原子 fit、顺序编排、索引和权重对齐、运行审计、失败策略及 Python-native artifact。
- Python 最低版本由 3.10 提升到 3.12；NumPy ≥2.5、pandas ≥3.0、sklearn ≥1.9、
  joblib ≥1.6；SciPy ≥1.18 为稀疏输出处理的直接依赖。主版本上界限定已验证的兼容范围。
- OptBinning 0.21 的 metadata 要求 sklearn ≥1.6、OR-Tools ≥9.4,<9.12；
  uv 实际解析为 OR-Tools 9.11.4210，无额外手写的 solver 依赖约束。
  该 solver 没有 Python 3.13 wheel，因此 binning 当前验收范围为 Python 3.12；base CI 覆盖 3.12/3.13。
- 显式设置 quantile_method 和随机种子，保留新 sklearn API；增加 base/binning CI 分支。
- 新增完整合成数据、第三方对照、状态冻结、扩展与持久化测试和示例。

## 0.1 系列 — 漏斗扩展

### 新增

- 动态漏斗 `Funnel / Stage / Transition`：任意阶段数，相邻、从首阶段或显式指定转化。
- `Sum / 条件计数 / Ratio / Col`：数值求和、向量条件计数及命名指标相除。
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
