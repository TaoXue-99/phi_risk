# 变更记录

当前源码版本 0.3.0；此记录不代表已发布 PyPI 或 GitHub Release。

## Unreleased — DataPrep 显式生命周期

- BasePrepStep 收敛为转换与审计协议；新增 StatelessPrepStep / FittedPrepStep，删除 Prep requires_fit。
- 无状态转换不创建伪训练属性，ValueMapper.mapping 明确为配置；DataPrep 仍 clone 并固定计划。
- DataPrep 按类型顺序调度，新增逐步骤输出 schema 检查，保留 transactional fit、权重及 backend 能力。
- 新增 Clip、LogTransform、LogitTransform，保留 null、索引和审计，新增可选 PrepAudit.kind。
- 迁移：自定义有状态 Prep 必须继承 FittedPrepStep；stateless 直接 transform，不再调用 fit。
  旧 DataPrep artifact 需重新 fit/save；单个 stateless 的持久化通过 DataPrep 完成。
- 不改依赖、包版本、DataQuality/Analysis 或现有第三方算法。

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
