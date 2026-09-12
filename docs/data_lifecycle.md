# 数据生命周期 0.2

从 [README 的三个简单 API](../README.md#data-quality) 开始；可运行的完整流程在
[examples/data_workflow_risk.py](../examples/data_workflow_risk.py)。

## 边界与版本

- `analysis` 计算指标和多维分析；现有 BaseTransformer、QuantileBinner、CubeResult 不改契约。
- `data_quality` 学习 reference 并返回判断，不依赖 prep/workflow。
- `data_prep` 学习转换规则并提供审计，不依赖 quality/workflow。
- `data_workflow` 只编排 stage 协议，不识别 sklearn/OptBinning 具体类型。
- `_data.py` 负责小型共享契约；`_artifact.py` 负责持久化；没有通用 DAG 或调度器。

采用 Python ≥3.12，与项目当前 3.12.13 环境一致。基础依赖采用较新的稳定系列：
NumPy 2.5、pandas 3.0、sklearn 1.9、joblib 1.6、SciPy 1.18；允许系列后续兼容版本，
下一个主版本需重新验证。`uv.lock` 固定可复现安装，库 metadata 表达支持范围。
SciPy 直接用于识别稀疏矩阵，保留稀疏输出，不默认转 dense。

OptBinning 使用官方 0.21 系列；实际包 metadata 要求 sklearn ≥1.6、OR-Tools ≥9.4,<9.12。
本次解析仍保留 sklearn 1.9，求解器由 uv 按上游约束解析为 9.11.4210。
该求解器发布物没有 CPython 3.13 wheel，官方 binning extra 当前使用 Python 3.12；
基础功能支持 Python ≥3.12，CI 分别验证 3.12/3.13，binning 只验证 3.12。
不会通过静默切换非官方 fork 或降低整个基础栈来掩盖此约束。
升级 adapter 时先验证原生输出、缺失/特殊值、列名、权重、状态和保存，再放宽支持范围。

参考：[sklearn KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)、
[OptBinning BinningProcess](https://gnpalencia.org/optbinning/binning_process.html)。

## 生命周期与输入契约

输入必须是列名为字符串的 pandas DataFrame。fit reference 要求列名唯一；SchemaCheck 的
current 可包含重复列并返回 FAIL。需要按列执行的其他检查遇到重复列或缺少必需列会抛异常。
普通检查 FAIL 不短路；执行错误不会被伪装成 PASS/SKIP。

所有组合器 fit 时 clone 声明实例，成功后一次提交 fitted state。失败的 refit 保留上一份有效状态。
构造器保持 sklearn 参数语义，学习结果保存在尾随 `_` 的属性中。使用 `set_params()` 修改配置
会清除旧 fitted state；组合器与 FittedPrepStep 需重新 fit，StatelessPrepStep 可直接使用新配置。
不要直接修改已拟合组合中的参数或 backend 属性。

Prep fit 按顺序处理训练数据。transform 输入列名及顺序必须匹配其 fit reference；dtype 可以变化，
从而允许 ToNumeric 修复字符串数据。保留行数、顺序、索引；pandas Series 标签和权重要求索引
完全一致，array 按位置对齐。重复行索引可用，不做隐式 reindex。禁止删除/重排样本的 transformer。

`feature_names_in_ / feature_names_out_` 在 fit 内确定，transform 不更新。`get_feature_names_out()`
返回整个输出 DataFrame 的列名。空 DataPrep/Workflow 是保持输入的合法组合；空 QualityReport 为 SKIP。
第三方 transformer 对零行输入的限制沿用其自身语义，不伪造转换结果。

`BaseQualityCheck.requires_fit=False` 的质量检查仍可直接 validate（本次未改质量模块）。
Prep 通过 StatelessPrepStep / FittedPrepStep 类型表达生命周期，不再使用 requires_fit flag。
StatelessPrepStep 只有配置和 transform/run，不提供 fit 或训练列名属性；fit_transform 只是 transform 别名。
FittedPrepStep 和 DataPrep 提供 fit/get_feature_names_out；拟合后 transform/run 不更新训练状态。
DataPrep 即使只包含 stateless 步骤也必须 fit，以 clone 配置并固定整个计划及各步骤输出 schema。
输入 DataFrame 不原地修改；结果 `.data` 返回防御性复制，公开的结果 mapping/sequence 递归冻结。
插件也应遵守不修改嵌套 object 单元格、不学习 current 的契约；不对任意第三方代码做沙箱隔离。

## Quality 语义

| Check | 语义 |
|---|---|
| SchemaCheck | 缺列、多列、dtype、顺序、重复列；allow_extra/check_order 可控制 |
| RowCountCheck | 最小行数、相对变化绝对值；零 reference 到正行数相对变化为 infinity |
| MissingRateCheck | 当前缺失率上限、相对 reference 的增加量（百分点） |
| MissingLikeCheck | 默认仅识别空字符串/null/none/nan/n/a；真正 null 由 MissingRateCheck 处理 |
| UniqueCheck | 复合键，duplicate_count 统计重复组中的所有行（keep=False） |
| CategorySetCheck | 分别记录新类别和消失类别；null 不属于类别；两个开关独立控制告警 |
| CardinalityCheck | 非 null 唯一值数量的相对变化绝对值 |
| ConstantCheck | 含 null 的最大占比，达到 max_dominant_rate 即 FAIL |
| NumericConvertibleCheck | 在非 null 源值中计算成功率；无可转换样本返回 SKIP |
| FiniteCheck | 数值列的 NaN/+inf/-inf；可允许 NaN |
| RangeCheck | 忽略 null，支持双闭、左闭、右闭、双开；没有非 null 样本时 SKIP |

一般阈值在**超过**时告警；最小成功率在低于时失败；常量检查在达到占比时失败。
WARN/FAIL 阈值必须有限、非负且 warn≤fail；比率阈值不能超过 1。
MissingRateCheck 的空 current 为 SKIP；空 reference 的缺失率为 NaN，此时 delta 不可用，
建议设置 max_rate 或提供非空 reference。类别消失和行数为零仍可作为有意义的异常判断。

聚合严重程度为 FAIL > WARN > PASS > SKIP。`.passed` 仅表示 PASS，WARN 不等于 PASS。
`to_frame()` 固定为 check/status/columns/observed/expected/message；细节存于 results[].details。

DataQuality 仅检查数据质量，不提供 PSI，不导入 analysis 或 metrics。
需要分布比较时，在质量检查流程之外显式使用 `Cube.compute_comparison`，
见 [比较分析](comparative_analysis.md)。两种功能的执行和结果互相独立。

DataProfile 保存全部列的轻量统计，类别 top values 默认最多 20 个，不保存完整高基数集合。
数值 min/max 包括无穷值；均值、标准差和分位数只使用有限值。count 表示非 null 数量。

## Prep adapter 与审计

- ToNumeric/ToDatetime 使用 pandas，默认 errors=raise；coerce 的新增 null/NaT 和示例出现在 audit。
- ValueMapper 支持 error/keep/value；未映射的非 null 才算 unknown。mapping 是配置，
  不创建 mapping_；DataPrep.fit 通过 clone 隔离调用者的配置。
- MissingImputer 委托 SimpleImputer，默认 median、keep_empty_features=True，以保持全空列。
  支持 callable strategy、add_indicator、自定义 missing_values。统计量不加权，audit 明示此策略。
- KBinsStep 委托 KBinsDiscretizer，默认 ordinal/quantile、random_state=0、
  quantile_method=averaged_inverted_cdf；支持 uniform/kmeans 和其他上游 quantile_method。
- SklearnStep clone 原 estimator。支持权重时传入；不支持时收到权重会明确报错。
  优先 pandas set_output；稀疏 encoder 保留稀疏并通过 pandas SparseDtype 表达。
- 默认同宽输出覆盖指定输入列；维度变化时使用 backend 名称或显式 output_columns。
  replace 删除指定输入列并插入生成列；append 保留输入，必须确保输出命名不冲突。
  无关列保持顺序，生成列从第一个被替换输入列的位置插入。

`transform` 只执行转换；`run` 额外对相关列生成 before/after 快照和 step 语义审计。
不默认扫描所有单元格计算 changed_cells，也不 profile 未涉及的全部列。
PrepAudit 增加可选 kind（stateless/fitted），保留原有字段与位置参数；audit_frame 增加 kind 列。
Clip、LogTransform、LogitTransform 的详细语义与迁移说明见 [生命周期重构](data_prep_refactor.md)。
输入/输出列名、每步名称和 workflow metadata 构成基础 lineage；未实现特征级 DAG lineage。
显式命名使用 `DataPrep([("income_parse", ToNumeric(["income"]))])`，匿名同名步骤自动加后缀。

### OptBinning

`fit(X, y, sample_weight)` 要求标签同时包含 0/1，所选列不可包含 label。
参数通过 BinningProcess 传递，显式参数优先于 binning_process_kwargs；逐变量参数遵循原生库语义。
支持类别列、special_codes、选择标准和 n_jobs。append 输出命名为 `column__metric`。
权重遵循上游限制：仅 binary target、cart prebinning；本版 wrapper 使用 binary target。
非 cart 配置传入权重会明确拒绝，避免被底层静默忽略。

`summary()` 返回原生汇总；`get_binned_variable()` 返回 native OptimalBinning 的防御性副本，
允许使用其完整 API 而不意外修改 fitted rules；`binning_table()` 在副本上 build，避免统计缓存污染。
需要直接调试 backend 时仍可访问 `binning_process_`，但应只读使用。
运行 audit 记录 backend/version、selected columns、null/special 数量、solver 状态和 bin 分布，
不为每批次生成完整分箱表。

## Workflow 与持久化

QualityStage 默认 on_fail=raise，DataWorkflowError 携带 report/stage；warn 发出 UserWarning 后继续；
continue 保留 FAIL 报告并继续。策略处理数据判断失败，不吞执行异常。字段缺失时应先用严格
schema gate 拦截；继续到依赖该列的 prep 仍会抛 DataPrepError。

Workflow 支持任意 Quality/Prep 顺序。`.quality_reports` 是 stage name 到报告的 mapping，
`.prep_audits` 是 prep stage name 到 audits tuple 的 mapping。stage_results 保留阶段结果，
因此 run 比 transform 占用更多内存；只要数据输出时使用 transform。

三个组合器及拟合后的 FittedPrepStep 支持 save/load。StatelessPrepStep 不制造拟合标记，
如需保存固定规则，请包入 DataPrep 后 fit/save。旧 DataPrep artifact 缺少逐步骤 schema，
须显式重新 fit 后保存；本次不做隐式迁移。artifact envelope version=1，
包含包版本、Python 版本、依赖版本和 payload；临时文件写完后原子替换，不破坏已有 artifact。
加载检测类型和 envelope version；环境变化会给出 trained/current 明细警告。
joblib 基于 pickle，**只加载可信来源**；envelope 验证不能使不可信 pickle 安全。
OptBinning 已拟合变量出现 solver=mip 时提前拒绝保存；请使用 cp 重新 fit。
本版是 Python-native artifact，不承诺跨版本 portable export。

## 自定义扩展

```python
from phl_risk.data_quality import BaseQualityCheck, CheckResult, CheckStatus
from phl_risk.data_prep import StatelessPrepStep

class PositiveCheck(BaseQualityCheck):
    requires_fit = False
    def _validate(self, X):
        status = CheckStatus.PASS if X["x"].gt(0).all() else CheckStatus.FAIL
        return CheckResult("positive", status, ("x",), "x must be positive")

class AddOne(StatelessPrepStep):
    def _transform(self, X):
        return X.assign(x=X.x + 1)
```

有状态 Prep 插件继承 FittedPrepStep 并实现 `_fit`；无状态插件继承 StatelessPrepStep。
状态命名尾随 `_`，构造器只保存显式参数以兼容 sklearn clone。
自定义 prep 可覆盖 audit_columns 和 _audit_details。未知 backend 异常保留原始上下文；
不要捕获所有异常后自动重新 fit。真实底层算法只在 adapter 内实现/调用。

## 验证与后续范围

测试包含 11 类检查的独立文件、生命周期/clone/原子 refit/状态指纹、backend equivalence、
完整数据异常场景、无 extra import、atomic save、元数据差异及自定义 check/prep。
CI 配置 base 的 Python 3.12/3.13，以及 binning 的 Python 3.12，并执行示例和 wheel 构建。
本地验证不等同于远程 CI 已通过；当前 Prep 重构验证见 [data_prep_refactor.md](data_prep_refactor.md)；
[data_lifecycle_review.md](data_lifecycle_review.md) 保留 0.2 历史验收。

未实现 portable export、Polars/Spark、DAG/并行 workflow、监控服务或自动阈值。
阶段/adapter/metric 协议允许以后扩展，不为这些功能预建空框架。
