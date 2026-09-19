# LightGBM 二分类实验 V1

`Plan = declaration`，`Experiment = execution`，`Run = one immutable training attempt`。
Hydra 是可选配置来源；RFE 只生成特征候选；compare 是人工实验决策视图。本模块不是 AutoML。
现有 Goal / Strategy / ModelPlan / DataPlan API 未修改，声明层仍只使用标准库。

## 安装与直接运行

```bash
pip install 'phl-risk[lightgbm,hydra]'
python examples/modeling/lightgbm_experiment.py --root ./experiments
```

可选后端支持 `lightgbm>=4.0,<5`，最低支持 **4.0.0**。
4.0 起提供本框架所需的 `early_stopping(min_delta=...)`。
对于 4.0–4.3，内部适配 NumPy 2 / pandas 3 的旧转换接口；对于 4.0–4.5，
适配 sklearn 的 `force_all_finite → ensure_all_finite` 参数改名。
适配只修改已加载 LightGBM 模块自身的内部引用，不修改 NumPy/sklearn 公共函数，
不改变原生训练算法；同一进程中其他 LightGBM 调用也会使用这些兼容引用。
4.6 及以上无需上述适配。Run metadata 的 `lightgbm_compatibility` 记录启用项。
这属于 Python API 兼容支持，不保证不同 LightGBM 版本训练结果逐位一致。
见 [early_stopping 官方文档](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.early_stopping.html)。

核心 dict 配置只需 `phl-risk[lightgbm]`（包含 YAML 文件写入依赖 PyYAML）。
macOS 的 LightGBM wheel 还需要 OpenMP：`brew install libomp`。
示例用 1,200 行、20 个合成特征，包含权重、baseline、RFE、三轮参数实验与恢复验证。
真实问题有 100 个特征时，可将 RFE 候选设为 `[80, 60, 50, 40]`。

## 最小完整使用

```python
from sklearn.datasets import make_classification
import pandas as pd
from phl_risk.modeling import ModelPlan, DataPlan
from phl_risk.modeling.goal import BinaryClassification
from phl_risk.modeling.strategy import LightGBM
from phl_risk.modeling.plan import RoleSpec, FeatureSpec, SplitSpec, ColumnSplitter, PartitionSpec
from phl_risk.modeling.experiment.lightgbm import LightGBMExperiment

X, y = make_classification(n_samples=600, n_features=10, random_state=2026)
features = [f"x{i}" for i in range(10)]
data = pd.DataFrame(X, columns=features)
data["label"] = y
data["partition"] = ["train"] * 360 + ["valid"] * 80 + ["test"] * 80 + ["oot"] * 80
model_plan = ModelPlan(BinaryClassification(), LightGBM())
data_plan = DataPlan(
    RoleSpec(target="label"), FeatureSpec(numerical=features),
    SplitSpec(ColumnSplitter("partition"), PartitionSpec(train="train", valid="valid", test="test", oot="oot")),
)
exp = LightGBMExperiment(name="risk_demo", root="./experiments", data=data,
                         model_plan=model_plan, data_plan=data_plan)
baseline_config = {
    "model": {"params": {"learning_rate": 0.03, "num_leaves": 7, "max_depth": 3,
                         "num_threads": 2, "verbosity": -1, "seed": 2026}},
    "train": {"num_boost_round": 100, "early_stopping": {"stopping_rounds": 10},
              "log_evaluation": {"enabled": False}},
}
baseline = exp.run(name="baseline", config=baseline_config)
comparison = exp.compare()  # pandas.DataFrame; numerical values stay float
```

无需再传 target、weight、categorical 或 partition column。目标必须非空且只有 0/1；每个
partition 必须非空、两个类别均有正有效权重。无效 AUC 直接报错，不悄悄保存 NaN。
DataPlan 中的全部 role/split 字段不能混入特征。特征子集不能重复，最终顺序始终遵循 FeatureSpec。
特征名不能含空白、控制字符或 LightGBM 不支持的 JSON 标点；请显式改名，框架不会悄悄改写。
Experiment 拷贝 attach 时的各 partition；之后修改用户 DataFrame 不改变正在进行的实验数据。

## 数据分区的科学含义

- train：拟合模型、学习类别词表、执行 RFE 删除决策。
- valid：早停、比较超参数与候选特征集合；默认 compare 展示 train_auc、valid_auc 和两者 gap。
- test：模型方案固定后的独立留出评估，不参与参数或特征选择。
- oot：独立时间外样本，用于检验跨时间泛化，不参与调参。

反复基于 valid 选择模型会使其指标带有选择偏差；最终效果应看未参与决策的 test/OOT。
若根据 test/OOT 再修改方案，它们也就参与了选择，不能继续被称为未触碰的最终评估集。
这些是数据用途，不是简单列名：历史 Run 显式指定 `validation_partition="test"` 仍可恢复，
但其中的 test 实际承担 valid 职责，不能解读为独立测试成绩。
新配置缺省寻找 valid，找不到会报错，不自动降级使用 test。现有 Run 配置和名称不会被改写。
旧实验改变分区契约时应新建 experiment 名称；仅恢复旧 Run 无需迁移磁盘文件。

## 推荐迭代流程

合理 baseline → RFE 粗筛 → 固定特征后手工调参 → compare → 小范围特征微调 → 最终/OOT 验证。

```python
selection = exp.select_features(base_run=baseline.run_id, candidate_counts=[8, 6, 4], step=0.1)
selected_features = selection.runs[1].features  # 人工选择 6 个特征候选
selection.compare()  # 委托同一个 compare 实现，包含 base 和这些候选
recommendation = selection.within_tolerance(metric="valid_auc", tolerance=0.001)
# recommendation 只是一个 Run；不会设置 reference 或“最终模型”。
```

每个数量独立调用 sklearn RFE，estimator 是 LGBMClassifier，轮数取 baseline.best_iteration。
只使用 train 的 X/y/weight，既不传 validation eval_set，也不使用 OOT。
RFE 的 sklearn 实现会转成 ndarray，故候选排序阶段将类别字段转换为 train 词表的 ordinal codes，
按数值参与重要性排序；这不是原生类别搜索。所有候选最终均重新构造 Dataset，并按 DataPlan 的
类别语义用 `lgb.train()` 正式训练。具有大量无序类别的场景，应结合这一筛选近似人工判断。
权重通过当前 sklearn 正式 `fit(..., sample_weight=...)` API 传递，在局部 context 中关闭
metadata routing，退出后恢复调用者设置。原始 RFE estimator 的 score 不进入 Run 指标。
RFE V1 不自动重映射 `monotone_constraints`、interaction constraints、按特征惩罚或强制分裂/分箱文件
中的位置引用；这些非空配置会在筛选前明确报错。可用无约束 baseline 生成候选，再按候选特征顺序
显式调整约束后调用普通 `exp.run()`；原生训练仍支持这些 LightGBM 能力。

```python
from phl_risk.modeling.experiment.lightgbm import compose_lightgbm_config
cfg = compose_lightgbm_config(
    config_dir="examples/modeling/conf", config_name="baseline",
    overrides=["model.params.max_depth=2", "model.params.bagging_freq=3"],
)
run = exp.run(name="depth2_bag3", config=cfg.config, overrides=cfg.overrides,
              features=selected_features)
exp.compare(reference=baseline.run_id)
exp.compare(params=["max_depth", "num_leaves", "bagging_freq", "lambda_l2"])
exp.compare(params="all")
exp.compare(include_test=True, include_oot=True)  # 模型方案固定后才检查独立留出集
```

Hydra Compose 解析 defaults、override 和 interpolation，不改变 cwd、不接管落盘目录。
同一进程内 Hydra 全局初始化需由调用者协调；V1 不实现 multirun orchestration。

## compare 语义

默认 reference 是第一个成功完成的 Run；`exp.set_reference(run_id)` 可以持久设置，
`exp.compare(reference=...)` 只作用于当次比较。唯一的 display name 也可使用；重名时必须传 run_id。
所有 delta 相对于 reference，而不是相邻一行：`delta_valid_auc = current - reference`。
`auc_gap = train_auc - validation_auc`；训练历史同时记录 train 和 validation，early stopping
仅由非训练 validation 决定。`validation_partition` 默认 valid，允许其它已声明名称，但不能为 train/oot。
若比较中的 validation partition 不同，会展示这些 partition 的 AUC；不同口径的 delta_auc_gap 为 NaN。

默认列：run、run_id、created_at、n_features、feature_change、best_iteration、train_auc、
validation AUC、auc_gap、delta validation AUC、delta_auc_gap、param_changes。
所有分区 AUC 都计算并持久化，但默认 compare 隐藏独立 test/OOT；
分别用 `include_test=True` / `include_oot=True` 显式展示。不会据单一 AUC 自动选最佳模型。

模型参数与 train 控制参数递归 flatten 后，以固定键顺序生成变化文本。
`params="all"` 额外展开所有模型参数，列表形式只展开指定参数。
特征数量变化显示如 `100→60 (-40, -40.0%)`，同数量替换显示 `100→100 (+5/-5)`。
特征集合与 reference 相同才显示 `-`，不会因“上一行相同”而隐藏变化。

## 配置与执行边界

只允许 BinaryClassification + LightGBM。Plan 的 `binary_logloss` / `weighted_binary_logloss`
是学习语义；execution 映射为 `objective=binary`，指标固定 `metric=auc`。
缺省值由框架补齐，显式冲突报错。轮数、早停、类别字段由 train/DataPlan 统一控制，不能用
LightGBM 参数别名绕过；seed/boosting/verbosity 的常见别名会规范化，冲突值报错。
其它原生模型参数透传，不重新实现 LightGBM 参数系统。
DART 必须显式禁用 early stopping，因为 LightGBM 不支持 DART 的早停。

每次 Run 新建 Dataset，以声明顺序传入特征和权重。类别词表仅来自 train，未知类别转换成缺失值；
直接用 `run.model.predict()` 时，调用者需要提供具有相同字段顺序和兼容 category dtype 的数据。
所有框架内预测、重要性和 save_model 都显式使用 best_iteration；无有效早停轮数时用完整 iteration。

ColumnSplitter 遇到缺失或未声明的分区值报错，避免悄悄丢行。
HashSplitter 用 SHA256 对 seed 和带类型标签的结构化 key 编码，取高 53 位映射到 [0,1)，
按 PartitionSpec 声明顺序的累计比例分配。支持字符串、有限数值和 bool 的单键/复合键；
整数和等值浮点数编码一致，字符串与数值区分。缺失值、非有限值、不支持的 key 类型报错。
不依赖 row order、Python hash() 或进程 hash seed。
RandomSplitter / TimeSplitter 可以在 Plan 中声明，但 V1 runtime 明确报 unsupported。

## 落盘与恢复

```text
experiments/risk_demo/
├── experiment.json                # schema version、名字、ModelPlan/DataPlan 和分区顺序契约
├── reference.json                 # 仅 set_reference 后存在
└── runs/
    └── <UTC timestamp>_<UUID>_<slug>/
        ├── run.json               # 状态、版本、schema/counts、指标、配置、文件校验和
        ├── config.yaml            # 默认值合并后的实际 model/train 配置
        ├── overrides.yaml         # Hydra overrides 或调用者传入的 changes（默认 []）
        ├── features.json          # 最终有序特征
        ├── metrics.json           # 所有 partition AUC 和 auc_gap
        ├── eval_history.json      # 完整 train/validation history，包括早停等待阶段
        ├── feature_importance.csv # gain/split 重要性和排名，gain 降序
        └── model.txt              # 原生 Booster.save_model，保存选定轮数
```

Run 名称可重复，run_id 为 UTC 秒时间戳 + 完整 UUID + slug。目录独占创建，关键文件临时写入后
原子替换，completed manifest 最后提交。完成或失败后 Store 拒绝再次提交同一个 Run。
训练异常记录 failed / error_type / error_message，然后抛出异常。进程被强制终止时可能留有 running
目录，compare 只读取 completed；不把未完成目录视为成功。不提供 V1 自动恢复失败训练。

```python
from phl_risk.modeling.experiment.lightgbm import LightGBMRun
restored = LightGBMExperiment(name="risk_demo", root="./experiments", data=data,
                              model_plan=model_plan, data_plan=data_plan)
restored.runs  # tuple of completed runs
one = LightGBMRun.load(baseline.path)  # 无需重新 attach data，可独立读取结果
model = one.model  # 此时才加载 lightgbm.Booster，可调用 dump_model/predict 等原生 API
```

恢复时严格比较 ModelPlan/DataPlan 契约及 partition 顺序。不会持久化原始数据或预测；metadata
保存 Python/phl-risk/LightGBM/sklearn/Hydra 等版本、seed、字段、类别词表、样本数和权重字段。
版本信息可追踪环境，config.yaml 包含框架解析后的完整配置；未显式配置的 LightGBM 参数仍使用
该记录版本的后端默认值（原生 model.txt 也保留模型参数）。相同数据与环境仍由使用者负责提供；
V1 不做完整数据内容指纹、不保证跨平台训练逐位一致。

Run 是冻结快照，dict 属性返回独立副本，feature_importance 每次读取新 DataFrame，model 每次
懒加载独立 Booster。调用者修改模型对象不会改写历史磁盘文件。校验和检测文件损坏，
不构成对恶意修改的数字签名或文件系统权限保护。

## V1 范围

包含：0/1 二分类、权重、Column/Hash split、原生训练、三种 callbacks、完整历史、AUC、
importance、不可覆盖 Run、恢复/校验、reference compare、可选 Hydra、train-only RFE、tolerance helper。
不包含：Random/Time split execution、其它模型/学习目标、自动最优模型、Optuna/grid/random search、
自动 sweep、SHAP、校准、KS/PSI 等扩展指标、部署/Serving、Dashboard、数据库/MLflow/WandB、分布式训练。

实现所依赖的接口见 [LightGBM early_stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)
和 [sklearn RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)。

### 安装固定的 4.0.0

```bash
python -m pip install -e '.[lightgbm,hydra]' 'lightgbm==4.0.0'
```

如果平台没有适用 wheel、转为源码构建，现代 CMake 可能拒绝旧版的 minimum policy。
本机 macOS ARM64 使用 `libomp`，并通过下列构建选项安装成功：

```bash
python -m pip install 'lightgbm==4.0.0' \
  --config-settings=cmake.define.CMAKE_POLICY_VERSION_MINIMUM=3.5
```

3.x 不在支持范围；本框架没有为旧版本删减 `min_delta` 或静默改变 early stopping 语义。
