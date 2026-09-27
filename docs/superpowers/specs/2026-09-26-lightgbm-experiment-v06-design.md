# LightGBM Experiment 0.6.0 设计

状态：用户已确认并完成实现。基于 2026-09-26 完整代码审计及用户允许全新迭代的说明。

## 目标与取舍

Experiment 接受 prepared partitions，完成训练、记录事实、查询和自由比较。
允许替换 0.5 的公共调用方式和内部实现，不维护旧 Plan 驱动的双执行路径。
保留原生 LightGBM、sklearn RFE、已有可靠落盘机制与 LightGBM >=4.0,<5 支持。
不实现其他 backend、通用继承框架、自动最佳模型选择或数据处理。
不修改工作区正在进行的 Plan 目录迁移及无关 Notebook 改动。

## 公共入口

保留 phl_risk.modeling.experiment.lightgbm 路径，不新增 tree 目录。
主要导出 LightGBMExperiment、LightGBMConfig、LightGBMRun、RunRecord。
Hydra 合成保留为可选 adapter，核心 Config/Experiment 不依赖 Hydra。

```python
exp = LightGBMExperiment(name="risk_demo", root="./experiments")
config = LightGBMConfig(
    params={"num_leaves": 7, "max_depth": 3, "reg_lambda": 15.0},
    num_boost_round=5000,
    early_stopping_rounds=200,
)
run = exp.run(
    name="baseline",
    partitions={"train": train_df, "valid": valid_df, "oot_june": oot_df},
    target="label",
    features=["feature_a", "feature_b"],
    categorical_features=[],
    weight=None,
    config=config,
)
config2 = config.with_params(num_leaves=15, reg_lambda=20.0)
index = exp.runs()
frame = exp.compare(params=["num_leaves", "reg_lambda"])
frame["gap"] = frame["train_auc"] - frame["valid_auc"]
restored = exp.get_run(run.run_id)
model = restored.model
```

## 输入

Experiment 构造只接收 name/root，读取历史不要求数据或后端可用。
run 明确接收 partitions、target、features、categorical_features、weight 和 config。
内部轻量输入对象只在当前执行中存在，不持久化 DataFrame。
固定 train 为训练分区，validation_partition 默认 valid；其他名称任意。
每个分区必须是非空且列名唯一的 DataFrame；名称必须是非空字符串。
所有入模字段必须存在；特征非空、唯一、不与 target/weight 重叠，严格保留传入顺序。
保持模型 feature name 安全校验和训练后 schema 核对。
0/1 target 无缺失，AUC 评估分区须有两个有效类别；权重有限、非负、有效类别权重均为正。
不能计算 AUC 时明确报错，不静默跳过或伪造指标。
数值特征必须是后端支持的数值类型；类别特征必须已准备为 categorical，跨分区类别 schema 一致。
不自动切分、编码、学习类别词表、填补或把未知类别变成缺失值。
输入校验不得修改用户 DataFrame。同步执行期间调用方不得并发修改输入数据。

## Config

LightGBMConfig 是深层不可变的 specification，保存显式 supplied 配置。
提供 with_params 和 with_overrides，均返回新对象；不原地改变旧配置。
with_params 只修改模型参数，with_overrides 修改结构化 model/train 配置。
继续支持原有 model.params + train 结构的普通 Mapping，避免要求 Hydra。
保留 defaults、框架字段 unknown-key 检查、受控参数保护和别名冲突检查。
不重写 LightGBM 全参数 schema；其他 backend 参数交给 LightGBM 处理。
resolve_config 产生独立 ResolvedConfig，含 supplied 和 resolved 两份快照。
resolved 是框架实际交给 Dataset/Trainer 的配置，不宣称枚举了所有 LightGBM 隐式默认值。
同时记录后端版本及训练后 Booster params，明确原生默认值仍受版本影响。
保持 objective=binary、metric=auc 和 DART/early stopping 不兼容检查。
validation 名称只做训练分区冲突等必要检查，不按 oot/test 字符串强行制定使用政策。
early_stopping_rounds=None 表示关闭早停，validation 仍用于记录学习曲线。

## Record 与 artifacts

在 run.py 中定义 RunRecord，不新增通用模型框架。
RunRecord 的源信息是不可变、可序列化的事实：

- identity：run_id、name、status、created_at、started_at、completed_at/failed_at。
- model：family=tree、backend=lightgbm、backend_version。
- input：target、weight、完整有序 features、categorical_features、partition rows 和必要 dtype schema。
- configuration：supplied、resolved；可选配置来源信息，仅在实际使用时记录。
- result：嵌套 metrics、best_iteration、best_score、training_seconds、Booster params。
- environment：Python、phl-risk、LightGBM、NumPy、pandas、sklearn；使用 Hydra 时才记录其版本。
- artifacts：语义名称、相对路径、文件格式和 sha256。
- failure：error_type、error_message；不伪造模型或指标。

metrics 采用 {partition: {"auc": float}}。不持久化 gap、delta、reference、parent 或 winner。
不持久化原始数据或预测。完整 evaluation history 单独作为 artifact，避免索引加载大曲线。
n_features 从 features 推导，避免多个事实来源。
不增加并发自增 sequence；按 started_at 和 run_id 排序，序号只作为展示信息。

LightGBMRun 包装 Record 与文件访问，暴露 config/features/metrics/artifacts/path 等属性。
model 保留原生 Booster，按需加载并验证 checksum 和特征顺序。
running/failed 可读取记录；访问不存在的模型产生清晰 RunError。

## Store 和生命周期

保留 Path、唯一 UUID、排他目录创建、临时文件原子替换和完整性校验。
输入和配置校验通过后分配 running 记录，再训练、评估、收集文件，最后提交 completed manifest。
训练或落盘失败记录 failed，保留原始异常并重新抛出；失败记录写入失败时给原异常加 note。
不承诺捕获进程终止或机器断电；这种情况下保留 running，不擅自认定失败。

新 artifact_version=2。Experiment manifest 只记录实验身份、版本、创建时间，不绑定数据/Plan。
completed/failed 记录不重写。
轻量 runs 索引只读取 manifest，不扫描全部模型文件。
明确加载 Run 时执行完整校验，model/importance 在使用时再次验证对应 artifact。
读取相对 artifact 路径时验证不能逃逸 Run 目录。

旧 artifact v1 不自动迁移、不覆写。打开旧目录明确报版本不支持，并建议新建实验目录。
本轮不实现旧构造参数、旧 runs 属性、旧平铺 metrics、set_reference 或 tolerance recommendation 的兼容分支。
文档提供迁移示例；旧文件保留，可用旧版本读取。

## Execution

Experiment.run 仅描述流程：校验输入 → resolve → start → build → fit → evaluate → importance → complete。
保留 LGBDatasetBuilder、LightGBMTrainer、callback 工厂，不新增执行器基类。
每 Run 新建 Dataset；仅 train/validation 进入训练 callbacks。
预测明确使用 best_iteration，必要时回退到完整 iteration。
evaluation.py 只计算各分区 AUC；importance.py 保存 gain/split importance 与排序。
_compat.py 保留 4.0 兼容。optional dependency 继续懒加载。

## 查询和比较

exp.runs(status=None) 返回所有状态的轻量 DataFrame，可按状态筛选。
get_run 接受 run_id 或唯一名称；同名歧义要求 run_id。
compare(runs=None, metrics=None, partitions=None, params=None, features=False, metadata=None)
返回 completed records 的 DataFrame，选择运行时保持调用方指定顺序。
默认展示身份、backend、特征数、训练轮数、耗时、train/validation 指标。
partitions="all" 展示任意额外分区；显式列表仅展开所选分区。
metrics 选择指标名称，当前 evaluator 只有 auc。
params 支持列表或 "all"；第一版不保留 params="changed"，用户直接取参数列比较。
features=True 展开完整有序特征列表；metadata 选择已记录事实字段，不计算派生结论。
列命名避免参数和身份/指标相互覆盖；指标保持 float。
不自动对不同数据的运行作可比性保证，用户可通过输入摘要判断。

## RFE

保留可选 select_features 入口，但每次显式传入 prepared partitions/target/weight/categorical_features。
base_run 只作为用户选择的模型参数和候选特征来源，不构成持久 Run 关系。
辅助函数只接收明确输入和配置，返回候选特征子集，不读取 Experiment 私有属性。
Experiment 使用候选子集调用普通 run，生成同等地位的独立结果。
仅 train 参与 RFE；保留 sample_weight、位置约束校验、sklearn routing 和 native 最终训练。
删除自动容差推荐；候选结果支持普通 compare 查询。

## 文件与实施顺序

1. run.py/store.py 和新增 record/store 测试：记录、状态、artifacts、schema 版本。
2. config.py/hydra.py/__init__.py 和 config 测试：不可变配置、supplied/resolved。
3. runtime.py 改为纯输入校验；移除 split.py；调整 dataset/callbacks/trainer/evaluation；新增 importance.py；重写 experiment 编排。
4. comparison.py 和测试：自由字段展开，无强制差异。
5. feature_selection.py 和测试：显式输入、候选训练。
6. examples、README、使用文档、迁移说明、CHANGELOG、0.6.0 版本和锁文件。

每阶段先说明改动，实施后运行相关测试并报告，再进入下一阶段。
既有 split 执行不迁入声明层，外部准备数据的例子用普通 pandas 表达。
Plan 教程中旧 Experiment 执行能力引用仅作局部文档更新，不重构 Plan。

## 验收

覆盖不可变配置、defaults/aliases/冲突、字段缺失、类别/权重、任意分区、状态转换、失败不伪造结果、原子落盘、损坏检测、模型预测 roundtrip、记录查询无需数据、比较无 gap/delta/winner、RFE train-only。
保留已有 backend-free 导入和可选依赖错误验证，新增核心执行模块无 Plan/Split 依赖的边界测试。
运行完整 pytest -W error、Ruff lint/format、新示例，复验 LightGBM 4.0 和当前版本；现有中间版本 CI 保留。
不推送、不打 tag、不发布 PyPI，除非用户另行要求。
