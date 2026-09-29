# 变更记录

当前源码版本 0.10.0；此记录不代表已发布 PyPI 或 GitHub Release。

## 0.10.0 — Experiment 解耦与 analysis 分箱扩展（2026-09-29）

- 配置数据、快照转换独立于会话；Record 不导入 Hydra adapter，原 ComposedConfig/读 YAML 入口保留。
- MethodExperiment 直接使用 Store/Session/查询函数，不再嵌套构造 Experiment；运行中 metadata 统一由 Store 写入。
- initialize 可显式保存 comparison_partitions，方法级 compare 优先采用该策略，避免遗漏某次 Run 设置后展示 OOT。
- 修复 comparison_partitions 生成器被消费、重复分区未拒绝的问题；已有未声明策略的空间保持旧行为。
- start_run(task=...) 可记录模型无关的 objective/metric/feval 展示名称；旧配置路径仍可读，不控制训练。
- Run.load/get_run 保持默认完整校验，新增 verify=False 与 verify_artifacts；attempt.record 仅读取 metadata，访问文件仍验哈希。
- 损坏 method/sequence metadata 给出领域异常，新增依赖边界和非 LGB 记录回归测试。
- 更新 10 万行 LGB Guide、原生脚本及职责文档；保留旧实验，最新 Guide 使用独立 native_guide_v010 项目。
- Record schema 3 与已有公共入口兼容；不新增 Trainer、深度学习封装或自动选优。
- analysis 新增 EqualWidthBinner 等距分箱、FixedBinner 自定义完整边界，复用 QuantileBinner 提取的公共区间实现。
- 保持现有等频 API；新分箱支持 BinDimension、布局/总计及 PSI，固定边界创建后可直接 compute。
- analysis 完整 Notebook 新增第 9.2–9.4 节和功能树入口，补充分箱边界、越界与缺失语义及一致性测试。

## 0.9.4 — 比较表默认展示实验名称（2026-09-27）

- compare 默认在 run 后展示 name，将 Notebook 的 start_run 名称与执行编号对应。
- 同名重复执行仍保留多行；已有 fields=["name"] 写法不产生重复列。
- 旧 Run 可直接使用新展示，无须重新训练或修改记录。

## 0.9.3 — 比较表展示训练目标与监控名称（2026-09-27）

- compare 默认增加 objective、metric 名称列，有记录时增加 feval 名称列。
- 自定义函数展示短名称，完整 callable 身份和摘要保持在原始记录中；多指标保持列表。
- 支持当前嵌套参数与旧平铺/model.params 记录，缺失显示“未记录”，不从结果或 metadata 猜测。
- 更新 Guide 内置目标/自定义 loss 对照、参数比较说明，并按授权重建 experiments。

## 0.9.2 — 原生 callback 参数直传（2026-09-27）

- 初始化 baseline 移除 early_stopping/log_evaluation 的框架 enabled 开关。
- 脚本、Guide 和 README 直接从 cfg 取值，在 lgb.train 中明确列出原生 callbacks。
- 生成的 callback 参数块可以直接通过 ** 传入原生函数；无需过滤或动态拼装。
- 原生日志 period=0 用于静默；不使用早停的案例直接不传早停 callback。
- 更新真实训练测试，覆盖生成配置与 LightGBM callback 接口的兼容性。
- 按用户要求重跑 Guide 并替换 experiments；旧用户配置需删除 enabled 后再用于直传，库不自动改写。

## 0.9.1 — 独立派生配置与方案保存（2026-09-27）

- 明确 baseline 不变、每次 Hydra Compose 派生独立完整配置、选择配置执行的流程。
- ComposedConfig.save(path) 原子导出完整方案，拒绝覆盖已有文件；函数仍只保存身份。
- 修复生成器形式 overrides 在输入校验时被提前消耗的问题。
- 新增配置独立性、嵌套列表隔离、baseline 不变和另存保护测试。
- Guide 先准备 baseline/depth3/depth3_lr，再分别训练并比較；新增导出及读取方案示例。
- 本次用新 Guide 的 experiments 替换旧示例结果；普通用户重复执行 Guide 仍追加 Run，不自动删除历史。

## 0.9.0 — 完整配置初始化与配置快照（2026-09-27）

- 新建 LightGBM 空间要求显式 objective/metric；metric 接受字符串或列表，objective 接受字符串或函数。
- baseline.yaml 分 params/train，包含树结构、采样、正则、种子、设备、轮数和 early stopping/logging 控制。
- 自定义函数在 YAML 中保存身份与源码摘要；运行时显式绑定实际函数，不做自动导入或执行。
- start_run(config=...) 支持 Mapping、YAML 路径与 Hydra ComposedConfig；自动保存最终配置、overrides 和入口源文件。
- Hydra 外部参数覆盖沿用原生 Compose；config 模式禁止 log_params 以防记录分叉，纯 params 模式保持兼容。
- 比较支持嵌套参数完整路径；原生训练与一次指标提交保持不变。
- 兼容说明：已有 baseline 不改写；打开已有空间用 open_method；旧 Record schema 3 保持可读。
- 更新原生脚本与 10 万行 Guide，新增完整配置、双指标、自建 YAML、Hydra 与函数 objective 初始化案例。

## 0.8.0 — 原生模型代码与轻量实验记录（2026-09-27）

- 删除 execution 目录及强制 LightGBMConfig/Hooks/Trainer/DatasetBuilder/评估策略/RFE helper；不新增通用 Executor。
- 新增 start_run 记录上下文：原生 Python 自由执行，正常提交 completed，异常/KeyboardInterrupt 记录 failed。
- 提供参数、输入描述、指标、JSON、文件与 serializer 记录；文件在 log 时快照，后续模型变化不影响已记录文件。
- 方法初始化不绑定训练器，可创建 lgb/xgb/torch 等平行空间。创建空间不宣称已实现或验证该模型训练。
- LightGBM 适配器仅保存和加载原生 Booster，轮数显式指定；可选 importance。4.0 依赖桥接改为显式选择。
- YAML/Hydra 保持可选，配置读取不再校验模型 task/objective 等；compose_config 为模型无关合成函数。
- 用户记录实际参数，callable 只记录身份与可取得的源码 hash，不伪装完整程序复现。
- Guide 改为原生 CPU 训练、分类/回归/自建 loss、callback、继续训练、categorical/missing、sparse/CV、RFE、ranking 和可选 GPU/CUDA。
- 新测试围绕记录生命周期、快照、损坏检测、并发、原生模型 roundtrip 与跨模型空间；删除已移除训练封装的专用测试。
- Record schema 保持 3；旧同 schema 事实不改写。0.7 的训练调用需迁移为原生代码加记录上下文。
- 保留编号、Asia/Shanghai 时间、原子落盘与比较。本轮未改 Plan API，未发布或推送。

## 0.7.0 — 显式执行配置与 LightGBM 扩展（2026-09-27）

- 保持 Initialization / Execution / Record / Adapters 边界；扩展 LightGBM 为二分类、多分类和回归。
- 删除框架隐式 binary/AUC、轮数、验证分区、早停和日志默认决策；关键字段缺失即报错。
- 初始化 baseline.yaml 生成待填写模板；完整教学配置独立置于 examples/modeling/conf。
- 新增 LightGBMHooks：原生 objective、feval、callbacks，报告预测转换与自定义 metric；要求代码 revision，记录 callable 身份，不 pickle 函数。
- 自定义 objective 明确 raw/custom 预测语义，原生 early stopping 与 YAML 重复配置报错。
- 训练 metric 与最终报告指标分离；支持 train-only、显式禁用评估和回归加权指标。
- RFE 根据任务使用分类/回归 wrapper，拒绝无法还原 Python hooks 的 baseline。
- compare_params 增加 hooks 身份差异；跨任务比较仍由使用者显式选择 runs 和指标。
- 重写 100,000 行 × 60 特征 Guide，覆盖分类/回归/quantile、自建 loss、callbacks、Hydra、RFE、恢复和目录解释。
- 不兼容执行配置变更：旧配置须显式迁移；旧 artifact 不自动改写。Plan 公共 API 无本轮调整。
- 保持 LightGBM >=4.0,<5、后端可选依赖和 Asia/Shanghai 时间；本轮验收详见 docs/lightgbm_experiment_review.md。

## 0.6.0 — Execution / Record 实验结构与 LightGBM 执行（2026-09-26）

- 实验时间统一使用 Asia/Shanghai（+08:00）：创建、开始、结束、失败及目录命名；历史 UTC 记录在查询视图转换展示，不改写历史文件或目录。
- 修正默认落盘层级：Experiment(root="./experiments") 直接使用根目录，LightGBM 方法目录固定为 lgb；不再由 Guide 添加随机项目目录。显式 name 仅保留旧 root/name 用法兼容。
- 新增 initialization 与薄 MethodExperiment：方法独立 configs/runs/reports，显式 YAML 路径输入；初始化不覆盖，open_method 只读恢复。
- 方法 Run 采用 lgb_run_01 + Asia/Shanghai 时间目录，filelock 与持久化计数器保护并发编号；失败不复用。项目支持汇总方法记录，旧 schema 3 根 Run 可继续读取。
- compare 改为 run/time/AUC/params(dict) 紧凑视图，额外字段按需；compare_params 对比两次 resolved 配置，支持 only_changed 与缺失/None 区分。
- 增加普通 YAML adapter（PyYAML 随 LightGBM extra），Hydra 保持显式可选；保存入口快照、overrides、resolved YAML 和 metrics 导出。
- 重写小型脚本和 10 万行 Guide 为初始化与 YAML 主流程。
- Experiment 核心按 Execution / Record 拆分；execution 下设 tree/lightgbm 和 deep 扩展位置。Record 不导入执行层，不解释 LightGBM config、AUC 或模型文件。
- Experiment 构造只需 name/root，每次 run 显式接受 prepared partitions/target/features/weight/categorical features；删除 Plan/Split 强依赖、内部数据切分和类别转换。
- 新增不可变 LightGBMConfig，with_params/with_overrides 产生新配置；分别记录 supplied/resolved，保留原生参数语义与别名冲突保护。
- 新增 RunRecord，记录模型身份、完整有序特征、分区摘要、嵌套指标、训练结果、环境及 artifacts；删除持久化 gap、reference、RFE 父子来源等派生关系。
- runs() 返回 running/completed/failed 轻量索引；compare 支持选定 runs/metrics/partitions/params/features/metadata，无默认 gap/delta/winner。
- 保留 native Dataset/train/Booster、加权 AUC、train-only RFE、4.0+ 接口兼容、原子落盘及 checksum。
- RFE 使用显式数据输入，不读取 Experiment private runtime；删除 within_tolerance 和自动推荐。
- Hydra 保留可选 adapter，并记录实际来源；核心 JSON Store 不依赖 PyYAML，普通 YAML adapter 与后端导出使用 LightGBM extra 的 PyYAML。
- artifact schema 升为 3。0.5 和中间 schema 2 目录明确拒绝且不改写；旧记录使用对应原版本读取。
- 删除 experiment.lightgbm 旧入口；改用 Experiment + LightGBMExecution。Run.model 为模型身份，load_model(run) 返回 Booster。RFE 为独立后端 helper。
- PreparedExecution / ExecutionResult 与 Artifact serializer 是扩展边界；Deep 仅保留命名空间，未实现深度模型。
- 不兼容变更：移除旧构造参数、runs 属性、平铺 metrics、set_reference、params="changed"、include_test/include_oot；使用新 API，参见迁移文档。
- modeling 顶层对 Plan 采用按需导出，保持原类身份，同时允许 Experiment 不加载声明实现。
- 更新 README 功能图、独立示例、使用/迁移文档和边界测试。
- 清理旧 experiment/lightgbm 缓存目录；新增 10 万行、60 特征完整 Notebook，固定特征演示 LightGBM 参数实验、诊断和恢复。
- 本轮验证结果见 docs/lightgbm_experiment_review.md；本地结果不代表远程 CI 或已发布。

## Unreleased — Modeling Plan 目录重组

- 声明实现按 plan/model_plan 与 plan/data_plan 归属组织；两个 Plan 类位于各自 definition.py。
- Goal/Strategy 迁入 model_plan；模型兼容性解析独立到 resolver.py。
- 数据侧 split 拆为 SplitSpec、PartitionSpec 与 Splitter 三个职责文件。
- 保留 modeling 和 plan 的 Plan/Spec 导出；删除顶层 Goal/Strategy 兼容目录，统一从 plan.model_plan 导入。
- 原叶子实现模块路径随迁移调整；不提供旧 pickle 路径兼容。JSON、describe、校验和执行行为保持不变。
- 更新示例、文档与执行层导入，新增公开入口类身份和无第三方依赖导入测试。

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
