# Experiment 0.9.3 配置流程本地验收

日期：2026-09-27，Asia/Shanghai。记录本地验证，不代表发布、远程 CI 或推送。

## 本轮变更

- compare 在时间后增加 objective、metric 名称，有记录时展示 feval 名称。
- 自定义函数只展示短名称，多指标保留列表；完整身份/摘要和原始参数不变。
- 新增平铺/嵌套/旧参数布局、缺失值、自定义函数列表与记录不变的测试。
- Guide 增加内置 binary 与 logistic_loss / logistic_feval 的实际比较表。
- 新结果 15 条成功、1 条故意失败，旧 experiments 已按授权替换并清理。

- 初始化模板移除 callback enabled 开关，配置块直接兼容原生 early_stopping/log_evaluation 参数。
- Guide、脚本和文档直接从 cfg 取值，在 lgb.train 中列出 callbacks，不再动态拼装。
- 不使用早停的案例直接不传早停；日志静默使用原生 period=0。
- 真实训练测试新增对生成配置的 callback 直传校验；LightGBM 4.0 与当前版本均通过。

- 基于固定 baseline 派生多套独立完整配置，Guide 先准备三套，再选择执行。
- ComposedConfig.save 原子导出配置方案且拒绝覆盖，修复生成器 overrides 被提前消费的问题。
- 新增配置隔离、baseline 不变和导出安全测试。

- LightGBM 初始化显式指定 objective/metric，生成包含 params/train 的可用起步 YAML。
- metric 支持字符串和列表；objective 支持函数，YAML 保存函数身份与源码摘要。
- start_run(config=...) 接收 Mapping、YAML 路径和 Hydra ComposedConfig，自动保存配置快照与来源。
- Hydra 外部 overrides 可调整模型参数、指标列表和训练控制。
- 原生 lgb.train、callbacks、自定义 objective/feval、预测与指标计算保持自由。
- config 模式禁止追加 log_params，防止 config.yaml 与 run.json 分叉；原有 params 模式保留。
- compare 默认完整展开嵌套参数路径到 params 单元格；compare_params 支持完整路径或唯一后缀。
- 源码和 uv.lock 版本 0.9.3；Record schema 3，旧记录无需迁移。Plan 未修改。

## 本地验证

| 检查 | 结果 |
| --- | --- |
| 全仓库 pytest -q -W error | **469 passed** |
| 隔离 LightGBM 4.0 Experiment tests | **53 passed** |
| 无可选后端初始化 | 本轮全仓测试中的隔离导入检查通过 |
| Ruff check | 通过 |
| Ruff format --check | 198 files already formatted |
| git diff --check | 通过 |
| 命令行脚本外部 Hydra overrides | 训练、保存、恢复、比较通过 |
| Guide | 42 个单元格，20 个代码单元格从头执行并保存输出、1 幅曲线 |
| Guide Run | 15 completed + 1 故意失败示例 |
| Artifact | config.yaml 与 run.json 快照、模型恢复、上海时间和 0.9.3 版本核对通过 |

新增测试验证初始化必填项、非法参数不产生方法目录、已有文件保护、指标列表、函数 objective、
真实原生训练及保存恢复、Hydra 外部覆盖、独立 YAML、源文件变化后的快照、嵌套比较与配置不可追加修改。
基础环境 skips 为可选依赖集成。初始化仍不需要 LightGBM/Hydra/PyYAML；config= 写 YAML 需要 yaml extra。
LightGBM 4.0 在仓库现代依赖下显式使用兼容辅助函数；不声称未调整的旧版本能直接兼容全部现代依赖。

## Guide 结果与范围

100,000 行 × 60 特征。原生分类、回归、quantile、自定义 loss、callbacks、多分类、
categorical/missing、继续训练、sparse CV、RFE 和 ranking 保留并执行。
新增从初始化 baseline 开始的双指标配置、Hydra 外部覆盖、独立 YAML 和函数 objective 初始化。
GPU/CUDA 分支未执行。已查看输出表格与导出学习曲线；未进行整份 Notebook 的浏览器页面视觉验收。

| Run | Train AUC | Valid AUC | 参数 |
| --- | ---: | ---: | --- |
| lgb_run_01 | 0.819129 | 0.812458 | depth=4, learning_rate=0.05 |
| lgb_run_02 | 0.811324 | 0.808854 | depth=3, learning_rate=0.05 |
| lgb_run_03 | 0.808662 | 0.807623 | depth=3, learning_rate=0.03 |

这些数值验证记录流程，不构成模型推荐。oot_demo 仍是 IID 示例，不能替代真实时间外验证。

本次输出位于 `experiments/lgb/runs/`，按用户要求替换旧实验结果，暂存的旧目录已清理。
同项目 xgb 目录仅初始化；custom_objective_demo 子项目展示函数初始化，不产生额外伪训练记录。
本轮中间试跑移出后清理，正式 experiments 目录仅保留本轮执行的结果。重复执行仍追加编号。

## 使用与迁移

- 已有空间用 open_method；新建 lgb 必须提供 objective/metric，已有 baseline 不自动改写或升级。
- 函数无法靠 YAML 还原，读取后显式绑定真实函数；保留代码与数据版本。
- config= 下参数在 start_run 调用时快照，之后不得修改训练参数而沿用旧记录。
- config.source.yaml 只保存 Hydra 入口源文件；所有组合后的值在 config.yaml，配置组源码仍需版本管理。
- 指标通过一次 log_metrics 提交，Hydra 不替代评估；recording_seconds 不等于纯训练耗时。
- 不新增 Trainer/Executor、自动最优模型、自动调参或可执行函数恢复。

主要文件：initialization/lightgbm.py、record/configuration.py、record/session.py、experiment.py、
method.py、record/comparison.py、adapters/yaml.py、配置测试、Guide、原生脚本、baseline.yaml、README、CHANGELOG 与版本文件。
