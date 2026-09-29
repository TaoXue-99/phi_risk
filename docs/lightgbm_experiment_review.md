# Experiment 0.10.0 解耦验收

日期：2026-09-28，Asia/Shanghai。本记录是本地验证，不代表发布或远程 CI。

## 变更与职责

- 配置对象及 YAML 快照移到 configuration.py；参数/函数身份转换放在 _snapshot.py。
- Hydra adapter 返回中立配置数据，ComposedConfig 与原 YAML 读取入口继续可用。
- Record 与初始化模板不导入配置 adapter；参数模式可在无 YAML/Hydra/模型后端时记录运行。
- MethodExperiment 直接使用 Store/Session/query，不再在内部构造 Experiment。
- 运行中事实与完成/失败状态由 Store 写入，禁止修改身份及已结束记录。
- 方法空间可保存比较分区策略；生成器只消费一次，重复名称明确报错。
- task 可显式提供 objective/metric/feval 名称，支持非 LGB 原生参数结构；不控制训练。
- attempt.record 读取 metadata；Run.load/get_run 保持默认完整校验，并支持 verify=False。
  单文件访问仍校验哈希，verify_artifacts 支持显式全量检查。
- Record schema 保持 3，不改写旧记录；版本更新为 0.10.0。

## 验证范围

- 全仓 `pytest -q -W error`：478 passed。
- experiment 专项：62 passed。
- `ruff check .` 通过。
- `ruff format --check src tests examples README.md docs/lightgbm_experiment.md`：201 files already formatted。
- Guide：46 个单元格，其中 21 个代码单元格，连续两次从头执行成功并保存第二次输出。
- 每次产生 15 completed + 1 故意 failed；独立项目累计 30 completed + 2 failed，编号不覆盖。
- 数据为 100,000 行、60 特征；本机 Python 3.12.13、LightGBM 4.7.0。
- 本轮没有重跑隔离 LightGBM 4.0 环境；兼容辅助实现未修改，不将历史验收当作本轮结果。
- 已检查保存的表格与学习曲线；未做整份 Notebook 的浏览器页面视觉验收。

新增测试覆盖分区迭代器、策略重开和优先级、非 LGB 任务名称、Store 更新保护、
大文件按需校验与损坏检测、损坏 method manifest、导入边界。
隔离导入测试禁止 LightGBM/Torch/sklearn/NumPy/pandas/Hydra/YAML 和 adapters，
仍可创建 deep 方法空间，记录参数、指标与任意文件；这不代表运行了 PyTorch 模型。

## 实际结果与目录

最新输出位于 `experiments/native_guide_v010/lgb/`。旧 experiments 内容保留。
同项目 xgb 只初始化，未伪造训练记录。重复运行本指南继续追加编号。

| 第二次执行 Run | name | Train AUC | Valid AUC |
| --- | --- | ---: | ---: |
| lgb_run_17 | cpu_baseline | 0.819129 | 0.812458 |
| lgb_run_18 | depth3 | 0.811324 | 0.808854 |
| lgb_run_19 | depth3_lr | 0.808662 | 0.807623 |

结果与第一次一致；用于验证记录流程，不构成模型推荐。
Guide 保留原生二分类、回归/quantile、自定义 loss、callbacks、多分类、类别/缺失值、
继续训练、稀疏 CV、RFE 与 ranking，并补充方法级策略、task 展示、文件校验和深度学习接入说明。
GPU/CUDA 未执行，PyTorch 仅说明接入方式。

## 保留边界

- 原生训练/预测/评估仍由用户控制，没有新增 Trainer、模型基类或自动最优模型。
- 配置在 start_run 调用时快照；函数仅保存身份/摘要，复现仍需代码、依赖和数据版本。
- 方法级比较策略不改变早停或指标保存。旧无策略空间保持原行为；显式 compare(partitions=...) 可控制显示。
- 失败 Run 保留错误与 facts，暂存 artifact 仍清理；未增加强杀恢复或失败诊断产物恢复。
- 通用 RunRecord 的动态属性和历史配置字段继续兼容，本轮不为内部优化迁移记录格式。
