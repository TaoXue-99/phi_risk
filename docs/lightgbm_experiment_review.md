# LightGBM Experiment V1 验收记录

日期：2026-09-19。

后续修正：默认验证分区已改为 valid；下方数值是修正前的历史验收，
其中 test 实际用于早停/调参，承担 validation 职责，不是独立测试成绩。
当前 example 使用 train/valid/test/OOT 四分区，不能将历史表解释为当前示例输出。
当前仓库的本地实现与验证；未提交 Git commit、未推送、未运行远程 CI。

## 交付与兼容性

六阶段已完成：运行时契约/Store/Run → 原生训练 → compare → Hydra → RFE → 文档与验收。
现有 ModelPlan、DataPlan、Goal、Strategy、metrics 和 _artifact.py 无修改。
学习目标 binary_logloss / weighted_binary_logloss 映射至原生 binary objective；weight 仍完全来自 DataPlan。
初始验收采用 LightGBM >=4.6,<5；后续兼容性验收将支持下限降低至 4.0.0，见文末记录。
旧版实际问题来自 NumPy/pandas 转换和 sklearn 校验参数改名，现已有局部适配。

## 验证结果

- Python 3.12.13 / LightGBM 4.7.0 / sklearn 1.9.0 / Hydra 1.3.7。
- 完整开发环境：`.venv/bin/python -m pytest -q -W error` → **457 passed**。
- 新增 LightGBM 实验测试：**65 passed**。
- 只安装 LightGBM 的隔离环境（未安装 Hydra）：实验测试 **64 passed, 1 skipped**。
- 基础隔离环境（无 LightGBM、Hydra、YAML、OptBinning）：全量 **424 passed, 27 skipped**。
  跳过可选 backend/Hydra 的实际集成；声明层、配置、split、缺依赖错误等仍执行。
  与全依赖环境收集数量不同，是已有 OptBinning 模块级 skip 导致，不代表测试丢失。
- LightGBM **4.6.0** + 当前项目依赖的隔离环境：实验测试 **65 passed**。
- `ruff check src/phl_risk tests examples benchmarks` → **All checks passed**。
- `ruff format --check src/phl_risk tests examples benchmarks` → **179 files already formatted**。
- `git diff --check` 通过。
- 完整 example：baseline、两个 RFE、三个参数实验，共 6 个 completed Run；恢复后 compare 完全一致。
- wheel / sdist 构建通过；实验包各模块均包含在 wheel 内。
- CI 新增 LightGBM/Hydra 的 Python 3.12/3.13 矩阵；本轮没有本地 Python 3.13 运行结果或远程 CI 结果。
- macOS 实训开始前缺失 libomp，已通过 Homebrew 安装，之后原生训练与加载通过。

## 实际 compare 输出

以下来自完整 synthetic example，20 特征、1200 行、固定 seed=2026、sample weight。
省略较宽的 run_id / created_at 两列，数值仅在本 Markdown 中取六位小数；DataFrame 保留 float。
所有变化都相对 baseline，非相对上一行。

| run | n_features | feature_change | best_iteration | train_auc | test_auc | auc_gap | delta_test_auc | delta_auc_gap | param_changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 20 | - | 16 | 0.911965 | 0.870061 | 0.041903 | 0.000000 | 0.000000 | - |
| rfe_12 | 12 | 20→12 (-8, -40.0%) | 138 | 0.947331 | 0.893678 | 0.053653 | 0.023616 | 0.011750 | - |
| rfe_8 | 8 | 20→8 (-12, -60.0%) | 5 | 0.890971 | 0.881449 | 0.009522 | 0.011387 | -0.032381 | - |
| depth2 | 12 | 20→12 (-8, -40.0%) | 141 | 0.923459 | 0.880160 | 0.043300 | 0.010098 | 0.001396 | max_depth: 3→2 |
| depth2_bag3 | 12 | 20→12 (-8, -40.0%) | 141 | 0.923614 | 0.876314 | 0.047300 | 0.006253 | 0.005396 | bagging_freq: 1→3; max_depth: 3→2 |
| lambda20 | 12 | 20→12 (-8, -40.0%) | 24 | 0.887189 | 0.856512 | 0.030677 | -0.013549 | -0.011226 | bagging_freq: 1→3; lambda_l2: 15.0→20; max_depth: 3→2 |

这些数值只验证实验链路，不是对模型效果或“最佳模型”的声明。

## 审查与回归

独立只读代码审查发现的两处问题已用回归测试修复：

1. LightGBM 会改写含空格的特征名。现于 runtime 拒绝不安全名称，并在完成提交前核对 Booster schema。
2. 原始 RFE 无法随逐步删列同步映射 monotone/interaction/feature penalty 等按位置参数。
   V1 提前明确拒绝这些 RFE 配置；普通原生训练不受此限制，用户可显式对齐候选约束。

另补充了 seed 别名覆盖、metadata/config 文件一致性、跨进程 hash seed、并发唯一路径、
类别词表仅从 train 学习、无早停预测轮数、无权重 AUC、故障落盘和缺依赖的真实边界测试。

## V1 限制

- RFE 类别排序使用 train ordinal codes；最终 candidate 仍按 DataPlan 原生 categorical 训练。
- RFE 不重映射按特征位置绑定的约束或外部强制分裂/分箱文件，提前抛 ExperimentError。
- RandomSplitter、TimeSplitter 仅 declaration，不提供 runtime execution。
- 不保存原始数据或预测；重新 attach 的数据内容由调用者负责，不提供数据内容指纹。
- 不实现 AutoML、自动最佳模型、自动 sweep、其它模型/任务、非 AUC 指标、部署或实验数据库。
- Run 不可变是 API/Store 保证，校验和可检测损坏，不是对外部恶意修改的数字签名。

## 文件清单

新增：

- `docs/lightgbm_experiment.md`
- `docs/lightgbm_experiment_review.md`
- `docs/superpowers/plans/2026-09-19-lightgbm-experiment.md`
- `examples/modeling/conf/baseline.yaml`
- `examples/modeling/lightgbm_experiment.py`
- `src/phl_risk/modeling/experiment/__init__.py`
- `src/phl_risk/modeling/experiment/lightgbm/__init__.py`
- `src/phl_risk/modeling/experiment/lightgbm/_utils.py`
- `src/phl_risk/modeling/experiment/lightgbm/callbacks.py`
- `src/phl_risk/modeling/experiment/lightgbm/comparison.py`
- `src/phl_risk/modeling/experiment/lightgbm/config.py`
- `src/phl_risk/modeling/experiment/lightgbm/dataset.py`
- `src/phl_risk/modeling/experiment/lightgbm/evaluation.py`
- `src/phl_risk/modeling/experiment/lightgbm/experiment.py`
- `src/phl_risk/modeling/experiment/lightgbm/feature_selection.py`
- `src/phl_risk/modeling/experiment/lightgbm/hydra.py`
- `src/phl_risk/modeling/experiment/lightgbm/run.py`
- `src/phl_risk/modeling/experiment/lightgbm/runtime.py`
- `src/phl_risk/modeling/experiment/lightgbm/split.py`
- `src/phl_risk/modeling/experiment/lightgbm/store.py`
- `src/phl_risk/modeling/experiment/lightgbm/trainer.py`
- `tests/modeling/experiment/lightgbm/conftest.py`
- `tests/modeling/experiment/lightgbm/test_comparison.py`
- `tests/modeling/experiment/lightgbm/test_edges.py`
- `tests/modeling/experiment/lightgbm/test_experiment.py`
- `tests/modeling/experiment/lightgbm/test_hydra.py`
- `tests/modeling/experiment/lightgbm/test_rfe.py`
- `tests/modeling/experiment/lightgbm/test_runtime.py`

修改：

- `src/phl_risk/exceptions.py`：ExperimentError / RunError。
- `pyproject.toml`、`uv.lock`：可选依赖与锁定版本。
- `README.md`：入口和工作流说明。
- `.github/workflows/tests.yml`：可选后端测试与完整示例。
- `.gitignore`：忽略仓库根目录 experiments 输出。

核心对象职责和实际调用代码详见 [使用文档](lightgbm_experiment.md)及
[完整 example](../examples/modeling/lightgbm_experiment.py)。

## valid 默认语义修正验收

默认 validation_partition 已改为 valid；默认 compare 显示 train/valid，
独立 test/OOT 分别通过 include_test/include_oot 显示。
RFE tolerance 默认跟随候选的 validation partition，拒绝使用其独立 test 指标作推荐。
显式配置 test 作为 validation 的历史 Run 保持可读，但其 test 不具备独立评估含义。

- 全量 `pytest -q -W error`：**461 passed**。
- 实验测试：**69 passed**，新增四分区、无隐式 test 回退、legacy 配置兼容测试。
- ruff check：通过；format：**180 files already formatted**。
- 更新后的四分区 example：6 个 Run，恢复后的 compare 一致，运行通过。

## LightGBM 4.0 兼容性补充验收（2026-09-19）

项目版本仍为本次新增功能的 **0.5.0**，可选依赖下限改为 `lightgbm>=4.0,<5`。
新增 `_compat.py`，只适配 LightGBM 模块内部引用，不修改 NumPy/sklearn 公共函数：

- 4.0–4.3：旧 NumPy dtype/array 与 pandas rename 调用。
- 4.0–4.5：sklearn 校验函数的 `force_all_finite` 参数改名。
- 4.6+：不启用兼容改写。
- 每个 Run 保存 `metadata.lightgbm_compatibility`；重复初始化幂等，lazy Booster 同样适用。

修复前实测：4.0.0 为 21 failed / 48 passed；4.4.0 和 4.5.0 各 4 failed / 65 passed。
修复后增加测试覆盖 nullable DataFrame 原生预测、依赖公共函数保持不变、幂等初始化、
兼容元数据记录。原有训练、RFE、权重、类别、落盘恢复与比较测试一同运行。

当前环境全量 `pytest -q -W error`：**463 passed**。
Ruff check 通过，format check **182 files already formatted**。
CI 增加 Python 3.12 + LightGBM 4.0.0–4.6.0 的逐版本矩阵；远程 CI 尚未执行。
本机旧版 4.0/4.1/4.2 源码构建使用 `CMAKE_POLICY_VERSION_MINIMUM=3.5`，
安装说明已补充到使用文档。最低支持 4.0.0，不承诺 3.x 或不同版本间逐位相同的模型结果。

本地 Python 3.12.13 + 当前项目依赖逐版本结果（全部带 `-W error`）：

| LightGBM | 实验测试 |
| --- | --- |
| 4.0.0 | 71 passed |
| 4.1.0 | 71 passed |
| 4.2.0 | 71 passed |
| 4.3.0 | 71 passed |
| 4.4.0 | 71 passed |
| 4.5.0 | 71 passed |
| 4.6.0 | 71 passed |
| 4.7.0 | 71 passed |
