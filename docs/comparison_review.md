# Comparative analysis 阶段验收与最终审查

验收日期：2026-09-11。Python 3.12.13、macOS arm64；NumPy 2.5.2、pandas 3.0.5。
本记录为本地验证结果，不代表远程 CI 状态。原有 Python 3.13 CI 配置保留，但本任务不以其为验收要求。

## Phase 0：现状与最小修改

Cube 编排，Measure.compile 生成 Plan 节点，PandasEngine 归约，CubeResult 保存结果，Layout 展示。
Dimension/BinDimension 与 QuantileBinner 已有显式 fit/transform 生命周期；InvalidPolicy 提供 nan/warn/raise。
0.3.0 已移除旧计数 PSI 和数据质量独立算法，DataQuality 已移除漂移检查并与 analysis 解耦。
新增比较入口放在 Cube，模式校验位于 Cube/Plan，后端新增非 abstract 的 execute_comparison，避免破坏旧后端。
不添加 AnalysisInputs 或复杂调度器；两个明确的方法签名已经足够。

## 初次比较框架实现的阶段记录

以下测试数量为阶段历史快照；0.3.0 统一 PSI 的追加验收见文末。

| 阶段 | 决策、核心实现与文件 | 新增验证与阶段结果 | 风险与下一步 |
|---|---|---|---|
| 1 Protocol | `_comparison_types.py`、`measures/_base.py`、`_nodes.py`、`_plan.py`、`_cube.py`、引擎基类；比较节点通过协议执行，旧 BaseMeasure 默认 single | 错误入口、混合指标、Plan 模式、旧后端回归；177 passed | 私有节点仍不承诺外部 API 稳定；进入数值层 |
| 2 Kernel | `metrics/_psi.py` / `_policy.py`；1D/2D 最后轴向量化，唯一占比内核 | 独立算式、两侧零占比、epsilon、dtype、shape、逐行无效策略；相关29 passed | epsilon 统一为1e-8；进入分布构建 |
| 3 Distribution | `_distribution.py`；私有参考 binner、类别并集、独立缺失编码、bincount profile | 边界、超范围、批量计数、类别/缺失字符串冲突；分布+kernel共20 passed | 稠密矩阵可能大；有分配上限，进入 Measure |
| 4 PSI | `measures/_psi.py`；字段配置与批量计算分离，返回 ComparativeNode | 整体、相同分布、位移分布、不同样本量 | 不隐式权重、不缓存；接通共同维度 |
| 5 Dimensions | `engines/_comparison.py`；两侧各转换一次，整数轴编码与一次 joint factorize | 0–4 维与独立 pd.cut/value_counts 逐值对照，缺组、None、categorical 轴、显式 BinDimension.fit | 日期必须相同 key；不广播、不 match_by；进入批量字段 |
| 6 Batch | 一次 groups 对象传给每个 calculation；每字段向量化，无逐组 PSI 循环 | 3/100字段、调用计数、重名、顺序、非 PSI 测试插件；协议+PSI共23 passed | 每个字段仍独立拟合，没有跨调用缓存；进入实测 |
| 7 Performance | `benchmarks/benchmark_comparison.py`；24种规模对照，优化内置 binner 复制和类别重编码 | 数值对照通过；最大组合另做3次取中位数 | 无分层未显示优势，不做绝对最快承诺；进入完整回归 |
| 8 Result/docs | 继续使用 CubeResult，无第二套结果类型；新增示例/文档/CI demo | 完整225 passed；Ruff通过；原有analysis/funnel和新PSI示例通过；比较文档4段代码顺序执行通过 | 比较 totals/details 不支持，明确拒绝；完成打包审查 |

阶段记录对应当时测试快照；最终完整集合包含新增矩阵上限、极端非法占比和额外边界回归。

## 性能数据

每侧行数相同，随机种子42，数值正态样本，10个分箱；每维3类，最多27个观察组。
下表各组合单次采样，包含分箱拟合/转换、计算和结果构建，存在运行噪声，不是 SLA。
对照实现也仅按字段拟合一次 reference，不做逐组重复拟合；主要差别是每字段重复 pandas groupby/value_counts。
全部24种组合均核对了结果数值。独立单元测试进一步按实际 joint key 逐值核对。

| 每侧行数 | 字段数 | 维度数 | 共享编码秒 | pandas秒 | pandas/共享 |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 1 | 0 | 0.012 | 0.006 | 0.48× |
| 100,000 | 1 | 1 | 0.017 | 0.008 | 0.46× |
| 100,000 | 1 | 2 | 0.020 | 0.009 | 0.44× |
| 100,000 | 1 | 3 | 0.023 | 0.010 | 0.41× |
| 100,000 | 10 | 0 | 0.054 | 0.052 | 0.95× |
| 100,000 | 10 | 1 | 0.064 | 0.070 | 1.10× |
| 100,000 | 10 | 2 | 0.068 | 0.078 | 1.15× |
| 100,000 | 10 | 3 | 0.070 | 0.086 | 1.23× |
| 100,000 | 100 | 0 | 0.541 | 0.516 | 0.95× |
| 100,000 | 100 | 1 | 0.538 | 0.692 | 1.29× |
| 100,000 | 100 | 2 | 0.547 | 0.781 | 1.43× |
| 100,000 | 100 | 3 | 0.551 | 0.848 | 1.54× |
| 1,000,000 | 1 | 0 | 0.056 | 0.047 | 0.85× |
| 1,000,000 | 1 | 1 | 0.160 | 0.059 | 0.37× |
| 1,000,000 | 1 | 2 | 0.188 | 0.065 | 0.35× |
| 1,000,000 | 1 | 3 | 0.219 | 0.074 | 0.34× |
| 1,000,000 | 10 | 0 | 0.501 | 0.467 | 0.93× |
| 1,000,000 | 10 | 1 | 0.600 | 0.589 | 0.98× |
| 1,000,000 | 10 | 2 | 0.631 | 0.739 | 1.17× |
| 1,000,000 | 10 | 3 | 0.669 | 0.707 | 1.06× |
| 1,000,000 | 100 | 0 | 5.236 | 4.757 | 0.91× |
| 1,000,000 | 100 | 1 | 5.156 | 6.003 | 1.16× |
| 1,000,000 | 100 | 2 | 5.263 | 6.618 | 1.26× |
| 1,000,000 | 100 | 3 | 5.242 | 7.181 | 1.37× |

最大组合每侧100万行、100字段、3维，另做3次：共享编码中位数 **5.206s**，
pandas 中位数 **7.055s**，约 **1.36×**。
两个输入 DataFrame 合计约 **1571.7 MiB**，此数字不是进程峰值内存。
每组27×10的计数矩阵很小，主要瓶颈为100列上的分位点拟合、分箱和扫描；分层越多，共享编码越有价值。
无维度时没有分组工作可共享，框架校验与结果封装会产生额外开销。

执行方式：

```bash
uv run python benchmarks/benchmark_comparison.py --rows 100000 1000000 --fields 1 10 100
uv run python benchmarks/benchmark_comparison.py --rows 1000000 --fields 100 --dimensions 3 --repeats 3
```

## 最终架构审查

- Cube/Engine 没有 PSI 类型分支；`RowDifference` 仅在测试中证明第二种计算可直接复用入口。
- PSI 不知道 Dimension 字段或日期。分组信息由 Engine 提供，数值内核只处理 NumPy 占比。
- 所有维度共同匹配，取两侧 key 并集。每个字段共享同一组编码，100字段测试真实拦截计数为1。
- 参考全局学习边界后应用两侧；缺箱补零、类别并集、缺失独立桶和 min_samples 已覆盖。
- 新比较路径放在引擎内部辅助模块，避免继续扩大 PandasEngine 单文件；它不是新 Engine 类型。
- 结果复用 CubeResult；比较型 totals 暂不支持。Layout 不访问原始数据，不计算 PSI。
- BaseMeasure/旧后端兼容，AUC/KS 数值调用未变；DataQuality 不再计算 PSI，模块独立性由隔离测试验证。
- pandas 耦合集中于分组与分布适配；数学内核和 invalid policy 无 pandas 依赖。
- 原生 QuantileBinner 路径避免额外源列复制；自定义 binner 输入保留防御性复制；用户 DataFrame 不被修改。
- Profile/协议仅为轻量内部 dataclass/ABC；没有新结果类型、调度器或通用表达式系统。
- 本期未实现其他比较指标、时间 DSL、match_by、缓存、权重、bin details 或比较总计。

接口与架构图见 [comparative_analysis.md](comparative_analysis.md)。


## 0.3.0 统一 PSI 初次验收（历史快照，质量适配器随后移除）

- 删除计数 PSI 函数、PSIState、PSIMetric 和旧 DriftMetric/DRIFT_METRICS；唯一业务路径为比较 Cube。
- DistributionDriftCheck 保存所选参考列，validate 一次批量调用比较框架；逐值一致性与调用次数有测试。
- 类别并集、常量单箱、无效 PSI 为 SKIP 和新状态持久化有回归；旧 artifact 需重新 fit。
- pyproject、包 __version__、uv.lock 一致为0.3.0；离线锁文件检查通过。
- Python3.12 完整测试 **228 passed**；README14段、比较文档4段Python示例顺序运行通过。
- analysis、funnel、psi、quality、prep、OptBinning、risk-workflow七个脚本运行通过。
- 0.3.0 wheel/sdist构建通过，从wheel导入版本、确认旧接口删除、运行比较与质量检查通过。
- README以全框架功能图和函数入口表开篇，环境依赖移至安装部分；源码提交不等同于 PyPI 发布。


## 0.3.0 模块隔离最终验收

已删除 DistributionDriftCheck 及其适配代码、导出与质量测试；PSI 仅保留在 analysis/metrics。
数据工作流不再嵌入 PSI，PSI 示例保持独立的六步分析。DataQuality 当前11类检查。
新增子进程导入隔离与源码依赖测试，确认 DataQuality 不导入 analysis/metrics。
Python3.12 完整测试226 passed；Ruff通过；PSI、质量与数据工作流示例通过。
上述“初次验收”保留为历史快照，其质量适配器已移除，不能作为当前使用方式。
