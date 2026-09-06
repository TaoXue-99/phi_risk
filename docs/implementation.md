# analysis V0.1 逐阶段实现说明

本文按 design decision → why → public API → internal API → code → tests → limitations → next extension 记录各阶段。
源码均位于 `src/phl_risk`。数值层不依赖 analysis；Cube 不调用 pandas；只有具体维度、PandasEngine 和 Result/Layout 适配 pandas。

## Phase 0 — Architecture review

1. **Design decision**：Dimension 描述分析维度，Measure 编译计算需求，Transformer 学习行级转换；Cube 编排 Plan 和 Engine；Result/Layout 分离。
2. **Why**：分析语义、执行方式、展示顺序各自变化时不需要重写另外两层。
3. **Public API**：`Cube(dimensions=[...], measures=[...])`、`fit(reference)`、`compute(current)`、`result.layout(rows=..., columns=...)`。
4. **Internal API**：轻量 immutable specification，原生归约、组指标和派生指标三类计算节点。
5. **Code**：`analysis/` 下分 dimensions、transforms、measures、engines；Cube、Plan、Axis、Result、Layout 各有私有实现文件，公共入口通过 `__all__` 导出。
6. **Tests**：围绕数值、生命周期、计划合并、0–2 维分析和交叉布局分组验收。
7. **Limitations**：第一版 pandas 结果容器；没有调度器、复杂优化器或多 backend。
8. **Next extension**：先完成独立 numerical kernels，为引擎和其他调用方复用。

## Phase 1 — Metrics kernel

1. **Design decision**：AUC 复用 sklearn roc_auc_score；KS 复用 roc_curve 的完整阈值；同时请求时仍各自调用，不共享曲线或切换 AUC 实现；EventRate 复用 NumPy 求和与比例运算。
2. **Why**：排序中的 ties 必须作为整体，避免结果依赖原始行顺序；缺失与无效策略必须跨指标一致。
3. **Public API**：`auc_score(y_true, y_score, sample_weight=None, on_invalid="nan")`、`ks_score(...)`、`event_rate(target, event_value=1, sample_weight=None, on_invalid="nan")`。
4. **Internal API**：`prepare_binary` 处理框架输入策略；`auc_score` 与 `ks_score` 分别直接调用对应 sklearn 公共接口；`weights`、`invalid` 统一验证和策略。
5. **Code**：`metrics/_common.py`、`_auc.py`、`_ks.py`、`_event_rate.py`。没有自写排序、累计分布或积分。极端有限分数跨度超出 float64 时，使用 np.unique 的有序编码保留每个顺序/tie，避免 sklearn 差分溢出。
6. **Tests**：`tests/test_metrics.py`，20 组随机有/无权重 AUC 对照 sklearn；KS 独立阈值枚举；缺失、单类别、零权重、极端值、非二元事件。
7. **Limitations**：AUC/KS 是二元 0/1 指标；不支持 multiclass、partial AUC；纯函数指无用户输入副作用，不要求排除 pandas 缺失识别。
8. **Next extension**：其他标准指标同样优先适配成熟库公共 API；先验证语义，再测量性能，不维护无明确必要性的自定义算法。

## Phase 2 — Transform

1. **Design decision**：`QuantileBinner` 保存不可变配置，fit 后保存参考边界；输出 ordered categorical。
2. **Why**：OOT 只能使用 reference 的分位点；实际箱数、标签和区间顺序必须稳定可检查。
3. **Public API**：`BaseTransformer`；`QuantileBinner(n_bins=5, duplicates="drop", labels=None, include_lowest=True, precision=3)`；`fit`、`transform`、`fit_transform`。
4. **Internal API**：`is_fitted`、`metadata()`；内部边界保存为 tuple，`bin_edges_` 返回副本；构造器没有数据学习。
5. **Code**：`analysis/transforms/_base.py`、`_quantile.py`。有限参考值学习，两端扩为无穷；precision 只作用于展示；fit 验证完成后提交状态。
6. **Tests**：`tests/test_transforms_dimensions.py`，reference/current、越界、无穷、missing、重复边界、常量、labels、精度、重复索引及极端有限值。
7. **Limitations**：单列数值变换，线性分位数；`include_lowest=False` 让最低端点 -inf 缺失；不支持学习时权重。请求 5 箱不保证得到 5 箱。
8. **Next extension**：Pipeline 可串联同样的 fit/transform 接口。V0.1 不交付占位 Pipeline 类。

## Phase 3 — Dimension

1. **Design decision**：ColumnDimension 只选择字段；BinDimension 组合 Transformer 并稳定命名。
2. **Why**：分箱属于维度变换；分箱代码不能进入 Cube 或 Engine 的业务分支。
3. **Public API**：`BaseDimension`、`ColumnDimension("dt", name=None)`、`BinDimension("score", transformer, name=None)`。
4. **Internal API**：`required_columns()`、`output_name`、`requires_fit`、`is_fitted`、`metadata()`；具体 pandas 维度产生同索引 Series。
5. **Code**：`analysis/dimensions/`；字符串在 Cube 中归一化；字段错误包含源列名，输出保留名在构造时验证。
6. **Tests**：维度命名、缺字段、未拟合、索引保持、字符串归一化、输出名冲突，以及 Cube 间状态隔离。
7. **Limitations**：内置具体维度依赖 pandas Series，抽象 BaseDimension 不规定 backend 容器。自定义变换应保持长度、索引且不修改输入；`y` 为预留参数，内置无监督变换不使用它。
8. **Next extension**：DateDimension、DerivedDimension 可实现相同接口；引擎适配不同 backend 的行级输入。

## Phase 4 — Measure

1. **Design decision**：指标是 frozen configuration，其 `compile(context)` 返回已解析的 MeasureSpec。
2. **Why**：命名、依赖与计算方法分开，Context 不成为 global state；同一指标可复用 backend 计算节点。
3. **Public API**：Count、Share、EventRate、AUC、KS；显式 `name=` 消除默认名称碰撞。
4. **Internal API**：`BaseMeasure.compile`、`required_columns(context)`；MeasureSpec 含 name、node、empty_value。
5. **Code**：`analysis/measures/` 与 `_nodes.py`。Count/Share 不读取 Context 权重；EventRate/AUC/KS 显式配置优先于 Context。
6. **Tests**：`tests/test_measures_plan.py`，依赖、名称、Context 优先级、自定义计数 Measure 扩展；10/100 与 3/10 回归测试固定 Share 与 EventRate 的区别。
7. **Limitations**：Share 只接受 all；Count 只计算行数；不支持显式 None 来取消 Context 的权重（None 表示继承）。
8. **Next extension**：新 Measure 可复用现有节点；新的操作必须由目标 Engine 显式支持。

## Phase 5 — LogicalPlan

1. **Design decision**：CubePlan 保存维度状态快照、MeasureSpec、filters、Context、Policy；聚合按语义相同的节点去重。
2. **Why**：Count 与 Share 共用 row_count；相同 target 的 EventRate 共用 valid_count。优化依据计算语义而非指标显示名。
3. **Public API**：`cube.plan(context=None, totals=False)`、`cube.explain(format="table", totals=False)`、CubePlan、FilterExpression。explain 默认返回三列表格，format="text" 返回对齐文本。
4. **Internal API**：AggregateNode、GroupMetricNode、DerivedMetricNode；`plan.aggregates`、`plan.group_metrics`、`plan.required_columns()`。
5. **Code**：`analysis/_plan.py`、`_nodes.py`。派生层实现最小比例节点，没有通用 DAG、优化器或 Python eval。
6. **Tests**：共享节点数、指标类型、字段依赖、frozen 行为、explain 文本，以及 refit 后旧 plan 结果不变。
7. **Limitations**：Plan 是 frozen 外壳并隔离 Cube 的拟合状态，内部 Transformer 仍有 fit 方法；调用者应把 Plan 作为只读对象。callable 闭包捕获的外部状态无法深度冻结。
8. **Next extension**：扩展操作词汇和 backend capability；必要时再实现真正的变换规格序列化和深冻结，而非提前建立 AST 系统。

## Phase 6 — PandasEngine

1. **Design decision**：过滤 → 维度变换 → 缺失策略 → 整数坐标 → 一次 native aggregation → group kernels → derived ratios。
2. **Why**：整数坐标统一处理 categorical、missing 和多维组合；临时辅助列不污染源数据，也不会与原字段名碰撞。
3. **Public API**：`PandasEngine.execute(plan, data, context=None)`、BaseCubeEngine；Cube 支持 `engine="pandas"` 或 Engine 对象。
4. **Internal API**：`prepare_fit` 为 reference 应用过滤；groupby 对象复用分组索引，原生辅助列统一 sum；weighted EventRate 用 `np.maximum.at` 按组有效 target 缩放，避免另一次 groupby。AUC/KS 共用分组索引与字段数组，但各自调用 auc_score/ks_score；完全相同的指标节点仍去重。
5. **Code**：`analysis/engines/_base.py`、`_pandas.py`。没有 iterrows、逐行 apply 或 groupby.apply。AUC/KS 遍历组，其他指标向量化执行。
6. **Tests**：`tests/test_cube_engine.py` 覆盖 0–2 维、缺失、权重、过滤、空数据、输入保护、异常、backend dispatch；审查极端权重跨组缩放。
7. **Limitations**：同输入 AUC/KS 也分别计算，可能重复排序；组数量很多时仍有 sklearn 校验和 Python 调用开销。过滤 callable 是 pandas-specific；尚不提供任务并行。
8. **Next extension**：保持标准指标的固定公共调用路径，以 benchmark 评估性能；Polars/DuckDB 适配相同逻辑节点，无需把 executor 引入 Engine。

## Phase 7 — CubeResult

1. **Design decision**：稀疏 canonical long 数据与完整轴域分开保存；metric 永远是最后一个逻辑轴。
2. **Why**：避免计算阶段立即构造巨大笛卡尔积；reference 空箱仍能在布局阶段完整显示。
3. **Public API**：`data_`、`dimensions_`、`measures_`、`metadata_`、`axes`、`shape`、`ndim`、`total(over=[...])`。总计是独立保存的预计算结果，不改变 canonical shape。
4. **Internal API**：AxisSpec(name, role, values)，坐标缺失规范化为 None；Result 构造验证 schema、坐标唯一性、域包含关系、measure/metric 一致性。
5. **Code**：`analysis/_axis.py`、`_result.py`。dimensions_ 是 AxisSpec 元数据，具体变换元数据位于 metadata_["dimensions"]。
6. **Tests**：canonical 列顺序、轴域大小、稀疏观察数、metadata/data 防御性副本、重名坐标拒绝。
7. **Limitations**：容器为 pandas DataFrame，数值 value 在布局时转换为 float；逻辑 shape 的乘积可以大于 canonical 行数。
8. **Next extension**：select/sort 可在 Result 层返回新结果，并同步更新轴域；无需修改 Cube。

## Phase 8 — Layout

1. **Design decision**：TableLayout 完整排列所有轴，按 metric 的 empty value 补齐结构性空组合，然后 transpose/reshape。
2. **Why**：将指标轴放在 rows 第一层或 columns 第一层只是同一张结果的不同轴排列；不能隐式聚合遗漏的轴。
3. **Public API**：`result.layout(rows, columns, max_cells=..., totals=False, total_label="Total", bin_labels="code")`；`TableLayout(...).render(result)` 返回 DataFrame。
4. **Internal API**：整数坐标赋值到展示 tensor；按明确 levels/codes 构建 MultiIndex，避免 pandas 默认排序改变 metric 顺序。
5. **Code**：`analysis/_layout.py`。只有展示阶段物化完整网格；max_cells 在分配内存前检查。
6. **Tests**：`tests/test_result_layout.py`，5×5×3 精确索引和矩阵，三个轴所有排列及每种 rows/columns 划分；零维行/列、missing 坐标、空箱、重复轴和遗漏轴。
7. **Limitations**：不支持轴省略、隐式聚合或可格式化 table wrapper。总计要求 compute(totals=True) 预计算；layout 仅放置总计坐标。返回原生 pandas 表，后续 pandas 操作遵循 pandas 自身语义。
8. **Next extension**：独立格式化/导出层可以消费 table 或 Result，不影响计算层。

## Phase 9 — Cube

1. **Design decision**：Cube 只规范化配置、拟合、编译和调用 Engine；拥有独立的拟合维度副本。
2. **Why**：两个 Cube 复用同一个 BinDimension 输入时不能共享 mutable state；多维 refit 中途失败不能破坏旧状态。
3. **Public API**：声明式构造、fit、compute、fit_compute、plan、explain、dimensions、dimensions_、measures、is_fitted、repr。
4. **Internal API**：配置维度和拟合维度分离；拟合所有维度成功后提交；Plan 深拷贝当前维度状态。
5. **Code**：`analysis/_cube.py`、公共 `__init__.py`；库使用命名 logger，记录 fit/compute 事件，没有 basicConfig。
6. **Tests**：状态隔离、未拟合拒绝、refit 原子性、旧 Plan 重放、字符串归一化、invalid backend、输入无副作用。
7. **Limitations**：不实现通用 get_params/set_params；替换配置应构建新对象。fit/compute 不是承诺线程安全的并发操作；使用者应分别持有 Cube 实例或只读 Plan。
8. **Next extension**：必要时增加显式 configuration replacement 或 clone 协议，而非让任意 set_params 留下陈旧拟合状态。

## Phase 10 — 真实案例验收

1. **Design decision**：使用固定、小型、可解释的数据验收两个真实场景，并以 pytest 固定结果。
2. **Why**：随机 smoke test 只能验证可运行；固定矩阵和 MultiIndex 顺序才能防止静默变化。
3. **Public API**：案例只使用公共导出，不要求使用者理解节点或 Engine 内部细节。
4. **Internal API**：案例通过 assertions 验证 learned edges 不变、逻辑 shape 正确和 metric 优先顺序。
5. **Code**：`examples/analysis_examples.py`，直接执行打印 explain、分层表和两种交叉布局。
6. **Tests**：固定四个分组 AUC/KS 为 `[.75,.5]`、`[1,1]`、`[0,1]`、`[.5,0]`；交叉每格 Count=1、Share=.04、EventRate 随列取 0/1。
7. **Limitations**：本地案例为合成验收数据，不代表真实业务性能或统计稳定性。
8. **Next extension**：独立 benchmark 可监控 1M rows、10/100 groups 和多指标；先建立基线，再按测量结果优化。

## 增量改进 — 总计、解释表格与区间

1. **Design decision**：Engine 在 totals=True 时从相同有效样本计算全部较低维度的分组结果，Result 保存总计，Layout 决定展示哪些维度的总计；explain 输出二维描述表。
2. **Why**：AUC、KS 与 EventRate 不具有对已有数值直接求和/平均的总计语义；必须保存计算阶段得到的 pooled statistics，保持 Layout 无执行职责。
3. **Public API**：compute/fit_compute/plan 的 totals 参数，layout 的 totals/total_label/bin_labels，result.total(over=...)，explain(format="table"|"text")。
4. **Internal API**：PandasEngine._compute_prepared 复用过滤后的数据和转换后 Series；低维计划关闭 totals，防止递归展开；总计按保留维度名称索引。validate_field 命名明确表达字段验证职责。
5. **Code**：总计不保留原始数据，不重新 fit/transform，不重复执行 filters；Layout 以独立总计坐标填充已有统计结果，区间标签仅作显示映射。exceptions.py 包含带解释的继承树。
6. **Tests**：pooled AUC=.75 而两个日 AUC 均为 1；weighted/unweighted EventRate 总计；行列与整体 Share；同一过滤总体；转换次数；空箱/空总体；总计标签冲突；unstack 等价性；explain 表格与文本。
7. **Limitations**：总计增加计算工作；n 个维度总共需要 2**n 个分组粒度（最多 64），无任意聚合公式。explain 默认类型改为 DataFrame，旧字符串用途需 format="text"。
8. **Next extension**：有性能依据时共享总计的可加归约中间量；AUC/KS 继续固定调用 sklearn 公共函数，不从分组指标近似推导。
