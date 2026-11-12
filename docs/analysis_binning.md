# analysis 分箱维度

`BinDimension(column, transformer)` 用转换后的箱标签分层。算法由 transformer 决定，
无需增加新的 Cube 或 Dimension 类型。三种分箱器都从 `phl_risk.analysis` 导入。

| 方案 | 入口 | 边界来源 | 生命周期 |
|---|---|---|---|
| 等频 | `QuantileBinner(n_bins=5)` | 参考有限值分位点 | fit 后 transform |
| 等距 | `EqualWidthBinner(n_bins=5)` | 参考有限范围等距切点 | fit 后 transform |
| 自定义 | `FixedBinner(edges=[0, 20, 50, 100])` | 完整、严格递增的指定边界 | 创建后可直接 transform |

## 等距：参考范围决定切点

```python
from phl_risk.analysis import BinDimension, Count, Cube, EqualWidthBinner

cube = Cube([BinDimension("score", EqualWidthBinner(5))], [Count()])
cube.fit(reference)
result = cube.compute(current, totals=True)
table = result.layout(bin_labels="interval", totals=True)
# 在当前样本自身学习并计算：cube.fit_compute(current)
```

参考范围 0–100、四箱时，内部切点是 25、50、75。首尾扩展为 `-inf` 和 `inf`，
因此首尾箱覆盖新样本越界值，并非有限等宽区间。需要严格有限区间时使用
`FixedBinner(np.linspace(0, 100, 5))`。等距不保证每箱人数相同，且会受参考极值影响。

拟合忽略 NaN 和无穷；无有限参考值报错。常量产生一箱；浮点数无法区分的重复边界合并。
`n_bins_` 是实际箱数，`bin_edges_` 返回边界副本，`reference_range_` 保留参考有限范围。
`fit_field` 的用法与等频相同：

```python
BinDimension("score_b", EqualWidthBinner(5), fit_field="score_a")
```

## 自定义：完整边界，不学习样本

```python
import numpy as np
from phl_risk.analysis import FixedBinner

binner = FixedBinner(
    [-np.inf, 0.2, 0.5, 0.8, np.inf],
    labels=["low", "medium", "high", "very_high"],
    precision=3,
)
result = Cube([BinDimension("score", binner)], [Count()]).compute(current)
```

`edges` 是完整边界，至少两个实数，严格递增，不允许 NaN、重复边界或隐式排序。
数组或列表在创建时保存为不可变快照。标签数量必须等于边界数量减一，默认 B1、B2…。
`FixedBinner` 创建后 `is_fitted=True`；`fit()` 为兼容统一接口而验证输入，不改变边界。
Cube 可以直接 compute；也可以 fit_compute。固定方案不需要 fit_field。

有限边界 `[0, 20, 50, 100]` 对应 `[0,20]`、`(20,50]`、`(50,100]`。
范围外值产生缺失标签；Cube 的维度缺失策略默认丢弃这些行，设置
`ComputePolicy(missing=MissingPolicy(dimension="keep"))` 可保留缺失组。
缺失组不区分原始 NaN 和越界；显式传入 ±inf 可覆盖全部数值。

## 共用语义与组合

- 右闭区间；`include_lowest=True` 默认包含第一个端点，False 排除它。
- NaN 保持缺失，不修改用户数据，Series 索引和名称保持不变。
- `precision=6` 只控制 metadata/layout 区间文字，必要时自动增加精度，不改实际边界。
- 两个字段共享自定义区间：分别配置相同 FixedBinner 边界；共享等距边界：相同参考、
  相同配置和 fit_field。
- 可以与 Count、Sum、EventRate、Funnel、AUC/KS 及单样本 totals 组合。
- PSI 的 `binner=` 也支持这两种 transformer；固定边界外样本按 PSI 的 missing 策略
  进入缺失桶或丢弃，比较完整分布建议覆盖 ±inf。

可执行案例与结果见 [完整 Notebook 的第 9.2–9.4 节](../examples/analysis_complete_guide.ipynb)。

## 实现复用关系

三种分箱器共用从原 QuantileBinner 提取的内部实现，而非分别复制分箱代码：

```text
BaseTransformer
└── _NumericBinner（内部实现）
    ├── QuantileBinner：np.quantile 学习边界
    ├── EqualWidthBinner：np.linspace 学习边界
    └── FixedBinner：校验用户完整边界
```

公共层统一数值输入、标签与显示精度校验、边界状态保存、NumPy searchsorted
区间分配、缺失处理、索引保持、边界副本和 metadata 输出。等频与等距还共用
箱数校验和有限参考值提取。拟合在校验通过后才保存状态，失败 refit 保留之前状态。

QuantileBinner 的入参、分位点算法、重复边界策略和结果保持兼容。
_NumericBinner 属于内部实现；外部扩展仍使用公开 BaseTransformer 协议。

## 按客群分别拟合：fit_by

默认 `BinDimension("score", QuantileBinner(5))` 在整个参考样本上学习一套边界，
再按 Cube 的各维度交叉统计。维度排列不表示先分层再分箱。

希望观察各客群内部低分至高分档的风险时，显式指定 `fit_by`：

```python
cube = Cube(
    ["mute_group", BinDimension("score", QuantileBinner(5), fit_by=["mute_group"])],
    [Count(), EventRate("label", name="bad_rate")],
)
result = cube.fit_compute(oot, totals=True)
table = result.layout(
    rows=["mute_group", "metric", "score_bin"],
    totals=True, total_label="总计", bin_labels="interval",
)
```

- 全局分箱适合比较相同分数区间的客群风险；分层分箱适合比较各客群内部的分位档位。
- 每层独立拟合，B1 在不同层可能对应不同区间。重复分数默认合并边界，实际箱数可不同。
- `fit_by` 支持多个原始字段，要求它们同时以未重命名的 ColumnDimension 出现在 Cube 中。
  参数可与 `fit_field`、等距分箱并用；不依赖维度排列顺序。
- `cube.fit(reference).compute(current)` 复用各层参考边界，current 不保证等频。
  新客群明确报错；组内没有有限参考值报错并标明分层键，不自动回退到全局边界。
- 分层字段缺失遵循维度 drop/keep 策略；keep 时缺失键作为独立层。
- `totals=True` 只生成保留所有 fit_by 字段的汇总。本例只有各客群总计，
  没有跨客群同箱汇总或总体总计。显式折叠 fit_by 字段报错。
- 推荐客群与分箱放在同一行轴或列轴。拆在两侧且显示 interval 时，分箱轴取各层区间并集，
  不适用的交叉格为 NaN，不能当成零样本箱。
- `cube.dimensions_[i].metadata()` 的 `groups` 保存分层键 `key` 与独立的 `transform`
  metadata（精确边界、标签、实际箱数），不从四舍五入的显示文本还原边界。

完整可执行案例、全局/分层对照、参考边界复用和多字段分层见
[实践手册 9.5](../examples/analysis_complete_guide.ipynb#s9-stratified)。

## 确认拟合范围：INFO 日志与 explain

存在普通分组、学习型 BinDimension 未设置 fit_by 时，成功 `Cube.fit()` 会为每个符合条件
的分箱维度记录一条英文 INFO 说明，指出使用准备后的参考样本整体拟合，并建议 fit_by。
这不是错误警告，不改变边界、指标或总计。后续 compute/layout 不重复提示；
无普通分组、显式 fit_by 或 FixedBinner 不生成此提示。

库不配置日志 handler、级别或根 logger，默认 INFO 可能不可见。应用可按需启用
`logging.getLogger("phl_risk.analysis")` 的 INFO 日志；用 record.event 等于
`analysis.binning.global_scope` 的过滤器可以只显示这一类说明。

`cube.explain()` 无需拟合或启用日志，直接说明每个分箱维度的 global / within groups /
fixed boundaries 范围，并显示分层总计保留的 fit_by 字段。

临时捕获英文提示、恢复 logger 设置及对照 explain 的可执行案例见
[实践手册 9.6](../examples/analysis_complete_guide.ipynb#s9-binning-scope)。

### 层内等距分箱

`fit_by` 同样适用于 EqualWidthBinner，无需额外开关：

```python
BinDimension("score", EqualWidthBinner(5), fit_by=["mute_group"])
```

每层按自己的参考有限最小值/最大值计算等距切点；人数不保证相等，可能出现空箱。
外尾仍延伸至 ±inf。与层内等频一样，compute 复用参考边界，totals=True 仅保留各层
内部总计。不传 fit_by 则继续全局拟合、保留原有全量总计。

偏斜数据下“等距各箱 8/1/0/0/1 条”与“等频每箱 2 条”的可执行对照见
[实践手册 9.5.1](../examples/analysis_complete_guide.ipynb#s9-stratified-width)。
