# 动态漏斗分析

`Funnel` 把有序阶段展开成普通 Measure，沿用 Cube 的分组、分箱、过滤和总计。
数量可以是逐行 0/1，也可以是已汇总的数值；默认统一求和。

## 最小用法

```python
import pandas as pd
from phl_risk.analysis import Cube, Funnel

df = pd.DataFrame({
    "dt": ["2026-09-01", "2026-09-02"],
    "戳额": [100, 10], "有额": [60, 8], "发标": [30, 6], "提现": [15, 4],
})
funnel = Funnel(stages=["戳额", "有额", "发标", "提现"], rates="adjacent")
result = Cube(["dt"], funnel.measures()).compute(df, totals=True)
table = result.layout(totals=True, total_label="总计")
```

`funnel.measures()` 只生成配置，不读取数据、不 fit、不计算。上例展开为：

| 指标 | 计算 |
|---|---|
| 戳额数量 / 有额数量 / 发标数量 / 提现数量 | 对相应列分别 Sum |
| 戳额→有额转化率 | 有额数量 / 戳额数量 |
| 有额→发标转化率 | 发标数量 / 有额数量 |
| 发标→提现转化率 | 提现数量 / 发标数量 |

返回 tuple，可与其他指标组合，例如 `[*funnel.measures(), Count()]`。
转化率输出 0–1 比例，不自动乘 100。总计先汇总原始有效行上的数量再相除，
例如整体戳额→有额为 `68/110`，不会取两个每日比例的平均值。

## 任意阶段数、任意指定转化

`stages` 顺序代表漏斗顺序，名字不得重复。n 个阶段的数量如下：

| rates | 生成的率 |
|---|---|
| `"adjacent"`（默认） | 相邻阶段，n−1 个 |
| `"from_first"` | 首阶段到各后续阶段，n−1 个 |
| `"both"` | 上述两种合并去重；n≥2 时 2n−3 个 |
| `[]` | 仅数量 |
| `[Transition(...), ...]` | 仅指定的转化，按指定顺序 |

单阶段只有数量；空阶段报错。可直接输入转化前后的阶段名：

```python
from phl_risk.analysis import Transition

funnel = Funnel(
    ["戳额", "有额", "发标", "提现"],
    rates=[
        Transition(before="戳额", after="有额"),
        Transition(before="戳额", after="提现", name="整体提现转化率"),
    ],
)
```

默认率名称为 `前阶段→后阶段转化率`。未知阶段、逆向、自身转化、重复边和输出重名均报错。
如果使用 Stage，Transition 引用 Stage 的显示名，不是底层源列名。

## 列名与显示名分离、条件计数

```python
from phl_risk.analysis import Col, CountWhere, Stage, Sum

funnel = Funnel([
    Stage("浏览", Sum("view_count", missing="zero")),
    Stage("点击", Sum("click_count", missing="zero")),
    Stage("确认", Sum("confirm_count", missing="zero")),
])

# 逐行状态，阶段数量由条件计数。下游状态仍属于上游阶段。
conditional = Funnel([
    Stage("浏览", CountWhere(Col("status").isin(["view", "click", "done"]))),
    Stage("点击", CountWhere(Col("status").isin(["click", "done"]))),
    Stage("确认", CountWhere((Col("status") == "done") & (Col("valid") == 1))),
])
```

Stage 支持 Sum、CountWhere、Count，Funnel 将它们的输出名统一为 `阶段名数量`，覆盖原有 `name`。
`aggregation` 当前只支持 `"sum"`；复杂计数通过显式 Stage 表达。
Col 支持 `== != > >= < <=`、`isin`、`isna`、`notna`，以及带括号的 `& | ~` 组合；
不要使用 Python 的 `and/or/not`。比较缺失值会保留未知状态，取反后仍未知，
CountWhere 最终只计 True；需要计缺失时使用 `isna()`。这些条件也可用于 Cube 的 filters。

## 分箱与普通分层使用相同指标

```python
from phl_risk.analysis import BinDimension, QuantileBinner

cube = Cube([BinDimension("score", QuantileBinner(5))], funnel.measures())
# 自己学习分箱并统计
result = cube.fit_compute(detail_df, totals=True)

# 或：参考样本学习边界，应用于当前样本
cube.fit(reference_df)
result = cube.compute(current_df, totals=True)
result.layout(totals=True, total_label="总计", bin_labels="interval")
```

上述 `detail_df/reference_df/current_df` 需要 score 以及所配 Stage 的源列。
参考边界不随 current 改变，current 中各箱人数不保证相等。
不分箱时维度可用 `["dt"]`、`["dt", "channel"]` 或 `[]`。
只有日期和已汇总数量的表没有个体分数，不能还原个体等频分箱。
二维布局可使用 `rows=["metric", "dt"], columns=["channel"]`；行列总计与总总计均重新计算。

## 通用 Ratio 与数据口径

```python
from phl_risk.analysis import Ratio

measures = [
    Sum("view_count", name="views"),
    Sum("click_count", name="clicks"),
    Ratio(numerator="clicks", denominator="views", name="ctr"),
]
```

Ratio 引用同一个 Cube 中的指标输出名；需要显式提供这些指标。支持引用其他 Ratio，
计划按依赖顺序计算，展示仍按声明顺序。缺失引用和循环依赖在 plan/explain 阶段报错。
它不接受任意公式字符串，也不对输入逐行相除再平均。

| 边界 | 行为 |
|---|---|
| Sum 源列 | 实数数值或布尔 dtype；不自动转换数字字符串；拒绝复数和无穷 |
| Sum 缺失 | 默认 `missing="propagate"`：组内任意缺失则数量未知；`"zero"` 显式按 0 |
| 空组合 / 空总体 | Sum、CountWhere、Count 为 0；Ratio 为 NaN |
| Ratio 分母为 0 | 遵循 ComputePolicy.on_invalid：默认 NaN，可 warn/raise |
| 有限输入的求和 / 相除溢出 | 按 on_invalid 返回 NaN、警告或报错 |
| Ratio 上游缺失 | 传播 NaN，不把未知数量自动解释成 0 |
| 权重 | Sum 和 CountWhere 不读取 AnalysisContext.weight；按源数量或命中行数统计 |
| 去重 | 不隐式按 user_id 去重；用户需先确定每行含义与统计单位 |
| 漏斗关系 | 假定阶段数量单位一致且后阶段来自前阶段；不验证逐行嵌套，也不裁剪大于 1 的率 |

通用 Sum 可用于负数数量，通用 Ratio 允许负的非零分母；漏斗所需的非负、嵌套业务口径由输入保证。
当前数值结果统一为 float64，超大整数超过其精确范围时不保证逐单位精确。

完整合成数据与断言示例：[funnel_examples.py](../examples/funnel_examples.py)。
执行 `uv run python examples/funnel_examples.py`；测试 `uv run pytest tests/test_funnel.py -q -W error`。
