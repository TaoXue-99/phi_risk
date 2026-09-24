# Count / Sum 指标内筛选

Count 和 Sum 使用相同的可选关键字参数 where。它筛选组内参与当前指标的记录，
不改变其他指标、维度域或分箱学习数据。Cube(filters=...) 则作用于整个分析。

```python
from phl_risk.analysis import Cube, Col, Count, Sum

condition = Col("category").isin(["A", "B"]) & (Col("age") >= 18)
cube = Cube(
    dimensions=["dt"],
    measures=[
        Count(name="全部数量"),
        Count(where=condition, name="筛选数量"),
        Sum("amount", where=condition, name="筛选金额"),
        Sum("amount", where=Col("amount") > 10, name="大于10金额"),
    ],
)
result = cube.compute(df, totals=True)
result.layout(totals=True)
```

## 条件类型

| 场景 | 写法 |
|---|---|
| 单类别 | `Col("category") == "A"` |
| 多类别 | `Col("category").isin(["A", "B"])` |
| 数值阈值 | `Col("amount") > 10` |
| 数值区间 | `(Col("amount") >= 10) & (Col("amount") < 100)` |
| 缺失 / 非缺失 | `Col("category").isna()` / `.notna()` |
| 或 / 非 | `(条件1) \| (条件2)` / `~(条件)` |

使用带括号的 &、|、~，不要使用 Python 的 and、or、not。
普通比较中的缺失保留为未知，组合条件后最终未知不匹配。
因此 `~(Col("category") == "A")` 不会选中缺失类别；选缺失请显式使用 isna()。
不隐式转换列类型，数值比较使用数字，字符串类别使用字符串。
Count 统计行数，不去重；Sum 的计算列仍须为实数数值类型，筛选列可以是类别或数值。

## 缺失、零匹配与诊断

- where=None 保持原有行为；没有记录匹配时，Count、Sum 返回 0。
- Sum 先应用条件，再对匹配行执行原有 missing 策略。
- 默认 missing="propagate"：匹配行存在缺失即返回 NaN，诊断为 missing_propagated。
- missing="zero"：匹配行的缺失按零求和。
- 被排除行的缺失或无穷值不影响求和；匹配行的无穷值仍报错。
- diagnostics 的 affected_rows 仅统计匹配行中的缺失；input_rows 仍是该分组分析总行数，不是匹配行数。
- 总计在合并样本上应用相同条件计算，不平均各组结果。

## 兼容与执行

旧 Count(name) 和 Sum(column, name, missing) 的位置参数保持不变，where 仅支持关键字传入。
条件计数统一使用 Count(where=condition, name=...)，不再提供单独的条件计数类。
Count 默认名称仍为 count，Sum 仍为 sum__字段；声明同类多指标时请用 name 区分，重复名称仍报错。
本次不向 AUC、KS、EventRate、Share 增加 where。

Engine 复用现有 Predicate 表达式和字段依赖检查；一次聚合内共用相同条件的布尔掩码，
不逐组重新筛选。总计的不同聚合粒度仍各自执行原有聚合流程。

可运行且含结果断言的示例：[conditional_measures.py](../examples/conditional_measures.py)。
