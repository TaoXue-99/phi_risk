# Analysis diagnostics 框架预览

保留当前 NaN 和 missing/on_invalid 策略，额外记录原因，不自动修改输入数据。
本轮是可审阅的最小框架，尚未覆盖全部指标原因。

```python
result = cube.compute(df, totals=True)
result.layout()  # 原有数值与结构不变
report = result.diagnostics()
result.total(over=["segment"]).diagnostics()
```

索引为各维度和 metric；无维度时只有 metric 索引。
列固定为 reason、field、input_rows、affected_rows、dependency、message。
数量列使用 nullable Int64；不适用的数量为空，不假定为零。
一个结果可能有多个原因，索引允许重复，不应直接相加 affected_rows。
返回新 DataFrame，修改报告不会影响原结果。维度名称与诊断列名可相同。

## 原因码与对应说明

以下四种是当前框架实际会返回的原因码。原因码供程序筛选；message 是具体上下文说明，
当前代码输出英文，中文含义见下表。不要通过匹配 message 文本判断原因。

| reason | 中文说明 | 当前触发条件 | 建议检查 |
|---|---|---|---|
| `missing_propagated` | 输入缺失传播导致指标为空 | Sum 使用 `missing="propagate"`，该组被求和字段至少有一个缺失值 | 根据 field 定位原始列，检查该分组的缺失记录；affected_rows 是缺失行数 |
| `upstream_missing` | 依赖的上游指标为空 | Ratio（包括漏斗转化率）的分子或分母指标为 NaN | 根据 dependency 找到同一分组的上游指标诊断，继续追查原因 |
| `zero_denominator` | 分母为零，无法计算比率 | Ratio（包括漏斗转化率）的分母等于 0，结果为 NaN | 检查 dependency 指向的分母指标、原始数量及该分组是否存在转化基数 |
| `reason_not_recorded` | 结果为空，但尚未记录详细原因 | 实际计算结果中存在 NaN，且该结果没有具体诊断记录 | 结合指标规则检查输入；这表示诊断覆盖不足，不表示没有问题，也不是对数据原因的判断 |

`zero_denominator` 当前只对 Ratio 路径提供详细记录。EventRate 等其他路径即使也因
分母为零而返回 NaN，现阶段仍可能显示 `reason_not_recorded`，不能据此推断真实原因。

总计使用自身计算过程的证据；没有实际计算空值时返回同结构空表。
同一比率可能同时记录上游缺失和零分母；报告不是每个指标仅保留一个原因。

## 诊断字段含义

| 字段 | 含义与当前填充规则 |
|---|---|
| 维度 + metric（索引） | 定位分组及指标；无维度时仅有 metric。可用 `report.reset_index()` 转为普通列，但需注意维度名称与诊断列名冲突 |
| reason | 上表中的原因码 |
| field | 原始数据字段名；当前在 Sum 缺失传播诊断中填写 |
| input_rows | 过滤及维度缺失处理后，参与该分组分析的行数；不是整个原始数据集行数，也不是该指标非缺失样本数 |
| affected_rows | 当前在 Sum 诊断中表示该字段的缺失行数；其他原因未统计时为空 |
| dependency | 上游指标名称，不一定是原始列名；当前在 Ratio 诊断中填写 |
| message | 具体说明文字，包含字段、缺失数量或依赖名称等上下文 |

`reason_not_recorded` 当前仅提供 reason 和 message，其他诊断字段为空。
诊断表中的空字段表示“不适用或尚未采集”，与原指标的 NaN 不是同一含义。

## 示例：1000 行中一条缺失

假设同一 segment 有 1000 行，before 全为 1，after 中 999 行为 1、一行为 NaN，
使用默认 `Funnel(["before", "after"])`：

| metric | 计算结果 | reason | input_rows | affected_rows | dependency |
|---|---|---|---:|---:|---|
| before数量 | 1000 | 无诊断行 | — | — | — |
| after数量 | NaN | missing_propagated | 1000 | 1 | — |
| before→after转化率 | NaN | upstream_missing | 1000 | — | after数量 |

对应 message 示例：

- `after: 1 missing values; missing='propagate'.`：after 有一条缺失，当前策略将缺失传播到求和结果。
- `Dependency 'after数量' is missing.`：转化率依赖的 after数量为空。
- `Denominator is zero.`：分母为零。
- `This calculation has not recorded a detailed cause.`：该计算路径尚未记录详细原因。

可以对返回的 DataFrame 进行筛选，无须重新计算指标：

```python
report = result.diagnostics()
missing_inputs = report.loc[report["reason"].eq("missing_propagated")]
upstream_issues = report.loc[report["reason"].eq("upstream_missing")]
```

## 内部结构

PandasEngine 使用已有缺失计数及依赖值收集不可变 Diagnostic 记录；
CubeResult 保存记录；diagnostics() 转成带维度和指标索引的 DataFrame。
内部按 canonical 行位置关联，不保留原始 DataFrame，不重新计算指标。
其他 Engine 不提供记录仍兼容。Diagnostic 暂不作为公开导入接口。

## 尚未实现

AUC/KS、EventRate、PSI 和溢出的详细原因仍待接入。
layout 补出的无样本单元格不在本报告中；空报告不能证明 layout 没有空值。
暂不提供原始行定位、诊断筛选和自动告警；on_invalid="raise" 仍直接抛错。

完整运行示例：[analysis_diagnostics.py](../examples/analysis_diagnostics.py)。
