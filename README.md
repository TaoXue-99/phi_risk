# phl-risk

面向风控、模型评估和数据分析的声明式多维分析框架。Python ≥ 3.10；运行时依赖 NumPy、pandas、scikit-learn。

```text
Dimension × Measure × Transform → Cube → CubePlan → Engine → CubeResult → Layout
```

Cube 描述分析任务，Engine 执行计算，CubeResult 保存 canonical long 结果，Layout 决定轴如何展示。
源码包名为 **`phl_risk`**，发行包名为 **`phl-risk`**。

## 安装与验证

在仓库根目录运行：

```bash
uv sync --group dev
uv run pytest -q -W error
uv run ruff check src/phl_risk tests examples
uv run python examples/analysis_examples.py
```

也可使用 `python -m pip install -e .` 安装运行时包。
scikit-learn 是正式运行时依赖，负责 AUC/ROC 数值计算；框架保留输入策略与组合编排。

## 优先复用成熟库

| 计算 | 实现 |
|---|---|
| AUC（单独或与其他指标组合） | `sklearn.metrics.roc_auc_score` |
| KS | `sklearn.metrics.roc_curve(drop_intermediate=False)` + `np.max(abs(tpr-fpr))` |
| 参考分位数 / 当前分箱 | `np.quantile` / `np.searchsorted` + pandas 有序类别 |
| Count / Share / EventRate | pandas 共享 groupby sum + NumPy 向量化比例 |
| 结果布局 | NumPy broadcast/transpose/reshape + pandas MultiIndex |

`phl_risk.metrics` 是薄适配层，统一缺失、权重、二元类别和 `on_invalid` 策略，
不维护自写的 ROC 排序、累计分布或 AUC 积分算法，也不调用 sklearn 私有 API。
测试验证三方库参数、边界、共享调用次数以及分析结果，不重新实现三方库本身。

AUC 和 KS 始终独立调用各自的 sklearn 公共函数，不因指标组合切换实现路径。
两者可能重复排序，以保持调用路径清晰；仅完全相同的指标节点（例如重命名的 AUC）复用结果。
使用 `python benchmarks/benchmark_ranking.py` 可分别测量两个适配函数的耗时。

## 分层指标分析

```python
from phl_risk.analysis import Cube, AUC, KS

cube = Cube(
    dimensions=["dt", "user_type"],
    measures=[AUC(target="label", score="score"), KS(target="label", score="score")],
)
result = cube.compute(df)
print(cube.explain())
print(result.data_)       # dt, user_type, metric, value
print(result.shape)       # 每个维度轴的长度 + metric 轴的长度
table = result.layout(rows=["dt"], columns=["user_type", "metric"])
```

`dimensions=[]` 表示全局分析；`measures=None` 默认使用 `Count()`，显式空列表报错。
无状态维度无需 fit。V0.1 正式保证 0–2 个维度；实现使用通用维度序列。

## Reference 分箱与 OOT 交叉分析

```python
from phl_risk.analysis import Cube, BinDimension, QuantileBinner, Count, Share, EventRate

cube = Cube(
    dimensions=[
        BinDimension("score_a", QuantileBinner(n_bins=5)),
        BinDimension("score_b", QuantileBinner(n_bins=5)),
    ],
    measures=[Count(), Share(denominator="all"), EventRate("label", name="event_rate")],
)
cube.fit(train)
result = cube.compute(oot)

# count 的 B1–B5，随后 share 的 B1–B5，随后 event_rate 的 B1–B5。
vertical = result.layout(rows=["metric", "score_a_bin"], columns=["score_b_bin"])
horizontal = result.layout(rows=["score_a_bin"], columns=["metric", "score_b_bin"])
```

两次 layout 不重新执行 Cube，也不修改 `result.data_`。`fit_compute(df)` 是显式 fit + compute 的便利方法。
参考边界可通过 `cube.dimensions_[0].transformer.bin_edges_` 或结果 metadata 查看。

## 整体指标、行列总计与区间展示

总计表示**合并原始样本后重新计算指标**，不是已有单元格的均值。
AUC、KS 不能从每日指标推导整体值；EventRate 也必须合并事件数与有效分母，不能平均各格比例。
因此显式启用预计算：

```python
result = cube.compute(df, totals=True)

# dt × dataset：底部增加每个 dataset 的整体 AUC/KS。
table = result.layout(
    rows=["dt"],
    columns=["metric", "dataset"],
    totals=["dt"],       # 指定要折叠并增加 Total 的维度
    total_label="总计",
)

# 等价地沿用先 layout 再 unstack 的写法：
table = result.layout(totals=["dt"], total_label="总计").unstack(level="dataset")
```

交叉分箱的每个 metric 块都可增加底部总计和右侧总计：

```python
cube.fit(train)
result = cube.compute(oot, totals=True)
vertical = result.layout(
    rows=["metric", "new_score_bin"],
    columns=["old_score_bin"],
    totals=True,             # 为所有分析维度添加总计；metric 轴不汇总
    total_label="总计",
    bin_labels="interval",  # 展示真实拟合边界，如 [-inf, 0.5]、(0.5, inf]
)
```

- 每个 metric 块按各箱 → 总计的顺序显示；右下角是该指标的整体值。
- Count 的整体值是总行数；非空总体的 Share 整体值是 1；EventRate 是整体事件比例。
- `totals=["new_score_bin"]` 只添加该维度的总计，`totals=True` 添加所有维度的总计。
- `bin_labels="interval"` 仅改变展示，canonical 数据、B1/B2 标签、轴域和 learned edges 不变。
  区间文字使用实际 float 边界，不按 precision 四舍五入，避免不同边界显示成相同区间。
- `result.total(over=["dt"])` 直接取出折叠 dt 后的预计算 CubeResult；
  `result.total(over=["dt", "dataset"])` 取出整体结果。
- 总计只使用通过 filters 和全部维度 missing policy 的同一批有效分析行，不让被排除行重新进入分母。
- 分箱转换和过滤只执行一次；总计复用已转换维度，并按额外分组粒度计算指标。
  两个维度会额外计算三个粒度，耗时高于普通 compute；总分组粒度上限为 64。
- `layout()` 不保留、不访问原始样本、不重新执行引擎。未启用 `compute(..., totals=True)` 时请求总计会明确报错。
- 默认 `totals=False`、`bin_labels="code"`，既有展示不变；`shape`、`data_` 不计入总计，
  总计保存在独立的预计算结果中；展示的 `max_cells` 检查包括新增总计。
- 总计标签与维度已有值冲突时会报错，可换用 `total_label=`。

## 表格形式的 explain

```python
cube.explain()                     # Notebook 原生 DataFrame 表格
cube.explain(totals=True)          # 同时说明总计计算计划
print(cube.explain(format="text")) # 终端对齐文本表格，不截断字段
```

表格按 `Section / Item / Description` 展示维度、源字段、拟合状态和边界、指标输入、
引擎、共享聚合数量以及缺失/无效策略。explain 仅解释计划，不执行计算。
默认返回类型由旧版字符串改为 DataFrame；需要字符串时显式使用 `format="text"`。

## 语义约定

| 项目 | V0.1 行为 |
|---|---|
| Count | 当前单元格的行数，不受 label/score 缺失或 Context 权重影响 |
| Share | 单元格行数 / 过滤及维度缺失处理后进入分析的总行数 |
| EventRate | 指定事件的有效 target 数 / 全部有效 target 数；支持非二元事件值及权重 |
| AUC / KS | target 为 0/1；高分预测 1；KS 取累计分布最大绝对差 |
| 指标名称 | `count`、`share`、`event_rate__label__1`、`auc__score`、`ks__score`；支持 `name=` |
| 重名 | 维度输出名和指标名分别检查唯一性；维度名 `metric`、`value` 保留 |
| 顺序 | 维度按声明，指标按声明，普通字段按有效数据首次出现，categorical 按类别顺序 |
| 输入 | 内置变换与计算不修改用户 DataFrame；使用位置编码，支持重复行索引 |
| 结果 | 稀疏 long data + 完整轴域；`shape` 表示逻辑轴域，不等于长表行数 |
| 空组合 | layout 补齐：Count=0；非空总体的 Share=0；EventRate/AUC/KS=NaN |
| 空总体 | 全局 Count=0，其他指标 NaN；普通字段无轴值，参考分箱/声明类别仍保留 |

例如单元格有 10 人、总体 100 人、其中 3 人发生事件且 target 均有效：**Share=10%，EventRate=30%**。

```python
from phl_risk.analysis import AnalysisContext, ComputePolicy, MissingPolicy

context = AnalysisContext(target="label", weight="sample_weight")
cube = Cube(
    dimensions=["dt"],
    measures=[AUC(score="score"), EventRate()],
    policy=ComputePolicy(missing=MissingPolicy(dimension="keep"), on_invalid="nan"),
)
result = cube.compute(df, context=context)
```

Measure 显式 target/weight 覆盖 Context；`None` 表示继承默认值。
权重用于 AUC、KS、EventRate，必须有限、非负；Count/Share 始终是行数语义。
默认维度缺失删除，可设置 `dimension="keep"`，缺失坐标在 AxisSpec 中表示为 `None`。
target 缺失按指标删除；score 缺失及非有限值只从 AUC/KS 删除，不影响其他指标。
单类别、零有效权重或零分母默认 NaN；可选择 `on_invalid="warn"` 或 `"raise"`。
无效配置、缺字段、非法权重始终报错，不被 `on_invalid` 吞掉。
结构性空箱不会触发无效组警告，layout 使用明确的空单元格值。

## 分箱细节

- `fit()` 只使用 reference 中的有限数值，全部无效时报错。
- `duplicates="drop"` 合并重复分位点；常量列产生 1 箱。`duplicates="raise"` 严格报错。
- 两端扩展为 `-inf`、`+inf`，current 越界及无穷进入两端箱，missing 仍为 missing。
- 右闭区间；`include_lowest=True` 包含 `-inf`，设为 False 时仅该最低端点视为缺失。
- 默认标签为有序 `B1…Bk`，实际箱数保存在 `n_bins_`，自定义 labels 长度必须匹配实际箱数。
- `precision` 只控制 metadata 中的区间文字，不修改拟合边界或分箱判定。
- 构造器参数不可变；拟合字段以 `_` 结尾暴露；`bin_edges_` 返回防御性副本。

## 过滤、计划与扩展

```python
cube = Cube(["dt"], [Count(), Share()], filters=[lambda frame: frame["eligible"]])
plan = cube.plan()
print(cube.explain())
result = cube.compute(df, engine="pandas")
```

filters 顺序应用，fit reference 和 compute current 使用相同过滤规则；布尔掩码必须对齐索引，缺失掩码视为 False。
callable 是 pandas 临时入口；`FilterExpression` 提供 `required_columns()` 和 `evaluate(data, backend=...)` 扩展协议。
过滤函数应为纯函数；内置引擎隔离常规 DataFrame 赋值，不保证递归复制 object 单元格里的可变对象。

`CubePlan` 编译指标依赖并合并重复聚合节点。Engine 共用一次 native groupby aggregation，再执行组指标和派生比例。
AUC/KS 共用分组索引和原始字段数组，但分别调用各自的数值函数。
`BaseDimension`、`BaseTransformer`、`BaseMeasure`、`BaseCubeEngine` 都有公共导出；节点细节是 V0.1 私有 API。

`TableLayout` 必须恰好放置所有轴；遗漏、重复或未知轴会报错，绝不隐式求均值。
默认最多物化 1,000,000 个单元格，可通过 `max_cells=` 显式调整。
`data_`、`metadata_` 和 Cube 的维度访问器返回防御性副本；不提供通用 `set_params`。

## 文档与边界

- [逐阶段设计、代码与测试说明](docs/implementation.md)
- [代码审查与验证记录](docs/review.md)
- [完整可运行案例](examples/analysis_examples.py)

V0.1 不实现其他 backend、任务并行、Pipeline、SQL AST、任意派生 DAG、Excel/绘图/格式化系统或通用 select/sort API。
Result 的 pandas 容器是显式边界；未来引擎可以适配相同结果契约。
浮点统计不保证任意精度；极端权重比例小于浮点最小可表示范围时可能舍入为零。
