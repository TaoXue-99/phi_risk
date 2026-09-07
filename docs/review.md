# V0.1 代码审查与验证记录

当前实现完成 Phase 0–11，并进行了功能、数值、状态、布局、扩展和打包审查。
本记录是本地验证快照，不代表 PyPI 发布或远程 CI 状态。

## 本地验证

- Python **3.12.13**；NumPy **2.5.2**；pandas **3.0.5**；scikit-learn **1.9.0**。
- `python -m pytest -q -W error`：**120 passed**，把非预期 warnings 视为失败。
- Ruff lint 与 format 检查通过。
- `examples/analysis_examples.py` 与 `examples/funnel_examples.py` 执行及内嵌 assertions 通过。
- 所有源码通过 Python 3.10 grammar 的 AST 解析检查。
- 离线构建 wheel 和 sdist 成功；wheel 检查并通过从产物导入的 smoke test。
- GitHub Actions 已配置 Python 3.10 / 3.12 / 3.13 测试；本轮未远程执行。

**兼容性验证缺口**：此验证环境没有执行 Python 3.10 或 pandas 2.x 运行时测试。
语法解析、依赖锁文件的版本解析和 CI 配置不能代替实际运行；最低支持版本仍需 CI 验证。

## 审查结果

| 方面 | 结论与证据 |
|---|---|
| API consistency | 统一 fit/compute；常用对象从 analysis 导出；专门 namespaces 同时可用；没有 Cube.transform 或 builder 链 |
| Naming / business neutrality | Dimension/Measure/Axis 分离；Count、Share、EventRate 语义固定；没有 BadRate 等业务名称进入基础层 |
| NumPy axis semantics | canonical metric 轴固定在最后；Layout 自由转置；所有三轴排列及 rows/columns 分割均测试；shape 包括 reference 空箱 |
| pandas efficiency | 一次 groupby，共享归约；没有 iterrows、apply(axis=1)、groupby.apply；AUC/KS 只提取依赖数组，不复制整组的无关列 |
| sklearn lifecycle | 显式学习边界，compute 不 fit；未拟合报错；构造器参数不可变；拟合副本隔离及 pickle roundtrip 通过 |
| Composition / SOLID | BinDimension 组合 Transformer；BaseMeasure 编译节点；BaseCubeEngine 负责 backend；Result 不知道 Cube，不可能回调计算 |
| Typing | Python 3.10 语法；Protocol、ABC、dataclass、Literal、类型别名与 py.typed；本轮未运行完整静态类型检查器 |
| Error handling | 包级异常层次；字段、重复名、无效布局明确拒绝；invalid policy 只处理未定义统计，不吞配置错误 |
| Testing | sklearn AUC 对照、KS 独立阈值参考、固定分组回归、5×5 矩阵、缺失/空数据/权重/自定义扩展及输入不变性 |
| Future engine compatibility | 计划节点没有 pandas lambda；具体 pandas Dimension 和 Result 容器的耦合明确；其他 Engine 通过适配满足契约 |
| Future execution compatibility | Engine 无线程/进程/调度语义；仅对命名 Ratio 依赖排序，没有并行 executor 或通用任务调度系统 |
| State / immutability | frozen config、Plan 快照、原子 refit；DataFrame 与 metadata 对外防御性复制；不承诺深冻结任意 callable 闭包 |
| Performance / copies | 只建临时归约列，组指标使用依赖数组；过滤回调复制为输入保护；结果访问复制为一致性保护；布局才物化笛卡尔积 |
| Premature abstractions | 未实现空 Pipeline、通用 AST、多个 backend、任意 optimizer、格式化/绘图/Excel 系统 |
| God classes | Cube 仅编排，Measure 仅描述；pandas 执行细节集中于唯一 backend，未扩散到 Cube/Result；数值 kernels 独立 |

## 审查中修复的问题

1. 分位点的浮点舍入使整数边界产生约 `7e-15` 的差异：测试改为容差比较，未通过边界舍入掩盖问题。
2. 极端相反有限数的分位点插值可能溢出：只在极端区间内缩放后插值再还原。
3. sklearn 对极端有限分数差分可能溢出：适配层仅在跨度超出 float64 时使用 NumPy 有序编码，保留所有顺序和 ties；普通输入不增加排序。
4. 全局权重缩放可能让其他组或 missing target 的大权重压低当前组有效权重：EventRate 按每组有效 target 缩放，分子分母复用同一尺度。
5. 空 labels 列表原先可能被当作“未指定”：改为明确区分 None 和空序列，错误长度报错。
6. group metric 原先切片整组 DataFrame：改为预提取 target/score 数组，只按组切片依赖数组。

## 性能冒烟检查

本机单次执行，随机种子 7，Count + Share + EventRate，无权重；不作为正式 benchmark 或 SLA：

| 输入行数 | 分组 | compute 墙钟时间 | 验证 |
|---:|---|---:|---|
| 1,000,000 | 10 组 | 约 0.017 秒 | Count 总和 1,000,000；共享 3 个归约节点 |
| 1,000,000 | 100 组 | 约 0.014 秒 | Count 总和 1,000,000；共享 3 个归约节点 |
| 1,000,000 | 5×5 参考分箱 | 约 0.036 秒 | shape=(5,5,3)；Count 总和 1,000,000 |

这里只检查主干是否存在明显重复扫描或规模错误；正式性能比较需要预热、重复测量、内存统计和机器信息固定。
AUC/KS 采用固定的独立调用路径，可能重复排序。`benchmarks/benchmark_ranking.py`
分别测量两个公共适配函数的耗时，并验证与直接调用 sklearn 的结果一致。
不再提供共享曲线分支，也不使用此前共享曲线的加速比描述当前实现。

## 成熟库复用审查

- sklearn 从 dev dependency 移到正式 runtime dependency；wheel 安装自动带入该依赖。
- 删除本地 binary_curve 排序/累计实现和 AUC 手工积分。
- AUC 始终直接使用 roc_auc_score；KS 始终使用完整 roc_curve；移除共享曲线调度器和 sklearn auc 积分分支。
- 不用 scipy 的 KS 检验替代这里的 weighted binary separation：其统计检验接口与本项目权重语义不同。
- EventRate 的求和、加权缩放、比例，分箱的 quantile/searchsorted，布局的 transpose/reshape 已是 NumPy 原生计算，继续保留。
- 保留 pandas 原生共享聚合；不把 Count/Share/EventRate 改成逐组第三方函数调用。
- 新增 `tests/test_library_integration.py`，验证实际公共 API 调用、数据清理、权重传递、完整阈值、独立调用路径、相同指标去重次数、异常透传和 NumPy 分箱与 pandas.cut 等价性。
- 保留固定预期与 KS 独立参考，防止仅使用相同函数自我对照。框架的缺失/无效策略测试继续执行。

参考公共 API：[roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)、
[roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)、
[pandas.cut](https://pandas.pydata.org/docs/reference/api/pandas.cut.html)。

## 明确保留的限制

- 总计是 compute(totals=True) 的额外计算，Layout 仅显示预计算值；不能从一个未预计算总计的 CubeResult 恢复整体 AUC/KS。
- explain 默认返回 DataFrame；字符串调用方需迁移至 format="text"。

- 不保证超过两个 Dimension 的公共兼容性，尽管内部没有写死一维/二维字段分支。
- 只支持 pandas Engine 和 pandas Result 容器；没有并行执行。
- Plan 的 frozen 外壳不是任意插件对象的深度不可变系统，调用者应只读使用它。
- 不提供 set_params/通用 clone；配置替换使用新对象；没有 Pipeline、select/sort/format。
- 数值为浮点统计，极端到无法表示的权重比例仍可能舍入为零。
- 结构性空组合保存在轴域并由 Layout 补齐；canonical data 本身是稀疏的。
- 支持 Ratio 引用命名指标及依赖排序；未实现任意公式或 SQL 编译。

## 总计与展示改进审查

- 异常树明确标出 InvalidMetricError 同时继承 PhlRiskError 和 ValueError，且不属于 AnalysisError。
- measures 的 field 已统一改名 validate_field。
- 过滤与维度变换只执行一次，所有总计使用原分析有效样本总体；不会因折叠缺失维度而重新纳入被删行。
- AUC/KS 总计调用对应 sklearn 函数；EventRate 按 pooled target/weights 计算，不平均各格比率。
- 总计默认关闭；两维增加三个分组粒度；max_cells 计算包含总计坐标，标签碰撞明确报错。
- 布局区间映射保留 canonical 标签和数据；区间使用实际拟合浮点边界，空箱、缺失组及总计顺序有测试。
- 已有 layout().unstack('dataset') 用法可保留；也可直接布局 rows=['dt'], columns=['metric','dataset']。


## 漏斗扩展审查

- 数量使用 pandas 原生共享归约，条件使用向量表达式；没有逐行 Python 计数。
- Sum 默认传播未知数量，显式 missing="zero" 才按零；条件未知不计入 CountWhere。
- 动态阶段和指定转化均生成普通指标；参考分箱、日期分层、二维布局复用现有接口。
- 比率总计按有效原始行重新汇总分子分母，测试使用不等组规模防止误用平均率。
- 原有 AUC/KS 固定调用 sklearn 公共函数的路径未改变，原有回归测试继续执行。
- README、implementation 与新增 funnel 文档同步了公开 API、空组、权重和缺失语义。

- 现有 `examples/demo_cube_cross.ipynb` 修复 oot 定义顺序；27 个非空代码单元在全新 IPython 会话按顺序运行通过，原有保存输出保留。
