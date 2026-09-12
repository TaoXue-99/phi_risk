# DataPrep：显式 Stateless / Fitted 生命周期

本次基于 0.3.0 增量修改，保留现有类名、构造参数、第三方算法、审计和组合器入口。
未改依赖、版本号、`_data.py`、异常、Analysis 或 DataQuality。

## 继承与调度

```text
BasePrepStep                         DataFrame、行/索引保护、run、audit
├── StatelessPrepStep                configuration → transform
│   ├── ToNumeric
│   │   └── ToDatetime
│   ├── ValueMapper
│   ├── Clip
│   ├── LogTransform
│   └── LogitTransform
└── FittedPrepStep                    configuration → fit → state_ → transform
    ├── SklearnStep
    │   ├── MissingImputer
    │   └── KBinsStep
    └── OptBinningStep
```

BasePrepStep 不定义 fit/_fit/训练属性，也不使用 requires_fit。StatelessPrepStep 提供
fit_transform 便利别名，但只调用 transform；DataPrep 不调用该别名，只直接 transform。
FittedPrepStep 保留 `clone → _fit → checked transform → commit_fitted`，训练失败不覆盖旧状态。

DataPrep 仍统一提供 fit/transform/fit_transform/run/get_feature_names_out/save/load。
fit 时 clone 所有配置、逐步执行；只有 isinstance(step, FittedPrepStep) 才拟合。
下一步的 fit 接收前面所有步骤转换后的训练数据。即使全为 stateless，DataPrep 仍需 fit。
组合器拥有 feature_names_in_/out_ 和 step_feature_names_out_，不将这些属性写进 stateless 步骤。
transform/run 均检查每一步输出列名和顺序；dtype 可按转换规则改变。

## 数值步骤

```python
from phl_risk.data_prep import Clip, LogTransform, LogitTransform

# 三者均可直接 transform/run，无须 fit。
clip = Clip(columns=["score"], lower=0, upper=1)
log = LogTransform(column="income", output_column="log_income")
logit = LogitTransform(column="prob", output_column="prob_logit", eps=1e-6)
```

- 输入须为实数数值列；字符串先经 ToNumeric。布尔和复数明确拒绝，null 传播。
- Clip 的边界是有限实数或 None；lower≤upper。保持原整数精度，分数边界需要时提升为浮点。
  lower/upper 都为 None 时不裁剪。audit 记录各列 clipped_count。
- LogTransform 是自然对数。非 null 值必须 >0，不自动平移、裁剪或学习参数；+inf 保持 +inf。
- LogitTransform 先裁剪到 [eps, 1-eps]，再计算 log(x)-log1p(-x)，避免不必要的比例溢出。
  eps 必须在 (0, 0.5)，且 1-eps 在浮点数中可区分于 1。越界值、无穷值均裁剪，audit 记录数量。
- output_column=None 或与输入列相同表示覆盖；新的列名表示追加。与其他已有列冲突会报错。
  新增列会出现在 after snapshot/output_columns 中，原始输入不改动。

可运行混合例子：[data_prep_lifecycle.py](../examples/data_prep_lifecycle.py)。

## 文件清单与目的

| 文件 | 目的 |
|---|---|
| `src/phl_risk/data_prep/_base.py` | 拆分三个基类；公共检查/审计留在 Base，事务式拟合迁入 Fitted |
| `src/phl_risk/data_prep/_prep.py` | 按类型顺序调度；保存/检查逐步骤 schema；旧计划给出显式 refit 提示 |
| `src/phl_risk/data_prep/_audit.py` | 在原字段末尾增加可选 kind，不改变原位置参数 |
| `src/phl_risk/data_prep/_result.py` | audit_frame 增加 kind 展示列 |
| `src/phl_risk/data_prep/steps/convert.py` | ToNumeric/ToDatetime 进入 Stateless，删除 flag |
| `src/phl_risk/data_prep/steps/mapping.py` | mapping 作为配置使用，删除 _fit、mapping_ 和状态回退 |
| `src/phl_risk/data_prep/steps/sklearn.py` | 仅将继承改为 Fitted，保留原 adapter 行为 |
| `src/phl_risk/data_prep/integrations/optbinning.py` | 仅将继承改为 Fitted，保留原 artifact/solver 检查 |
| `src/phl_risk/data_prep/steps/numeric.py` | 新增 Clip、LogTransform、LogitTransform 与针对性审计 |
| `src/phl_risk/data_prep/steps/__init__.py` | 导出三个新增数值步骤 |
| `src/phl_risk/data_prep/__init__.py` | 导出两个新生命周期基类及三个数值步骤，保留旧导入 |
| `tests/data_prep/test_framework.py` | 迁移原有两个测试插件的基类 |
| `tests/integrations/test_risk_workflow.py` | 迁移自定义 Prep 插件；Quality 插件保持原协议 |
| `tests/data_prep/test_lifecycle.py` | 显式生命周期、混合调度、schema、行保护、原子 refit、配置隔离及持久化 |
| `tests/data_prep/test_numeric.py` | 数学对照、边界/null/非法输入、碰撞、新列审计、整数精度及 roundtrip |
| `examples/data_prep_lifecycle.py` | 完整混合流程，展示 stateless 不创建训练属性及 audit.kind |
| `README.md` | 简单 API 和两类生命周期入口 |
| `examples/README.md` | 新示例导航、fit 使用对照和迁移入口 |
| `docs/data_lifecycle_review.md` | 标明 0.2 历史快照并链接当前验收，保留历史记录 |
| `docs/data_lifecycle.md` | 更新主文档中的生命周期、mapping、持久化和插件约定 |
| `docs/data_prep_refactor.md` | 本交付记录、文件清单、迁移和后续建议 |
| `CHANGELOG.md` | Unreleased 增量变化与迁移说明 |

MissingImputer 和 KBinsStep 通过 SklearnStep 间接进入 Fitted，无需修改各自实现。
PrepSnapshot 和 StepResult 保持不变；PrepResult 数据保护逻辑保持不变。

## 兼容性变化

现有 `DataPrep([...]).fit(...).transform(...)`、run 和已有步骤构造器保持兼容。
所有原有测试在仅迁移测试插件继承后通过。

以下使用需要迁移：

1. 单独的 ToNumeric/ToDatetime/ValueMapper 不再提供 fit 或训练列名属性，直接 transform/run。
   可调用 fit_transform 便利别名，但不会产生任何 state_。
2. 自定义有状态 Prep 必须从 BasePrepStep 改为 FittedPrepStep，实现 _fit；
   无状态插件改为 StatelessPrepStep，删除 requires_fit。
3. DataPrep.fit 会 clone 配置，原配置后续修改不改变已拟合计划；独立 ValueMapper 直接使用当前 mapping。
4. audit_frame 增加 kind 列；依赖精确列集合的调用者需调整。PrepAudit 原位置参数兼容，kind 默认 None。
5. 旧 DataPrep artifact 没有逐步骤 schema，加载后运行会明确要求 refit/save，不会自动从 current 学习。
   包含旧 DataPrep 的 Workflow 同样需重新 fit。新对象 roundtrip 和单独 FittedPrepStep 保存能力保留。
6. 无状态步骤的单独 save 仍受共有 DataEstimator artifact 的拟合前置条件限制；请封装为 DataPrep 后保存。
   本次不扩大到配置 artifact 协议重构。

## 验证

本地 Python 3.12.13、现有完整 binning 环境；未安装新的依赖或临时解释器。

- 修改前基线：`pytest -q -W error`，226 passed。
- 生命周期拆分及已有步骤迁移后：226 passed。
- 全部实现与新增用例完成后：269 passed（净增加 43 个参数化测试案例），无 skip。
- `ruff check .` 通过；`ruff format --check src/phl_risk tests examples benchmarks` 通过（129 文件）。
- 新增混合流程示例运行通过；git diff --check 通过。
- 现有 sklearn/OptBinning 对照、权重、稀疏输出、feature names、audit、Workflow 和 artifact 测试继续通过。

## Follow-up suggestions

这些问题只记录，未扩大本次重构范围：

- DataEstimator 的 save/load 仍假定对象已拟合；以后可独立设计 stateless configuration artifact，
  不应为了保存配置再次制造 _is_fitted_。
- 重复索引下，len + index.equals 无法识别“相同索引标签之间交换行”的插件错误；
  若未来需要严格核验样本身份，应单独定义 row identity 协议。当前插件仍必须保证不重排样本。
- 任意第三方插件可自行修改内部状态或 object 单元格中的可变对象；深层不可变性需要专门的插件约束，
  本次保留现有防御性复制边界，没有引入运行沙箱或每次 transform 全对象复制。
