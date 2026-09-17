# Modeling Plan Layer

第一阶段仅声明、resolve 和校验。`ModelPlan` 与 `DataPlan` 是平行核心，
前者约束后者；两者均不持有 DataFrame，不进行训练、切分或转换。
本模块只导入标准库和包内模块，不增加依赖。

## 目录与公开入口

```text
src/phl_risk/modeling/
├── __init__.py              # 仅重导出 ModelPlan / DataPlan
├── _utils.py               # 名称、列元组和 JSON 描述工具
├── goal/
│   ├── __init__.py
│   ├── _base.py            # ModelingGoal
│   ├── classification.py   # BinaryClassification
│   ├── regression.py       # Regression
│   ├── causal.py           # CausalEffect
│   └── survival.py         # Survival
├── strategy/
│   ├── __init__.py
│   ├── _base.py            # ModelStrategy
│   ├── tree.py             # LightGBM
│   └── deep_learning.py    # MLP
└── plan/
    ├── __init__.py
    ├── _objective.py       # ObjectiveOptions
    ├── _requirements.py    # RoleRequirements
    ├── model.py            # ModelPlan 和内部 resolver
    ├── data.py             # DataPlan
    ├── role.py             # RoleSpec
    ├── feature.py          # FeatureSpec
    └── split.py            # BaseSplitter、四种 splitter、PartitionSpec、SplitSpec
```

`goal` 导出 ModelingGoal 和四种 Goal；`strategy` 导出 ModelStrategy、LightGBM、MLP。
`plan` 导出两个 Plan、ObjectiveOptions、RoleRequirements、三个一级 Spec、
PartitionSpec、BaseSplitter 和四种 Splitter。不向包根 `phl_risk` 增加重导出。
没有建立 BasePlan：两个 Plan 的共享逻辑不足以支持一层公共继承。

## 契约与 resolve

| 对象 | 责任 |
|---|---|
| ModelingGoal | 学习问题的 name、family、required_roles、optional_roles；不接收真实列名 |
| ModelStrategy | 技术路线、execution_family、data_family、支持的 Goal/objective 语义与权重/类别能力 |
| ModelPlan | 校验 Goal + Strategy，解析 objective、执行族、数据族与角色需求 |
| RoleRequirements | 解析后的 required/optional 语义角色集合 |
| RoleSpec | 动态 semantic role → tuple[str, ...] 列绑定 |
| FeatureSpec | 有序 numerical/categorical 列声明 |
| SplitSpec | splitter + partitions；根据 splitter 的 partition_mode 联合校验 |
| DataPlan | roles + features + split；自校验和对 ModelPlan 的联合校验 |

Resolver 按以下顺序工作：

1. 检查 Goal/Strategy 类型以及 Strategy 是否声明支持 Goal family。
2. 取 Goal 有序 objective_families 与 Strategy supported_objective_families 的交集。
   根据 objective_role_requirements 过滤 Strategy 不支持的加权目标。
3. 交集第一个 objective 是 default；未指定时 selected 等于 default，
   指定时必须在 available 中。没有匹配时明确失败。
4. 合并 Goal 必需角色和 selected objective 的额外角色；从 optional 中排除 required。
   Strategy 不支持 weight 时移除 optional weight，若必需 weight 则失败。
5. 固化 ObjectiveOptions、RoleRequirements、execution_family、data_family。

没有按具体 Goal/Strategy 类名进行分派。自定义扩展使用 frozen dataclass 和
ClassVar tuple/frozenset 能力声明；`objective_role_requirements` 为
`tuple[tuple[str, frozenset[str]], ...]`。能力声明应保持不变。
`validate()` 会重新 resolve 并检查核心已解析契约是否发生漂移。

| Goal + Strategy | available（依优先级） | default | 必需角色 |
|---|---|---|---|
| BinaryClassification + LightGBM / MLP | binary_logloss、weighted_binary_logloss | binary_logloss | target；加权目标额外要求 weight |
| Regression + LightGBM / MLP | mse、mae | mse | target |
| CausalEffect + 普通 LightGBM / MLP | 不支持 | — | Goal 语义为 outcome、treatment |
| Survival + 普通 LightGBM / MLP | 不支持 | — | Goal 语义为 duration、event |

Objective 名称是本模块语义标识，不能直接当作第三方后端参数传入。
`weighted_binary_logloss` 声明使用样本权重的学习目标；本阶段不计算 loss。
CausalEffect 的 `causal_effect` 和 Survival 的 `cox` 仅是最小语义声明，
当前没有对应内置 Strategy。测试通过独立的测试 Strategy 验证它们的角色契约。
MLP 当前只支持数值特征，不隐式编码类别；LightGBM 声明支持类别特征。
`supports_custom_objective=False`，本阶段不接收可调用 objective。

## 使用

```python
from phl_risk.modeling.goal import BinaryClassification
from phl_risk.modeling.strategy import LightGBM
from phl_risk.modeling.plan import (
    ModelPlan, DataPlan, RoleSpec, FeatureSpec,
    SplitSpec, PartitionSpec, HashSplitter,
)

model_plan = ModelPlan(BinaryClassification(), LightGBM())
data_plan = DataPlan(
    roles=RoleSpec(
        target="1m_30", sample_key=["user_id", "dt"], time="dt",
        weight="sample_weight",
    ),
    features=FeatureSpec(
        numerical=["age", "income", "beh_score"],
        categorical=["education", "channel"],
    ),
    split=SplitSpec(
        splitter=HashSplitter(key=["user_id"], random_state=2026),
        partitions=PartitionSpec(train=0.7, test=0.3),
    ),
)
data_plan.validate()
data_plan.validate_against(model_plan)
print(model_plan.describe())
print(data_plan.describe())
```

完整可运行脚本：[modeling_plan.py](../examples/modeling_plan.py)。

## Role / Feature 规则

RoleSpec 接收 `**roles`，没有固定 target/outcome 等字段列表。内部复制并按 role 名排序，
使用 MappingProxyType 保存 `bindings`；查询方式为 `roles.bindings["target"]`。
列字符串/list/tuple 均归一化为 tuple。sample_key 可有多列，其他角色当前恰好一列。
不传 time 表示没有时间角色；显式空列表、None、空白字符串均报错。
同一列可以承担多个角色；单个角色绑定内的重复列会报错，不静默去重。
角色名称是否被当前学习问题接受，由联合校验决定。

FeatureSpec 将输入复制为不可变 tuple，验证组内不重复、两组不重叠、
列名非空且为字符串；允许空特征组。`all` 返回 numerical + categorical。
不会根据角色自动删掉特征，也不在本阶段推断 dtype 或转换数据。

`DataPlan.validate_against(model_plan)` 先校验两侧声明，然后一次性报告缺失角色、
多余角色、数据族不匹配和类别特征能力不匹配。
不检查真实列存在性、样本唯一性、缺失率、数据泄漏或训练标签取值。

## Split 声明

| Splitter | 声明 | partition_mode |
|---|---|---|
| HashSplitter | key 非空列元组；非负整数 random_state，默认 2026 | ratios |
| RandomSplitter | 非负整数 random_state，默认 2026；可选单列 stratify | ratios |
| ColumnSplitter | 已有分区字段 column | values |
| TimeSplitter | 显式 time 列；按升序时间及 partition 声明顺序分配比例 | ratios |

HashSplitter.key 与 sample_key 完全独立，可以仅按 user_id 将多次 observation
绑定在同一分区。当前没有执行 hash，也没有定义最终哈希键编码协议。

PartitionSpec 接收 `**partitions`，内部保存复制后的只读有序 mapping `definitions`。
至少包含 train，分区名为非空字符串；可以增加 calibration、holdout 等任意名称。
重复关键字由 Python 调用语义直接拒绝。比例模式要求每个值为 (0, 1] 内数值，
排除 bool，使用 fsum + isclose 检查总和为 1，容差 1e-9。
数值有限性在 PartitionSpec 构造时检查，比例语义在 SplitSpec 检查。

```python
from phl_risk.modeling.plan import ColumnSplitter, PartitionSpec, SplitSpec

split = SplitSpec(
    splitter=ColumnSplitter(column="dtype"),
    partitions=PartitionSpec(train="train", test="test", oot="oot"),
)
```

ColumnSplitter 支持字符串、整数、有限浮点数、bool 源值；不要求总和为 1，
不同分区不能匹配相同源值（包括 Python 等值的 1 / True / 1.0）。
不支持 None 和复合匹配条件。TimeSplitter 仅声明升序比例切分；
日期区间、边界同时间样本处理、缺失时间处理和舍入规则留待执行层明确。
分区顺序有语义，to_dict 保留插入顺序；交换顺序会改变声明。

## 不可变、异常与序列化

Plan/Spec 和内置 Goal/Strategy 使用 frozen dataclass；集合采用 tuple/frozenset，
mapping 为复制后只读视图。to_dict 返回可独立修改的纯 JSON 兼容值，
describe 返回字符串且不打印。没有承诺 pickle、hash 或 from_dict 重建协议。
角色集合的输出排序固定，objective 和特征保持声明顺序。

异常沿用现有体系：`PhlRiskError → ModelingError → PlanError → CompatibilityError`。
PlanError 同时继承 ValueError；避免为每个 Spec 增设相似异常。
对象构造时即校验；两个 Plan 和 SplitSpec 另提供显式 validate()，成功返回 None。

## 下一阶段需明确的边界

- 语义 objective 到后端参数/loss 的映射，以及训练权重语义。
- Hash key 编码、缺失值、稳定性版本；Random、Time 的分层与边界策略。
- 源字段值不属于任一 Column 分区时的行为。
- Causal / Survival 专用 Strategy 的估计目标、假设与进一步 objective 选择。
- 声明 schema 版本和反序列化协议，应在真正消费 manifest 时确定。

本次没有实现 MultiTask 空壳、Layout、DataBuilder、Engine、训练、调参、
预测、Artifact、评估、数据转换或任何真实 split，也没有接入 LightGBM/PyTorch。
