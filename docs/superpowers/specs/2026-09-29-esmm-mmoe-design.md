# ESMM + MMoE 与 Experiment

用户已在对话确认本设计，并要求直接安装依赖、实现及编写全面指南。

## 边界

- `models/esmm_mmoe` 提供原生 PyTorch 网络与损失，不依赖训练器、Hydra 或 Experiment。
- `training/torchkeras` 提供专用 KerasModel/runner，保留原生 fit、callbacks 和 TensorBoard。
- Experiment 只增加 baseline 与原生 PyTorch 保存适配，不接管训练。
- PyTorch、torchkeras、TensorBoard 使用独立可选 extras；初始化无需这些后端。
- Python >=3.12，沿用现有 numpy/pandas/sklearn 约束。

## 网络与任务

共享网络（分类 embedding + 连续输入）→ MMoE → 两个 tower。
输出 dict：ctr、cvr、ctcvr；ctcvr = ctr * cvr。支持空分类输入。
显式构造参数，外部负责设备迁移。ESMMLoss 对 ctr/ctcvr 加权 BCE，
校验二元标签和转化链路 y_ctcvr <= y_ctr，返回总损失及分量。
保留原网络 ELU、dropout、可选 BatchNorm、L2 归一化及专家/gate bias。

## 训练与指标

batch = (x_cat, x_cont, y_ctr, y_ctcvr)。原生循环与 torchkeras 共用网络/损失。
专用 runner 按样本聚合 loss，按全集计算 AUC；单类别 AUC 省略并给出原因。
scheduler 支持显式 epoch/batch 更新；默认 epoch，避免改变原 StepLR 语义。
第一版验收单进程 CPU；不宣称分布式和混合精度已验证。
torchkeras 不做全局 monkey patch；TensorBoard writer 生命周期由调用方管理。

## 配置与持久化

`initialize(method="esmm_mmoe")` 生成 baseline，已有文件不覆盖。
配置分为 model/loss/optimizer/scheduler/data/train/torchkeras。
输入维数、分类基数由预处理 schema 确定，在 start_run 前解析为完整配置。
record_pytorch 保存 CPU state_dict、显式模型配置和后端版本；
load_pytorch 加载至调用者提供的原生网络，不自动执行配置中的 Python。
只承诺恢复推理；完整 optimizer/RNG/sampler 断点续训不在本轮范围。

## 指南与验证

examples/modeling 提供中文 Markdown + 可执行脚本 + Notebook。
固定种子合成转化链路，先划分再拟合预处理，训练/验证/测试互不混用。
两条路径展示初始化、配置派生、TensorBoard、训练、最佳模型重评估、
权重/预处理保存、恢复原始 DataFrame 推理、compare/compare_params。
验证公式、梯度、输入边界、尾 batch 加权、scheduler、单类别指标、
artifact 快照/篡改检测、可选依赖隔离、Notebook 顺序执行和既有完整测试。

## 用户追加：500 万～1000 万行指南

用户在执行中明确要求大数据规模示例。新增独立大数据脚本与 Notebook，默认 500 万行，
支持参数改为 1000 万行；分块生成三份 mmap 原始数组，训练区间增量拟合标准化，
以训练均值填补缺失，IterableDataset 直接产出 batch，按块重排训练顺序。
实际验收 500 万行、60/20/20 切分，两条训练路径各 3 epochs。
保留小型完整 API 指南，方便学习、单元测试与故障定位。
