# ESMM + MMoE：Experiment、torchkeras 与原生 PyTorch 全面指南

配套文件：

- [可顺序执行的 Notebook](esmm_mmoe_complete_guide.ipynb)：逐步解释、两种训练、调参、比较及恢复，保留执行输出。
- [完整 Python 脚本](esmm_mmoe_experiment.py)：可直接运行，所有数据均为合成。
- [架构说明](../../docs/superpowers/specs/2026-09-29-esmm-mmoe-design.md)。

## 1. 安装与运行

在仓库根目录执行：

```bash
uv sync --extra torchkeras --extra tensorboard --extra hydra --group dev
uv run --extra torchkeras --extra tensorboard python examples/modeling/esmm_mmoe_experiment.py
```

如果同时保留 LightGBM、OptBinning 等现有环境，使用 `uv sync --all-extras --group dev`。
uv 的同步会按选定的 extras 整理环境；后续 `uv run` 也需要保留所需 extras，或在已同步环境中使用 `.venv/bin/python`。

纯 PyTorch 路径：

```bash
uv sync --extra pytorch --group dev
uv run --extra pytorch python examples/modeling/esmm_mmoe_experiment.py \
  --backend native --no-tensorboard
```

该路径不导入 torchkeras、accelerate、torchmetrics 或 TensorBoard。
仅使用网络时只需 PyTorch；`pytorch` extra 附带 YAML，供实验配置读取。

脚本默认 2,400 行、6 epochs，两条路径分别形成一个 Run。可以修改：

```bash
uv run --extra torchkeras --extra tensorboard python examples/modeling/esmm_mmoe_experiment.py \
  --root experiments/my_esmm --rows 5000 --epochs 15 --backend both
```

重复运行会创建新 Run，不覆盖此前实验。baseline 已存在时保持原样。
Notebook 选择本项目 `.venv` 内核，从头运行全部单元格；无需下载业务数据。

## 2. 三个独立层次

| 层次 | 责任 | 可以独立使用吗 |
|---|---|---|
| `modeling.models.esmm_mmoe` | PyTorch 网络与 ESMM 损失 | 可以，不需要 Experiment 或 torchkeras |
| `modeling.training.torchkeras` | ESMM 专用 runner，沿用 KerasModel.fit | 可以，不需要 Experiment |
| `modeling.experiment` | 配置快照、Run、artifact、查询与比较 | 可以记录任意原生训练代码 |

没有统一 Trainer 工厂、自动训练入口或全局 torchkeras monkey patch。示例的
`train_native`、`run_one` 等是可编辑的教学代码，不是必须遵守的框架接口。

## 3. 转化链路与模型

令 C 表示第一阶段事件，V 表示后续转化：

- `ctr = P(C | x)`；
- `cvr = P(V | C, x)`；
- `ctcvr = ctr × cvr = P(C,V | x)`。

输入是分类 ID `int64[N,C]` 与连续特征浮点数 `[N,D]`。
分类 embedding 与连续特征拼接，经共享 MLP、MMoE 专家和两个 gate，进入两个任务塔。
模型输出具有 `ctr`、`cvr`、`ctcvr` 三个键的字典，每个值为 `[N,1]`。

```python
from phl_risk.modeling.models.esmm_mmoe import ESMMMoE, ESMMLoss

net = ESMMMoE(
    num_continuous=6,
    categorical_cardinalities=[4, 3],  # 每列自己的基数，包含未知类别 ID
    embedding_dim=4,
    share_dim=24,
    base_dim=12,
    expert_units=8,
    num_experts=4,
)
loss_fn = ESMMLoss(ctr_weight=1.0, ctcvr_weight=1.0)
```

`ESMMLoss` 用全空间的 CTR、CTCVR 标签计算加权 BCE，不对所有样本直接监督 CVR。
标签只能为 0/1，必须满足 `y_ctcvr <= y_ctr`；向量 `[N]` 和列向量 `[N,1]` 均可。
未到观察窗口的标签、缺失标签、负采样或样本权重需要另外明确业务口径，当前 loss 不隐式处理。

纯连续模型使用 `categorical_cardinalities=[]`，输入空 `int64[N,0]`。
当前要求至少一个连续特征。BatchNorm 可选；启用时训练 batch 不能只有一个样本，
应明确调整 batch_size/drop_last，不能通过偷偷复制样本掩盖问题。

## 4. 数据切分与预处理

脚本生成六个连续特征和 channel/device 两个分类特征，按转化链路生成标签。
先将原始 DataFrame 划分为 train/valid/test，再仅使用 train：

1. 学习连续特征的中位数填补和 StandardScaler；
2. 学习分类词表，保留 ID 0 表示缺失或未知类别；
3. 固定特征名称、顺序与每列类别数；
4. 对 valid/test 仅执行 transform。

```python
partitions = split_data(make_data(2400))
state = fit_preprocessor(partitions["train"], [f"x{i}" for i in range(6)], ["channel", "device"])
cat, cont = transform_features(partitions["valid"], state)
```

示例函数定义见脚本和 Notebook。真实业务可以替换为自己的 DataPrep 工作流，
只要训练和推理使用同一份已拟合状态、特征顺序及类别映射即可。
不能分别对验证集/测试集 fit，也不能直接把任意字符串编码后的整数当作连续特征。
真实时间链路通常应使用时间切分；本指南的随机分层只是合成数据演示。

## 5. 初始化与 baseline

```python
from phl_risk.modeling.experiment import Experiment
from phl_risk.modeling.experiment.configuration import read_yaml_config

exp = Experiment(root="experiments/esmm_mmoe_guide")
space = exp.initialize(method="esmm_mmoe", comparison_partitions=["valid", "test"])
cfg = read_yaml_config(space.config_dir / "baseline.yaml")
```

配置包含 model/loss/optimizer/scheduler/data/train/torchkeras。`num_continuous` 默认
为 null，必须从准备好的 schema 解析，不能直接使用未解析模板构造模型。

```python
cfg["model"]["num_continuous"] = len(state["continuous_features"])
cfg["model"]["categorical_cardinalities"] = [
    len(state["vocabularies"][column]) + 1 for column in state["categorical_features"]
]
cfg["data"]["continuous_features"] = state["continuous_features"]
cfg["data"]["categorical_features"] = state["categorical_features"]
```

初始化本身不导入 PyTorch、torchkeras、TensorBoard 或 YAML。baseline 不是模型质量保证。
已有方法使用 `exp.open_method('esmm_mmoe')`。ESMM 的初始化不接受 LightGBM 专用的
objective/metric 参数；需要描述任务时传给 `start_run(task=...)`。

## 6. 配置派生与 Hydra

纯 Python 使用 `deepcopy(cfg)`，再改 optimizer、网络或损失权重，避免多个方案共享字典。
Hydra 也可直接复用：

```python
from phl_risk.modeling.experiment.adapters.hydra import compose_config

variant = compose_config(
    config_dir=space.config_dir,
    config_name="baseline",
    overrides=["optimizer.lr=0.0005", "model.num_experts=6", "loss.ctcvr_weight=2.0"],
)
# 训练前在 variant.config 中解析 num_continuous、categorical_cardinalities 和特征配置。
# 然后 with space.start_run(config=variant) as run: ...，保留 overrides 来源。
```

本指南的 `run_one` 接受已解析的普通字典。要保留 Hydra 来源，可在自定义记录代码中
直接将 `ConfigSource` 传给 `start_run`，用其 `.config` 构造对象。
配置快照在 start_run 时确定，之后不要更改实际训练参数；若需要改参，应开启新 Run。
模型配置不自动执行 Python 对象：优化器、自定义 loss 和 callbacks 由代码明确构造。

## 7. DataLoader 合同

一个 batch 为 `(x_cat, x_cont, y_ctr, y_ctcvr)`。原生 PyTorch 和 torchkeras 使用相同合同。
训练集可以 shuffle；验证/测试集不 shuffle、不 drop_last，最终评估也包括完整训练集。

DataLoader 的随机生成器使用固定种子。示例每次重新构建网络、优化器和 loader，
两条路径不会接着使用上一条路径训练后的权重。跨设备或库版本不承诺逐位相同结果。

## 8. 原生 PyTorch 路径

```python
optimizer = torch.optim.Adam(net.parameters(), lr=cfg["optimizer"]["lr"])
for x_cat, x_cont, y_ctr, y_ctcvr in train_loader:
    optimizer.zero_grad()
    predictions = net(x_cat, x_cont)
    targets = {"ctr": y_ctr, "ctcvr": y_ctcvr}
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()
```

完整 `train_native` 包括设备迁移、梯度裁剪、scheduler、验证、TensorBoard、早停、
最佳权重保存和恢复。只使用 valid 选模，训练结束后才评估 test。
两个示例驱动器都支持 `--device cpu`（默认）和 `--device cuda`；
CUDA 不可用时立即报错。Mac MPS 不在本轮验收范围。

## 9. torchkeras 路径

```python
from phl_risk.modeling.training.torchkeras import ESMMKerasModel

trainer = ESMMKerasModel(
    net,
    loss_fn,
    optimizer=optimizer,
    lr_scheduler=scheduler,
    scheduler_interval="epoch",
    max_grad_norm=1.0,
)
history = trainer.fit(
    train_data=train_loader,
    val_data=valid_loader,
    epochs=cfg["train"]["epochs"],
    patience=cfg["train"]["patience"],
    monitor="val_loss",
    mode="min",
    ckpt_path="best.pt",
    plot=False,
    quiet=True,
    cpu=True,
)
```

`fit` 使用 torchkeras 原生接口，返回 DataFrame，结束时加载最佳权重。
专用 runner 处理多输入、损失分量、按样本加权的 epoch loss、整集合 AUC。
默认 scheduler 按 epoch 调用，保持原模型脚本语义；按 batch 调用必须明确设置 interval。
ReduceLROnPlateau 需要验证指标，使用自定义 validation callback，不放进参数为空的 step 路径。

当前支持单进程训练；多进程会明确报错，避免错误聚合指标或重复写入实验文件。
本轮验证 CPU、float32、无梯度累积。混合精度/梯度累积仍由原生 fit 暴露，使用前需专项验证。
网络需要自行改变时，可以直接替换原生训练代码，无需继承本项目 Trainer。

## 10. TensorBoard 与监控

```bash
uv run --extra tensorboard tensorboard --logdir experiments/esmm_mmoe_guide
```

每个 Run 的 `tensorboard/` 是独立日志目录。原生路径直接调用 SummaryWriter；
torchkeras 路径用示例中的 `TensorBoardHistory` callback 写入 epoch 指标。
训练框架本身还支持其他 callbacks 和绘图选项。

writer 由调用方创建，在 try/finally 中 flush/close，Experiment 不替用户关闭资源。
完成后将 event 文件归档为校验和保护的 `tensorboard.zip`；实时目录本身不是校验 artifact。
`--no-tensorboard` 可完全跳过导入和写日志。不要在每个 batch 保存模型 artifact。

## 11. 把训练放进 Experiment

```python
with space.start_run(
    config=cfg,
    name="native",
    metadata={"training_framework": "native"},
    task={"objective": "ESMM weighted BCE", "metric": ["ctr_auc", "ctcvr_auc", "loss"]},
) as run:
    # ...用户自己的原生循环，或 trainer.fit(...)
    # ...使用选定权重重新评估完整 train/valid/test
    run.log_metrics(metrics)
    run.log_json("history", history.to_dict(orient="list"))
    record_pytorch(run, net, model_config=net.get_config())
```

完整脚本另外保存特征描述、数据指纹、训练框架版本、预处理状态、缺失指标原因和最佳 epoch。
异常会保留 failed Run 并抛出；completed 只说明记录上下文完成，不代表模型达到质量要求。

## 12. 正确比较结果

最终指标由 `evaluate_esmm(net, loader, loss_fn)` 返回 `(metrics, undefined_reasons)`。
它保持模型的设备和调用前的 train/eval 状态。loss 按样本数加权，AUC 在全集上计算，
不会平均 batch AUC。单类别 AUC 不存在时省略该字段并返回原因。

```python
comparison = space.compare(
    metrics=["ctr_auc", "ctcvr_auc", "loss"],
    partitions=["valid", "test"],
    params=["optimizer.lr", "model.num_experts"],
    metadata=["metadata.training_framework"],
    fields=["best_epoch"],
)
print(space.compare_params(native_run.run_id, keras_run.run_id))
```

metadata 使用记录中的完整路径。与 LightGBM 比较 CTCVR AUC 时，必须保证其标签、
测试样本和观察窗口一致。不同损失权重下总 loss 的口径不同，不应直接据此排名；
优先比较 AUC、同口径分任务 loss。CVR 应在 C=1 子集上评估，不能把联合概率拿去替代条件概率。
本指南默认不生成 CVR 指标，可以在恢复预测后显式圈选并计算。

## 13. 恢复模型和原始 DataFrame 推理

```python
saved = space.get_run(run_id)
net = ESMMMoE(**saved.read_json("model_config"))
net = load_pytorch(saved, net).eval()
state = joblib.load(saved.artifact_path("preprocessor"))
predictions = predict_frame(net, new_frame, state)
```

`predict_frame` 定义在配套示例里。加载权重前校验 artifact，采用 `weights_only=True`；
不会通过配置自动导入类或加载整个 pickled 模型对象。预处理 joblib 仅加载自己生成且可信的文件。
恢复时使用保存的特征顺序和词表，不重新 fit。未知分类值映射到 ID 0，缺少必要列明确报错。

`record_pytorch` 保存瞬时 CPU state_dict，之后继续训练或修改网络不会改变 artifact。
它也适用于其他原生 nn.Module，调用方负责创建兼容网络并提供正确结构配置。
加载不替用户调用 `.eval()`，设备及模式由调用方控制。

## 14. 权重恢复与断点续训

本轮 artifact 足以恢复推理，并不包含完整训练状态。完整续训还需 optimizer、scheduler、
epoch、随机数状态、数据采样进度和混合精度 scaler 等。用户可以使用 `run.log_artifact`
显式保存自己的 checkpoint；加载旧权重重新训练也应创建新 Run，并记录父 Run 的 ID。

## 15. 替换成真实数据

1. 定义转化链路、观察窗口与有效标签；检查 `y_ctcvr <= y_ctr`。
2. 指定连续列、分类列以及标签映射。
3. 按时间/业务边界切分；仅 train 拟合预处理。
4. 从训练 schema 解析网络输入维数和类别数。
5. 决定 valid 监控指标及方向，不用 test 选模。
6. 根据需要选择原生 PyTorch 或 torchkeras；两者都显式保存最终配置。
7. 保存最佳权重、预处理和特征信息；在独立对象上做重载预测对照。

## 16. 范围与参考

这次保留原实现的数学结构，并移除了 cfg/device/业务路径耦合；零向量归一化增加 epsilon。
不承诺旧业务 checkpoint 自动迁移，旧文件还需匹配特征、结构参数和预处理。
当前数据接口不包含序列、多模态、样本权重或缺失标签 mask。
精确 AUC 会保存 O(N) 的预测/标签；超大数据集可换成自己定义的指标实现。

- [原始网络](https://github.com/TaoXue-99/ModelMatrix/blob/main/models/dl/essm_mmoe/model/esmm.py)
- [torchkeras 官方仓库](https://github.com/lyhue1991/torchkeras)
- [原有 Experiment 设计](../../docs/lightgbm_experiment.md)

版本与实际验收结果见 [验收记录](../../docs/esmm_mmoe_review.md)。

## 17. 500 万～1000 万行大数据版

[大数据 Notebook](esmm_mmoe_large_scale_guide.ipynb) 和
[大数据脚本](esmm_mmoe_large_scale.py) 默认真实生成 **5,000,000** 行不同的随机合成数据，
按 300 万 / 100 万 / 100 万划分，两条训练路径各训练 3 epochs。
修改 `--rows 10000000` 可使用 1000 万行。

```bash
uv run --extra torchkeras --extra tensorboard python examples/modeling/esmm_mmoe_large_scale.py
uv run --extra torchkeras --extra tensorboard python examples/modeling/esmm_mmoe_large_scale.py \
  --rows 10000000 --epochs 3 --batch-size 4096 --chunk-size 100000
```

设计区别：

- 分块生成三份紧凑 .npy 数组，使用 mmap 按批读取；500 万行约 143 MiB 磁盘。
- 连续特征通过 train 区间的 `StandardScaler.partial_fit` 拟合，使用训练均值填补缺失。
  小型指南采用中位数填补，两个指南的预处理口径有所不同。
- 类别词表由合成生成器预先声明；真实数据需仅从训练区间建立业务词表。
- IterableDataset 直接产出完整 batch，训练按块随机重排，避免全量行索引和逐样本 Python 调用。
- 单进程、默认 CPU（可选 CUDA）、float32、4 个 CPU 计算线程；函数结束恢复原来的线程数。
- 保存完整数据 SHA256、行数、切分、训练时间、epoch 耗时、吞吐量和模型 artifact。
- AUC 仍精确计算，因此需要 O(N) 分数存储；mmap 不等于训练进程没有额外内存开销。

吞吐量是“实际访问的训练样本数 / fit 总耗时”，包含验证和监控。
torchkeras 默认还记录 train AUC，原生循环只在训练结束后完整评估，耗时不应理解为纯框架性能比较。
大数据示例会显式记录所选设备、batching、epoch scheduler 和模型演示尺寸；这些最终值全部写入 Run。
数据保存在 `experiments/esmm_mmoe_large/dataset_5000000/`，不提交 Git。
复用数据时校验指纹；参数不同或已有不完整数据会报错，请选择新路径。

## 18. 本机 macOS 的 OpenMP 兼容边界

本轮在 macOS 上复现：同一 Python 进程同时执行 PyTorch 与 LightGBM，可能因两套
OpenMP 动态库发生 SIGSEGV。系统崩溃栈落在 PyTorch 和 Homebrew 的 libomp 之间；
两种 import 顺序都不是本机可靠的解决办法。分别运行两套后端没有此问题。
这是已知上游问题，参见 [LightGBM FAQ](https://lightgbm.readthedocs.io/en/latest/FAQ.html)
和 [LightGBM issue #6595](https://github.com/lightgbm-org/LightGBM/issues/6595)。

建议 LightGBM 和 PyTorch 使用独立进程或各自的 Notebook 内核，训练完成后再用
不需要加载后端的 Experiment 读取、比较记录。本项目不改写系统动态库，也不设置
`KMP_DUPLICATE_LIB_OK` 掩盖冲突。安装两种后端不意味着需要在同一进程导入它们。

本地全量测试按以下两组独立进程执行，涵盖全部测试，而非跳过旧功能：

```bash
.venv/bin/python -m pytest -q -W error \
  --ignore=tests/examples/test_esmm_guide.py --ignore=tests/modeling/models \
  --ignore=tests/modeling/training --ignore=tests/modeling/experiment/pytorch
.venv/bin/python -m pytest -q -W error tests/examples/test_esmm_guide.py \
  tests/modeling/models tests/modeling/training tests/modeling/experiment/pytorch
```

另外，OptBinning 0.21 所用 OR-Tools 9.11 约束 protobuf 5.26.x；与 TensorBoard 一起安装时，
uv.lock 选择 TensorBoard 2.20.0。binning extra 明确约束 protobuf，避免解析器为新版 TensorBoard
静默退回更老 OR-Tools。protobuf 5.26.x 的 Python 3.12 C 扩展有已知导入弃用警告；
仅对应 TensorBoard 集成测试对这个精确的上游警告设置过滤，不全局关闭 warnings。


## 19. Linux CPU 与 NVIDIA CUDA 部署

Mac 用于开发；Linux CPU 和单张 NVIDIA GPU 都有显式运行入口。
建议先使用 Python 3.12。Linux CPU CI 同时配置 Python 3.12/3.13，
覆盖原生 PyTorch、torchkeras、Experiment 和磁盘数据完整流程。
本机无法执行 Linux/CUDA；这些 CI 配置尚未在远程运行，不能视为 Linux 验收结果。

在 Linux 项目根目录创建独立环境（不复制 Mac 的 `.venv`）：

```bash
uv venv --python 3.12 .venv-linux
# CPU 服务器：明确选择 CPU wheel，避免下载 CUDA 运行库。
uv pip install --python .venv-linux/bin/python 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-linux/bin/python -e '.[torchkeras,tensorboard,hydra]' pytest
.venv-linux/bin/python examples/modeling/esmm_mmoe_large_scale.py --device cpu --backend both --rows 5000000
```

GPU 服务器使用独立环境：先通过 `nvidia-smi` 检查驱动，依据
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
选择与驱动兼容的 CUDA wheel 索引，在上面的 torch 安装命令中替换 CPU 索引。
不同服务器可能需要不同 CUDA 构建，不将某一 CUDA 版本写死到通用库依赖中。
uv 的索引与后端配置见 [uv 官方 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/)。

```bash
.venv-linux/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
CUDA_VISIBLE_DEVICES=0 .venv-linux/bin/python examples/modeling/esmm_mmoe_large_scale.py --device cuda --backend both --rows 5000000
# 小型完整指南同样支持 --device cuda。
.venv-linux/bin/python -m pytest tests/examples/test_esmm_guide.py -k cuda -q
```

最后一条会执行两条后端的 CUDA 训练与跨设备权重恢复测试；没有 NVIDIA GPU 时会跳过
两条训练测试，跳过不代表通过 GPU 验收。先用 `--rows 1000 --epochs 1` 验证环境，
再运行 500 万或 1000 万行。生产部署需固定实际验收的 torch/CUDA 版本和环境依赖；
以上是安装入口，不是已在目标服务器验收的环境锁定文件。
使用该独立环境的 Python 执行，避免另一次默认 `uv sync` 改变所选 torch 构建。

模型在创建 optimizer 前迁移设备；原生循环搬运每批 tensor，torchkeras 通过
Accelerate 搬运。最佳权重保存后仍可恢复到 CPU，Experiment artifact 不绑定 CUDA。
为避免 Accelerate 的进程级状态干扰，切换 CPU/CUDA 时启动新 Python 进程或重启内核。
当前支持单进程、单 GPU、float32；不使用 `torchrun` 启动本示例。多卡 DDP、AMP 和
梯度累积需要独立扩展与验收。大数据 loader 固定 `num_workers=0`，避免 Linux fork
或多 worker 重复读取；后续并行读取需要先实现分片。
