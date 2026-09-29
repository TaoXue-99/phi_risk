# ESMM + MMoE 本地验收

日期：2026-09-29。分支：`codex/esmm-mmoe-experiment`。
这是本地验收，未执行远程 CI、发布、推送或合并。

## 交付

- 可独立使用的 ESMMMoE 与 ESMMLoss，支持纯连续输入。
- torchkeras 专用 runner、精确 AUC、样本加权损失及 epoch/batch scheduler。
- ESMM baseline、PyTorch state_dict 记录与校验加载；Experiment 核心记录 schema 不变。
- 小型完整 API 指南（Markdown、Notebook、脚本）。
- 500 万～1000 万行指南（Notebook、脚本），默认两种训练路径各 3 epochs。
- 单独的 CPU 深度学习 CI job；远程尚未运行。

## 环境

Python 3.12.13，macOS ARM64，CPU 4 个计算线程。
uv 安装并记录 extras：pytorch、torchkeras、tensorboard；Notebook 验证工具加入 dev group。

| 包 | 验收版本 |
|---|---|
| torch | 2.14.0 |
| torchkeras | 3.9.9（固定适配版本） |
| accelerate | 1.15.0 |
| torchmetrics | 1.9.0 |
| tensorboard | 2.20.0 |
| numpy / pandas / sklearn | 2.5.2 / 3.0.5 / 1.9.0 |
| LightGBM | 4.7.0 |
| OptBinning / OR-Tools / protobuf | 0.21.0 / 9.11.4210 / 5.26.1 |

## 测试与数学对照

- 原有功能：**508 passed**，包含 LightGBM、OptBinning、数据处理与原 Experiment。
- 新增功能：**28 passed**，包含两种真实训练、TensorBoard、数据分块、配置消费、
  可选依赖隔离、权重保存与篡改校验、预处理恢复、标签和类别边界。
- `ruff check src/phl_risk tests examples benchmarks` 通过。
- `ruff format --check src/phl_risk tests examples benchmarks` 通过。
- wheel/sdist 构建成功；使用现有 0.10.0 开发版本号，未发布。
- 与用户原 GitHub ESMM 源码对照 8 组：分类列 0/2、BatchNorm false/true、
  normalization none/divide；迁移相同 state_dict 后，前向概率和全部参数梯度一致。
- 零向量归一化增加 epsilon；这项边界行为有意改进，避免原除零问题。

## 500 万行实测

实际数组行数 **5,000,000**，不是少量数据重复拼接。
train=3,000,000，valid=1,000,000，test=1,000,000。
原始磁盘数组共 **150,000,384 字节（143.1 MiB）**。
内存映射仍有操作系统页缓存、batch、网络和精确 AUC 的额外内存需求。

第一次脚本实测（固定随机种子 2026）：

| 路径 | 实际 epochs | 最佳 epoch | fit 秒数 | 训练样本/秒 | test CTR AUC | test CTCVR AUC |
|---|---:|---:|---:|---:|---:|---:|
| 原生 PyTorch | 3 | 2 | 18.17 | 495,441 | 0.741067 | 0.774499 |
| torchkeras | 3 | 2 | 22.48 | 400,275 | 0.741067 | 0.774499 |

吞吐量分母包含 fit 中的验证、监控与保存；torchkeras 额外计算 train AUC。
这不是通用硬件 benchmark。模型是小型表格网络，不代表大型网络的速度。
两条路径最终都使用最佳验证权重重新评估，测试集不参与选模。

数据位于 `experiments/esmm_mmoe_large/dataset_5000000/`；Run 与 TensorBoard
位于同项目 `esmm_mmoe/runs/`。这些运行产物被 Git 忽略，未复制进源码或 wheel。
Notebook 重跑产生新的 Run，速度可能略有变化；Notebook 保存本次实际输出。
1000 万行是可配置入口，本轮不声称已实际训练 1000 万行。

## Notebook

两份 Notebook 都用项目内核顺序执行，保存真实输出：

- `examples/modeling/esmm_mmoe_complete_guide.ipynb`：两种训练、配置派生、第三次调参、
  Hydra、TensorBoard、完整 DataFrame 恢复及条件 CVR 示例。
- `examples/modeling/esmm_mmoe_large_scale_guide.ipynb`：实际 500 万行，两种训练各 3 epochs、
  耗时/AUC 曲线、模型和训练统计重载、所有 artifact 校验。

## 已复现的上游兼容边界

单进程执行所有后端测试在本机触发 SIGSEGV。原生系统崩溃栈显示多个 libomp
运行时混用，分别来自 PyTorch/其他 Python 数值包和 Homebrew（LightGBM）。
LightGBM 先导入和 PyTorch 先导入都未可靠解决。单独运行各后端成功。
参考 [LightGBM FAQ](https://lightgbm.readthedocs.io/en/latest/FAQ.html) 和
[上游 issue #6595](https://github.com/lightgbm-org/LightGBM/issues/6595)。

采用独立 Python 进程完成全部测试：

```bash
.venv/bin/python -m pytest -q -W error \
  --ignore=tests/examples/test_esmm_guide.py --ignore=tests/modeling/models \
  --ignore=tests/modeling/training --ignore=tests/modeling/experiment/pytorch
.venv/bin/python -m pytest -q -W error tests/examples/test_esmm_guide.py \
  tests/modeling/models tests/modeling/training tests/modeling/experiment/pytorch
```

不是跳过失败测试，也不是宣称原始单进程 pytest 已通过。实际业务建议后端使用
独立进程/内核，Experiment 在无需加载训练后端的情况下比较历史。
未修改系统动态库，未全局设置 OpenMP 绕过变量。

OptBinning 的旧 OR-Tools 与新版 TensorBoard 的 protobuf 约束不同；binning extra
显式约束 protobuf 5.26.x，uv 选择兼容的 TensorBoard 2.20.0。
仅对对应测试精确过滤 protobuf C 扩展的已知导入弃用警告；不关闭产品 warnings。

## 范围

验证了单进程 CPU/float32。GPU、混合精度、梯度累积、完整断点续训未专项验收；
多进程 runner 明确拒绝执行。当前 loss 不含样本权重/缺失标签 mask。
独立 reviewer 启动因账号额度失败，未提供审阅；完成了主执行者审阅与上述实际验证。


## Linux 支持补充（2026-09-29）

- 两个 CLI 都增加 `--device cpu|cuda`，在 optimizer 创建前迁移模型。
- CUDA 不可用时提前失败；权重跨设备恢复使用容差比较。
- Linux x86_64 / Python 3.12 的 torchkeras、TensorBoard、Hydra 依赖解析成功。
  wheel-only 检查因 Hydra 的 antlr4-python3-runtime 4.9 只有源码发行而失败；
  正常允许源码包的解析通过。这不是 Linux 运行时测试。
- Ubuntu CPU CI 增加 Python 3.12/3.13 及两种后端 CLI 冒烟，尚未远程执行。
- 本机新增深度学习测试为 **29 passed, 2 skipped**；两个 skip 是实际 CUDA 测试。
- Mac 上已有 500 万行 CPU 实测；Linux CPU / NVIDIA GPU 实际训练仍需目标环境验收。
