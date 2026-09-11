# Data lifecycle 0.2 验收记录

日期：2026-09-11。本记录是本地验证，不代表远程 CI 或发布已完成。

## 完成范围

Task 0 仓库审计已完成；Task 1–9 分别保存 Git checkpoint；Task 10 包含 README、
数据生命周期协议、版本限制、CI 和本记录。新增 12 类质量检查、7 类 Prep、
任意 Quality/Prep 顺序的 Workflow、统一版本化 artifact，以及完整示例与扩展测试。

## 本地证据

- Python 3.12.13，NumPy 2.5.2、pandas 3.0.5、sklearn 1.9.0、SciPy 1.18.1、joblib 1.6.0。
- 完整 binning 环境：`pytest -q -W error` **176 passed**。
- 独立基础安装环境、不安装 OptBinning：完整测试快照 **167 passed, 2 skipped**；
  随后新增的 Quality-only 重复列策略测试所在 workflow 目录再验收 **10 passed**。
  skips 为可选 OptBinning 集成，不是基础功能缺失。
- Ruff lint 和 format 检查通过；115 个 Python 文件格式检查通过。
- 原有 analysis/funnel 示例和新增 quality/prep/OptBinning/risk-workflow 示例全部运行通过。
- README 四个数据模块 Python 代码块按顺序运行通过。
- wheel/sdist 构建及直接从 wheel 导入、运行生命周期冒烟通过。
- sklearn/OptBinning 原生输出对照、缺失与特殊值、分类变量、权重、列选择、
  状态指纹、原子 refit、元数据警告、持久化 roundtrip 和扩展插件验收通过。

## 上游兼容性限制

OptBinning 0.21 metadata 要求 sklearn ≥1.6、OR-Tools ≥9.4,<9.12。
实际解析为 OR-Tools 9.11.4210；Python 3.13 wheel-only 解析实测失败，原因是
上述 OR-Tools 版本只提供 cp38–cp312 wheel。官方 binning 当前使用 Python 3.12。
基础功能 CI 配置为 3.12/3.13，binning CI 为 3.12；本轮未执行 Python 3.13 运行时测试或远程 CI。

## 清理范围

本次独立基础测试环境、临时 wheel/sdist、测试脚本、README 测试 artifact、
临时 Matplotlib/uv 缓存及为解析检查下载的 Python 3.13.13 均作为临时验收资源清理。
项目 .venv 中的正式依赖、uv.lock、源代码、正式测试与示例保留。
未改动本次工作开始之前已有的临时目录或其他 Python 安装。
