"""Public exception hierarchy shared by numerical and analysis layers."""


# Exception                              Python 异常基类
# └── PhlRiskError                       本包所有异常的统一捕获入口
#     ├── AnalysisError                  分析定义、执行和展示相关异常
#     │   ├── CubeError                  Cube 配置或生命周期错误
#     │   ├── DimensionError             维度定义、依赖字段或输出错误
#     │   ├── MeasureError               指标配置、命名或依赖错误
#     │   ├── TransformError             行级数据变换错误
#     │   │   └── NotFittedError         需要学习的变换尚未 fit
#     │   ├── LayoutError                结果轴、布局或展示请求错误
#     │   └── EngineError                后端输入、能力或执行错误
#     └── InvalidMetricError             样本上统计量未定义；同时继承 ValueError
#         └── also ValueError            可被 except ValueError 捕获（多重继承）
# 捕获建议：按具体异常处理；except AnalysisError 捕获分析层异常，
# except PhlRiskError 捕获所有包级异常。InvalidMetricError 不属于 AnalysisError。


class PhlRiskError(Exception):
    """Base package error."""


class AnalysisError(PhlRiskError):
    """Invalid analysis definition or execution."""


class CubeError(AnalysisError):
    """Invalid cube configuration."""


class DimensionError(AnalysisError):
    """Invalid dimension definition or values."""


class MeasureError(AnalysisError):
    """Invalid measure configuration."""


class TransformError(AnalysisError):
    """Transformation failure."""


class NotFittedError(TransformError):
    """A required fit has not completed."""


class LayoutError(AnalysisError):
    """Invalid result layout."""


class EngineError(AnalysisError):
    """Unsupported or invalid backend execution."""


class InvalidMetricError(PhlRiskError, ValueError):
    """A statistic is undefined for the supplied observations."""
