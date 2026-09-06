"""Declarative Dimension × Measure analysis with explicit fit/compute lifecycle."""

from ._axis import AxisSpec
from ._context import AnalysisContext, ComputePolicy, MissingPolicy
from ._cube import Cube
from ._layout import TableLayout
from ._plan import CubePlan, FilterExpression
from ._result import CubeResult
from .dimensions import BaseDimension, BinDimension, ColumnDimension, DimensionLike
from .engines import BaseCubeEngine, EngineLike, PandasEngine
from .measures import AUC, KS, BaseMeasure, Count, EventRate, MeasureLike, Share
from .transforms import BaseTransformer, QuantileBinner

__all__ = [
    "Cube",
    "AnalysisContext",
    "ComputePolicy",
    "MissingPolicy",
    "BaseDimension",
    "ColumnDimension",
    "BinDimension",
    "BaseTransformer",
    "QuantileBinner",
    "BaseMeasure",
    "Count",
    "Share",
    "EventRate",
    "AUC",
    "KS",
    "CubePlan",
    "BaseCubeEngine",
    "PandasEngine",
    "CubeResult",
    "AxisSpec",
    "TableLayout",
    "FilterExpression",
    "DimensionLike",
    "MeasureLike",
    "EngineLike",
]
