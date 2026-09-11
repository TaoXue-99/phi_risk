"""Declarative orchestration only; no pandas grouping or presentation here."""

import logging
from copy import deepcopy
from typing import Literal, Sequence

from phl_risk.exceptions import CubeError, DimensionError, EngineError, MeasureError

from ._context import AnalysisContext, ComputePolicy
from ._plan import CubePlan, FilterExpression, FilterLike
from ._result import CubeResult
from .dimensions import BaseDimension, ColumnDimension, DimensionLike
from .dimensions._base import validate_name
from .engines import BaseCubeEngine, EngineLike, PandasEngine
from .measures import BaseMeasure, Count, MeasureLike

logger = logging.getLogger("phl_risk.analysis")


def _engine(engine: EngineLike | None) -> BaseCubeEngine:
    if engine is None or (isinstance(engine, str) and engine == "pandas"):
        return PandasEngine()
    if isinstance(engine, BaseCubeEngine):
        return engine
    raise EngineError(f"Unsupported engine {engine!r}; V0.1 includes 'pandas'")


class Cube:
    """Describe WHAT to analyze; fit dimensions explicitly, compute through an engine.

    Missing measures default to Count(). An explicit empty measure list is an
    error. Constructor inputs and exposed dimension configurations are copied
    so fitted state cannot leak across cubes.
    """

    def __init__(
        self,
        dimensions: Sequence[DimensionLike] | None = None,
        measures: Sequence[MeasureLike] | None = None,
        filters: Sequence[FilterLike] | None = None,
        engine: EngineLike | None = None,
        *,
        policy: ComputePolicy | None = None,
    ) -> None:
        if isinstance(dimensions, str):
            raise CubeError("dimensions must be a sequence, e.g. ['dt']")
        normalized = tuple(
            ColumnDimension(d) if isinstance(d, str) else deepcopy(d)
            for d in (() if dimensions is None else dimensions)
        )
        if any(not isinstance(d, BaseDimension) for d in normalized):
            raise DimensionError("dimensions must contain column names or BaseDimension objects")
        names = [d.output_name for d in normalized]
        for name in names:
            validate_name(name)
        if len(names) != len(set(names)):
            raise DimensionError(f"Duplicate dimension output names: {names}")
        normalized_measures = tuple(deepcopy((Count(),) if measures is None else measures))
        if not normalized_measures or any(
            not isinstance(m, BaseMeasure) for m in normalized_measures
        ):
            raise MeasureError("measures must contain at least one BaseMeasure")
        modes = {m.mode for m in normalized_measures}
        if not modes <= {"single", "comparative"} or len(modes) != 1:
            raise MeasureError("Cannot mix single-sample and comparative measures")
        predicates = tuple(() if filters is None else filters)
        if any(not callable(f) and not isinstance(f, FilterExpression) for f in predicates):
            raise CubeError("filters must contain callables or FilterExpression implementations")
        self._dimensions = normalized
        self._measures = normalized_measures
        self._filters = predicates
        self._engine = _engine(engine)
        self._policy = policy or ComputePolicy()
        if not isinstance(self._policy, ComputePolicy):
            raise CubeError("policy must be ComputePolicy")
        self._fitted_dimensions: tuple[BaseDimension, ...] | None = None

    @property
    def dimensions(self) -> tuple[BaseDimension, ...]:
        return deepcopy(self._dimensions)

    @property
    def measures(self) -> tuple[BaseMeasure, ...]:
        return deepcopy(self._measures)

    @property
    def dimensions_(self) -> tuple[BaseDimension, ...]:
        return deepcopy(self._active_dimensions)

    @property
    def _active_dimensions(self) -> tuple[BaseDimension, ...]:
        return self._dimensions if self._fitted_dimensions is None else self._fitted_dimensions

    @property
    def is_fitted(self) -> bool:
        return all(d.is_fitted for d in self._active_dimensions)

    def fit(
        self, reference: object, y: object = None, context: AnalysisContext | None = None
    ) -> "Cube":
        logger.info("analysis.cube.fit.start")
        prepared = self._engine.prepare_fit(reference, self._filters)
        dimensions = deepcopy(self._dimensions)
        for dimension in dimensions:
            dimension.fit(prepared, y, context)
        # Commit a complete state only after every dimension fit succeeds.
        self._fitted_dimensions = dimensions
        logger.info("analysis.cube.fit.end")
        return self

    def plan(self, context: AnalysisContext | None = None, *, totals: bool = False) -> CubePlan:
        if not isinstance(totals, bool):
            raise CubeError("totals must be bool")
        context = context or AnalysisContext()
        if not isinstance(context, AnalysisContext):
            raise CubeError("context must be AnalysisContext")
        measures = tuple(m.compile(context) for m in self._measures)
        names = [m.name for m in measures]
        if len(names) != len(set(names)):
            raise MeasureError(f"Duplicate measure names: {names}; supply explicit name=")
        plan = CubePlan(
            deepcopy(self._active_dimensions),
            measures,
            deepcopy(self._filters),
            context,
            self._policy,
            totals,
        )
        if plan.mode != self._measures[0].mode:
            raise MeasureError("Measure mode must match its compiled node type")
        return plan

    def explain(
        self,
        context: AnalysisContext | None = None,
        *,
        format: Literal["table", "text"] = "table",
        totals: bool = False,
    ):
        return self.plan(context, totals=totals).explain(self._engine.name, format=format)

    def compute(
        self,
        data: object,
        context: AnalysisContext | None = None,
        *,
        engine: EngineLike | None = None,
        totals: bool = False,
    ) -> CubeResult:
        self._validate_mode("single")
        logger.info("analysis.cube.compute.start")
        try:
            backend = self._engine if engine is None else _engine(engine)
            result = backend.execute(self.plan(context, totals=totals), data)
        except Exception:
            logger.exception("analysis.cube.compute.failed")
            raise
        logger.info("analysis.cube.compute.end")
        return result

    def _validate_mode(self, expected: str) -> None:
        if self._measures[0].mode != expected:
            actual = self._measures[0].mode
            target = "compute_comparison" if actual == "comparative" else "compute"
            raise MeasureError(
                f"{type(self._measures[0]).__name__} is a {actual} measure; "
                f"use Cube.{target}() instead"
            )

    def compute_comparison(
        self,
        reference: object,
        current: object,
        context: AnalysisContext | None = None,
        *,
        engine: EngineLike | None = None,
    ) -> CubeResult:
        """Compare matching dimension keys in two prepared populations.

        Dimension transformers retain their explicit fit lifecycle. Measure-owned
        distributions learn temporary reference definitions for this call only.
        """
        self._validate_mode("comparative")
        backend = self._engine if engine is None else _engine(engine)
        return backend.execute_comparison(self.plan(context), reference, current)

    def fit_compute(
        self,
        data: object,
        y: object = None,
        context: AnalysisContext | None = None,
        *,
        totals: bool = False,
    ) -> CubeResult:
        self._validate_mode("single")
        return self.fit(data, y, context).compute(data, context, totals=totals)

    def __repr__(self) -> str:
        return (
            f"Cube(dimensions={self._dimensions!r}, measures={self._measures!r}, "
            f"engine={self._engine!r}, fitted={self.is_fitted})"
        )
