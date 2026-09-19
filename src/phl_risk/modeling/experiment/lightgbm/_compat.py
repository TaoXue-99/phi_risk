"""Narrow adapters for LightGBM 4.x against this project's modern dependencies.

Only references owned by LightGBM are adapted. NumPy and sklearn public functions
remain untouched; the native training and feature selection algorithms are unchanged.
"""

from functools import wraps
from threading import RLock

from phl_risk.exceptions import OptionalDependencyError

_LOCK = RLock()


class _LegacyNumPy:
    def __init__(self, numpy):
        self._numpy = numpy

    def __getattr__(self, name):
        return getattr(self._numpy, name)

    def find_common_type(self, array_types, scalar_types):
        # LightGBM's pandas conversion always supplies an empty scalar_types list.
        if scalar_types:
            raise TypeError("Legacy LightGBM dtype adapter requires empty scalar_types")
        return self._numpy.result_type(*array_types)

    def array(self, *args, **kwargs):
        if kwargs.get("copy") is False:
            kwargs["copy"] = None
        return self._numpy.array(*args, **kwargs)


def _pandas_adapter(function):
    @wraps(function)
    def convert(data, feature_name, categorical_feature, pandas_categorical):
        import pandas as pd

        if isinstance(data, pd.DataFrame) and (feature_name is None or feature_name == "auto"):
            data = data.rename(columns=str)
            feature_name = list(data.columns)
        return function(data, feature_name, categorical_feature, pandas_categorical)

    return convert


def _validation_adapter(function):
    @wraps(function)
    def validate(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return function(*args, **kwargs)

    return validate


def prepare_backend(backend):
    """Initialize compatibility once, including for lazy-loaded native Boosters."""
    with _LOCK:
        if hasattr(backend, "_phl_risk_compatibility"):
            return backend
        release = tuple(int(part) for part in backend.__version__.split(".")[:2])
        if not (4, 0) <= release < (5, 0):
            raise OptionalDependencyError("LightGBM Experiment requires lightgbm>=4.0,<5")
        applied = []
        if release < (4, 4):
            backend.basic.np = _LegacyNumPy(backend.basic.np)
            backend.basic._data_from_pandas = _pandas_adapter(backend.basic._data_from_pandas)
            applied.append("legacy_numpy_pandas")
        if release < (4, 6):
            for name in ("_LGBMCheckXY", "_LGBMCheckArray"):
                setattr(backend.sklearn, name, _validation_adapter(getattr(backend.sklearn, name)))
            applied.append("sklearn_ensure_all_finite")
        backend._phl_risk_compatibility = tuple(applied)
        return backend
