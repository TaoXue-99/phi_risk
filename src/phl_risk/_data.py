"""Small shared contracts; no dependency on the three data modules."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from .exceptions import NotFittedError, PhlRiskError


def frame(X, error=PhlRiskError, *, unique=True):
    if not isinstance(X, pd.DataFrame):
        raise error("Expected a pandas DataFrame")
    if any(not isinstance(c, str) for c in X.columns):
        raise error("Column names must be strings")
    if unique and not X.columns.is_unique:
        raise error("Duplicate column names are not executable here")
    return X


def columns_of(X, columns, error=PhlRiskError):
    if isinstance(columns, (str, bytes)):
        raise error("columns must be a sequence of column names")
    names = tuple(X.columns if columns is None else columns)
    if not names or len(set(names)) != len(names):
        raise error("columns must be nonempty and unique")
    missing = tuple(c for c in names if c not in X.columns)
    if missing:
        raise error(f"Missing columns: {missing}")
    return names


def check_fitted(obj):
    if not getattr(obj, "_is_fitted_", False):
        raise NotFittedError(f"{type(obj).__name__} requires fit(reference) first")


def aligned(value, X, name, error):
    if value is None:
        return None
    if isinstance(value, pd.Series) and not value.index.equals(X.index):
        raise error(f"{name} index must exactly match X.index")
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != len(X):
        raise error(f"{name} must be one-dimensional with len(X) values")
    if name == "sample_weight":
        try:
            valid = np.isfinite(array).all() and (array >= 0).all()
        except TypeError as exc:
            raise error("sample_weight must be numeric") from exc
        if not valid or not (array > 0).any():
            raise error("sample_weight must be finite, nonnegative, with positive total")
    return value.copy() if hasattr(value, "copy") else deepcopy(value)


def commit_fitted(obj, candidate):
    """Publish a complete fit, replacing stale fitted attributes on refit."""
    for key in tuple(vars(obj)):
        if key.endswith("_") and not key.startswith("__"):
            delattr(obj, key)
    for key, value in vars(candidate).items():
        if key.endswith("_") and not key.startswith("__"):
            setattr(obj, key, value)
    obj._is_fitted_ = True
    return obj


class DataEstimator(BaseEstimator):
    """sklearn parameters with explicit fitted-state invalidation on reconfiguration."""

    def set_params(self, **params):
        result = super().set_params(**params)
        if params:
            for key in tuple(vars(self)):
                if key.endswith("_") and not key.startswith("__"):
                    delattr(self, key)
        return result

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted_", False)


def named_estimators(items, expected, error):
    if isinstance(items, (str, bytes)):
        raise error("Expected a sequence of estimators")
    import re

    result, names = [], set()
    for item in items:
        explicit = isinstance(item, tuple)
        if explicit:
            if len(item) != 2:
                raise error("Named entries must be (name, estimator)")
            name, estimator = item
        else:
            estimator = item
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(item).__name__).lower()
            base, count = name, 2
            while name in names:
                name, count = f"{base}_{count}", count + 1
        if not isinstance(name, str) or not name or "__" in name or name in names:
            raise error(f"Invalid or duplicate name: {name!r}")
        if not isinstance(estimator, expected):
            raise error(f"{name} must be a {expected.__name__}")
        names.add(name)
        result.append((name, clone(estimator)))
    return tuple(result)


@dataclass(frozen=True)
class FrozenMapping(Mapping):
    """Recursively immutable, pickle-friendly mapping for result snapshots."""

    _items: tuple

    def __init__(self, value=()):
        object.__setattr__(self, "_items", tuple((k, freeze(v)) for k, v in dict(value).items()))

    def __getitem__(self, key):
        for k, value in self._items:
            if k == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (k for k, _ in self._items)

    def __len__(self):
        return len(self._items)


def freeze(value):
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (tuple, list, np.ndarray, pd.Index)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def thaw(value):
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [thaw(v) for v in value]
    return deepcopy(value)
