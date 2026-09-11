import pandas as pd
import pytest

from phl_risk._data import FrozenMapping, aligned, check_fitted, frame
from phl_risk.exceptions import DataPrepError, NotFittedError, OptionalDependencyError


def test_frame_and_lifecycle_contracts():
    with pytest.raises(DataPrepError):
        frame([1], DataPrepError)
    duplicate = pd.DataFrame([[1, 2]], columns=["x", "x"])
    assert frame(duplicate, unique=False) is duplicate
    with pytest.raises(DataPrepError):
        frame(duplicate, DataPrepError)
    with pytest.raises(NotFittedError):
        check_fitted(object())
    assert issubclass(OptionalDependencyError, ImportError)


def test_alignment_and_nested_immutable_mapping():
    X = pd.DataFrame({"x": [1, 2]}, index=[5, 6])
    with pytest.raises(DataPrepError, match="index"):
        aligned(pd.Series([1, 0]), X, "y", DataPrepError)
    value = FrozenMapping({"nested": {"a": [1, 2]}})
    with pytest.raises(TypeError):
        value["nested"]["a"] = 3
    assert value["nested"]["a"] == (1, 2)
