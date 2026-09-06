import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis.dimensions import BinDimension, ColumnDimension
from phl_risk.analysis.transforms import QuantileBinner
from phl_risk.exceptions import DimensionError, NotFittedError, TransformError


def test_reference_current_and_boundaries():
    binner = QuantileBinner(n_bins=5)
    with pytest.raises(NotFittedError):
        binner.transform([1])
    assert binner.fit(np.arange(101)) is binner
    np.testing.assert_allclose(binner.bin_edges_, [-np.inf, 20, 40, 60, 80, np.inf])
    source = pd.Series(
        [-np.inf, -1, 0, 20, 20.1, 100, 1000, np.inf, np.nan], index=list("abcdefghi")
    )
    result = binner.transform(source)
    assert result.index.equals(source.index)
    assert result.iloc[:8].tolist() == ["B1", "B1", "B1", "B1", "B2", "B5", "B5", "B5"]
    assert pd.isna(result.iloc[8])
    assert result.cat.ordered
    edges = binner.bin_edges_
    edges[1] = -100
    assert binner.bin_edges_[1] == 20
    assert binner.transform([50]).iloc[0] == "B3"


def test_duplicate_constant_empty_labels_precision():
    binner = QuantileBinner().fit([1, 1, 1, None, np.inf])
    assert binner.n_bins_ == 1
    assert binner.transform([-100, 100]).tolist() == ["B1", "B1"]
    with pytest.raises(TransformError, match="Duplicate"):
        QuantileBinner(duplicates="raise").fit([1, 1])
    with pytest.raises(TransformError, match="no finite"):
        QuantileBinner().fit([None, np.inf])
    with pytest.raises(TransformError, match="learned"):
        QuantileBinner(labels=["a", "b"]).fit([1, 1])
    a = QuantileBinner(2, labels=["low", "high"], precision=1)
    b = QuantileBinner(2, labels=["low", "high"], precision=10)
    pd.testing.assert_series_equal(a.fit_transform([0, 0.123, 1]), b.fit_transform([0, 0.123, 1]))
    assert pd.isna(QuantileBinner(1, include_lowest=False).fit([0]).transform([-np.inf])[0])
    for kwargs in ({"n_bins": 0}, {"n_bins": 1.5}, {"labels": ["a", "a"]}):
        with pytest.raises(TransformError):
            QuantileBinner(**kwargs)


def test_dimensions():
    df = pd.DataFrame({"score": [0, 1, 2, 3]}, index=[5, 5, 7, 9])
    column = ColumnDimension("score", name="raw")
    assert column.fit(df) is column
    assert column.output_name == "raw"
    assert column.required_columns() == ("score",)
    dim = BinDimension("score", QuantileBinner(2), name="band")
    with pytest.raises(NotFittedError):
        dim.transform(df)
    assert dim.fit(df) is dim
    assert dim.transform(df).tolist() == ["B1", "B1", "B2", "B2"]
    assert dim.transform(df).index.equals(df.index)
    with pytest.raises(DimensionError, match="Missing"):
        ColumnDimension("absent").transform(df)
    with pytest.raises(DimensionError, match="reserved"):
        ColumnDimension("metric")


def test_extreme_finite_reference_and_empty_labels():
    binner = QuantileBinner(2).fit([-1e308, 1e308])
    assert binner.bin_edges_[1] == 0
    assert binner.transform([-1e308, 0, 1e308]).tolist() == ["B1", "B1", "B2"]
    with pytest.raises(TransformError, match="learned"):
        QuantileBinner(labels=[]).fit([1, 1])
