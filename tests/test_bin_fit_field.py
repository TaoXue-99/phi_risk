import numpy as np
import pandas as pd
import pytest

from phl_risk.analysis import BinDimension, Count, Cube, QuantileBinner
from phl_risk.exceptions import DimensionError


def test_cross_shared_boundaries_order_and_fit_compute():
    data = pd.DataFrame({"a": [0, 2, 4, 6], "b": [0, 0, 1, 6]})
    original = data.copy(deep=True)
    cube = Cube(
        [BinDimension("b", QuantileBinner(2), fit_field="a"), BinDimension("a", QuantileBinner(2))],
        [Count()],
    )
    result = cube.fit_compute(data)
    b, a = cube.dimensions_
    np.testing.assert_array_equal(b.transformer.bin_edges_, [-np.inf, 3, np.inf])
    np.testing.assert_array_equal(a.transformer.bin_edges_, b.transformer.bin_edges_)
    assert b.transform(data).tolist() == ["B1", "B1", "B1", "B2"]
    assert result.layout(rows=["b_bin"], columns=["a_bin", "metric"]).to_numpy().tolist() == [
        [2, 1],
        [0, 1],
    ]
    assert b.metadata()["fit_field"] == "a"
    assert b.metadata()["column"] == "b"
    assert "Fit source: a" in cube.explain(format="text")
    pd.testing.assert_frame_equal(data, original)


def test_fit_only_needs_learning_field_compute_only_needs_output_field():
    cube = Cube([BinDimension("b", QuantileBinner(2), "band", fit_field="a")], [Count()])
    cube.fit(pd.DataFrame({"a": [0, 2, 4, 6]}))
    current = pd.DataFrame({"b": [1, 2, 6]}, index=[5, 5, 7])
    assert cube.dimensions_[0].transform(current).tolist() == ["B1", "B1", "B2"]
    assert cube.compute(current).layout().to_numpy().ravel().tolist() == [2, 1]
    with pytest.raises(DimensionError, match="Missing.*a"):
        cube.fit(current)
    # A failed refit must preserve the previous usable fitted state.
    assert cube.compute(current).layout().to_numpy().ravel().tolist() == [2, 1]


@pytest.mark.parametrize("field", ["", 0, False, []])
def test_invalid_fit_field(field):
    with pytest.raises(DimensionError, match="fit_field"):
        BinDimension("b", QuantileBinner(2), fit_field=field)


def test_duplicate_learning_column_rejected():
    dim = BinDimension("b", QuantileBinner(2), fit_field="a")
    with pytest.raises(DimensionError, match="Duplicate.*a"):
        dim.fit(pd.DataFrame([[1, 2]], columns=["a", "a"]))
