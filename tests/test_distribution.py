import numpy as np
import pandas as pd

from phl_risk.analysis._distribution import distribution_codes, profile
from phl_risk.analysis.transforms import QuantileBinner


def test_reference_bins_and_batched_counts():
    ref = pd.Series([0.0, 1.0, 2.0, 3.0, np.nan])
    cur = pd.Series([-100.0, 100.0, np.nan])
    binner = QuantileBinner(2)
    a, b, bins, metadata = distribution_codes(ref, cur, binner, "bucket")
    assert not binner.is_fitted
    assert bins == 3 and metadata["missing_bucket"] == 2
    np.testing.assert_array_equal(a, [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(b, [0, 1, 2])
    result = profile(np.array([0, 0, 1, 1, 1]), a, 2, bins)
    np.testing.assert_array_equal(result.counts, [[2, 0, 0], [0, 2, 1]])
    np.testing.assert_allclose(result.proportions.sum(axis=1), [1, 1])


def test_category_union_and_missing_collision():
    a, b, bins, meta = distribution_codes(
        pd.Series(["A", "B", None]), pd.Series(["A", "C", "__MISSING__"]), None, "bucket"
    )
    assert bins == 5
    assert meta["labels"] == ("A", "B", "C", "__MISSING__")
    assert a[-1] != b[-1]


def test_profile_allocation_guard(monkeypatch):
    import pytest

    import phl_risk.analysis._distribution as distribution
    from phl_risk.exceptions import EngineError

    monkeypatch.setattr(distribution, "MAX_PROFILE_CELLS", 3)
    with pytest.raises(EngineError, match="profile exceeds"):
        profile(np.array([0]), np.array([0]), 2, 2)
