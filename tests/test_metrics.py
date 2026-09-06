import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from phl_risk.exceptions import InvalidMetricError
from phl_risk.metrics import auc_score, event_rate, ks_score


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("weighted", [False, True])
def test_auc_matches_sklearn(seed, weighted):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 200)
    scores = rng.integers(0, 10, 200) / 10
    weight = rng.random(200) if weighted else None
    assert auc_score(y, scores, weight) == pytest.approx(
        roc_auc_score(y, scores, sample_weight=weight), abs=1e-12
    )


def test_ks_ties_and_weighted_reference():
    y = np.array([0, 1, 0, 1, 1])
    score = np.array([0.1, 0.1, 0.3, 0.5, 0.9])
    w = np.array([1, 2, 3, 4, 5])
    expected = max(
        abs(
            w[(score <= t) & (y == 1)].sum() / w[y == 1].sum()
            - w[(score <= t) & (y == 0)].sum() / w[y == 0].sum()
        )
        for t in score
    )
    assert ks_score(y, score, w) == pytest.approx(expected)
    assert ks_score([0, 1, 0, 1], [1, 1, 1, 1]) == 0
    assert auc_score([0, 1], [1, 1]) == 0.5


@pytest.mark.parametrize("fn", [auc_score, ks_score])
def test_binary_missing_invalid_and_weights(fn):
    assert fn([0, 1, None, 1], [0, 1, 0.5, np.inf]) == 1
    assert fn(pd.Series([0, 1, pd.NA], dtype="Int64"), [0, 1, None]) == 1
    assert np.isnan(fn([], []))
    assert np.isnan(fn([1, 1], [0, 1]))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(fn([1], [1], on_invalid="warn"))
    with pytest.raises(InvalidMetricError):
        fn([0, 2], [0, 1], on_invalid="raise")
    with pytest.raises(ValueError, match="finite non-negative"):
        fn([0, 1], [0, 1], [1, -1])
    with pytest.raises(ValueError, match="equal lengths"):
        fn([0, 1], [0])
    assert np.isnan(fn([0, 1], [0, 1], [0, 1]))
    assert fn([0, 1], [0, 1], [1e308, 1e308]) == 1
    assert fn([0, 1], [-1e308, 1e308]) == 1


def test_event_rate():
    assert event_rate([0, 1, None, 1]) == pytest.approx(2 / 3)
    assert event_rate(["SUCCESS", "FAIL", None], "SUCCESS", [2, 1, 99]) == pytest.approx(2 / 3)
    assert event_rate(pd.Series([1, 0, pd.NA], dtype="Int64")) == 0.5
    assert np.isnan(event_rate([None]))
    assert np.isnan(event_rate([1], sample_weight=[0]))
    with pytest.raises(InvalidMetricError):
        event_rate([], on_invalid="raise")
    with pytest.raises(ValueError):
        event_rate([1], event_value=None)
