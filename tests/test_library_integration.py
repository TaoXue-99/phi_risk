"""Verify library adapters, fixed execution paths, and preserved framework policies."""

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as sklearn_metrics

from phl_risk.analysis import AUC, KS, Cube, QuantileBinner
from phl_risk.metrics import auc_score, ks_score


def test_auc_delegates_prepared_inputs_to_public_sklearn(monkeypatch):
    original = sklearn_metrics.roc_auc_score
    calls = []

    def record(y_true, y_score, *, sample_weight):
        calls.append((y_true, y_score, sample_weight))
        return original(y_true, y_score, sample_weight=sample_weight)

    monkeypatch.setattr(sklearn_metrics, "roc_auc_score", record)
    result = auc_score(
        pd.Series([0, 1, pd.NA, 1, 0], dtype="Int64"),
        [0.1, 0.9, 0.5, np.inf, 0.8],
        [2, 4, 9, 9, 0],
    )
    assert result == 1
    assert len(calls) == 1
    target, score, weight = calls[0]
    np.testing.assert_array_equal(target, [0, 1])
    np.testing.assert_array_equal(score, [0.1, 0.9])
    np.testing.assert_array_equal(weight, [0.5, 1])
    assert target.dtype == np.int8


def test_ks_uses_all_roc_thresholds(monkeypatch):
    original = sklearn_metrics.roc_curve
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(sklearn_metrics, "roc_curve", record)
    assert ks_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.5
    assert calls == [{"pos_label": 1, "sample_weight": None, "drop_intermediate": False}]


@pytest.mark.parametrize("weighted", [False, True])
def test_paired_metrics_use_independent_public_functions(monkeypatch, weighted):
    original_auc = sklearn_metrics.roc_auc_score
    original_roc = sklearn_metrics.roc_curve
    calls = {"auc": 0, "ks": 0}

    def record_auc(*args, **kwargs):
        calls["auc"] += 1
        return original_auc(*args, **kwargs)

    def record_roc(*args, **kwargs):
        calls["ks"] += 1
        return original_roc(*args, **kwargs)

    def reject_auc_integration(*args, **kwargs):
        raise AssertionError("Framework must call roc_auc_score, never integrate a shared ROC")

    monkeypatch.setattr(sklearn_metrics, "roc_auc_score", record_auc)
    monkeypatch.setattr(sklearn_metrics, "roc_curve", record_roc)
    monkeypatch.setattr(sklearn_metrics, "auc", reject_auc_integration)
    frame = pd.DataFrame(
        {
            "g": ["a"] * 4 + ["b"] * 4,
            "y": [0, 0, 1, 1] * 2,
            "s": [0.1, 0.4, 0.35, 0.8] * 2,
            "w": [1, 2, 3, 4] * 2,
        }
    )
    weight = "w" if weighted else None
    result = Cube(
        ["g"],
        [
            KS("s", "y", weight=weight),
            AUC("s", "y", weight=weight),
            AUC("s", "y", weight=weight, name="auc_again"),
        ],
    ).compute(frame)
    # Each unique metric runs once per group; a renamed identical AUC reuses its value.
    assert calls == {"auc": 2, "ks": 2}
    assert "ranking_input_batches" not in result.metadata_
    for key, group in frame.groupby("g"):
        w = group.w if weighted else None
        expected_auc = sklearn_metrics.roc_auc_score(group.y, group.s, sample_weight=w)
        table = result.layout()
        assert table.loc[key, "auc__s"] == pytest.approx(expected_auc)
        assert table.loc[key, "auc_again"] == pytest.approx(expected_auc)
        assert table.loc[key, "ks__s"] == pytest.approx(ks_score(group.y, group.s, w))


def test_distinct_dependencies_are_not_shared():
    frame = pd.DataFrame({"y": [0, 0, 1, 1], "s": [0.1, 0.4, 0.35, 0.8], "w": [1, 2, 3, 4]})
    result = Cube(measures=[AUC("s", "y"), KS("s", "y", weight="w")]).compute(frame)
    table = result.layout()
    assert table.loc[0, "auc__s"] == pytest.approx(auc_score(frame.y, frame.s))
    assert table.loc[0, "ks__s"] == pytest.approx(ks_score(frame.y, frame.s, frame.w))


def test_extreme_range_preserves_ties_and_tiny_distinct_scores():
    target = [0, 1, 0, 1, 0, 1]
    scores = [-1e308, -1e-308, 0, 1e-308, 1e308, 1e308]
    expected = sklearn_metrics.roc_auc_score(target, [0, 1, 2, 3, 4, 4])
    assert auc_score(target, scores) == pytest.approx(expected)
    assert ks_score(target, scores) == pytest.approx(ks_score(target, [0, 1, 2, 3, 4, 4]))


def test_third_party_errors_are_not_hidden(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unexpected upstream error")

    monkeypatch.setattr(sklearn_metrics, "roc_auc_score", fail)
    with pytest.raises(ValueError, match="unexpected upstream"):
        auc_score([0, 1], [0, 1], on_invalid="nan")


@pytest.mark.parametrize("include_lowest", [False, True])
def test_numpy_bin_assignment_matches_pandas_cut(include_lowest):
    reference = np.linspace(-1, 1, 101)
    binner = QuantileBinner(5, include_lowest=include_lowest).fit(reference)
    edges = binner.bin_edges_
    current = pd.Series(
        np.r_[
            -np.inf,
            -100,
            edges[1:-1],
            np.nextafter(edges[1:-1], np.inf),
            100,
            np.inf,
            np.nan,
        ]
    )
    expected = pd.cut(
        current,
        bins=edges,
        labels=binner.labels_,
        right=True,
        include_lowest=include_lowest,
        ordered=True,
    )
    pd.testing.assert_series_equal(binner.transform(current), expected)
