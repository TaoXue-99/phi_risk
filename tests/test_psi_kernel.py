import numpy as np
import pytest

from phl_risk.exceptions import InvalidMetricError
from phl_risk.metrics import psi_from_proportions


@pytest.mark.parametrize("dtype", [np.float32, np.float64, int])
def test_identical_and_matrix(dtype):
    p = np.array([[0, 1], [1, 0]], dtype=dtype)
    np.testing.assert_allclose(psi_from_proportions(p, p), [0, 0])
    assert isinstance(psi_from_proportions(p[0], p[0]), float)


@pytest.mark.parametrize("epsilon", [1e-8, 0.01, 2.0])
def test_independent_arithmetic(epsilon):
    p, q = np.array([0.0, 0.4, 0.6]), np.array([0.5, 0.5, 0.0])
    a, b = np.maximum(p, epsilon), np.maximum(q, epsilon)
    a, b = a / a.sum(), b / b.sum()
    expected = sum((y - x) * np.log(y / x) for x, y in zip(a, b))
    assert psi_from_proportions(p, q, epsilon=epsilon) == pytest.approx(expected)


@pytest.mark.parametrize("p,q", [([], []), ([1], [0, 1]), ([[[1]]], [[[1]]])])
def test_shapes(p, q):
    with pytest.raises(ValueError):
        psi_from_proportions(p, q)


@pytest.mark.parametrize("p", [[np.nan, 1], [-1, 2], [0, 0], [np.inf, 0], [0.2, 0.2]])
def test_invalid_row_policy(p):
    ref = np.array([[0.5, 0.5], p])
    got = psi_from_proportions(ref, [[0.5, 0.5], [0.5, 0.5]])
    assert got[0] == 0 and np.isnan(got[1])
    with pytest.warns(RuntimeWarning):
        psi_from_proportions(p, [0.5, 0.5], on_invalid="warn")
    with pytest.raises(InvalidMetricError):
        psi_from_proportions(p, [0.5, 0.5], on_invalid="raise")


@pytest.mark.parametrize("epsilon", [0, -1, np.nan, np.inf])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        psi_from_proportions([1], [1], epsilon=epsilon)


def test_large_invalid_proportions_use_policy_without_numpy_warning():
    assert np.isnan(psi_from_proportions([1e308, 1e308], [0.5, 0.5]))


def test_psi_independent_formula_and_empty_populations():
    p, q = np.array([0.2, 0.8]), np.array([0.4, 0.6])
    assert psi_from_proportions(p, q) == pytest.approx(((q - p) * np.log(q / p)).sum())
    assert psi_from_proportions([0, 1], [1, 0]) > 1
    for p, q in [([], []), ([0], [1]), ([-1], [1]), ([1, 2], [1])]:
        with pytest.raises(ValueError):
            psi_from_proportions(p, q, on_invalid="raise")
