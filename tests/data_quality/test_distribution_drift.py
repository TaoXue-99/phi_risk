import pandas as pd

from phl_risk.data_quality.checks import DistributionDriftCheck


def test_incident():
    import joblib
    import numpy as np

    rng = np.random.default_rng(5)
    X = pd.DataFrame({"x": rng.normal(size=1000)})
    check = DistributionDriftCheck(["x"]).fit(X)
    before = joblib.hash(check)
    assert check.validate(X).passed
    assert check.validate(X + 5).failed
    assert joblib.hash(check) == before
    cat = DistributionDriftCheck(["x"]).fit(pd.DataFrame({"x": ["a", "b", None]}))
    assert cat.validate(pd.DataFrame({"x": ["z"] * 10})).failed
