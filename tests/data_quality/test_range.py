import pandas as pd

from phl_risk.data_quality.checks import RangeCheck


def test_incident():
    X = pd.DataFrame({"x": [0.0, 1.0, None]})
    assert RangeCheck("x", 0, 1).validate(X).passed
    assert RangeCheck("x", 0, 1, inclusive="neither").validate(X).failed
