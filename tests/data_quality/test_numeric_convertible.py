import pandas as pd

from phl_risk.data_quality.checks import NumericConvertibleCheck


def test_incident():
    X = pd.DataFrame({"x": ["1000 PHP", "unknown", "20", None]})
    r = NumericConvertibleCheck(["x"]).validate(X)
    assert r.failed and r.details["x"]["failed_count"] == 2
