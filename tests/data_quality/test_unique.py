import pandas as pd

from phl_risk.data_quality.checks import UniqueCheck


def test_incident():
    X = pd.DataFrame({"id": [1, 1, 1], "dt": [1, 1, 2]})
    r = UniqueCheck(["id", "dt"]).validate(X)
    assert r.failed and r.details["duplicate_count"] == 2
