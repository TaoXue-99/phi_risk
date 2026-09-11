import pandas as pd

from phl_risk.data_quality.checks import MissingLikeCheck


def test_incident():
    X = pd.DataFrame({"x": [" NULL ", "", -999, None, "ok"]})
    r = MissingLikeCheck(["x"]).validate(X)
    assert r.details["x"]["missing_like_count"] == 2
    assert (
        MissingLikeCheck(["x"], values=[-999]).validate(X).details["x"]["missing_like_count"] == 1
    )
