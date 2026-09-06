import pandas as pd
import pytest


@pytest.fixture
def fixture_analysis_df():
    return pd.DataFrame(
        {
            "dt": ["2026-01"] * 8 + ["2026-02"] * 8,
            "user_type": (["new"] * 4 + ["old"] * 4) * 2,
            "label": [0, 0, 1, 1] * 4,
            "score": [
                0.1,
                0.4,
                0.35,
                0.8,
                0.1,
                0.2,
                0.8,
                0.9,
                0.8,
                0.9,
                0.1,
                0.2,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
        }
    )
