"""Requires: uv sync --extra binning."""

import numpy as np
import pandas as pd

from phl_risk.data_prep.integrations import OptBinningStep

rng = np.random.default_rng(7)
train = pd.DataFrame({"score": rng.normal(size=400)})
y = (train.score + rng.normal(size=400) > 0).astype(int)
step = OptBinningStep(["score"], max_n_prebins=5).fit(train, y)
print(step.summary())
print(step.binning_table("score"))
assert step.transform(train).shape == train.shape
