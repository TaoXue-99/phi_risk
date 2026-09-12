"""Mixed fixed rules and learned transformers, using the existing project environment."""

import numpy as np
import pandas as pd

from phl_risk.data_prep import (
    Clip,
    DataPrep,
    KBinsStep,
    LogitTransform,
    MissingImputer,
    ToNumeric,
)

train = pd.DataFrame(
    {
        "income": [str(i * 100) for i in range(1, 101)],
        "prob": np.linspace(0, 1, 100).astype(str),
    }
)
train.loc[0, "income"] = None
prep = DataPrep(
    [
        ToNumeric(["income", "prob"]),
        Clip(["prob"], lower=1e-6, upper=1 - 1e-6),
        LogitTransform("prob", output_column="prob_logit"),
        MissingImputer(["income"], strategy="median"),
        KBinsStep(["income"], n_bins=10),
    ]
)
prep.fit(train)
for _, step in prep.steps_[:3]:
    assert not any(name.endswith("_") for name in vars(step))
oot = pd.DataFrame({"income": [None, "99999"], "prob": ["0", "1"]})
result = prep.run(oot)
assert result.data.index.equals(oot.index)
assert result.data.prob_logit.notna().all()
assert result.audit_frame().kind.tolist() == ["stateless"] * 3 + ["fitted"] * 2
print(result.data)
print(result.audit_frame()[["step", "kind", "input_columns", "output_columns"]])
