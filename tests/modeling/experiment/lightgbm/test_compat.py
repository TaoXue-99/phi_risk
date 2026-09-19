"""Exercise real old/new backends in the CI version matrix, without library-global patches."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import validation

from phl_risk.modeling.experiment.lightgbm._utils import require


def test_compatibility_is_idempotent_and_preserves_dependency_functions():
    pytest.importorskip("lightgbm")
    array, check_xy, check_array = np.array, validation.check_X_y, validation.check_array
    backend = require("lightgbm", "lightgbm")
    first = (
        backend.basic.np,
        backend.basic._data_from_pandas,
        getattr(backend.sklearn, "_LGBMCheckArray", None),
    )
    require("lightgbm", "lightgbm")
    assert first == (
        backend.basic.np,
        backend.basic._data_from_pandas,
        getattr(backend.sklearn, "_LGBMCheckArray", None),
    )
    assert np.array is array
    assert validation.check_X_y is check_xy
    assert validation.check_array is check_array
    assert isinstance(backend._phl_risk_compatibility, tuple)
    if tuple(int(part) for part in backend.__version__.split(".")[:2]) >= (4, 6):
        assert backend._phl_risk_compatibility == ()
        assert backend.basic.np is np


def test_nullable_dataframe_native_prediction_and_compat_metadata(exp, config, inputs):
    run = exp.run(name="compat", config=config)
    data = inputs["data"][list(run.features)].astype("Float64")
    original = data.copy()
    predictions = run.model.predict(data, num_iteration=run.best_iteration)
    expected = run.model.predict(data.to_numpy(dtype=float), num_iteration=run.best_iteration)
    np.testing.assert_allclose(predictions, expected)
    pd.testing.assert_frame_equal(data, original)
    assert run.metadata["lightgbm_compatibility"] == list(
        require("lightgbm", "lightgbm")._phl_risk_compatibility
    )
