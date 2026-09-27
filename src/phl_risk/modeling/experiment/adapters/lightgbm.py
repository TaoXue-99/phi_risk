"""Optional native Booster persistence. No training, prediction or task policy."""

from phl_risk.exceptions import RunError

from .._utils import require


def enable_lightgbm_compatibility():
    """Explicit opt-in for LightGBM 4.0–4.5 with modern NumPy/pandas/sklearn.

    This changes only references inside LightGBM. It does not wrap train().
    Prefer a recent LightGBM for an unmodified native environment.
    """
    from ._lightgbm_compat import load_backend

    return load_backend()


def record_lightgbm(run, model, *, num_iteration, importance=True):
    """Snapshot the explicitly selected native model iterations into an active Run.

    Metrics and eval history remain user-provided. Continued training after this
    call cannot alter the saved model. No objective, callbacks or data are inferred.
    """
    backend = require("lightgbm", "lightgbm")
    if not isinstance(model, backend.Booster):
        raise RunError("record_lightgbm requires a native lightgbm.Booster")
    if type(num_iteration) is not int or not 1 <= num_iteration <= model.current_iteration():
        raise RunError("num_iteration must explicitly select existing model iterations")
    run.log_model_info(family="tree", backend="lightgbm", backend_version=backend.__version__)
    run.log_input(features=list(model.feature_name()))
    run.log_artifact(
        "model",
        "model.txt",
        lambda p: model.save_model(str(p), num_iteration=num_iteration),
        format="lightgbm",
    )
    run.log_result(
        saved_iteration=num_iteration,
        best_iteration=model.best_iteration,
        booster_params=model.params,
    )
    if importance:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "feature": model.feature_name(),
                "importance_gain": model.feature_importance(
                    importance_type="gain", iteration=num_iteration
                ),
                "importance_split": model.feature_importance(
                    importance_type="split", iteration=num_iteration
                ),
            }
        )
        for kind in ("gain", "split"):
            frame[f"{kind}_rank"] = (
                frame[f"importance_{kind}"].rank(method="min", ascending=False).astype(int)
            )
        frame = frame.sort_values("importance_gain", ascending=False, kind="stable")
        run.log_artifact(
            "feature_importance",
            "feature_importance.csv",
            lambda p: frame.to_csv(p, index=False),
            format="csv",
        )


def load_lightgbm(run):
    """Load a verified native Booster; Python functions/transforms are not restored."""
    backend = require("lightgbm", "lightgbm")
    if (
        run.model["backend"] != "lightgbm"
        or run.artifacts.get("model", {}).get("format") != "lightgbm"
    ):
        raise RunError("Run has no native LightGBM model")
    try:
        model = backend.Booster(model_file=str(run.artifact_path("model")))
        if tuple(model.feature_name()) != run.features:
            raise ValueError("model feature names differ from record")
        expected = run.result.get("saved_iteration", run.result.get("best_iteration"))
        if expected is not None and model.current_iteration() != expected:
            raise ValueError("model iteration differs from record")
        return model
    except (ValueError, OSError, backend.basic.LightGBMError) as error:
        raise RunError(f"Cannot load LightGBM model: {error}") from error
