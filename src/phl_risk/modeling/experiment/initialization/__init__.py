"""Create model spaces without requiring a training backend or an executor."""

from phl_risk.exceptions import ExperimentError

from .._utils import slug
from ..record.artifact import atomic_write, read_json
from ..record.store import ExperimentStore


def _identity(method, family=None):
    if family is not None and (not isinstance(family, str) or not family):
        raise ExperimentError("family must be a non-empty string")
    if method in ("lgb", "lightgbm"):
        return {"name": "lightgbm", "family": family or "tree", "prefix": "lgb"}
    if method == "esmm_mmoe":
        return {"name": method, "family": family or "deep", "prefix": method}
    if (
        not isinstance(method, str)
        or slug(method) != method
        or method in (".", "..", "runs", "configs", "reports")
    ):
        raise ExperimentError("Method must be a safe, non-reserved directory name")
    return {"name": method, "family": family or "unspecified", "prefix": method}


def _stored_family(manifest):
    value = read_json(manifest)
    family = value.get("family") if isinstance(value, dict) else None
    if not isinstance(family, str) or not family:
        raise ExperimentError(f"Invalid method manifest: {manifest}")
    return family


def initialize(
    project, method, *, family=None, objective=None, metric=None, comparison_partitions=None
):
    spec = _identity(method, family)
    manifest = project.path / spec["prefix"] / "method.json"
    if manifest.exists() and family is None:
        spec["family"] = _stored_family(manifest)
    baseline = manifest.parent / "configs" / "baseline.yaml"
    supplied = objective is not None or metric is not None
    if baseline.exists() and supplied:
        raise ExperimentError(
            "An existing baseline is never overwritten. Open the method and edit YAML or use Hydra."
        )
    template = "# Optional user-owned model parameters.\n{}\n"
    if spec["name"] == "lightgbm" and not baseline.exists():
        from .lightgbm import baseline_template

        template = baseline_template(objective, metric)
    elif spec["name"] != "lightgbm" and supplied:
        raise ExperimentError(
            "objective / metric initialization is currently supported for LightGBM only"
        )
    elif spec["name"] == "esmm_mmoe" and not baseline.exists():
        from .esmm_mmoe import baseline_template

        template = baseline_template()
    store = ExperimentStore(
        project.path, spec["prefix"], method=spec, comparison_partitions=comparison_partitions
    )
    for directory in ("configs", "reports"):
        (store.path / directory).mkdir(exist_ok=True)
    try:
        atomic_write(
            store.path / "configs" / "baseline.yaml",
            lambda p: p.write_text(template, encoding="utf-8"),
            exclusive=True,
        )
    except FileExistsError:
        if supplied:
            raise ExperimentError("An existing baseline is never overwritten") from None
    from ..method import MethodExperiment

    return MethodExperiment(store)


def open_method(project, method):
    spec = _identity(method)
    manifest = project.path / spec["prefix"] / "method.json"
    if not manifest.is_file():
        raise ExperimentError(f"Method is not initialized: {method}")
    spec["family"] = _stored_family(manifest)
    from ..method import MethodExperiment

    return MethodExperiment(
        ExperimentStore(project.path, spec["prefix"], method=spec, create=False)
    )
