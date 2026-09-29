"""Optional model-independent Hydra composition. Native Hydra is equally usable."""

from pathlib import Path

from phl_risk.exceptions import ExperimentError

from .._utils import require

# Backward-compatible name; Record depends only on the neutral ConfigSource.
from ..configuration import ConfigSource as ComposedConfig


def compose_config(*, config_dir, config_name, overrides=()):
    """Derive a new complete parameter set using native Hydra override syntax.

    Never writes the baseline or reuses a previous composed configuration.
    """
    hydra = require("hydra", "hydra")
    omega = require("omegaconf", "hydra")
    if isinstance(overrides, str):
        raise ExperimentError("Hydra overrides must contain strings")
    try:
        overrides = tuple(overrides)
    except TypeError as error:
        raise ExperimentError("Hydra overrides must contain strings") from error
    if any(not isinstance(v, str) for v in overrides):
        raise ExperimentError("Hydra overrides must contain strings")
    root = Path(config_dir).expanduser().resolve()
    path = root / config_name
    if path.suffix not in (".yaml", ".yml"):
        path = path.with_suffix(".yaml")
    try:
        source = path.read_text(encoding="utf-8")
        with hydra.initialize_config_dir(config_dir=str(root), version_base=None):
            config = hydra.compose(config_name=config_name, overrides=list(overrides))
        value = omega.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        if not isinstance(value, dict):
            raise ExperimentError("Composed parameters must be a mapping")
        return ComposedConfig(value, tuple(overrides), source, path)
    except ExperimentError:
        raise
    except Exception as error:
        raise ExperimentError(f"Could not compose configuration: {error}") from error
