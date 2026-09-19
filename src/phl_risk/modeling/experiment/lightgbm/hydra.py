"""Optional Hydra Compose adapter; no decorators, cwd changes or output management."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from phl_risk.exceptions import ExperimentError

from ._utils import detached, require


@dataclass(frozen=True)
class ResolvedConfig:
    config: dict
    overrides: tuple[str, ...]


def compose_lightgbm_config(
    *, config_dir: str | Path, config_name: str, overrides: Sequence[str] = ()
) -> ResolvedConfig:
    hydra = require("hydra", "hydra")
    omega = require("omegaconf", "hydra")
    if isinstance(overrides, str) or any(not isinstance(v, str) for v in overrides):
        raise ExperimentError("Hydra overrides must be a sequence of strings")
    try:
        with hydra.initialize_config_dir(
            config_dir=str(Path(config_dir).expanduser().resolve()), version_base=None
        ):
            config = hydra.compose(config_name=config_name, overrides=list(overrides))
        plain = omega.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        if not isinstance(plain, dict):
            raise ExperimentError("Composed configuration must be a mapping")
        return ResolvedConfig(detached(plain), tuple(overrides))
    except ExperimentError:
        raise
    except Exception as exc:
        raise ExperimentError(f"Could not compose LightGBM config: {exc}") from exc
