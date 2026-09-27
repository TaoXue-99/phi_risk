"""Optional model-independent Hydra composition. Native Hydra is equally usable."""

from dataclasses import dataclass
from pathlib import Path

from phl_risk.exceptions import ExperimentError

from .._utils import require


@dataclass(frozen=True)
class ComposedConfig:
    """One independent, resolved configuration derived from an unchanged source.

    ``config`` is an ordinary dict for native libraries and runtime callable binding.
    Each compose call owns its values. Finish edits before starting a Run.
    """

    config: dict
    overrides: tuple[str, ...]
    source_text: str
    source_path: Path

    def save(self, path: str | Path) -> Path:
        """Export a complete YAML snapshot to a new file; never overwrite a source.

        Callable values are identity records, not executable functions. Rebind them
        in Python when loading the exported configuration. Hydra provenance remains
        on this object and in Runs that receive it directly.
        """
        from ..record.artifact import atomic_write
        from ..record.session import parameter_snapshot

        yaml = require("yaml", "yaml")
        target = Path(path).expanduser().resolve()
        text = yaml.safe_dump(parameter_snapshot(self.config), allow_unicode=True, sort_keys=False)
        try:
            atomic_write(target, lambda p: p.write_text(text, encoding="utf-8"), exclusive=True)
        except FileExistsError as error:
            raise ExperimentError(f"Configuration already exists: {target}") from error
        except OSError as error:
            raise ExperimentError(f"Could not save configuration: {error}") from error
        return target


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
