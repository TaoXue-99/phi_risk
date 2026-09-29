"""Configuration data and snapshots; no training, sessions or Hydra imports."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from phl_risk.exceptions import ExperimentError, RunError

from ._snapshot import parameter_snapshot
from ._utils import require


@dataclass(frozen=True)
class ConfigSource:
    """One independent, resolved configuration derived from an unchanged source.

    ``config`` is an ordinary dict for native libraries and runtime callable binding.
    Each compose call owns its values. Finish edits before starting a Run.
    """

    config: dict
    overrides: tuple[str, ...]
    source_text: str
    source_path: Path
    source_kind: str = "hydra"

    def save(self, path: str | Path) -> Path:
        """Export a complete YAML snapshot to a new file; never overwrite a source.

        Callable values are identity records, not executable functions. Rebind them
        in Python when loading the exported configuration. Hydra provenance remains
        on this object and in Runs that receive it directly.
        """
        from .record.artifact import atomic_write

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


def read_yaml_config(path, *, with_source=False):
    yaml = require("yaml", "yaml")
    try:
        text = Path(path).read_text(encoding="utf-8")
        value = yaml.safe_load(text)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ExperimentError(f"Cannot read YAML: {error}") from error
    if not isinstance(value, dict):
        raise ExperimentError("YAML parameters must be a mapping")
    if "defaults" in value or "${" in text:
        raise ExperimentError("Use compose_config for Hydra composition/interpolation")
    return (value, text) if with_source else value


def configuration_snapshot(config):
    yaml = require("yaml", "yaml")
    source = None
    origin = {}
    overrides = []
    if isinstance(config, ConfigSource):
        value = config.config
        source = config.source_text
        overrides = list(config.overrides)
        origin = {"source_path": str(config.source_path), "source_kind": config.source_kind}
    elif isinstance(config, (str, Path)):
        path = Path(config).expanduser().resolve()
        # Parse exactly the text that is retained, even if the source changes later.
        value, source = read_yaml_config(path, with_source=True)
        origin = {"source_path": str(path), "source_kind": "yaml"}
    elif isinstance(config, Mapping):
        value = config
        origin = {"source_kind": "mapping"}
    else:
        raise RunError("config must be a mapping, YAML path, or ConfigSource")
    if not isinstance(value, Mapping):
        raise RunError("config must resolve to a mapping")
    snapshot = parameter_snapshot(value)
    facts = {"supplied": snapshot, "resolved": snapshot, "overrides": overrides, **origin}
    artifacts = {
        "config": ("config.yaml", yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False)),
        "overrides": ("overrides.yaml", yaml.safe_dump(overrides, allow_unicode=True)),
    }
    if source is not None:
        artifacts["config_source"] = ("config.source.yaml", source)
    return facts, artifacts
