"""Snapshot configuration sources without composing or executing model code."""

from collections.abc import Mapping
from pathlib import Path

from phl_risk.exceptions import RunError

from .._utils import require


def configuration_snapshot(config):
    from ..adapters.hydra import ComposedConfig
    from ..adapters.yaml import read_yaml_config
    from .session import parameter_snapshot

    yaml = require("yaml", "yaml")
    source = None
    origin = {}
    overrides = []
    if isinstance(config, ComposedConfig):
        value = config.config
        source = config.source_text
        overrides = list(config.overrides)
        origin = {"source_path": str(config.source_path), "source_kind": "hydra"}
    elif isinstance(config, (str, Path)):
        path = Path(config).expanduser().resolve()
        # Parse exactly the text that is retained, even if the source changes later.
        value, source = read_yaml_config(path, with_source=True)
        origin = {"source_path": str(path), "source_kind": "yaml"}
    elif isinstance(config, Mapping):
        value = config
        origin = {"source_kind": "mapping"}
    else:
        raise RunError("config must be a mapping, YAML path, or ComposedConfig")
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
