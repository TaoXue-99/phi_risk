"""Optional plain YAML reader. No model parameter schema or training defaults."""

from pathlib import Path

from phl_risk.exceptions import ExperimentError

from .._utils import require


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
