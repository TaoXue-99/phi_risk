"""Module separation is part of the public architecture contract."""

import ast
import subprocess
import sys
from pathlib import Path


def test_quality_has_no_analysis_or_metrics_dependency():
    for path in Path("src/phl_risk/data_quality").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("phl_risk.analysis", "phl_risk.metrics"))
    # A fresh interpreter detects indirect dependencies as well as direct imports.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import phl_risk.data_quality as quality
from phl_risk.data_quality import checks
assert not hasattr(quality, 'DistributionDriftCheck')
assert not hasattr(checks, 'DistributionDriftCheck')
assert not any(n.startswith(('phl_risk.analysis', 'phl_risk.metrics')) for n in sys.modules)
""",
        ],
        check=True,
    )


def test_version_and_removed_legacy_metric():
    import tomllib

    import phl_risk
    import phl_risk.metrics as metrics

    project = tomllib.loads(Path("pyproject.toml").read_text())
    lock = tomllib.loads(Path("uv.lock").read_text())
    assert phl_risk.__version__ == project["project"]["version"] == "0.5.0"
    assert next(p["version"] for p in lock["package"] if p["name"] == "phl-risk") == "0.5.0"
    assert not hasattr(metrics, "psi_score")
