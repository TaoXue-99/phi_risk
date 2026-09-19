"""Run from the repository: python examples/modeling/lightgbm_experiment.py.

Install phl-risk[lightgbm,hydra]. Output persists under ./experiments by default;
--root changes the parent directory. Repeated invocations append new unique runs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from phl_risk.modeling import DataPlan, ModelPlan
from phl_risk.modeling.experiment.lightgbm import LightGBMExperiment, compose_lightgbm_config
from phl_risk.modeling.goal import BinaryClassification
from phl_risk.modeling.plan import ColumnSplitter, FeatureSpec, PartitionSpec, RoleSpec, SplitSpec
from phl_risk.modeling.strategy import LightGBM


def main(root: str | Path = "./experiments") -> LightGBMExperiment:
    # Plan declares learning semantics and data organization; it performs no training.
    model_plan = ModelPlan(BinaryClassification(), LightGBM())
    features = [f"feature_{i:02d}" for i in range(20)]
    data_plan = DataPlan(
        roles=RoleSpec(target="label", weight="sample_weight"),
        features=FeatureSpec(numerical=features),
        split=SplitSpec(
            ColumnSplitter("partition"),
            PartitionSpec(train="train", valid="valid", test="test", oot="oot"),
        ),
    )
    x, y = make_classification(
        n_samples=1200, n_features=20, n_informative=8, class_sep=0.8, random_state=2026
    )
    data = pd.DataFrame(x, columns=features)
    data["label"] = y
    data["sample_weight"] = np.random.default_rng(2026).uniform(0.5, 1.5, len(data))
    data["partition"] = np.repeat(["train", "valid", "test", "oot"], [720, 160, 160, 160])
    attached = dict(
        name="demo_risk_model", root=root, data=data, model_plan=model_plan, data_plan=data_plan
    )
    exp = LightGBMExperiment(**attached)
    conf = Path(__file__).parent / "conf"
    cfg = compose_lightgbm_config(config_dir=conf, config_name="baseline")
    baseline = exp.run(name="baseline", config=cfg.config, overrides=cfg.overrides)
    exp.set_reference(baseline.run_id)
    print("\nBaseline\n", exp.compare().to_string(index=False))

    selection = exp.select_features(base_run=baseline.run_id, candidate_counts=[12, 8], step=0.25)
    print("\nRFE candidates\n", selection.compare().to_string(index=False))
    # The analyst chooses this candidate; no automatic "best model" is set.
    selected_features = selection.runs[0].features
    for name, overrides in [
        ("depth2", ["model.params.max_depth=2"]),
        ("depth2_bag3", ["model.params.max_depth=2", "model.params.bagging_freq=3"]),
        (
            "lambda20",
            [
                "model.params.max_depth=2",
                "model.params.bagging_freq=3",
                "model.params.lambda_l2=20",
            ],
        ),
    ]:
        changed = compose_lightgbm_config(
            config_dir=conf, config_name="baseline", overrides=overrides
        )
        exp.run(
            name=name,
            config=changed.config,
            overrides=changed.overrides,
            features=selected_features,
        )
    comparison = exp.compare()
    print("\nManual parameter experiments\n", comparison.to_string(index=False))
    # In real work: refine near the chosen feature count, then inspect OOT.
    print(
        "\nFinal review including test and OOT\n",
        exp.compare(include_test=True, include_oot=True).to_string(index=False),
    )

    reloaded = LightGBMExperiment(**attached)
    pd.testing.assert_frame_equal(comparison, reloaded.compare())
    restored = reloaded.get_run(baseline.run_id)
    print(f"\nReloaded {len(reloaded.runs)} runs from {reloaded.path}")
    print(f"Native model: {type(restored.model).__name__}; features: {restored.n_features}")
    return reloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./experiments"))
    main(parser.parse_args().root)
