"""Native LightGBM training with optional experiment recording.

python examples/modeling/lightgbm_experiment.py --root ./experiments --hydra
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.datasets import make_classification

from phl_risk.metrics import auc_score
from phl_risk.modeling.experiment import Experiment
from phl_risk.modeling.experiment.adapters.lightgbm import (
    enable_lightgbm_compatibility,
    load_lightgbm,
    record_lightgbm,
)


def main(root="./experiments", *, hydra=False, overrides=()):
    if tuple(int(v) for v in lgb.__version__.split(".")[:2]) < (4, 6):
        enable_lightgbm_compatibility()  # Explicit bridge for this repo's modern dependencies.
    x, y = make_classification(n_samples=1200, n_features=20, n_informative=8, random_state=2026)
    data = pd.DataFrame(x, columns=[f"feature_{i:02d}" for i in range(20)])
    project = Experiment(root=root)
    space = (
        project.open_method("lgb")
        if (project.path / "lgb" / "method.json").exists()
        else project.initialize(
            method="lgb",
            objective="binary",
            metric=["auc", "binary_logloss"],
            comparison_partitions=["train", "valid"],
        )
    )
    config_path = space.config_dir / "baseline.yaml"
    if hydra:
        from phl_risk.modeling.experiment.adapters.hydra import compose_config

        selected_config = compose_config(
            config_dir=config_path.parent, config_name=config_path.stem, overrides=overrides
        )
        cfg = selected_config.config
    else:
        if overrides:
            raise ValueError("External overrides require --hydra")
        cfg = yaml.safe_load(config_path.read_text())
        selected_config = config_path
    with space.start_run(
        name="native_binary", config=selected_config, comparison_partitions=["train", "valid"]
    ) as run:
        train = lgb.Dataset(data.iloc[:800], label=y[:800])
        valid = train.create_valid(data.iloc[800:], label=y[800:])
        history = {}
        model = lgb.train(
            params=cfg["params"],
            train_set=train,
            num_boost_round=cfg["train"]["num_boost_round"],
            valid_sets=[train, valid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.record_evaluation(history),
                lgb.early_stopping(**cfg["train"]["early_stopping"]),
                lgb.log_evaluation(**cfg["train"]["log_evaluation"]),
            ],
        )
        iteration = model.best_iteration or model.current_iteration()
        run.log_metrics(
            {
                "train": {
                    "auc": auc_score(
                        y[:800], model.predict(data.iloc[:800], num_iteration=iteration)
                    )
                },
                "valid": {
                    "auc": auc_score(
                        y[800:], model.predict(data.iloc[800:], num_iteration=iteration)
                    )
                },
            }
        )
        run.log_json("eval_history", history)
        record_lightgbm(run, model, num_iteration=iteration)
    restored = Experiment(root=root).open_method("lgb")
    assert load_lightgbm(restored.get_run(run.record.run)).feature_name() == list(data.columns)
    print(restored.compare(params=["max_depth", "num_leaves"]))
    return run.record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./experiments"))
    parser.add_argument("--hydra", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. params.max_depth=3")
    args = parser.parse_args()
    main(args.root, hydra=args.hydra, overrides=args.overrides)
