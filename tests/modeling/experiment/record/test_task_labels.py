from phl_risk.modeling.experiment import Experiment


def custom_loss(predictions, data):
    return predictions, predictions


def custom_score(predictions, data):
    return "score", 0.5, True


def test_compare_displays_task_names_without_changing_recorded_parameters(tmp_path):
    space = Experiment(root=tmp_path).initialize(method="generic")
    with space.start_run(
        params={"params": {"objective": "binary", "metric": ["auc", "binary_logloss"]}}
    ) as first:
        first.log_metrics({"valid": {"auc": 0.8}})
    with space.start_run(
        params={
            "params": {"objective": custom_loss, "metric": "None"},
            "train": {"feval": custom_score},
        }
    ) as second:
        second.log_metrics({"valid": {"auc": 0.7}})
    before = second.record.config
    frame = space.compare(params=[])
    assert frame.columns.tolist() == [
        "run",
        "name",
        "created_at",
        "objective",
        "metric",
        "feval",
        "valid_auc",
        "params",
    ]
    assert frame.objective.tolist() == ["binary", "custom_loss"]
    assert frame.metric.tolist() == [["auc", "binary_logloss"], "None"]
    assert frame.feval.tolist() == ["未记录", "custom_score"]
    assert frame.valid_auc.tolist() == [0.8, 0.7]
    assert second.record.config == before
    assert "source_sha256" in before["params"]["objective"]


def test_flat_legacy_and_missing_labels_do_not_guess_from_metadata(tmp_path):
    space = Experiment(root=tmp_path).initialize(method="generic")
    configurations = [
        {"objective": "regression", "metric": "rmse"},
        {"model": {"params": {"objective": "binary", "metric": "auc"}}},
        {"metadata": {"objective": "not a training objective"}},
    ]
    for config in configurations:
        with space.start_run(params=config):
            pass
    frame = space.compare(params=[])
    assert frame.objective.tolist() == ["regression", "binary", "未记录"]
    assert frame.metric.tolist() == ["rmse", "auc", "未记录"]
    assert "feval" not in frame
    assert space.compare(runs=[], params=[]).columns.tolist() == [
        "run",
        "name",
        "created_at",
        "objective",
        "metric",
        "params",
    ]


def test_multiple_custom_evaluation_names_and_missing_values(tmp_path):
    space = Experiment(root=tmp_path).initialize(method="generic")
    with space.start_run(
        params={"objective": None, "metric": None, "feval": [custom_score, custom_loss]}
    ):
        pass
    frame = space.compare(params=[])
    assert frame.iloc[0].objective == "未记录"
    assert frame.iloc[0].metric == "未记录"
    assert frame.iloc[0].feval == ["custom_score", "custom_loss"]


def test_compare_names_distinguish_repeated_execution_and_remain_compatible(tmp_path):
    space = Experiment(root=tmp_path).initialize(method="generic")
    for score in [0.7, 0.8]:
        with space.start_run(name="baseline", params={}) as attempt:
            attempt.log_metrics({"valid": {"auc": score}})
    frame = space.compare()
    assert frame.columns[:3].tolist() == ["run", "name", "created_at"]
    assert frame.name.tolist() == ["baseline", "baseline"]
    assert frame.run.nunique() == 2
    assert frame.valid_auc.tolist() == [0.7, 0.8]
    assert space.compare(fields=["name"]).equals(frame)
    assert space.compare(runs=[]).columns[:3].tolist() == ["run", "name", "created_at"]
