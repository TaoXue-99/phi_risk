import math

import pytest

torch = pytest.importorskip("torch")
DataLoader, TensorDataset = torch.utils.data.DataLoader, torch.utils.data.TensorDataset


def data(batch_size=4, one_class=False):
    torch.manual_seed(10)
    y1 = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.float)
    y12 = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float)
    if one_class:
        y1.zero_()
        y12.zero_()
    return DataLoader(
        TensorDataset(torch.empty(9, 0, dtype=torch.long), torch.randn(9, 2), y1, y12),
        batch_size=batch_size,
    )


def tiny():
    from phl_risk.modeling.models.esmm_mmoe import ESMMMoE

    return ESMMMoE(num_continuous=2, share_dim=6, base_dim=4, expert_units=3, dropout=0)


def test_evaluation_sample_weighted_and_auc_matches_whole_dataset():
    from sklearn.metrics import roc_auc_score

    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss
    from phl_risk.modeling.training.esmm import evaluate_esmm

    net, loss = tiny(), ESMMLoss(2, 3)
    net.train()
    scores, missing = evaluate_esmm(net, data(4), loss)
    assert net.training
    full = next(iter(data(9)))
    output = net(full[0], full[1])
    expected = loss.components(output, {"ctr": full[2], "ctcvr": full[3]})
    assert scores["loss"] == pytest.approx(expected["loss"].item(), rel=1e-6)
    assert scores["ctr_auc"] == pytest.approx(roc_auc_score(full[2], output["ctr"].detach()))
    assert not missing


def test_single_class_auc_omitted_empty_loader_rejected():
    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss
    from phl_risk.modeling.training.esmm import evaluate_esmm

    scores, missing = evaluate_esmm(tiny(), data(one_class=True), ESMMLoss())
    assert "ctr_auc" not in scores and "ctcvr_auc" not in scores
    assert set(missing) == {"ctr_auc", "ctcvr_auc"}
    assert all(math.isfinite(v) for v in scores.values())
    with pytest.raises(ValueError, match="empty"):
        evaluate_esmm(tiny(), [], ESMMLoss())


@pytest.mark.parametrize("interval,exponent", [("epoch", 2), ("batch", 6)])
def test_torchkeras_real_fit_scheduler_and_best_weights(tmp_path, interval, exponent):
    pytest.importorskip("torchkeras")
    from torchkeras import KerasModel

    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss
    from phl_risk.modeling.training.torchkeras import ESMMKerasModel

    original_runner = KerasModel.StepRunner
    net = tiny()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = ESMMKerasModel(
        net, ESMMLoss(), optimizer=optimizer, lr_scheduler=scheduler, scheduler_interval=interval
    )

    class Callback:
        epochs = 0

        def on_validation_epoch_end(self, model):
            self.epochs += 1
            assert "val_ctcvr_auc" in model.history

    callback = Callback()
    checkpoint = tmp_path / "best.pt"
    history = trainer.fit(
        data(),
        data(),
        epochs=2,
        patience=3,
        ckpt_path=str(checkpoint),
        plot=False,
        quiet=True,
        cpu=True,
        callbacks=[callback],
    )
    assert callback.epochs == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5**exponent)
    assert len(history) == 2
    assert KerasModel.StepRunner is original_runner
    torch.testing.assert_close(net.state_dict(), torch.load(checkpoint, weights_only=True))
    result = trainer.evaluate(data(), quiet=True)
    assert result["val_loss"] == pytest.approx(history["val_loss"].min(), rel=1e-6)


def test_native_training_reduces_loss():
    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss

    net, loss = tiny(), ESMMLoss()
    batch = next(iter(data(9)))
    targets = {"ctr": batch[2], "ctcvr": batch[3]}
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
    before = loss(net(batch[0], batch[1]), targets).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss(net(batch[0], batch[1]), targets).backward()
        optimizer.step()
    assert loss(net(batch[0], batch[1]), targets).item() < before * 0.9
