import math

import pytest

torch = pytest.importorskip("torch")


def network(**kwargs):
    from phl_risk.modeling.models.esmm_mmoe import ESMMMoE

    return ESMMMoE(num_continuous=3, share_dim=8, base_dim=4, expert_units=4, **kwargs)


@pytest.mark.parametrize("cardinalities", [[], [3, 5]])
def test_joint_probability_and_both_towers_receive_gradient(cardinalities):
    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss

    torch.manual_seed(7)
    net = network(categorical_cardinalities=cardinalities, dropout=0)
    cat = torch.zeros((6, len(cardinalities)), dtype=torch.long)
    output = net(cat, torch.randn(6, 3))
    assert set(output) == {"ctr", "cvr", "ctcvr"}
    torch.testing.assert_close(output["ctcvr"], output["ctr"] * output["cvr"])
    assert output["ctr"].shape == (6, 1)
    assert torch.all(output["ctcvr"] <= output["ctr"])
    target = {"ctr": torch.ones(6), "ctcvr": torch.tensor([0, 1, 0, 1, 0, 1])}
    ESMMLoss()(output, target).backward()
    for module in (net.share_network, net.mmoe, net.p1_network, net.p2_network):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum().item() for p in module.parameters()) > 0


def test_weighted_loss_matches_hand_calculation():
    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss

    output = {"ctr": torch.tensor([[0.8], [0.2]]), "ctcvr": torch.tensor([[0.4], [0.1]])}
    target = {"ctr": torch.tensor([1, 0]), "ctcvr": torch.tensor([1, 0])}
    components = ESMMLoss(ctr_weight=2, ctcvr_weight=3).components(output, target)
    assert components["ctr_loss"].item() == pytest.approx(-math.log(0.8))
    assert components["ctcvr_loss"].item() == pytest.approx(-math.log(0.36) / 2)
    assert components["loss"].item() == pytest.approx(-2 * math.log(0.8) - 1.5 * math.log(0.36))


@pytest.mark.parametrize(
    "target",
    [
        {"ctr": [0, 1], "ctcvr": [1, 0]},
        {"ctr": [0.5, 1], "ctcvr": [0, 1]},
        {"ctr": [float("nan"), 1], "ctcvr": [0, 1]},
        {"ctr": [1], "ctcvr": [1]},
    ],
)
def test_invalid_targets_rejected(target):
    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss

    output = {"ctr": torch.full((2, 1), 0.5), "ctcvr": torch.full((2, 1), 0.25)}
    with pytest.raises(ValueError):
        ESMMLoss()(output, {k: torch.tensor(v) for k, v in target.items()})


@pytest.mark.parametrize("weights", [(0, 0), (-1, 1), (float("nan"), 1)])
def test_invalid_loss_weights_rejected(weights):
    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss

    with pytest.raises(ValueError):
        ESMMLoss(*weights)


def test_schema_and_category_range_checks():
    net = network(categorical_cardinalities=[3])
    with pytest.raises(ValueError):
        net(torch.tensor([[3]]), torch.zeros(1, 3))
    with pytest.raises(ValueError):
        net(torch.tensor([[-1]]), torch.zeros(1, 3))
    with pytest.raises(ValueError):
        net(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 4))
    with pytest.raises(ValueError):
        net(torch.zeros(1, 1), torch.zeros(1, 3))


def test_zero_shared_vector_normalization_and_single_sample_inference():
    net = network(normalize_shared=True, batch_norm=True).eval()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        output = net(torch.empty(1, 0, dtype=torch.long), torch.zeros(1, 3))
    assert output["ctr"].item() == 0.5
    assert output["ctcvr"].item() == 0.25


def test_configuration_roundtrip():
    original = network(categorical_cardinalities=[4, 7], embedding_dim=2)
    restored = type(original)(**original.get_config())
    restored.load_state_dict(original.state_dict())
    original.eval()
    restored.eval()
    cat, cont = torch.zeros(3, 2, dtype=torch.long), torch.randn(3, 3)
    torch.testing.assert_close(original(cat, cont), restored(cat, cont))
