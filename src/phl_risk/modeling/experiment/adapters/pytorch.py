"""Optional PyTorch state_dict persistence, independent of the training framework."""

from phl_risk.exceptions import RunError

from .._utils import require


def record_pytorch(run, model, *, model_config):
    """Snapshot CPU tensors and explicit constructor parameters immediately.

    Pass an unwrapped nn.Module (e.g. accelerator.unwrap_model(net)). This is
    an inference checkpoint, not an optimizer/scheduler/RNG resume checkpoint.
    Feature schemas and preprocessing artifacts are recorded separately by callers.
    """
    torch = require("torch", "pytorch")
    if not isinstance(model, torch.nn.Module):
        raise RunError("record_pytorch requires a native torch.nn.Module")
    state = model.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise RunError("Only tensor state_dict entries are supported")
    snapshot = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
    # Preserve module version metadata used by load_state_dict migrations.
    from collections import OrderedDict

    snapshot = OrderedDict(snapshot)
    if hasattr(state, "_metadata"):
        from copy import deepcopy

        snapshot._metadata = deepcopy(state._metadata)
    run.log_model_info(family="deep", backend="pytorch", backend_version=str(torch.__version__))
    run.log_json("model_config", model_config)
    run.log_artifact("model", "model.pt", lambda path: torch.save(snapshot, path), format="pytorch")


def load_pytorch(run, model):
    """Verify and load weights strictly into a caller-constructed native model.

    Preserves the supplied model's device and train/eval mode. Call .eval() for
    inference. No automatic imports, pickled model objects or trainer restoration.
    """
    torch = require("torch", "pytorch")
    if not isinstance(model, torch.nn.Module):
        raise RunError("load_pytorch requires a native torch.nn.Module")
    if (
        run.model["backend"] != "pytorch"
        or run.artifacts.get("model", {}).get("format") != "pytorch"
    ):
        raise RunError("Run has no native PyTorch model")
    path = run.artifact_path("model")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError, TypeError, OSError) as error:
        raise RunError(f"Cannot load PyTorch model: {error}") from error
    return model
