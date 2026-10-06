"""Epoch checkpoints, explicit resume state, and model-only averaging."""
import random
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, state, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "state": state, "metadata": metadata, "rng": rng}
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, model=None, optimizer=None, restore_rng=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        torch.set_rng_state(payload["rng"]["torch"])
        random.setstate(payload["rng"]["python"])
        if torch.cuda.is_available() and "cuda" in payload["rng"]:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return payload


def average_checkpoints(paths, output, *, count=10):
    if count < 1 or len(paths) < count:
        raise ValueError(f"need at least {count} epoch checkpoints")
    payloads = [load_checkpoint(p) for p in paths[-count:]]
    metadata = payloads[-1]["metadata"]
    def comparable(value):
        result = dict(value)
        config = dict(result.get("config", {}))
        for limit in ("max_epochs", "max_updates"):
            config.pop(limit, None)
        if "config" in result:
            result["config"] = config
        return result
    if any(comparable(p["metadata"]) != comparable(metadata) for p in payloads):
        raise ValueError("cannot average checkpoints with different metadata")
    states = [p["model"] for p in payloads]
    if any(s.keys() != states[0].keys() for s in states):
        raise ValueError("checkpoint parameter names differ")
    average = {}
    for name, tensor in states[0].items():
        values = [s[name] for s in states]
        if any(v.shape != tensor.shape or v.dtype != tensor.dtype for v in values):
            raise ValueError(f"checkpoint tensor mismatch: {name}")
        if tensor.is_floating_point():
            average[name] = torch.stack([v.double() for v in values]).mean(0).to(tensor.dtype)
        else:
            if any(not torch.equal(v, tensor) for v in values):
                raise ValueError(f"non-floating buffer differs: {name}")
            average[name] = tensor.clone()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": average, "metadata": metadata,
                "averaged_checkpoints": [str(p) for p in paths[-count:]]}, output)
    return output
