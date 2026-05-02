from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def unwrap_model(model):
    return getattr(model, "module", model)


def collect_trainable_state_dict(model) -> dict[str, torch.Tensor]:
    base_model = unwrap_model(model)
    trainable_names = {
        name for name, param in base_model.named_parameters() if param.requires_grad
    }
    state_dict = base_model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if name in trainable_names
    }


def save_trainable_checkpoint(
    save_dir: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    extra: dict[str, Any] | None = None,
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    trainable_state = collect_trainable_state_dict(model)
    torch.save(trainable_state, save_path / "trainable_model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), save_path / "scheduler.pt")

    base_model = unwrap_model(model)
    trainable_names = sorted(
        name for name, param in base_model.named_parameters() if param.requires_grad
    )
    meta = {
        "checkpoint_version": 1,
        "num_trainable_tensors": len(trainable_state),
        "trainable_param_names": trainable_names,
    }
    if extra:
        meta.update(extra)
    (save_path / "trainer_state.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )


def load_trainable_checkpoint(
    load_dir: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    load_path = Path(load_dir)
    trainable_file = load_path / "trainable_model.pt"
    if not trainable_file.exists():
        raise FileNotFoundError(f"Missing trainable checkpoint: {trainable_file}")

    trainable_state = torch.load(trainable_file, map_location=map_location, weights_only=False)
    base_model = unwrap_model(model)
    incompatible = base_model.load_state_dict(trainable_state, strict=False)

    if optimizer is not None:
        optimizer_file = load_path / "optimizer.pt"
        if optimizer_file.exists():
            optimizer.load_state_dict(torch.load(optimizer_file, map_location=map_location, weights_only=False))

    if scheduler is not None:
        scheduler_file = load_path / "scheduler.pt"
        if scheduler_file.exists():
            scheduler.load_state_dict(torch.load(scheduler_file, map_location=map_location, weights_only=False))

    meta_path = load_path / "trainer_state.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "meta": meta,
    }
