from __future__ import annotations

from pathlib import Path

import torch

from .config import get_config
from .encoder import HaMeREncoder
from .model import HAMER


def _resolve_hamer_config_file(config_file: str | None, checkpoint: str | None) -> str:
    if config_file is not None:
        return str(config_file)
    if checkpoint is None:
        raise ValueError("config_file is required when checkpoint is not provided")
    checkpoint_path = Path(checkpoint)
    return str(checkpoint_path.parent.parent / "model_config.yaml")


def _strip_backbone_pretrained_weights(model_cfg):
    backbone_cfg = model_cfg.MODEL.BACKBONE
    has_weights = False
    getter = getattr(backbone_cfg, "get", None)
    if callable(getter):
        has_weights = getter("PRETRAINED_WEIGHTS", None) is not None
    elif hasattr(backbone_cfg, "PRETRAINED_WEIGHTS"):
        has_weights = getattr(backbone_cfg, "PRETRAINED_WEIGHTS") is not None

    if has_weights:
        model_cfg.defrost()
        if hasattr(backbone_cfg, "pop"):
            backbone_cfg.pop("PRETRAINED_WEIGHTS")
        elif hasattr(backbone_cfg, "PRETRAINED_WEIGHTS"):
            delattr(backbone_cfg, "PRETRAINED_WEIGHTS")
        model_cfg.freeze()
    return model_cfg


def _load_hamer_checkpoint_state_dict(
    checkpoint: str,
    map_location: str | torch.device = "cpu",
):
    checkpoint_obj = torch.load(str(checkpoint), map_location=map_location, weights_only=False)
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


def _make_hamer_model(
    *,
    config_file: str | None = None,
    checkpoint: str | None = None,
    pretrained: bool = True,
    cache_dir: str = "./_DATA",
    update_cachedir: bool = True,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
    model_cls=HAMER,
    eval_mode: bool = False,
    return_info: bool = False,
):
    resolved_config = _resolve_hamer_config_file(config_file=config_file, checkpoint=checkpoint)
    model_cfg = get_config(
        resolved_config,
        merge=True,
        cache_dir=cache_dir,
        update_cachedir=update_cachedir,
    )
    model_cfg = _strip_backbone_pretrained_weights(model_cfg)

    model = model_cls(model_cfg)
    incompatible = None

    if pretrained:
        if checkpoint is None:
            raise ValueError("checkpoint is required when pretrained=True")
        state_dict = _load_hamer_checkpoint_state_dict(checkpoint=checkpoint, map_location=map_location)
        incompatible = model.load_state_dict(state_dict, strict=strict)

    if eval_mode:
        model.eval()

    if return_info:
        return model, model_cfg, incompatible
    return model


def hamer(
    *,
    config_file: str | None = None,
    checkpoint: str | None = None,
    pretrained: bool = True,
    cache_dir: str = "./_DATA",
    update_cachedir: bool = True,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
    eval_mode: bool = False,
):
    return _make_hamer_model(
        config_file=config_file,
        checkpoint=checkpoint,
        pretrained=pretrained,
        cache_dir=cache_dir,
        update_cachedir=update_cachedir,
        strict=strict,
        map_location=map_location,
        eval_mode=eval_mode,
        return_info=False,
    )


def load_hamer(
    checkpoint_path: str,
    cache_dir: str = "./_DATA",
    map_location: str | torch.device = "cpu",
    strict: bool = False,
):
    return _make_hamer_model(
        checkpoint=checkpoint_path,
        pretrained=True,
        cache_dir=cache_dir,
        update_cachedir=True,
        strict=strict,
        map_location=map_location,
        return_info=True,
    )


def hamer_encoder(
    *,
    config_file: str | None = None,
    checkpoint: str | None = None,
    pretrained: bool = True,
    cache_dir: str = "./_DATA",
    update_cachedir: bool = True,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
    eval_mode: bool = False,
):
    return _make_hamer_model(
        config_file=config_file,
        checkpoint=checkpoint,
        pretrained=pretrained,
        cache_dir=cache_dir,
        update_cachedir=update_cachedir,
        strict=strict,
        map_location=map_location,
        model_cls=HaMeREncoder,
        eval_mode=eval_mode,
        return_info=False,
    )


def load_hamer_encoder(
    checkpoint_path: str,
    cache_dir: str = "./_DATA",
    map_location: str | torch.device = "cpu",
    strict: bool = False,
):
    return _make_hamer_model(
        checkpoint=checkpoint_path,
        pretrained=True,
        cache_dir=cache_dir,
        update_cachedir=True,
        strict=strict,
        map_location=map_location,
        model_cls=HaMeREncoder,
        return_info=True,
    )
