from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn

from .backbones import create_backbone
from .components.pose_transformer import TransformerDecoder


def _cfg_get(node, key, default=None):
    if isinstance(node, Mapping):
        return node.get(key, default)
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(node, key, default)


def _cfg_to_dict(node) -> dict:
    if node is None:
        return {}
    if isinstance(node, Mapping):
        return dict(node)
    if hasattr(node, "items"):
        return dict(node.items())
    return dict(node)


class HaMeRBackbone(nn.Module):
    """Query-only HaMeR backbone path.

    This module keeps the local HaMeR ViT feature extractor and transformer
    decoder path, but stops at the decoded query token instead of regressing
    MANO parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vit = create_backbone(cfg)

        mano_head_cfg = cfg.MODEL.MANO_HEAD
        joint_rep_type = _cfg_get(mano_head_cfg, "JOINT_REP", "6d")
        joint_rep_dim = {"6d": 6, "aa": 3}[joint_rep_type]
        npose = joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)

        self.input_is_mean_shape = _cfg_get(mano_head_cfg, "TRANSFORMER_INPUT", "zero") == "mean_shape"
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args.update(_cfg_to_dict(_cfg_get(mano_head_cfg, "TRANSFORMER_DECODER", {})))
        self.output_dim = int(transformer_args["dim"])
        self.transformer = TransformerDecoder(**transformer_args)

        self.decshape = nn.Linear(self.output_dim, 10)
        nn.init.zeros_(self.decshape.weight)
        nn.init.zeros_(self.decshape.bias)

        mean_params = np.load(Path(cfg.MANO.MEAN_PARAMS))
        init_betas = torch.from_numpy(mean_params["shape"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_betas", init_betas)

        if self.input_is_mean_shape:
            init_hand_pose = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
            init_cam = torch.from_numpy(mean_params["cam"].astype(np.float32)).unsqueeze(0)
            self.register_buffer("init_hand_pose", init_hand_pose)
            self.register_buffer("init_cam", init_cam)

    def _get_init_token(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.input_is_mean_shape:
            init_hand_pose = self.init_hand_pose.expand(batch_size, -1)
            init_betas = self.init_betas.expand(batch_size, -1)
            init_cam = self.init_cam.expand(batch_size, -1)
            return torch.cat([init_hand_pose, init_betas, init_cam], dim=1)[:, None, :].to(device=device, dtype=dtype)
        return torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)

    def forward(self, crops: torch.Tensor, hand_is_right: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del hand_is_right
        feats = self.vit(crops)
        context = einops.rearrange(feats, "b c h w -> b (h w) c")
        token = self._get_init_token(crops.shape[0], crops.device, context.dtype)
        token_out = self.transformer(token, context=context)
        query = token_out.squeeze(1)
        betas = self.decshape(query) + self.init_betas.expand(crops.shape[0], -1).to(device=query.device, dtype=query.dtype)
        return query, betas
