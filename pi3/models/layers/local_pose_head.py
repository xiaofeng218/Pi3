from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_head import ResConvBlock


class LocalPoseHeadTrunk(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, patch_h: int = 16, patch_w: int = 12):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.res_conv = nn.ModuleList([ResConvBlock(self.in_dim, self.in_dim) for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 4:
            leading_shape = feat.shape[:2]
            feat = feat.reshape(-1, feat.shape[-2], feat.shape[-1])
        elif feat.ndim == 3:
            leading_shape = feat.shape[:1]
        else:
            raise ValueError(f"Expected feat with shape (BN, HW, C) or (B, N, HW, C), got {tuple(feat.shape)}")

        bn, hw, c = feat.shape
        expected_hw = self.patch_h * self.patch_w
        if hw != expected_hw:
            raise ValueError(f"Expected {expected_hw} patch tokens, got {hw}")
        if c != self.in_dim:
            raise ValueError(f"Expected feature dim {self.in_dim}, got {c}")

        for block in self.res_conv:
            feat = block(feat)
        feat = self.avgpool(feat.permute(0, 2, 1).reshape(bn, c, self.patch_h, self.patch_w).contiguous())
        feat = feat.view(bn, -1)
        feat = self.more_mlps(feat)

        if len(leading_shape) == 2:
            return feat.reshape(*leading_shape, -1)
        return feat


def maybe_mask_feature(feat: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return feat
    if feat.shape[:2] != valid_mask.shape:
        raise ValueError(f"valid_mask must match leading feature shape {tuple(feat.shape[:2])}, got {tuple(valid_mask.shape)}")
    return feat * valid_mask.unsqueeze(-1).to(dtype=feat.dtype)
