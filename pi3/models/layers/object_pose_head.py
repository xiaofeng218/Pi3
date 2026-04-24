from __future__ import annotations

import torch
import torch.nn as nn


class ObjectPoseHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 1024):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rot_head = nn.Linear(hidden_dim, 6)
        self.trans_head = nn.Linear(hidden_dim, 3)
        self.scale_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, object_query_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(object_query_feat)
        rot6d = self.rot_head(hidden)
        trans = self.trans_head(hidden)
        log_scale = self.scale_head(hidden)
        scale = torch.exp(log_scale)
        return {
            "rot6d": rot6d,
            "trans": trans,
            "log_scale": log_scale,
            "scale": scale,
        }
