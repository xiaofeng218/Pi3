from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SCALE_MIN = -10.0
LOG_SCALE_MAX = 10.0


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
        self.trans_dir_head = nn.Linear(hidden_dim, 3)
        self.trans_scale_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.trans_scale_head.bias)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, object_query_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(object_query_feat)
        rot6d = self.rot_head(hidden)
        trans_dir = F.normalize(self.trans_dir_head(hidden), dim=-1, eps=1e-6)
        trans_log_scale = self.trans_scale_head(hidden)
        trans_scale = torch.exp(trans_log_scale.clamp(min=LOG_SCALE_MIN, max=LOG_SCALE_MAX))
        trans = trans_dir * trans_scale
        log_scale = self.scale_head(hidden)
        scale = torch.exp(log_scale.clamp(min=LOG_SCALE_MIN, max=LOG_SCALE_MAX))
        return {
            "rot6d": rot6d,
            "trans_dir": trans_dir,
            "trans_log_scale": trans_log_scale,
            "trans_scale": trans_scale,
            "trans": trans,
            "log_scale": log_scale,
            "scale": scale,
        }
