from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_pose_head import LocalPoseHeadTrunk, maybe_mask_feature

LOG_SCALE_MIN = -10.0
LOG_SCALE_MAX = 10.0


class ObjectPoseHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, patch_h: int = 16, patch_w: int = 12):
        super().__init__()
        self.trunk = LocalPoseHeadTrunk(in_dim=in_dim, hidden_dim=hidden_dim, patch_h=patch_h, patch_w=patch_w)
        self.rot_head = nn.Linear(hidden_dim, 6)
        self.trans_dir_head = nn.Linear(hidden_dim, 3)
        self.trans_scale_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.trans_scale_head.bias)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, object_query_feat: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.trunk(object_query_feat)
        batch_shape = hidden.shape[:2] if hidden.ndim == 3 else hidden.shape[:1]
        hidden = maybe_mask_feature(hidden, valid_mask) if hidden.ndim == 3 else hidden
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        rot6d = self.rot_head(hidden_flat)
        trans_dir = F.normalize(self.trans_dir_head(hidden_flat), dim=-1, eps=1e-6)
        trans_log_scale = self.trans_scale_head(hidden_flat)
        trans_scale = torch.exp(trans_log_scale.clamp(min=LOG_SCALE_MIN, max=LOG_SCALE_MAX))
        trans = trans_dir * trans_scale
        log_scale = self.scale_head(hidden_flat)
        scale = torch.exp(log_scale.clamp(min=LOG_SCALE_MIN, max=LOG_SCALE_MAX))

        if len(batch_shape) == 2:
            b, n = batch_shape
            return {
                "rot6d": rot6d.reshape(b, n, -1),
                "trans_dir": trans_dir.reshape(b, n, -1),
                "trans_log_scale": trans_log_scale.reshape(b, n, -1),
                "trans_scale": trans_scale.reshape(b, n, -1),
                "trans": trans.reshape(b, n, -1),
                "log_scale": log_scale.reshape(b, n, -1),
                "scale": scale.reshape(b, n, -1),
            }

        return {
            "rot6d": rot6d,
            "trans_dir": trans_dir,
            "trans_log_scale": trans_log_scale,
            "trans_scale": trans_scale,
            "trans": trans,
            "log_scale": log_scale,
            "scale": scale,
        }
