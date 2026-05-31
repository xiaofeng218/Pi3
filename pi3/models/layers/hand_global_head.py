from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hamer.geometry import rot6d_to_rotmat
from .local_pose_head import LocalPoseHeadTrunk, maybe_mask_feature

LOG_SCALE_MIN = -10.0
LOG_SCALE_MAX = 10.0


class HandGlobalHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, patch_h: int = 16, patch_w: int = 12):
        super().__init__()
        self.trunk = LocalPoseHeadTrunk(in_dim=in_dim, hidden_dim=hidden_dim, patch_h=patch_h, patch_w=patch_w)
        self.global_orient_head = nn.Linear(hidden_dim, 6)
        self.trans_dir_head = nn.Linear(hidden_dim, 3)
        self.trans_scale_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.trans_scale_head.bias)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, hand_feat: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.trunk(hand_feat)
        if hidden.ndim == 2:
            batch_shape = hidden.shape[:1]
        else:
            batch_shape = hidden.shape[:2]
        hidden = maybe_mask_feature(hidden, valid_mask) if hidden.ndim == 3 else hidden
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        global_orient_6d = self.global_orient_head(hidden_flat)
        global_orient = rot6d_to_rotmat(global_orient_6d).view(hidden_flat.shape[0], 1, 3, 3)
        transl_dir = F.normalize(self.trans_dir_head(hidden_flat), dim=-1, eps=1e-6)
        transl_log_scale = self.trans_scale_head(hidden_flat)
        transl_scale = torch.exp(transl_log_scale.clamp(min=LOG_SCALE_MIN, max=LOG_SCALE_MAX))
        transl = transl_dir * transl_scale
        log_scale = self.scale_head(hidden_flat)
        scale = torch.exp(log_scale.clamp(min=LOG_SCALE_MIN, max=LOG_SCALE_MAX))

        if len(batch_shape) == 2:
            b, n = batch_shape
            return {
                "pred_hand_global_orient_6d": global_orient_6d.reshape(b, n, -1),
                "pred_hand_global_orient": global_orient.reshape(b, n, 1, 3, 3),
                "pred_hand_transl_dir": transl_dir.reshape(b, n, -1),
                "pred_hand_transl_log_scale": transl_log_scale.reshape(b, n, -1),
                "pred_hand_transl_scale": transl_scale.reshape(b, n, -1),
                "pred_hand_transl": transl.reshape(b, n, -1),
                "pred_hand_log_scale": log_scale.reshape(b, n, -1),
                "pred_hand_scale": scale.reshape(b, n, -1),
            }

        return {
            "pred_hand_global_orient_6d": global_orient_6d,
            "pred_hand_global_orient": global_orient,
            "pred_hand_transl_dir": transl_dir,
            "pred_hand_transl_log_scale": transl_log_scale,
            "pred_hand_transl_scale": transl_scale,
            "pred_hand_transl": transl,
            "pred_hand_log_scale": log_scale,
            "pred_hand_scale": scale,
        }
