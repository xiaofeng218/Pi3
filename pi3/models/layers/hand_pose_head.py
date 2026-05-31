from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..hamer.config import resolve_mano_path_template
from ..hamer.geometry import aa_to_rotmat, rot6d_to_rotmat
from .local_pose_head import LocalPoseHeadTrunk, maybe_mask_feature


class HandPoseHead(nn.Module):
    def __init__(self, cfg, in_dim: int = 512, hidden_dim: int = 512, patch_h: int = 16, patch_w: int = 12):
        super().__init__()
        self.cfg = cfg
        self.trunk = LocalPoseHeadTrunk(in_dim=in_dim, hidden_dim=hidden_dim, patch_h=patch_h, patch_w=patch_w)

        model_cfg = getattr(cfg, "MODEL", None)
        mano_head_cfg = getattr(model_cfg, "MANO_HEAD", None) if model_cfg is not None else None
        if mano_head_cfg is None:
            mano_head_cfg = {}
        self.joint_rep_type = mano_head_cfg.get("JOINT_REP", "6d") if hasattr(mano_head_cfg, "get") else getattr(mano_head_cfg, "JOINT_REP", "6d")
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]
        self.npose = self.joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)

        self.decpose = nn.Linear(hidden_dim, self.npose)
        nn.init.zeros_(self.decpose.weight)
        nn.init.zeros_(self.decpose.bias)

        mano_data_dir = cfg.MANO.get("DATA_DIR", None) if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "DATA_DIR", None)
        mean_params_path = resolve_mano_path_template(cfg.MANO.MEAN_PARAMS, mano_data_dir)
        mean_params = np.load(Path(mean_params_path))
        init_hand_pose = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_hand_pose", init_hand_pose)

    def forward(self, hand_feat: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.trunk(hand_feat)
        batch_shape = hidden.shape[:2] if hidden.ndim == 3 else hidden.shape[:1]
        hidden = maybe_mask_feature(hidden, valid_mask) if hidden.ndim == 3 else hidden
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        batch_size = hidden_flat.shape[0]
        pred_hand_pose_coeffs = self.decpose(hidden_flat) + self.init_hand_pose.expand(batch_size, -1)

        joint_conversion_fn = {
            "6d": rot6d_to_rotmat,
            "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
        }[self.joint_rep_type]
        pred_hand_pose_rotmat = joint_conversion_fn(pred_hand_pose_coeffs).view(
            batch_size, self.cfg.MANO.NUM_HAND_JOINTS + 1, 3, 3
        )

        if len(batch_shape) == 2:
            b, n = batch_shape
            return {
                "pred_hand_pose_6d": pred_hand_pose_coeffs.reshape(b, n, -1),
                "pred_hand_global_orient": pred_hand_pose_rotmat[:, [0]].reshape(b, n, 1, 3, 3),
                "pred_hand_pose": pred_hand_pose_rotmat[:, 1:].reshape(b, n, self.cfg.MANO.NUM_HAND_JOINTS, 3, 3),
            }

        return {
            "pred_hand_pose_6d": pred_hand_pose_coeffs,
            "pred_hand_global_orient": pred_hand_pose_rotmat[:, [0]],
            "pred_hand_pose": pred_hand_pose_rotmat[:, 1:],
        }
