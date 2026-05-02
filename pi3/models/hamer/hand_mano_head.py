from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import aa_to_rotmat, rot6d_to_rotmat
from .mano_layer import build_mano_layer_pair
from .config import resolve_mano_path_template


class HandMANOHead(nn.Module):
    def __init__(self, cfg, in_dim: int = 2048, hidden_dim: int = 1024, mano_layer: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)

        model_cfg = getattr(cfg, "MODEL", None)
        mano_head_cfg = getattr(model_cfg, "MANO_HEAD", None) if model_cfg is not None else None
        if mano_head_cfg is None:
            mano_head_cfg = {}
        self.joint_rep_type = mano_head_cfg.get("JOINT_REP", "6d") if hasattr(mano_head_cfg, "get") else getattr(mano_head_cfg, "JOINT_REP", "6d")
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]
        self.npose = self.joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)

        self.project = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.decpose = nn.Linear(self.hidden_dim, self.npose)
        self.decshape = nn.Linear(self.hidden_dim, 10)
        self.dectransl_dir = nn.Linear(self.hidden_dim, 3)
        self.dectransl_scale = nn.Linear(self.hidden_dim, 1)
        self.decscale = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.dectransl_scale.bias)
        nn.init.zeros_(self.decscale.bias)

        mano_data_dir = cfg.MANO.get("DATA_DIR", None) if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "DATA_DIR", None)
        mean_params_path = resolve_mano_path_template(cfg.MANO.MEAN_PARAMS, mano_data_dir)
        mean_params = np.load(Path(mean_params_path))
        init_hand_pose = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params["shape"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_hand_pose", init_hand_pose)
        self.register_buffer("init_betas", init_betas)

        self.mano = mano_layer
        if self.mano is None and hasattr(cfg, "MANO"):
            try:
                mano_model_path = Path(resolve_mano_path_template(cfg.MANO.MODEL_PATH, mano_data_dir))
            except Exception:
                mano_model_path = None
            if mano_model_path is not None and mano_model_path.exists():
                mano_root = mano_model_path if mano_model_path.is_dir() else mano_model_path.parent
                right_model = mano_root / "MANO_RIGHT.pkl"
                left_model = mano_root / "MANO_LEFT.pkl"
                if not right_model.exists() or not left_model.exists():
                    raise FileNotFoundError(f"Missing MANO model files under {mano_root}")
                self.mano = build_mano_layer_pair(
                    mano_root=str(mano_root),
                    flat_hand_mean=cfg.MANO.get("FLAT_HAND_MEAN", False) if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "FLAT_HAND_MEAN", False),
                    ncomps=cfg.MANO.get("NCOMPS", 45) if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "NCOMPS", 45),
                    use_pca=cfg.MANO.get("USE_PCA", True) if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "USE_PCA", True),
                    center_idx=cfg.MANO.get("CENTER_IDX", None) if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "CENTER_IDX", None),
                    root_rot_mode=cfg.MANO.get("ROOT_ROT_MODE", "axisang") if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "ROOT_ROT_MODE", "axisang"),
                    joint_rot_mode=cfg.MANO.get("JOINT_ROT_MODE", "axisang") if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "JOINT_ROT_MODE", "axisang"),
                    robust_rot=cfg.MANO.get("ROBUST_ROT", False) if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "ROBUST_ROT", False),
                )

    def _mano_layer_for_side(self, is_right: bool | None) -> nn.Module | None:
        if self.mano is None:
            return None
        if isinstance(self.mano, nn.ModuleDict):
            if is_right is None:
                raise ValueError("hand_is_right is required when MANO is side-aware")
            side = "right" if is_right else "left"
            return self.mano[side]
        layer_side = getattr(self.mano, "side", None)
        if layer_side is not None and is_right is not None:
            requested_side = "right" if is_right else "left"
            if layer_side != requested_side:
                raise ValueError(f"mano layer side '{layer_side}' does not match requested side '{requested_side}'")
        return self.mano

    @staticmethod
    def _call_mano_layer(
        mano_layer: nn.Module,
        global_orient_rotmat: torch.Tensor,
        hand_pose_rotmat: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
    ):
        if not hasattr(mano_layer, "forward_rotmat"):
            raise TypeError("HandMANOHead requires a ManoLayer that implements forward_rotmat")
        return mano_layer.forward_rotmat(
            global_orient=global_orient_rotmat,
            hand_pose=hand_pose_rotmat,
            betas=betas,
            th_trans=transl,
        )

    def forward(
        self,
        hand_tokens: torch.Tensor,
        hand_is_right: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch_size = hand_tokens.shape[0]
        hidden = self.project(hand_tokens)

        pred_hand_pose = self.decpose(hidden) + self.init_hand_pose.expand(batch_size, -1)
        pred_betas = self.decshape(hidden) + self.init_betas.expand(batch_size, -1)
        pred_transl_dir = F.normalize(self.dectransl_dir(hidden), dim=-1, eps=1e-6)
        pred_transl_log_scale = self.dectransl_scale(hidden)
        pred_transl_scale = torch.exp(pred_transl_log_scale)
        pred_transl = pred_transl_dir * pred_transl_scale
        pred_log_scale = self.decscale(hidden)
        pred_scale = torch.exp(pred_log_scale)

        joint_conversion_fn = {
            "6d": rot6d_to_rotmat,
            "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
        }[self.joint_rep_type]
        pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(batch_size, self.cfg.MANO.NUM_HAND_JOINTS + 1, 3, 3)

        pred_mano_params = {
            "global_orient": pred_hand_pose[:, [0]],
            "hand_pose": pred_hand_pose[:, 1:],
            "betas": pred_betas,
        }
        output = {
            "pred_mano_params": pred_mano_params,
            "pred_hand_transl_dir": pred_transl_dir,
            "pred_hand_transl_log_scale": pred_transl_log_scale,
            "pred_hand_transl_scale": pred_transl_scale,
            "pred_hand_transl": pred_transl,
            "pred_hand_log_scale": pred_log_scale,
            "pred_hand_scale": pred_scale,
        }

        if self.mano is None:
            return output

        pred_vertices = torch.zeros((batch_size, 778, 3), device=hand_tokens.device, dtype=pred_scale.dtype)
        pred_joints_3d = torch.zeros((batch_size, 21, 3), device=hand_tokens.device, dtype=pred_scale.dtype)
        pred_scale = pred_scale.reshape(batch_size, 1)
        if isinstance(self.mano, nn.ModuleDict):
            if hand_is_right is None:
                raise ValueError("hand_is_right is required when MANO is side-aware")
        elif hand_is_right is None:
            default_side = bool(getattr(self.mano, "is_rhand", getattr(self.mano, "side", "right") == "right"))
            hand_is_right = torch.full((batch_size,), default_side, dtype=torch.bool, device=hand_tokens.device)
        if hand_is_right.shape[0] != batch_size:
            raise ValueError("hand_is_right must match batch size")

        for side_value in (True, False):
            side_mask = hand_is_right == side_value
            if not side_mask.any():
                continue
            mano_layer = self._mano_layer_for_side(side_value)
            if mano_layer is None:
                continue
            idx = side_mask.nonzero(as_tuple=False).squeeze(-1)
            mano_out = self._call_mano_layer(
                mano_layer,
                pred_hand_pose[idx, [0]],
                pred_hand_pose[idx, 1:],
                pred_betas[idx].float(),
                (pred_transl[idx] / pred_scale[idx]).float(), # 这里除scale是因为下面整体乘scale时会将pred_transl再乘上一个scale
            )
            pred_joints_3d[idx] = mano_out.joints.reshape(len(idx), -1, 3).to(dtype=pred_scale.dtype) * pred_scale[idx].unsqueeze(1) / 1000.0
            pred_vertices[idx] = mano_out.vertices.reshape(len(idx), -1, 3).to(dtype=pred_scale.dtype) * pred_scale[idx].unsqueeze(1) / 1000.0

        output.update(
            {
                "pred_hand_joints_3d": pred_joints_3d,
                "pred_hand_vertices": pred_vertices,
            }
        )
        return output
