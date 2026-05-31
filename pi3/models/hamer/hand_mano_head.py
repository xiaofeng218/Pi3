from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import resolve_mano_path_template
from .mano_layer import build_mano_layer_pair


class HandMANOHead(nn.Module):
    def __init__(self, cfg, mano_layer: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        self.mano = mano_layer
        if self.mano is None and hasattr(cfg, "MANO"):
            mano_data_dir = cfg.MANO.get("DATA_DIR", None) if hasattr(cfg.MANO, "get") else getattr(cfg.MANO, "DATA_DIR", None)
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
        hand_pose: torch.Tensor,
        global_orient: torch.Tensor,
        hand_betas: torch.Tensor,
        hand_is_right: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch_size = hand_pose.shape[0]
        if hand_pose.shape != (batch_size, self.cfg.MANO.NUM_HAND_JOINTS, 3, 3):
            raise ValueError(
                f"hand_pose must have shape {(batch_size, self.cfg.MANO.NUM_HAND_JOINTS, 3, 3)}, got {tuple(hand_pose.shape)}"
            )
        if global_orient.shape != (batch_size, 1, 3, 3):
            raise ValueError(f"global_orient must have shape {(batch_size, 1, 3, 3)}, got {tuple(global_orient.shape)}")
        if hand_betas.shape != (batch_size, 10):
            raise ValueError(f"hand_betas must have shape {(batch_size, 10)}, got {tuple(hand_betas.shape)}")
        if transl is None:
            transl = hand_pose.new_zeros((batch_size, 3))
        if scale is None:
            scale = hand_pose.new_ones((batch_size, 1))
        if transl.shape != (batch_size, 3):
            raise ValueError(f"transl must have shape {(batch_size, 3)}, got {tuple(transl.shape)}")
        if scale.shape != (batch_size, 1):
            raise ValueError(f"scale must have shape {(batch_size, 1)}, got {tuple(scale.shape)}")

        pred_mano_params = {
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "betas": hand_betas,
        }
        output = {
            "pred_mano_params": pred_mano_params,
            "pred_hand_mano_betas": hand_betas,
        }

        if self.mano is None:
            return output

        pred_vertices = torch.zeros((batch_size, 778, 3), device=hand_pose.device, dtype=scale.dtype)
        pred_joints_3d = torch.zeros((batch_size, 21, 3), device=hand_pose.device, dtype=scale.dtype)
        pred_vertices_local = torch.zeros((batch_size, 778, 3), device=hand_pose.device, dtype=scale.dtype)
        pred_joints_local = torch.zeros((batch_size, 21, 3), device=hand_pose.device, dtype=scale.dtype)
        scale = scale.reshape(batch_size, 1)
        if isinstance(self.mano, nn.ModuleDict):
            if hand_is_right is None:
                raise ValueError("hand_is_right is required when MANO is side-aware")
        elif hand_is_right is None:
            default_side = bool(getattr(self.mano, "is_rhand", getattr(self.mano, "side", "right") == "right"))
            hand_is_right = torch.full((batch_size,), default_side, dtype=torch.bool, device=hand_pose.device)
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
                global_orient[idx],
                hand_pose[idx],
                hand_betas[idx].float(),
                (transl[idx] / scale[idx]).float(),
            )
            local_mano_out = self._call_mano_layer(
                mano_layer,
                torch.eye(3, device=hand_pose.device, dtype=hand_pose.dtype).view(1, 1, 3, 3).expand(len(idx), -1, -1, -1),
                hand_pose[idx],
                hand_betas[idx].float(),
                torch.zeros((len(idx), 3), device=hand_pose.device, dtype=hand_betas.dtype),
            )
            pred_joints_3d[idx] = mano_out.joints.reshape(len(idx), -1, 3).to(dtype=scale.dtype) * scale[idx].unsqueeze(1) / 1000.0
            pred_vertices[idx] = mano_out.vertices.reshape(len(idx), -1, 3).to(dtype=scale.dtype) * scale[idx].unsqueeze(1) / 1000.0
            pred_joints_local[idx] = local_mano_out.joints.reshape(len(idx), -1, 3).to(dtype=scale.dtype) / 1000.0
            pred_vertices_local[idx] = local_mano_out.vertices.reshape(len(idx), -1, 3).to(dtype=scale.dtype) / 1000.0

        output.update(
            {
                "pred_hand_joints_3d": pred_joints_3d,
                "pred_hand_vertices": pred_vertices,
                "pred_hand_joints_local": pred_joints_local,
                "pred_hand_vertices_local": pred_vertices_local,
            }
        )
        return output
