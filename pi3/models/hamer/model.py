from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from yacs.config import CfgNode

from .backbones import create_backbone
from .config import resolve_mano_path_template
from .geometry import perspective_projection
from .mano_layer import build_mano_layer
from .heads import build_mano_head


class HAMER(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.backbone = create_backbone(cfg)
        self.mano_head = build_mano_head(cfg)

        mano_cfg = {key.lower(): value for key, value in dict(cfg.MANO).items()}
        mano_data_dir = mano_cfg.get("data_dir", None)
        mano_root_path = resolve_mano_path_template(mano_cfg.get("model_path"), mano_data_dir)
        mano_root_path = mano_root_path if mano_root_path is not None else mano_cfg.get("model_path")
        if mano_root_path is None:
            raise ValueError("cfg.MANO.MODEL_PATH is required for HAMER mano construction")
        mano_root = Path(mano_root_path)
        if mano_root.is_file():
            mano_root = mano_root.parent
        side = mano_cfg.get("side", "right")
        mano_model = mano_root / ("MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl")
        if not mano_model.exists():
            raise FileNotFoundError(f"Missing MANO model file under {mano_root}: {mano_model.name}")
        self.mano = build_mano_layer(
            side=side,
            mano_root=str(mano_root),
            flat_hand_mean=mano_cfg.get("flat_hand_mean", False),
            ncomps=mano_cfg.get("ncomps", 45),
            use_pca=mano_cfg.get("use_pca", True),
            center_idx=mano_cfg.get("center_idx", None),
            root_rot_mode=mano_cfg.get("root_rot_mode", "axisang"),
            joint_rot_mode=mano_cfg.get("joint_rot_mode", "axisang"),
            robust_rot=mano_cfg.get("robust_rot", False),
        )

    def forward_step(self, batch: Dict) -> Dict:
        x = batch["img"]
        batch_size = x.shape[0]

        conditioning_feats = self.backbone(x[:, :, :, 32:-32])
        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)

        output = {
            "pred_cam": pred_cam,
            "pred_mano_params": {key: value.clone() for key, value in pred_mano_params.items()},
        }

        device = pred_mano_params["hand_pose"].device
        dtype = pred_mano_params["hand_pose"].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )
        output["pred_cam_t"] = pred_cam_t
        output["focal_length"] = focal_length

        pred_mano_params["global_orient"] = pred_mano_params["global_orient"].reshape(batch_size, -1, 3, 3)
        pred_mano_params["hand_pose"] = pred_mano_params["hand_pose"].reshape(batch_size, -1, 3, 3)
        pred_mano_params["betas"] = pred_mano_params["betas"].reshape(batch_size, -1)
        mano_output = self.mano.forward_rotmat(
            global_orient=pred_mano_params["global_orient"].float(),
            hand_pose=pred_mano_params["hand_pose"].float(),
            betas=pred_mano_params["betas"].float(),
            th_trans=pred_cam_t.reshape(-1, 3).float(),
        )
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output["pred_keypoints_3d"] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output["pred_vertices"] = pred_vertices.reshape(batch_size, -1, 3)

        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t.reshape(-1, 3),
            focal_length=focal_length.reshape(-1, 2) / self.cfg.MODEL.IMAGE_SIZE,
        )
        output["pred_keypoints_2d"] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch)
