from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hamer.geometry import rot6d_to_rotmat
from pi3.utils.projection import project_points_cam_to_image_torch


_HAND_JOINT_VIS_GT_COLOR = (64, 196, 255, 255)
_HAND_JOINT_VIS_PRED_COLOR = (255, 96, 96, 255)
_HAND_JOINT_VIS_CORR_COLOR = (255, 220, 96, 255)
_HAND_JOINT_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


def _load_rerun():
    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("rerun is required for hand joint loss visualization") from exc
    return rr


def _solid_rgba(count: int, color: tuple[int, int, int, int]) -> np.ndarray:
    return np.repeat(np.array([[*color]], dtype=np.uint8), count, axis=0)


def _log_joint_correspondence_or_clear(rr, entity_path: str, gt_joints: np.ndarray, pred_joints: np.ndarray) -> None:
    if gt_joints.shape != pred_joints.shape or gt_joints.ndim != 2 or gt_joints.shape[1] != 3:
        rr.log(entity_path, rr.Clear(recursive=False))
        return
    strips = np.stack([gt_joints, pred_joints], axis=1)
    rr.log(
        entity_path,
        rr.LineStrips3D(
            strips=strips,
            colors=_solid_rgba(strips.shape[0], _HAND_JOINT_VIS_CORR_COLOR),
        ),
    )


def _log_hand_skeleton_or_clear(rr, entity_path: str, joints: np.ndarray, color: tuple[int, int, int, int]) -> None:
    if joints.ndim != 2 or joints.shape != (21, 3):
        rr.log(entity_path, rr.Clear(recursive=False))
        return
    strips = np.stack([joints[list(chain)] for chain in _HAND_JOINT_CHAINS], axis=0)
    rr.log(
        entity_path,
        rr.LineStrips3D(
            strips=strips,
            colors=_solid_rgba(strips.shape[0], color),
        ),
    )


def export_hand_joint_loss_rerun(
    output_path: str | Path,
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    sample_name: str = "hand_joint_loss_debug",
) -> Path:
    rr = _load_rerun()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_np = np.asarray(pred_joints.detach().cpu(), dtype=np.float32)
    gt_np = np.asarray(gt_joints.detach().cpu(), dtype=np.float32)

    rr.init(sample_name, spawn=False)
    rr.save(str(output_path))
    rr.log("world/camera", rr.ViewCoordinates.RDF)
    rr.log(
        "world/gt_hand_joints",
        rr.Points3D(
            positions=gt_np,
            colors=_solid_rgba(len(gt_np), _HAND_JOINT_VIS_GT_COLOR),
        ),
    )
    _log_hand_skeleton_or_clear(rr, "world/gt_hand_skeleton", gt_np, _HAND_JOINT_VIS_GT_COLOR)
    rr.log(
        "world/pred_hand_joints",
        rr.Points3D(
            positions=pred_np,
            colors=_solid_rgba(len(pred_np), _HAND_JOINT_VIS_PRED_COLOR),
        ),
    )
    _log_hand_skeleton_or_clear(rr, "world/pred_hand_skeleton", pred_np, _HAND_JOINT_VIS_PRED_COLOR)
    _log_joint_correspondence_or_clear(rr, "world/hand_joint_correspondence", gt_np, pred_np)
    return output_path


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=eps)


def _root_relative(points: torch.Tensor, root_index: int = 0) -> torch.Tensor:
    root = points[..., root_index:root_index + 1, :]
    return points - root


def _normalize_hand_geometry(
    points: torch.Tensor,
    root_index: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    points = _root_relative(points, root_index=root_index)
    scale = torch.sqrt((points ** 2).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)
    return points / scale


def _canonicalize_hand_geometry(
    points: torch.Tensor,
    global_orient: torch.Tensor,
    root_index: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    points = _root_relative(points, root_index=root_index)
    if global_orient.ndim >= 3 and global_orient.shape[-3] == 1:
        global_orient = global_orient.squeeze(-3)
    # Points are row-vectors, so multiplying by R maps global coordinates
    # back into the hand-local frame when global_orient is local->global.
    points = torch.matmul(points, global_orient)
    scale = torch.sqrt((points ** 2).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)
    return points / scale


def _as_scalar(tensor: torch.Tensor=None, default: float = 0.0) -> torch.Tensor:
    if tensor is None:
        return torch.tensor(default)
    if tensor.numel() == 0:
        return tensor.new_tensor(default)
    return tensor.detach().mean()


def estimate_scene_scale_from_depth(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    focus_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_depth.shape != gt_depth.shape or pred_depth.shape != valid_mask.shape:
        raise ValueError("pred_depth, gt_depth and valid_mask must have matching shapes")

    batch = pred_depth.shape[0]
    scale = []
    for b in range(batch):
        mask = valid_mask[b].bool()
        mask = mask & torch.isfinite(pred_depth[b]) & torch.isfinite(gt_depth[b])
        mask = mask & (pred_depth[b] > 0) & (gt_depth[b] > 0)
        if focus_mask is not None:
            mask = mask & focus_mask[b].bool()

        pred_vals = pred_depth[b][mask]
        gt_vals = gt_depth[b][mask]
        if pred_vals.numel() == 0:
            scale.append(pred_depth.new_tensor(1.0))
            continue

        ratio = pred_vals / (gt_vals + eps)
        ratio = ratio[torch.isfinite(ratio) & (ratio > 0)]
        if ratio.numel() == 0:
            scale.append(pred_depth.new_tensor(1.0))
            continue

        scale.append(ratio.median())

    return torch.stack(scale, dim=0)


@dataclass
class HandObjectLossDetails:
    hand_transl_scale_gt: torch.Tensor | None = None
    hand_scale_gt: torch.Tensor | None = None
    hand_global_orient_loss: torch.Tensor | None = None
    hand_pose_loss: torch.Tensor | None = None
    hand_adversarial_prior_loss: torch.Tensor | None = None
    hand_joints_3d_loss: torch.Tensor | None = None
    hand_2d_loss: torch.Tensor | None = None
    object_transl_scale_gt: torch.Tensor | None = None
    object_scale_gt: torch.Tensor | None = None
    object_rot_loss: torch.Tensor | None = None
    object_transl_scale_loss: torch.Tensor | None = None
    object_scale_loss: torch.Tensor | None = None


class HandObjectLoss(nn.Module):
    _DETAIL_KEYS = (
        "hand_transl_loss",
        "hand_full_scale_loss",
        "hand_global_orient_loss",
        "hand_pose_loss",
        "hand_adversarial_prior_loss",
        "hand_joints_3d_loss",
        "hand_2d_loss",
        "object_rot_loss",
        "object_transl_loss",
        "object_scale_loss",
    )

    def __init__(
        self,
        root_index: int = 0,
        geom_weight: float = 1.0,
        hand_transl_weight: float = 1.0,
        hand_full_scale_weight: float = 1.0,
        hand_global_orient_weight: float = 1.0,
        hand_pose_weight: float = 0.01,
        hand_beta_weight: float = 0.0,
        hand_joints_3d_weight: float = 1.0,
        hand_vertices_weight: float = 1.0,
        hand_2d_weight: float = 1.0,
        hand_adversarial_weight: float = 0.0,
        object_rot_weight: float = 1.0,
        object_transl_weight: float = 1.0,
        object_scale_weight: float = 0.1,
        object_2d_weight: float = 1.0,
        hand_discriminator_ckpt: str | None = None,
        hand_discriminator: nn.Module | None = None,
        debug_hand_joints_vis_enabled: bool = False,
        debug_hand_joints_vis_dir: str | None = None,
        debug_hand_joints_vis_max_exports: int = 0,
    ) -> None:
        super().__init__()
        self.root_index = int(root_index)
        self.geom_weight = float(geom_weight)
        self.hand_transl_weight = float(hand_transl_weight)
        self.hand_full_scale_weight = float(hand_full_scale_weight)
        self.hand_global_orient_weight = float(hand_global_orient_weight)
        self.hand_pose_weight = float(hand_pose_weight)
        self.hand_joints_3d_weight = float(hand_joints_3d_weight)
        self.hand_2d_weight = float(hand_2d_weight)
        self.hand_adversarial_weight = float(hand_adversarial_weight)
        self.object_rot_weight = float(object_rot_weight)
        self.object_transl_weight = float(object_transl_weight)
        self.object_scale_weight = float(object_scale_weight)
        self.debug_hand_joints_vis_enabled = bool(debug_hand_joints_vis_enabled)
        self.debug_hand_joints_vis_dir = None if debug_hand_joints_vis_dir in (None, "") else Path(debug_hand_joints_vis_dir)
        self.debug_hand_joints_vis_max_exports = int(debug_hand_joints_vis_max_exports)
        self._debug_hand_joints_vis_export_count = 0
        self.hand_discriminator = self._build_hand_discriminator(
            hand_adversarial_weight=self.hand_adversarial_weight,
            hand_discriminator_ckpt=hand_discriminator_ckpt,
            hand_discriminator=hand_discriminator,
        )

    def _build_hand_discriminator(
        self,
        hand_adversarial_weight: float,
        hand_discriminator_ckpt: str | None,
        hand_discriminator: nn.Module | None,
    ) -> nn.Module | None:
        if hand_discriminator is not None:
            discriminator = hand_discriminator
        elif hand_adversarial_weight > 0.0 and hand_discriminator_ckpt:
            discriminator = self._load_frozen_hand_discriminator(hand_discriminator_ckpt)
        else:
            discriminator = None

        if discriminator is None:
            return None

        discriminator.eval()
        for param in discriminator.parameters():
            param.requires_grad_(False)
        return discriminator

    def _load_frozen_hand_discriminator(self, checkpoint_path: str) -> nn.Module:
        from third_party.hamer.hamer.models.discriminator import Discriminator

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"hand discriminator checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        discriminator_state_dict = {
            key[len("discriminator."):]: value
            for key, value in state_dict.items()
            if key.startswith("discriminator.")
        }
        if not discriminator_state_dict:
            raise KeyError(f"no discriminator.* weights found in checkpoint: {checkpoint_path}")

        discriminator = Discriminator()
        discriminator.load_state_dict(discriminator_state_dict, strict=True)
        return discriminator

    def _batch_log_l1(
        self,
        pred_value: torch.Tensor,
        target_value: torch.Tensor,
        valid: torch.Tensor,
        pred_is_log: bool = False,
    ) -> torch.Tensor:
        pred_log = pred_value if pred_is_log else torch.log(pred_value.clamp_min(1e-6))
        target_log = torch.log(target_value.clamp_min(1e-6))
        pred_log, target_log = torch.broadcast_tensors(pred_log, target_log)
        loss = F.l1_loss(pred_log, target_log, reduction="none")
        if valid is not None:
            valid = valid.to(loss.dtype)
            valid = torch.broadcast_to(valid, loss.shape[:-1]).unsqueeze(-1)
            loss = loss * valid
        return loss.mean()

    def _batch_log_l2(
        self,
        pred_value: torch.Tensor,
        target_value: torch.Tensor,
        valid: torch.Tensor,
        pred_is_log: bool = False,
    ) -> torch.Tensor:
        pred_log = pred_value if pred_is_log else torch.log(pred_value.clamp_min(1e-6))
        target_log = torch.log(target_value.clamp_min(1e-6))
        pred_log, target_log = torch.broadcast_tensors(pred_log, target_log)
        loss = F.mse_loss(pred_log, target_log, reduction="none")
        if valid is not None:
            valid = valid.to(loss.dtype)
            valid = torch.broadcast_to(valid, loss.shape[:-1]).unsqueeze(-1)
            loss = loss * valid
        return loss.mean()

    def _masked_mean(self, loss: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if valid is not None:
            loss = loss[valid].mean() if valid.any() else loss.mean() * 0.0
        else:
            loss = loss.mean()
        return loss

    def _maybe_export_debug_hand_joints(
        self,
        pred_joints: torch.Tensor,
        gt_joints: torch.Tensor,
        per_sample_joint_loss: torch.Tensor,
        hand_valid: torch.Tensor | None,
        hand_is_right: torch.Tensor | None,
    ) -> None:
        if not self.debug_hand_joints_vis_enabled:
            return
        if self.debug_hand_joints_vis_max_exports <= 0:
            return
        if self._debug_hand_joints_vis_export_count >= self.debug_hand_joints_vis_max_exports:
            return
        if self.debug_hand_joints_vis_dir is None:
            return

        valid_mask = torch.ones_like(per_sample_joint_loss, dtype=torch.bool)
        if hand_valid is not None:
            valid_mask = valid_mask & hand_valid.bool()
        if not valid_mask.any():
            return

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        target_index = valid_indices[per_sample_joint_loss[valid_mask].argmax()].item()
        side_name = "unknown"
        if hand_is_right is not None and hand_is_right.numel() > target_index:
            side_name = "right" if bool(hand_is_right[target_index].item()) else "left"
        output_path = self.debug_hand_joints_vis_dir / (
            f"hand_joint_loss_{self._debug_hand_joints_vis_export_count:04d}_{side_name}_idx{target_index:03d}.rrd"
        )
        export_hand_joint_loss_rerun(
            output_path=output_path,
            pred_joints=pred_joints[target_index],
            gt_joints=gt_joints[target_index],
            sample_name=f"hand_joint_loss_{side_name}_{self._debug_hand_joints_vis_export_count:04d}",
        )
        self._debug_hand_joints_vis_export_count += 1

    def _project_hand_joints_2d(self, pred_joints_3d: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return project_points_cam_to_image_torch(pred_joints_3d, intrinsics)

    def _build_pred_object_vertices(
        self,
        pred_rot6d: torch.Tensor,
        pred_trans: torch.Tensor,
        pred_scale: torch.Tensor,
        template_vertices: torch.Tensor,
        normalization_center: torch.Tensor,
        normalization_scale: torch.Tensor,
    ) -> torch.Tensor:
        pred_rot = rot6d_to_rotmat(pred_rot6d.reshape(-1, 6)).reshape(*pred_rot6d.shape[:-1], 3, 3)
        template = template_vertices.to(dtype=pred_trans.dtype, device=pred_trans.device)
        center = normalization_center.to(dtype=template.dtype, device=template.device).reshape(-1, 1, 1, 3)
        norm_scale = normalization_scale.to(dtype=template.dtype, device=template.device).reshape(-1, 1, 1, 1).clamp_min(1e-6)
        normalized = (template.unsqueeze(1) - center) / norm_scale
        scaled = normalized * pred_scale.unsqueeze(-2)
        rotated = torch.matmul(scaled, pred_rot.transpose(-1, -2))
        return rotated + pred_trans.unsqueeze(-2)

    def forward(self, pred: dict[str, Any], gt: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = torch.zeros((), device=next((v.device for v in pred.values() if torch.is_tensor(v)), torch.device("cpu")))
        zero = total.detach() * 0.0
        details: dict[str, torch.Tensor] = {key: zero.clone() for key in self._DETAIL_KEYS}
        weighted_details: dict[str, torch.Tensor] = {key: zero.clone() for key in self._DETAIL_KEYS}

        # TODO：这里的object_normalization_scale、scene_scale_hand在外面设置好，不要在这里面进行设置。
        scene_scale = gt.get("scene_scale", None)

        # Hand branch.
        hand_valid = gt.get("hand_valid", None)
        if (
            "pred_hand_transl" in pred
            and "hand_transl" in gt
            and scene_scale is not None
        ):
            hand_transl = gt["hand_transl"]
            hand_scale = gt.get("hand_scale", None)
            if hand_scale is None:
                hand_owner_index = gt.get("hand_owner_index", None)
                if hand_owner_index is not None:
                    hand_scale = scene_scale[hand_owner_index[:, 0]].view(-1, 1)
                else:
                    hand_scale = scene_scale.view(*scene_scale.shape, *([1] * (hand_transl.ndim - scene_scale.ndim)))

            hand_transl_loss = F.l1_loss(
                pred["pred_hand_transl"], hand_transl, reduction="none"
            ).mean(dim=-1)
            if hand_valid is not None:
                hand_transl_loss = hand_transl_loss[hand_valid].mean() if hand_valid.any() else hand_transl_loss.mean() * 0.0
            else:
                hand_transl_loss = hand_transl_loss.mean()
            details["hand_transl_loss"] = _as_scalar(hand_transl_loss)
            weighted_details["hand_transl_loss"] = _as_scalar(self.hand_transl_weight * hand_transl_loss)

            hand_full_scale_loss = self._batch_log_l1(
                pred.get("pred_hand_log_scale", pred["pred_hand_scale"]),
                hand_scale,
                hand_valid,
                pred_is_log="pred_hand_log_scale" in pred,
            )
            details["hand_full_scale_loss"] = _as_scalar(hand_full_scale_loss)
            weighted_details["hand_full_scale_loss"] = _as_scalar(self.hand_full_scale_weight * hand_full_scale_loss)

            total = (
                total
                + self.hand_transl_weight * hand_transl_loss
                + self.hand_full_scale_weight * hand_full_scale_loss
            )

            if "pred_hand_mano_params" in pred and "hand_global_orient_rotmat" in gt:
                gt_global_orient = gt["hand_global_orient_rotmat"]
                hand_global_orient_loss = F.mse_loss(
                    pred["pred_hand_mano_params"]["global_orient"],
                    gt_global_orient,
                    reduction="none",
                ).mean(dim=(-1, -2, -3))
                hand_global_orient_loss = self._masked_mean(hand_global_orient_loss, hand_valid)
                details["hand_global_orient_loss"] = _as_scalar(hand_global_orient_loss)
                weighted_details["hand_global_orient_loss"] = _as_scalar(self.hand_global_orient_weight * hand_global_orient_loss)
                total = total + self.hand_global_orient_weight * hand_global_orient_loss

            if "pred_hand_mano_params" in pred and "hand_pose_rotmat" in gt:
                pred_pose_rotmat = pred["pred_hand_mano_params"]["hand_pose"]
                gt_pose_rotmat = gt["hand_pose_rotmat"]
                hand_pose_loss = F.mse_loss(pred_pose_rotmat, gt_pose_rotmat, reduction="none").mean(dim=(-1, -2, -3))
                hand_pose_loss = self._masked_mean(hand_pose_loss, hand_valid)
                details["hand_pose_loss"] = _as_scalar(hand_pose_loss)
                weighted_details["hand_pose_loss"] = _as_scalar(self.hand_pose_weight * hand_pose_loss)
                total = total + self.hand_pose_weight * hand_pose_loss

            if (
                self.hand_discriminator is not None
                and self.hand_adversarial_weight > 0.0
                and "pred_hand_mano_params" in pred
            ):
                pred_hand_pose = pred["pred_hand_mano_params"]["hand_pose"]
                pred_hand_betas = pred["pred_hand_mano_params"]["betas"]
                if hand_valid is not None:
                    if hand_valid.any():
                        pred_hand_pose = pred_hand_pose[hand_valid]
                        pred_hand_betas = pred_hand_betas[hand_valid]
                    else:
                        pred_hand_pose = pred_hand_pose[:0]
                        pred_hand_betas = pred_hand_betas[:0]
                if pred_hand_pose.shape[0] > 0:
                    disc_out = self.hand_discriminator(pred_hand_pose, pred_hand_betas)
                    hand_adv_prior_loss = ((disc_out - 1.0) ** 2).sum() / pred_hand_pose.shape[0]
                else:
                    hand_adv_prior_loss = total * 0.0
                details["hand_adversarial_prior_loss"] = _as_scalar(hand_adv_prior_loss)
                weighted_details["hand_adversarial_prior_loss"] = _as_scalar(self.hand_adversarial_weight * hand_adv_prior_loss)
                total = total + self.hand_adversarial_weight * hand_adv_prior_loss

            if ("pred_hand_joints_local" in pred or "pred_hand_joints_3d" in pred) and "hand_joints_3d" in gt:
                if "pred_hand_joints_local" in pred and "hand_global_orient_rotmat" in gt:
                    gt_global_orient = gt["hand_global_orient_rotmat"]
                    pred_joints = _normalize_hand_geometry(
                        pred["pred_hand_joints_local"],
                        root_index=self.root_index,
                    )
                    gt_joints = _canonicalize_hand_geometry(
                        gt["hand_joints_3d"],
                        gt_global_orient,
                        root_index=self.root_index,
                    )
                elif "pred_hand_mano_params" in pred and "hand_global_orient_rotmat" in gt:
                    gt_global_orient = gt["hand_global_orient_rotmat"]
                    pred_joints = _canonicalize_hand_geometry(
                        pred["pred_hand_joints_3d"],
                        pred["pred_hand_mano_params"]["global_orient"],
                        root_index=self.root_index,
                    )
                    gt_joints = _canonicalize_hand_geometry(
                        gt["hand_joints_3d"],
                        gt_global_orient,
                        root_index=self.root_index,
                    )
                else:
                    pred_joints = _root_relative(pred["pred_hand_joints_3d"], self.root_index)
                    gt_joints = _root_relative(gt["hand_joints_3d"], self.root_index)
                hand_joints_loss = F.l1_loss(pred_joints, gt_joints, reduction="none").mean(dim=(-1, -2)) * 10.0
                self._maybe_export_debug_hand_joints(
                    pred_joints=pred_joints,
                    gt_joints=gt_joints,
                    per_sample_joint_loss=hand_joints_loss,
                    hand_valid=hand_valid,
                    hand_is_right=gt.get("hand_is_right", None),
                )
                hand_joints_loss = self._masked_mean(hand_joints_loss, hand_valid)
                details["hand_joints_3d_loss"] = _as_scalar(hand_joints_loss)
                weighted_details["hand_joints_3d_loss"] = _as_scalar(self.geom_weight * self.hand_joints_3d_weight * hand_joints_loss)
                total = total + self.geom_weight * self.hand_joints_3d_weight * hand_joints_loss

            # TODO 1.2: 这个joint的2d损失先保留代码，但是可以先权重设置为0.
            if "pred_hand_joints_3d" in pred and "hand_joints_2d" in gt and "hand_camera_intrinsics" in gt:
                pred_hand_joints_2d, pred_hand_joints_2d_valid = self._project_hand_joints_2d(
                    pred["pred_hand_joints_3d"],
                    gt["hand_camera_intrinsics"],
                )
                gt_hand_joints_2d = gt["hand_joints_2d"]
                # Normalize by image dimensions: [0,1] range
                K = gt["hand_camera_intrinsics"]
                img_wh = torch.stack([2.0 * K[..., 0, 2].clamp_min(1.0),
                                      2.0 * K[..., 1, 2].clamp_min(1.0)], dim=-1)
                pred_2d_norm = pred_hand_joints_2d / img_wh.unsqueeze(-2)
                gt_2d_norm = gt_hand_joints_2d / img_wh.unsqueeze(-2)
                valid_2d = (
                    pred_hand_joints_2d_valid
                    & (pred_2d_norm >= 0.0).all(dim=-1)
                    & (pred_2d_norm <= 1.0).all(dim=-1)
                    & torch.isfinite(gt_2d_norm).all(dim=-1)
                )
                if hand_valid is not None:
                    valid_2d = valid_2d & hand_valid.unsqueeze(-1)
                if valid_2d.any():
                    safe_pred_2d_norm = torch.where(valid_2d.unsqueeze(-1), pred_2d_norm, torch.zeros_like(pred_2d_norm))
                    safe_gt_2d_norm = torch.where(
                        valid_2d.unsqueeze(-1),
                        torch.nan_to_num(gt_2d_norm, nan=0.0, posinf=0.0, neginf=0.0),
                        torch.zeros_like(gt_2d_norm),
                    )
                    hand_2d_error = F.smooth_l1_loss(safe_pred_2d_norm, safe_gt_2d_norm, reduction="none").mean(dim=-1)
                    hand_2d_loss = hand_2d_error[valid_2d].mean() if valid_2d.any() else hand_2d_error.mean() * 0.0
                    details["hand_2d_loss"] = _as_scalar(hand_2d_loss)
                    weighted_details["hand_2d_loss"] = _as_scalar(self.hand_2d_weight * hand_2d_loss)
                    total = total + self.hand_2d_weight * hand_2d_loss
                else:
                    details["hand_2d_loss"] = zero.clone()

        # Object branch.
        object_valid = gt.get("object_valid", None)
        if (
            "pred_object_rot6d" in pred
            and "object_pose_obj2cam" in gt
            and scene_scale is not None
        ):
            object_pose = gt["object_pose_obj2cam"]
            gt_rot = object_pose[..., :3, :3]
            gt_transl = object_pose[..., :3, 3]
            object_normalization_scale = gt.get("object_normalization_scale", None)
            if object_normalization_scale is None:
                object_normalization_scale = scene_scale

            pred_rot = rot6d_to_rotmat(pred["pred_object_rot6d"].reshape(-1, 6)).reshape(gt_rot.shape)
            object_rot_loss = F.mse_loss(pred_rot, gt_rot, reduction="none").mean(dim=(-1, -2))

            object_transl_loss = F.l1_loss(
                pred["pred_object_trans"], gt_transl, reduction="none"
            ).mean(dim=-1)

            object_scale_loss = self._batch_log_l1(
                pred.get("pred_object_log_scale", pred["pred_object_scale"]),
                object_normalization_scale,
                object_valid,
                pred_is_log="pred_object_log_scale" in pred,
            )

            if object_valid is not None:
                object_rot_loss = object_rot_loss[object_valid].mean() if object_valid.any() else object_rot_loss.mean() * 0.0
                object_transl_loss = object_transl_loss[object_valid].mean() if object_valid.any() else object_transl_loss.mean() * 0.0
            else:
                object_rot_loss = object_rot_loss.mean()
                object_transl_loss = object_transl_loss.mean()

            details["object_rot_loss"] = _as_scalar(object_rot_loss)
            weighted_details["object_rot_loss"] = _as_scalar(self.object_rot_weight * object_rot_loss)
            details["object_transl_loss"] = _as_scalar(object_transl_loss)
            weighted_details["object_transl_loss"] = _as_scalar(self.object_transl_weight * object_transl_loss)
            details["object_scale_loss"] = _as_scalar(object_scale_loss)
            weighted_details["object_scale_loss"] = _as_scalar(self.object_scale_weight * object_scale_loss)
            total = (
                total
                + self.object_rot_weight * object_rot_loss
                + self.object_transl_weight * object_transl_loss
                + self.object_scale_weight * object_scale_loss
            )
            
        details["_weighted_loss_details"] = weighted_details
        return total, details
