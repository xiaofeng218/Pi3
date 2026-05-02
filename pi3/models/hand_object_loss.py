from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hamer.geometry import rot6d_to_rotmat


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=eps)


def _geodesic_loss(pred_rot: torch.Tensor, gt_rot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rel = pred_rot.transpose(-2, -1) @ gt_rot
    trace = torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cosine)


def _root_relative(points: torch.Tensor, root_index: int = 0) -> torch.Tensor:
    root = points[..., root_index:root_index + 1, :]
    return points - root


def _as_scalar(tensor: torch.Tensor, default: float = 0.0) -> torch.Tensor:
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
    hand_joints_3d_loss: torch.Tensor | None = None
    hand_vertices_loss: torch.Tensor | None = None
    object_transl_scale_gt: torch.Tensor | None = None
    object_scale_gt: torch.Tensor | None = None
    object_rot_loss: torch.Tensor | None = None
    object_transl_scale_loss: torch.Tensor | None = None
    object_scale_loss: torch.Tensor | None = None


class HandObjectLoss(nn.Module):
    def __init__(
        self,
        root_index: int = 0,
        geom_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.root_index = int(root_index)
        self.geom_weight = float(geom_weight)

    def _batch_log_l1(self, pred_value: torch.Tensor, target_value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        pred_log = torch.log(pred_value.clamp_min(1e-6))
        target_log = torch.log(target_value.clamp_min(1e-6))
        pred_log, target_log = torch.broadcast_tensors(pred_log, target_log)
        loss = F.l1_loss(pred_log, target_log, reduction="none")
        if valid is not None:
            valid = valid.to(loss.dtype)
            valid = torch.broadcast_to(valid, loss.shape[:-1]).unsqueeze(-1)
            loss = loss * valid
        return loss.mean()

    def forward(self, pred: dict[str, Any], gt: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        details: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=next((v.device for v in pred.values() if torch.is_tensor(v)), torch.device("cpu")))

        scene_scale = gt.get("scene_scale", None)

        # Hand branch.
        hand_valid = gt.get("hand_valid", None)
        if (
            "pred_hand_transl_dir" in pred
            and "hand_transl" in gt
            and scene_scale is not None
        ):
            hand_transl = gt["hand_transl"]
            hand_transl_norm = hand_transl.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scene_scale_hand = scene_scale.view(*scene_scale.shape, *([1] * (hand_transl_norm.ndim - scene_scale.ndim)))
            hand_transl_scale_gt = hand_transl_norm
            hand_scale_gt = scene_scale_hand

            details["hand_transl_scale_gt"] = _as_scalar(hand_transl_scale_gt)
            details["hand_scale_gt"] = _as_scalar(hand_scale_gt)

            hand_dir_loss = 1.0 - F.cosine_similarity(
                _safe_normalize(pred["pred_hand_transl_dir"]),
                _safe_normalize(hand_transl),
                dim=-1,
            )
            if hand_valid is not None:
                hand_dir_loss = hand_dir_loss * hand_valid.to(hand_dir_loss.dtype)
                hand_dir_loss = hand_dir_loss[hand_valid].mean() if hand_valid.any() else hand_dir_loss.mean() * 0.0
            else:
                hand_dir_loss = hand_dir_loss.mean()
            hand_scale_loss = self._batch_log_l1(pred["pred_hand_transl_scale"], hand_transl_scale_gt, hand_valid)
            hand_full_scale_loss = self._batch_log_l1(pred["pred_hand_scale"], hand_scale_gt, hand_valid)

            total = total + hand_dir_loss + hand_scale_loss + 0.1 * hand_full_scale_loss

            if "pred_hand_joints_3d" in pred and "hand_joints_3d" in gt:
                pred_joints = _root_relative(pred["pred_hand_joints_3d"], self.root_index)
                gt_joints = _root_relative(gt["hand_joints_3d"], self.root_index)
                hand_joints_loss = F.l1_loss(pred_joints, gt_joints, reduction="none").mean(dim=(-1, -2)) * 10.0
                if hand_valid is not None:
                    hand_joints_loss = hand_joints_loss[hand_valid].mean() if hand_valid.any() else hand_joints_loss.mean() * 0.0
                else:
                    hand_joints_loss = hand_joints_loss.mean()
                details["hand_joints_3d_loss"] = _as_scalar(hand_joints_loss)
                total = total + self.geom_weight * hand_joints_loss
            if "pred_hand_vertices" in pred and "hand_vertices" in gt:
                pred_vertices = _root_relative(pred["pred_hand_vertices"], self.root_index)
                gt_vertices = _root_relative(gt["hand_vertices"], self.root_index)
                hand_vertices_loss = F.l1_loss(pred_vertices, gt_vertices, reduction="none").mean(dim=(-1, -2)) * 10.0
                if hand_valid is not None:
                    hand_vertices_loss = hand_vertices_loss[hand_valid].mean() if hand_valid.any() else hand_vertices_loss.mean() * 0.0
                else:
                    hand_vertices_loss = hand_vertices_loss.mean()
                details["hand_vertices_loss"] = _as_scalar(hand_vertices_loss)
                total = total + self.geom_weight * hand_vertices_loss

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

            object_transl_scale_gt = gt_transl.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            object_scale_gt = object_normalization_scale

            details["object_transl_scale_gt"] = _as_scalar(object_transl_scale_gt)
            details["object_scale_gt"] = _as_scalar(object_scale_gt)

            pred_rot = rot6d_to_rotmat(pred["pred_object_rot6d"].reshape(-1, 6)).reshape(gt_rot.shape)
            object_rot_loss = _geodesic_loss(pred_rot, gt_rot).reshape(gt_rot.shape[:-2])
            object_transl_dir_loss = 1.0 - F.cosine_similarity(
                _safe_normalize(pred["pred_object_transl_dir"]),
                _safe_normalize(gt_transl),
                dim=-1,
            )
            object_transl_scale_loss = self._batch_log_l1(pred["pred_object_transl_scale"], object_transl_scale_gt, object_valid)
            object_scale_loss = self._batch_log_l1(pred["pred_object_scale"], object_scale_gt, object_valid)

            if object_valid is not None:
                object_rot_loss = object_rot_loss[object_valid].mean() if object_valid.any() else object_rot_loss.mean() * 0.0
                object_transl_dir_loss = object_transl_dir_loss[object_valid].mean() if object_valid.any() else object_transl_dir_loss.mean() * 0.0
            else:
                object_rot_loss = object_rot_loss.mean()
                object_transl_dir_loss = object_transl_dir_loss.mean()

            details["object_rot_loss"] = _as_scalar(object_rot_loss)
            details["object_transl_scale_loss"] = _as_scalar(object_transl_scale_loss)
            details["object_scale_loss"] = _as_scalar(object_scale_loss)
            total = total + object_rot_loss + object_transl_dir_loss + object_transl_scale_loss + 0.1 * object_scale_loss

        return total, details
