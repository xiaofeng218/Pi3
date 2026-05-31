from __future__ import annotations

from typing import Any

import torch


def _infer_batch_device(batch: dict[str, Any]) -> torch.device | None:
    views = batch.get("views", None)
    if views:
        for view in views:
            for key in ("img", "depthmap", "camera_intrinsics", "camera_pose"):
                value = view.get(key, None)
                if torch.is_tensor(value):
                    return value.device
    payload = batch.get("object_multiview_payload", None)
    if isinstance(payload, dict):
        for value in payload.values():
            if torch.is_tensor(value):
                return value.device
    return None


def _as_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device)
    return tensor


def _stack_view_field(
    views,
    getter,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.stack([_as_tensor(getter(view), dtype=dtype, device=device) for view in views], dim=1)


def _sides_to_bool_tensor(sides: Any, *, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(sides, (list, tuple)):
        values = [side == "right" for side in sides]
    else:
        values = [sides == "right"]
    return torch.as_tensor(values, dtype=torch.bool, device=device)


def _extract_object_multiview_payload(batch: dict[str, Any]) -> dict[str, Any] | None:
    payload = batch.get("object_multiview_payload", None)
    if payload:
        return payload
    if not batch.get("views"):
        return None
    payload = batch["views"][0].get("object_multiview", {})
    if isinstance(payload, dict) and any(key in payload for key in ("img", "depthmap", "camera_intrinsics", "camera_pose")):
        return payload
    return None


def build_scene_inputs_from_views_batch(batch: dict[str, Any]) -> dict[str, Any]:
    views = batch["views"]
    device = _infer_batch_device(batch)
    imgs = _stack_view_field(views, lambda view: view["img"], dtype=torch.float32, device=device)
    depths = _stack_view_field(views, lambda view: view["depthmap"], dtype=torch.float32, device=device)
    intrinsics = _stack_view_field(views, lambda view: view["camera_intrinsics"], dtype=torch.float32, device=device)
    poses = _stack_view_field(views, lambda view: view["camera_pose"], dtype=torch.float32, device=device)

    hand_masks = _stack_view_field(views, lambda view: view["hand"]["mask"], dtype=torch.float32, device=device)
    hand_is_right = torch.stack(
        [_sides_to_bool_tensor(view["hand"]["mano_side"], device=device) for view in views],
        dim=1,
    )

    object_masks = _stack_view_field(views, lambda view: view["object"]["mask"], dtype=torch.bool, device=device)
    object_valid = _stack_view_field(views, lambda view: view["object"]["valid"], dtype=torch.bool, device=device)

    object_multiview_payload = _extract_object_multiview_payload(batch)
    object_multiview = None
    if object_multiview_payload is not None:
        object_multiview = {
            "img": _as_tensor(object_multiview_payload["img"], dtype=torch.float32, device=device),
            "depthmap": _as_tensor(object_multiview_payload["depthmap"], dtype=torch.float32, device=device),
            "camera_intrinsics": _as_tensor(object_multiview_payload["camera_intrinsics"], dtype=torch.float32, device=device),
            "camera_pose": _as_tensor(object_multiview_payload["camera_pose"], dtype=torch.float32, device=device),
        }

    return {
        "imgs": imgs,
        "depths": depths,
        "intrinsics": intrinsics,
        "poses": poses,
        "hand_masks": hand_masks,
        "hand_is_right": hand_is_right,
        "object_masks": object_masks,
        "object_valid": object_valid,
        "object_multiview": object_multiview,
    }


def build_gt_metric_from_views_batch(batch: dict[str, Any]) -> dict[str, Any]:
    views = batch["views"]
    device = _infer_batch_device(batch)
    object_multiview_shared = views[0]["object_multiview"]
    object_scale_values = []
    for view in views:
        scale_meta = view["object"].get("scale_meta", {})
        object_scale_values.append(
            scale_meta.get(
                "canonical_to_target_scale",
                scale_meta.get("canonical_to_scene_metric", view["object_multiview"]["normalization_scale"]),
            )
        )
    return {
        "hand_valid": _stack_view_field(views, lambda view: view["hand"]["valid"], dtype=torch.bool, device=device),
        "hand_pose_coeffs": _stack_view_field(views, lambda view: view["hand"]["pose_mano"], dtype=torch.float32, device=device),
        "hand_global_orient_rotmat_gt": _stack_view_field(
            views, lambda view: view["hand"]["global_orient_rotmat_gt"], dtype=torch.float32, device=device
        ),
        "hand_pose_rotmat_gt": _stack_view_field(
            views, lambda view: view["hand"]["pose_rotmat_gt"], dtype=torch.float32, device=device
        ),
        "hand_transl": _stack_view_field(views, lambda view: view["hand"]["hand_transl"], dtype=torch.float32, device=device),
        "hand_mano_betas": _stack_view_field(views, lambda view: view["hand"]["mano_betas"], dtype=torch.float32, device=device),
        "hand_joints_3d": _stack_view_field(views, lambda view: view["hand"]["joints_3d_cam"], dtype=torch.float32, device=device),
        "hand_joints_2d": _stack_view_field(views, lambda view: view["hand"]["joints_2d"], dtype=torch.float32, device=device),
        "hand_camera_intrinsics": _stack_view_field(views, lambda view: view["camera_intrinsics"], dtype=torch.float32, device=device),
        "hand_is_right": torch.stack(
            [_sides_to_bool_tensor(view["hand"]["mano_side"], device=device) for view in views],
            dim=1,
        ),
        "object_valid": _stack_view_field(views, lambda view: view["object"]["valid"], dtype=torch.bool, device=device),
        "object_pose_obj2cam": _stack_view_field(views, lambda view: view["object"]["pose_obj2cam"], dtype=torch.float32, device=device),
        "object_camera_intrinsics": _stack_view_field(views, lambda view: view["camera_intrinsics"], dtype=torch.float32, device=device),
        "object_normalization_center": _as_tensor(object_multiview_shared["normalization_center"], dtype=torch.float32, device=device),
        "object_normalization_scale": _as_tensor(object_multiview_shared["normalization_scale"], dtype=torch.float32, device=device),
        "object_scale_canonical_to_target": _as_tensor(object_scale_values, dtype=torch.float32, device=device),
    }


def build_gt_scale_meta_from_views_batch(batch: dict[str, Any]) -> dict[str, Any]:
    views = batch["views"]
    device = _infer_batch_device(batch)
    scene_focus_masks = torch.stack(
        [
            _as_tensor(view["hand"]["mask"], dtype=torch.bool, device=device) | _as_tensor(view["object"]["mask"], dtype=torch.bool, device=device)
            for view in views
        ],
        dim=1,
    )
    return {"scene_focus_masks": scene_focus_masks}


def build_precomputed_pi3x_sample(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "views": batch["views"],
        "scene_inputs": build_scene_inputs_from_views_batch(batch),
        "gt_metric": build_gt_metric_from_views_batch(batch),
        "gt_scale_meta": build_gt_scale_meta_from_views_batch(batch),
    }
