from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from debug.inspect_pi3x_pred_vs_gt import (
    load_batch,
    prepare_gt_like_pi3x,
)
from pi3.models.hand_object_loss import HandObjectLoss
from pi3.models.hamer.geometry import rot6d_to_rotmat
from pi3.visualization.pi3x_rerun_export import (
    _build_hand_mesh_vertices,
    _build_pred_object_mesh_vertices,
    _load_object_mesh_template,
    _transform_vertices,
    convert_scene_gt_to_pred_scale,
    export_pi3x_rerun_sample,
)


def _slice_batch_to_sample(batch, sample_index: int):
    sliced = []
    for view in batch:
        sample_view = {}
        for key, value in view.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > sample_index:
                sample_view[key] = value[sample_index : sample_index + 1]
            elif isinstance(value, list) and len(value) > sample_index:
                sample_view[key] = [value[sample_index]]
            else:
                sample_view[key] = value
        sliced.append(sample_view)
    return sliced


def _rotmat_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    return torch.cat([rotmat[..., :, 0], rotmat[..., :, 1]], dim=-1)


def _tensor_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _to_jsonable(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def build_reference_payloads(sample_batch, scene_gt, synthetic_depth_scale: float, data_root: str, mano_layer):
    display_scale = scene_gt["norm_factor"][0] * synthetic_depth_scale
    num_views = len(sample_batch)
    batch_size = sample_batch[0]["img"].shape[0]

    pred_vis = {
        "local_points": scene_gt["local_points"].clone() / synthetic_depth_scale,
        "norm_factor": scene_gt["norm_factor"].clone(),
    }
    pred_loss = {}
    gt_loss = {}

    hand_owner_index = []
    vis_hand_vertices = []
    vis_hand_joints = []
    loss_hand_vertices = []
    loss_hand_joints = []
    loss_hand_transl_dir = []
    loss_hand_transl_scale = []
    loss_hand_scale = []
    loss_hand_valid = []
    gt_hand_pose_mano = []
    gt_hand_transl = []
    gt_hand_betas = []
    gt_hand_joints = []

    object_rot6d = torch.zeros((batch_size, num_views, 6), dtype=torch.float32)
    object_trans = torch.zeros((batch_size, num_views, 3), dtype=torch.float32)
    object_scale_vis = torch.zeros((batch_size, num_views, 1), dtype=torch.float32)
    object_scale_loss = torch.zeros((batch_size, num_views, 1), dtype=torch.float32)
    object_transl_dir = torch.zeros((batch_size, num_views, 3), dtype=torch.float32)
    object_transl_scale_vis = torch.zeros((batch_size, num_views, 1), dtype=torch.float32)
    object_transl_scale_loss = torch.zeros((batch_size, num_views, 1), dtype=torch.float32)
    object_valid = torch.zeros((batch_size, num_views), dtype=torch.bool)
    gt_object_pose = torch.zeros((batch_size, num_views, 4, 4), dtype=torch.float32)

    for view_idx, view in enumerate(sample_batch):
        hand = view["hand"]
        object_multiview = view["object_multiview"]
        for b_idx in range(batch_size):
            hand_valid = bool(hand["valid"][b_idx])
            if hand_valid and bool(hand["mask"][b_idx].any()):
                hand_pose_mano = torch.cat([hand["pose_mano"][b_idx], hand["hand_transl"][b_idx]], dim=0)
                hand_verts_raw, hand_joints_raw, _ = _build_hand_mesh_vertices(
                    hand_pose_mano=hand_pose_mano,
                    mano_betas=hand["mano_betas"][b_idx],
                    mano_layer=mano_layer,
                    hand_side=str(hand["mano_side"][b_idx]),
                )
                hand_owner_index.append([b_idx, view_idx, 0])
                vis_hand_vertices.append(torch.as_tensor(hand_verts_raw, dtype=torch.float32) / display_scale)
                vis_hand_joints.append(torch.as_tensor(hand_joints_raw, dtype=torch.float32) / display_scale)

                loss_hand_vertices.append(torch.as_tensor(hand_verts_raw, dtype=torch.float32))
                loss_hand_joints.append(torch.as_tensor(hand_joints_raw, dtype=torch.float32))
                raw_transl = hand["hand_transl"][b_idx].detach().cpu().float()
                loss_hand_transl_dir.append(torch.nn.functional.normalize(raw_transl.unsqueeze(0), dim=-1, eps=1e-6)[0])
                loss_hand_transl_scale.append(raw_transl.norm().clamp_min(1e-6) * display_scale)
                loss_hand_scale.append(display_scale)
                loss_hand_valid.append(True)
                gt_hand_pose_mano.append(hand["pose_mano"][b_idx].detach().cpu().float())
                gt_hand_transl.append(raw_transl)
                gt_hand_betas.append(hand["mano_betas"][b_idx].detach().cpu().float())
                gt_hand_joints.append(hand["joints_3d_cam"][b_idx].detach().cpu().float())

            if bool(object_multiview["grasped_object_valid"][b_idx]):
                object_id = int(object_multiview["grasped_object_id"][b_idx])
                object_pose = object_multiview["grasped_object_pose_obj2cam"][b_idx].detach().cpu().float()
                if not torch.allclose(object_pose, torch.zeros_like(object_pose)):
                    template_vertices, _ = _load_object_mesh_template(data_root, object_id)
                    object_pose_np = object_pose.numpy()
                    rot = torch.from_numpy(object_pose_np[:3, :3]).float()
                    trans = torch.from_numpy(object_pose_np[:3, 3]).float()
                    center = torch.as_tensor(object_multiview["normalization_center"], dtype=torch.float32).reshape(3)
                    norm_scale = torch.as_tensor(object_multiview["normalization_scale"], dtype=torch.float32).reshape(1)
                    object_rot6d[b_idx, view_idx] = _rotmat_to_rot6d(rot.unsqueeze(0))[0]
                    object_trans[b_idx, view_idx] = (trans + rot @ center) / display_scale
                    object_scale_vis[b_idx, view_idx] = norm_scale / display_scale
                    object_scale_loss[b_idx, view_idx] = norm_scale * display_scale
                    object_transl_dir[b_idx, view_idx] = torch.nn.functional.normalize(trans.unsqueeze(0), dim=-1, eps=1e-6)[0]
                    object_transl_scale_vis[b_idx, view_idx] = trans.norm().clamp_min(1e-6) / display_scale
                    object_transl_scale_loss[b_idx, view_idx] = trans.norm().clamp_min(1e-6) * display_scale
                    object_valid[b_idx, view_idx] = True
                    gt_object_pose[b_idx, view_idx] = object_pose

    pred_vis["pred_hand_vertices"] = torch.stack(vis_hand_vertices, dim=0) if vis_hand_vertices else torch.zeros((0, 778, 3))
    pred_vis["pred_hand_joints_3d"] = torch.stack(vis_hand_joints, dim=0) if vis_hand_joints else torch.zeros((0, 21, 3))
    pred_vis["hand_owner_index"] = torch.tensor(hand_owner_index, dtype=torch.long) if hand_owner_index else torch.zeros((0, 3), dtype=torch.long)
    pred_vis["pred_object_rot6d"] = object_rot6d
    pred_vis["pred_object_trans"] = object_trans
    pred_vis["pred_object_scale"] = object_scale_vis
    pred_vis["pred_object_transl_dir"] = object_transl_dir
    pred_vis["pred_object_transl_scale"] = object_transl_scale_vis
    pred_vis["pred_object_log_scale"] = torch.log(object_scale_vis.clamp_min(1e-6))
    pred_vis["object_valid"] = object_valid

    pred_loss["pred_hand_transl_dir"] = torch.stack(loss_hand_transl_dir, dim=0) if loss_hand_transl_dir else torch.zeros((0, 3))
    pred_loss["pred_hand_transl_scale"] = torch.stack(loss_hand_transl_scale, dim=0).unsqueeze(-1) if loss_hand_transl_scale else torch.zeros((0, 1))
    pred_loss["pred_hand_scale"] = torch.stack(loss_hand_scale, dim=0).unsqueeze(-1) if loss_hand_scale else torch.zeros((0, 1))
    pred_loss["pred_hand_joints_3d"] = torch.stack(loss_hand_joints, dim=0) if loss_hand_joints else torch.zeros((0, 21, 3))
    pred_loss["pred_hand_vertices"] = torch.stack(loss_hand_vertices, dim=0) if loss_hand_vertices else torch.zeros((0, 778, 3))
    pred_loss["pred_object_rot6d"] = object_rot6d
    pred_loss["pred_object_trans"] = torch.zeros((batch_size, num_views, 3), dtype=torch.float32)
    pred_loss["pred_object_transl_dir"] = object_transl_dir
    pred_loss["pred_object_transl_scale"] = object_transl_scale_loss
    pred_loss["pred_object_scale"] = object_scale_loss
    pred_loss["object_valid"] = object_valid
    pred_loss["pred_object_log_scale"] = torch.log(object_scale_loss.clamp_min(1e-6))
    pred_loss["pred_object_trans"] = torch.zeros_like(object_trans)
    for b_idx in range(batch_size):
        for view_idx in range(num_views):
            if object_valid[b_idx, view_idx]:
                object_pose = gt_object_pose[b_idx, view_idx]
                pred_loss["pred_object_trans"][b_idx, view_idx] = object_pose[:3, 3]

    gt_loss["hand_valid"] = torch.tensor(loss_hand_valid, dtype=torch.bool) if loss_hand_valid else torch.zeros((0,), dtype=torch.bool)
    gt_loss["hand_pose_mano"] = torch.stack(gt_hand_pose_mano, dim=0) if gt_hand_pose_mano else torch.zeros((0, 48))
    gt_loss["hand_transl"] = torch.stack(gt_hand_transl, dim=0) if gt_hand_transl else torch.zeros((0, 3))
    gt_loss["hand_mano_betas"] = torch.stack(gt_hand_betas, dim=0) if gt_hand_betas else torch.zeros((0, 10))
    gt_loss["hand_joints_3d_cam"] = torch.stack(gt_hand_joints, dim=0) if gt_hand_joints else torch.zeros((0, 21, 3))
    gt_loss["object_valid"] = object_valid
    gt_loss["object_pose_obj2cam"] = gt_object_pose
    gt_loss["object_normalization_scale"] = torch.as_tensor(object_multiview["normalization_scale"], dtype=torch.float32)
    gt_loss["scene_scale"] = torch.as_tensor([display_scale], dtype=torch.float32)

    return pred_vis, pred_loss, gt_loss, display_scale


def run_selfcheck(sample_batch, scene_gt, pred_vis, pred_loss, gt_loss, data_root, sample_index=0, output_rrd=None, mano_layer=None):
    aligned_hand_errors = []
    aligned_object_errors = []

    display_scale = float(gt_loss["scene_scale"][0].item())
    for view_idx, view in enumerate(sample_batch):
        hand = view["hand"]
        object_multiview = view["object_multiview"]
        for b_idx in range(sample_batch[0]["img"].shape[0]):
            if bool(hand["valid"][b_idx]) and bool(hand["mask"][b_idx].any()):
                hand_pose_mano = torch.cat([hand["pose_mano"][b_idx], hand["hand_transl"][b_idx]], dim=0)
                hand_verts_raw, _, _ = _build_hand_mesh_vertices(
                    hand_pose_mano=hand_pose_mano,
                    mano_betas=hand["mano_betas"][b_idx],
                    mano_layer=mano_layer,
                    hand_side=str(hand["mano_side"][b_idx]),
                )
                pred_hand = pred_vis["pred_hand_vertices"][len(aligned_hand_errors)]
                aligned_gt_hand = torch.as_tensor(hand_verts_raw, dtype=torch.float32) / display_scale
                aligned_hand_errors.append((pred_hand - aligned_gt_hand).abs().max().item())

            if bool(object_multiview["grasped_object_valid"][b_idx]):
                object_id = int(object_multiview["grasped_object_id"][b_idx])
                object_pose = object_multiview["grasped_object_pose_obj2cam"][b_idx].detach().cpu().float()
                if not torch.allclose(object_pose, torch.zeros_like(object_pose)):
                    template_vertices, _ = _load_object_mesh_template(data_root, object_id)
                    center = torch.as_tensor(object_multiview["normalization_center"], dtype=torch.float32).reshape(1, 3)
                    norm_scale = torch.as_tensor(object_multiview["normalization_scale"], dtype=torch.float32).clamp_min(1e-6)
                    normalized_vertices = (template_vertices - center) / norm_scale
                    pose_vis = torch.eye(4, dtype=torch.float32)
                    pose_vis[:3, :3] = object_pose[:3, :3]
                    pose_vis[:3, 3] = (object_pose[:3, 3] + object_pose[:3, :3] @ center[0]) / display_scale
                    aligned_gt_object = _transform_vertices(
                        normalized_vertices * (norm_scale / display_scale),
                        pose_vis,
                    )
                    pred_object = _build_pred_object_mesh_vertices(
                        pred={
                            "pred_object_rot6d": pred_vis["pred_object_rot6d"][b_idx : b_idx + 1, view_idx : view_idx + 1],
                            "pred_object_trans": pred_vis["pred_object_trans"][b_idx : b_idx + 1, view_idx : view_idx + 1],
                            "pred_object_scale": pred_vis["pred_object_scale"][b_idx : b_idx + 1, view_idx : view_idx + 1],
                            "object_valid": torch.tensor([[True]]),
                        },
                        sample_index=0,
                        frame_idx=0,
                        template_vertices=template_vertices,
                        normalization_center=object_multiview["normalization_center"],
                        normalization_scale=object_multiview["normalization_scale"],
                    )
                    aligned_object_errors.append((pred_object - aligned_gt_object).abs().max().item())

    loss = HandObjectLoss()
    total_loss, details = loss(pred_loss, gt_loss)

    summary = {
        "display_scale": display_scale,
        "hand_max_abs_err": max(aligned_hand_errors) if aligned_hand_errors else None,
        "object_max_abs_err": max(aligned_object_errors) if aligned_object_errors else None,
        "loss": float(total_loss.detach().cpu().item()),
        "details": _to_jsonable(details),
    }
    if output_rrd is not None:
        scene_gt_vis = convert_scene_gt_to_pred_scale(
            scene_gt,
            torch.as_tensor([display_scale], dtype=torch.float32, device=scene_gt["local_points"].device),
        )
        export_pi3x_rerun_sample(
            output_path=output_rrd,
            batch=sample_batch,
            pred=pred_vis,
            gt=scene_gt_vis,
            sample_index=sample_index,
            data_root=data_root,
            mano_layer=mano_layer,
            item_name="pi3x_mesh_selfcheck",
        )
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--subject", default="20200709-subject-01")
    parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--frame-num", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--output-rrd", default=None)
    parser.add_argument("--synthetic-depth-scale", type=float, default=2.0)
    return parser


def main():
    args = build_parser().parse_args()
    batch = load_batch(args)
    sample_batch = _slice_batch_to_sample(batch, args.sample_index)
    scene_gt = prepare_gt_like_pi3x(sample_batch)
    sample_device = torch.device(args.device)
    scene_gt = {k: (v.to(sample_device) if torch.is_tensor(v) else v) for k, v in scene_gt.items()}

    mano_layer = None
    try:
        from pi3.visualization.pi3x_rerun_export import _build_visualization_mano_layer

        mano_layer = {
            "right": _build_visualization_mano_layer("right"),
            "left": _build_visualization_mano_layer("left"),
        }
    except Exception:
        pass

    pred_vis, pred_loss, gt_loss, display_scale = build_reference_payloads(
        sample_batch=sample_batch,
        scene_gt=scene_gt,
        synthetic_depth_scale=float(args.synthetic_depth_scale),
        data_root=str(Path(args.data_root).resolve()),
        mano_layer=mano_layer,
    )
    summary = run_selfcheck(
        sample_batch=sample_batch,
        scene_gt=scene_gt,
        pred_vis=pred_vis,
        pred_loss=pred_loss,
        gt_loss=gt_loss,
        data_root=str(Path(args.data_root).resolve()),
        sample_index=args.sample_index,
        output_rrd=args.output_rrd,
        mano_layer=mano_layer,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
