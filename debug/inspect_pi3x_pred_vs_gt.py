"""Run Pi3X-style GT preparation on a DexYCB batch and optionally compare model predictions.

This script is designed for debugging the DexYCB adapter before wiring it into
the main training stack. It mirrors Pi3X's training-time `prepare_gt` and
`normalize_pred` logic closely so predicted outputs can be compared against the
same normalized GT that training uses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_NUMPY_LEGACY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}
for _alias, _value in _NUMPY_LEGACY_ALIASES.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _value)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEXYCB_TOOLKIT_ROOT = REPO_ROOT / "dex-ycb-toolkit"
if DEXYCB_TOOLKIT_ROOT.exists() and str(DEXYCB_TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEXYCB_TOOLKIT_ROOT))

from pi3.utils.geometry import homogenize_points, se3_inverse
from datasets.base.utils import unified_collate_fn
from datasets.dexycb_dataset import DexYCBDataset
from pi3.models.pi3x import Pi3X
from pi3.models.hamer.mano_layer import build_mano_layer
from pi3.models.hamer.geometry import rot6d_to_rotmat
from pi3.visualization.pi3x_rerun_export import (
    _build_object_asset_transform,
    _load_textured_object_mesh,
    _log_textured_object_mesh_timeless,
    _log_textured_object_pose_or_clear,
    _set_frame_time,
)


_YCB_CLASS_NAMES = (
    "__background__",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
)

_YCB_COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (128, 0, 0),
    8: (0, 128, 0),
    9: (0, 0, 128),
    10: (128, 128, 0),
    11: (128, 0, 128),
    12: (0, 128, 128),
    13: (64, 0, 0),
    14: (0, 64, 0),
    15: (0, 0, 64),
    16: (64, 64, 0),
    17: (64, 0, 64),
    18: (0, 64, 64),
    19: (192, 0, 0),
    20: (0, 192, 0),
    21: (0, 0, 192),
}
_HAND_COLOR = (230, 230, 230)
_PRED_HAND_JOINT_COLOR = (64, 196, 255)
_HAND_JOINT_CORRESPONDENCE_COLOR = (255, 140, 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--subject", default="20200709-subject-01")
    parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--frame-num", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--output-rrd", default=None)
    parser.add_argument("--release", default="pi3x-pred-vs-gt")
    parser.add_argument("--dataset-name", default="dexycb")
    parser.add_argument("--item-id", default=None)
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Skip Pi3X forward and only inspect training-style GT normalization.",
    )
    parser.add_argument(
        "--hand-debug",
        action="store_true",
        help="Print detailed hand mesh diagnostics for the selected sample/frame.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to use for hand diagnostics.",
    )
    return parser


def load_batch(args):
    dataset = DexYCBDataset(
        data_root=args.data_root,
        mode=args.mode,
        subject=args.subject,
        resolution=[args.resolution],
        frame_num=args.frame_num,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=unified_collate_fn,
    )
    batch = None
    for idx, candidate in enumerate(loader):
        if idx == args.batch_index:
            batch = candidate
            break
    if batch is None:
        raise IndexError(f"batch_index={args.batch_index} is out of range")
    actual_batch_size = len(batch[0]["dataset"])
    if args.sample_index >= actual_batch_size:
        raise IndexError(
            f"sample_index={args.sample_index} is out of range for batch size {actual_batch_size}"
        )
    return batch


def batch_to_model_inputs(batch, device):
    imgs = torch.stack([view["img"] for view in batch], dim=1).to(device)
    depths = torch.stack([view["depthmap"] for view in batch], dim=1).to(device)
    intrinsics = torch.stack([view["camera_intrinsics"] for view in batch], dim=1).to(device)
    poses = torch.stack([view["camera_pose"] for view in batch], dim=1).to(device)
    dataset_names = batch[0]["dataset"]
    return dict(
        imgs=imgs,
        depths=depths,
        intrinsics=intrinsics,
        poses=poses,
        dataset_names=dataset_names,
    )


def run_pi3x_prediction(model, model_inputs):
    return model(
        imgs=model_inputs["imgs"],
        depths=model_inputs["depths"],
        intrinsics=model_inputs["intrinsics"],
        poses=model_inputs["poses"],
    )


def prepare_gt_like_pi3x(gt_raw, ref_idxs=None):
    gt_pts = torch.stack([view["pts3d"] for view in gt_raw], dim=1)
    masks = torch.stack([view["valid_mask"] for view in gt_raw], dim=1)
    poses = torch.stack([view["camera_pose"] for view in gt_raw], dim=1)
    imgs = torch.stack([view["img"] for view in gt_raw], dim=1)

    device = imgs.device
    B, N, H, W, _ = gt_pts.shape

    if ref_idxs is None:
        w2c_target = se3_inverse(poses[:, 0])
    else:
        w2c_target = se3_inverse(poses[torch.arange(B, device=device), ref_idxs])

    gt_pts = torch.einsum("bij, bnhwj -> bnhwi", w2c_target, homogenize_points(gt_pts))[..., :3]
    poses = torch.einsum("bij, bnjk -> bnik", w2c_target, poses)

    extrinsics = se3_inverse(poses)
    gt_local_pts = torch.einsum("bnij, bnhwj -> bnhwi", extrinsics, homogenize_points(gt_pts))[..., :3]

    valid_batch = masks.sum([-1, -2, -3]) > 0
    norm_factor = torch.ones((B,), device=device, dtype=gt_local_pts.dtype)
    if valid_batch.sum() > 0:
        B_ = int(valid_batch.sum().item())
        all_pts = gt_local_pts[valid_batch].clone()
        all_pts[~masks[valid_batch]] = 0
        all_pts = all_pts.reshape(B_, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor_valid = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

        gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor_valid[..., None, None, None, None]
        gt_local_pts[valid_batch] = gt_local_pts[valid_batch] / norm_factor_valid[..., None, None, None, None]
        poses[valid_batch, ..., :3, 3] /= norm_factor_valid[..., None, None]
        norm_factor[valid_batch] = norm_factor_valid

    gt_intrs = torch.stack([view["camera_intrinsics"] for view in gt_raw], dim=1)
    sparse_depth_masks = torch.stack([view["sparse_depth"] for view in gt_raw], dim=1) > 0

    return dict(
        imgs=imgs,
        global_points=gt_pts,
        local_points=gt_local_pts,
        sparse_depth_masks=sparse_depth_masks,
        valid_masks=masks,
        camera_poses=poses,
        camera_intrinsics=gt_intrs,
        dataset_names=gt_raw[0]["dataset"],
        norm_factor=norm_factor,
    )


def normalize_pred_like_pi3x(pred, gt, use_pred_normalize=True):
    pred = dict(pred)
    masks = gt["valid_masks"]
    local_points = pred["local_points"]
    B, N, _, _, _ = local_points.shape

    if use_pred_normalize:
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        norm_factor[masks.sum([-1, -2, -3]) == 0] = 1

        pred["local_points"] = pred["local_points"] / norm_factor[..., None, None, None, None]
        pred["camera_poses"] = pred["camera_poses"].clone()
        pred["camera_poses"][..., :3, 3] /= norm_factor.view(B, 1, 1)
        pred["norm_factor"] = norm_factor
    else:
        pred["norm_factor"] = torch.ones((B,), device=local_points.device)

    return pred


def summarize_pred_vs_gt(pred, gt):
    valid_masks = gt["valid_masks"]
    pred_local = pred["local_points"]
    gt_local = gt["local_points"]

    diff = (pred_local - gt_local).abs()
    valid = valid_masks[..., None].expand_as(diff)

    metrics = {
        "valid_ratio": float(valid_masks.float().mean().item()),
        "pred_norm_factor_mean": float(pred["norm_factor"].mean().item()),
        "gt_norm_factor_mean": float(gt["norm_factor"].mean().item()),
    }

    if valid.any():
        metrics["local_l1"] = float(diff[valid].mean().item())
        z_mask = valid_masks
        metrics["z_l1"] = float((pred_local[..., 2] - gt_local[..., 2]).abs()[z_mask].mean().item())
        metrics["pred_z_mean"] = float(pred_local[..., 2][z_mask].mean().item())
        metrics["gt_z_mean"] = float(gt_local[..., 2][z_mask].mean().item())
    else:
        metrics["local_l1"] = None
        metrics["z_l1"] = None
        metrics["pred_z_mean"] = None
        metrics["gt_z_mean"] = None

    return metrics


def _tensor_to_list(value, limit=None):
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    flat = value.detach().cpu().reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    return [float(x) for x in flat.tolist()]


def summarize_hand_mesh_debug(batch, pred, gt, sample_index: int, frame_idx: int):
    view = batch[frame_idx]
    hand = view["hand"]
    display_scale = float(gt["norm_factor"][sample_index].item()) if gt is not None else 1.0

    hand_pose_mano = torch.cat(
        [hand["pose_mano"][sample_index], hand["hand_transl"][sample_index]],
        dim=0,
    )
    gt_vertices, gt_faces = _build_hand_mesh_vertices(
        hand_pose_mano=hand_pose_mano,
        mano_betas=hand["mano_betas"][sample_index],
        mano_side=hand["mano_side"][sample_index],
    )
    gt_vertices = torch.as_tensor(gt_vertices, dtype=torch.float32)
    gt_vertices_aligned = gt_vertices / display_scale

    pred_vertices = _select_pred_hand_vertices(pred, sample_index, frame_idx) if pred is not None else None
    if pred_vertices is not None and not torch.is_tensor(pred_vertices):
        pred_vertices = torch.as_tensor(pred_vertices, dtype=torch.float32)
    elif pred_vertices is not None:
        pred_vertices = pred_vertices.detach().cpu().float()

    summary = {
        "hand_side": str(hand["mano_side"][sample_index]),
        "hand_valid": bool(hand["valid"][sample_index]),
        "display_scale": display_scale,
        "pose_mano_head": _tensor_to_list(hand["pose_mano"][sample_index], limit=9),
        "hand_transl": _tensor_to_list(hand["hand_transl"][sample_index], limit=3),
        "gt_source": "batch[frame].hand.pose_mano + hand_transl -> ManoLayer -> /display_scale",
        "pred_source": "pred.pred_hand_vertices",
        "gt_vertex0_raw": _tensor_to_list(gt_vertices[0], limit=3),
        "gt_vertex0_aligned": _tensor_to_list(gt_vertices_aligned[0], limit=3),
        "gt_centroid_raw": _tensor_to_list(gt_vertices.mean(dim=0), limit=3),
        "gt_centroid_aligned": _tensor_to_list(gt_vertices_aligned.mean(dim=0), limit=3),
        "gt_faces_count": int(gt_faces.shape[0]),
    }

    if pred_vertices is not None and pred_vertices.numel() > 0:
        pred_vertices = pred_vertices.reshape(-1, 3)
        summary.update(
            {
                "pred_vertex0": _tensor_to_list(pred_vertices[0], limit=3),
                "pred_centroid": _tensor_to_list(pred_vertices.mean(dim=0), limit=3),
                "pred_num_vertices": int(pred_vertices.shape[0]),
                "aligned_centroid_l1": float((pred_vertices.mean(dim=0) - gt_vertices_aligned.mean(dim=0)).abs().mean().item()),
                "aligned_vertex0_l1": float((pred_vertices[0] - gt_vertices_aligned[0]).abs().mean().item()),
            }
        )
    else:
        summary.update(
            {
                "pred_vertex0": None,
                "pred_centroid": None,
                "pred_num_vertices": 0,
                "aligned_centroid_l1": None,
                "aligned_vertex0_l1": None,
            }
        )

    return summary


def _load_rerun():
    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:
        raise ImportError("rerun is required for --output-rrd exports") from exc
    return rr


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _tensor_rgb_to_uint8(img):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def _compute_depth_vis_range(depth, mask):
    depth = depth.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy().astype(bool)
    if not mask.any():
        return 0.0, 1.0

    valid = depth[mask]
    lo, hi = np.percentile(valid, [2, 98])
    hi = max(float(hi), float(lo) + 1e-6)
    return float(lo), float(hi)


def _depth_to_uint8(depth, mask, lo=None, hi=None):
    depth = depth.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy().astype(bool)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if mask.any():
        if lo is None or hi is None:
            valid = depth[mask]
            lo, hi = np.percentile(valid, [2, 98])
            hi = max(hi, lo + 1e-6)
        scaled = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        out = (scaled * 255.0).round().astype(np.uint8)
        out[~mask] = 0
    return out


def _compute_gt_depth_alignment_factor(gt_depth, pred_depth, valid_mask, focus_mask=None):
    base_mask = valid_mask.clone()
    base_mask &= torch.isfinite(gt_depth) & (gt_depth > 0)
    if not base_mask.any():
        return gt_depth.new_tensor(1.0)

    align_mask = base_mask
    if focus_mask is not None:
        align_mask = align_mask & focus_mask.bool().to(device=align_mask.device)

    valid_gt = gt_depth[align_mask].reshape(-1)
    if valid_gt.numel() == 0:
        valid_gt = gt_depth[base_mask].reshape(-1)
        align_mask = base_mask

    kth = max(1, math.ceil(valid_gt.numel() * 0.5))
    foreground_threshold = valid_gt.kthvalue(kth).values

    overlap_mask = align_mask & (gt_depth <= foreground_threshold)
    overlap_mask &= torch.isfinite(pred_depth) & (pred_depth > 0)
    if not overlap_mask.any():
        return gt_depth.new_tensor(1.0)

    factor = (gt_depth[overlap_mask] / pred_depth[overlap_mask]).median()
    if not torch.isfinite(factor) or factor <= 0:
        return gt_depth.new_tensor(1.0)
    return factor


def _factor_like(factor, ref):
    if torch.is_tensor(factor):
        return factor.to(device=ref.device, dtype=ref.dtype)
    return ref.new_tensor(factor)


def _apply_alignment_factor_to_depth(depth, factor):
    return depth / _factor_like(factor, depth)


def _apply_alignment_factor_to_points(points, factor):
    return points / _factor_like(factor, points)


def _apply_alignment_factor_to_pose(pose, factor):
    aligned = pose.clone()
    factor = _factor_like(factor, aligned[..., :3, 3])
    aligned[..., :3, 3] /= factor
    return aligned


def _apply_alignment_factor_to_joint_positions(joints, factor):
    aligned = joints.clone()
    factor = _factor_like(factor, aligned)
    valid = ~(aligned == -1).all(dim=-1, keepdim=True)
    aligned[valid.expand_as(aligned)] /= factor
    return aligned


def _apply_alignment_factor_to_hand_pose_mano(hand_pose_mano, factor):
    aligned = hand_pose_mano.clone()
    factor = _factor_like(factor, aligned[..., 48:51])
    aligned[..., 48:51] /= factor
    return aligned


def _transform_vertices(vertices, pose):
    hom = torch.cat(
        [
            vertices,
            torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device),
        ],
        dim=1,
    )
    transformed = hom @ pose.transpose(0, 1)
    return transformed[:, :3]


def _apply_alignment_factor_to_view(view, factor):
    aligned = dict(view)

    if "depthmap" in aligned:
        aligned["depthmap"] = _apply_alignment_factor_to_depth(aligned["depthmap"], factor)
    if "camera_pose" in aligned:
        aligned["camera_pose"] = _apply_alignment_factor_to_pose(aligned["camera_pose"], factor)
    if "pts3d" in aligned:
        aligned["pts3d"] = _apply_alignment_factor_to_points(aligned["pts3d"], factor)
    if "hand" in aligned:
        hand = dict(aligned["hand"])
        if "joints_3d_cam" in hand:
            hand["joints_3d_cam"] = _apply_alignment_factor_to_joint_positions(
                hand["joints_3d_cam"], factor
            )
        if "hand_transl" in hand:
            hand["hand_transl"] = hand["hand_transl"].clone()
            hand["hand_transl"] /= _factor_like(factor, hand["hand_transl"])
        if "pose_mano" in hand and "hand_transl" in hand:
            hand["pose_mano_full"] = torch.cat([hand["pose_mano"], hand["hand_transl"]], dim=-1)
        aligned["hand"] = hand
    if "object_multiview" in aligned:
        object_multiview = dict(aligned["object_multiview"])
        if "grasped_object_pose_obj2cam" in object_multiview:
            object_multiview["grasped_object_pose_obj2cam"] = _apply_alignment_factor_to_pose(
                object_multiview["grasped_object_pose_obj2cam"], factor
            )
        aligned["object_multiview"] = object_multiview

    return aligned


def _solid_vertex_rgba(vertex_count, color):
    rgba = np.array([*color, 255], dtype=np.uint8)
    return np.repeat(rgba[None, :], vertex_count, axis=0)


def _load_trimesh():
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise ImportError("trimesh is required for object mesh export in --output-rrd") from exc
    return trimesh


@lru_cache(maxsize=32)
def _load_object_mesh_template(data_root_str, obj_id):
    if not 0 < obj_id < len(_YCB_CLASS_NAMES):
        raise ValueError(f"Unsupported YCB object id: {obj_id}")

    trimesh = _load_trimesh()
    obj_file = Path(data_root_str) / "models" / _YCB_CLASS_NAMES[obj_id] / "textured_simple.obj"
    mesh = trimesh.load(obj_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))

    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return vertices, faces


def _normalize_object_vertices(vertices, normalization_center, normalization_scale):
    if not torch.is_tensor(vertices):
        vertices = torch.as_tensor(vertices, dtype=torch.float32)
    center = torch.as_tensor(normalization_center, dtype=vertices.dtype, device=vertices.device).reshape(1, 3)
    scale = torch.as_tensor(normalization_scale, dtype=vertices.dtype, device=vertices.device).clamp_min(1e-6)
    return (vertices - center) / scale


@lru_cache(maxsize=2)
def _get_mano_layer(side):
    mano_root = DEXYCB_TOOLKIT_ROOT / "manopth" / "mano" / "models"
    return build_mano_layer(
        flat_hand_mean=False,
        ncomps=45,
        side=side,
        mano_root=str(mano_root),
        use_pca=True,
    ).eval()


def _build_hand_mesh_vertices(hand_pose_mano, mano_betas, mano_side):
    hand_pose_mano = hand_pose_mano.detach().cpu().float().reshape(1, 51)
    mano_betas = mano_betas.detach().cpu().float().reshape(1, 10)
    mano_pose = hand_pose_mano[:, :48]
    mano_trans = hand_pose_mano[:, 48:51]

    mano_layer = _get_mano_layer(str(mano_side))
    with torch.no_grad():
        if not hasattr(mano_layer, "forward"):
            raise TypeError("Expected ManoLayer.forward")
        mano_out = mano_layer.forward(
            mano_pose.float(),
            mano_betas.float(),
            th_trans=mano_trans.float(),
        )
        verts_mm = mano_out.vertices if hasattr(mano_out, "vertices") else mano_out[0]
    verts_m = verts_mm[0] / 1000.0
    faces = np.asarray(mano_layer.th_faces.detach().cpu().numpy(), dtype=np.int32)
    return verts_m, faces


def _select_pred_hand_vertices(pred, sample_index, frame_idx):
    pred_vertices = pred.get("pred_hand_vertices", None)
    if pred_vertices is None:
        return None

    if not torch.is_tensor(pred_vertices):
        pred_vertices = torch.as_tensor(pred_vertices)

    hand_owner_index = pred.get("hand_owner_index", None)
    if hand_owner_index is not None and torch.is_tensor(hand_owner_index) and hand_owner_index.numel() > 0:
        matches = (hand_owner_index[:, 0] == sample_index) & (hand_owner_index[:, 1] == frame_idx)
        if matches.any():
            return pred_vertices[matches.nonzero(as_tuple=False)[0, 0]]

    if pred_vertices.ndim >= 4 and pred_vertices.shape[0] > sample_index and pred_vertices.shape[1] > frame_idx:
        return pred_vertices[sample_index, frame_idx]

    if pred_vertices.ndim >= 3 and pred_vertices.shape[0] > sample_index:
        return pred_vertices[sample_index]

    return pred_vertices[0] if pred_vertices.ndim >= 3 and pred_vertices.shape[0] > 0 else None


def _select_pred_hand_joints(pred, sample_index, frame_idx):
    pred_joints = pred.get("pred_hand_joints_3d", None)
    if pred_joints is None:
        return None

    if not torch.is_tensor(pred_joints):
        pred_joints = torch.as_tensor(pred_joints)

    hand_owner_index = pred.get("hand_owner_index", None)
    if hand_owner_index is not None and torch.is_tensor(hand_owner_index) and hand_owner_index.numel() > 0:
        matches = (hand_owner_index[:, 0] == sample_index) & (hand_owner_index[:, 1] == frame_idx)
        if matches.any():
            return pred_joints[matches.nonzero(as_tuple=False)[0, 0]]

    if pred_joints.ndim >= 4 and pred_joints.shape[0] > sample_index and pred_joints.shape[1] > frame_idx:
        return pred_joints[sample_index, frame_idx]

    if pred_joints.ndim >= 3 and pred_joints.shape[0] > sample_index:
        return pred_joints[sample_index]

    return pred_joints[0] if pred_joints.ndim >= 3 and pred_joints.shape[0] > 0 else None


def _build_pred_object_mesh_vertices(pred, sample_index, frame_idx, template_vertices, normalization_center, normalization_scale):
    rot6d = pred.get("pred_object_rot6d", None)
    trans = pred.get("pred_object_trans", None)
    scale = pred.get("pred_object_scale", None)
    object_valid = pred.get("object_valid", None)
    if rot6d is None or trans is None or scale is None:
        return None

    if object_valid is not None:
        if torch.is_tensor(object_valid):
            if object_valid.ndim >= 2 and not bool(object_valid[sample_index, frame_idx]):
                return None
        elif not bool(object_valid):
            return None

    if not torch.is_tensor(rot6d):
        rot6d = torch.as_tensor(rot6d)
    if not torch.is_tensor(trans):
        trans = torch.as_tensor(trans)
    if not torch.is_tensor(scale):
        scale = torch.as_tensor(scale)

    if rot6d.ndim >= 3 and rot6d.shape[0] > sample_index and rot6d.shape[1] > frame_idx:
        rot6d = rot6d[sample_index, frame_idx]
    elif rot6d.ndim >= 2 and rot6d.shape[0] > sample_index:
        rot6d = rot6d[sample_index]
    else:
        rot6d = rot6d.reshape(-1, 6)[0]

    if trans.ndim >= 3 and trans.shape[0] > sample_index and trans.shape[1] > frame_idx:
        trans = trans[sample_index, frame_idx]
    elif trans.ndim >= 2 and trans.shape[0] > sample_index:
        trans = trans[sample_index]
    else:
        trans = trans.reshape(-1, 3)[0]

    if scale.ndim >= 3 and scale.shape[0] > sample_index and scale.shape[1] > frame_idx:
        scale = scale[sample_index, frame_idx]
    elif scale.ndim >= 2 and scale.shape[0] > sample_index:
        scale = scale[sample_index]
    else:
        scale = scale.reshape(-1, 1)[0]

    normalized_vertices = _normalize_object_vertices(template_vertices, normalization_center, normalization_scale)
    normalized_vertices = normalized_vertices * scale.reshape(1, 1)
    pose = torch.eye(4, dtype=normalized_vertices.dtype, device=normalized_vertices.device)
    pose[:3, :3] = rot6d_to_rotmat(rot6d.reshape(1, 6))[0].to(dtype=normalized_vertices.dtype, device=normalized_vertices.device)
    pose[:3, 3] = trans.to(dtype=normalized_vertices.dtype, device=normalized_vertices.device)
    return _transform_vertices(normalized_vertices, pose)


def _log_mesh_or_clear(rr, entity_path, vertices, faces, color):
    if vertices is None or faces is None:
        rr.log(entity_path, rr.Clear(recursive=False))
        return

    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=np.asarray(vertices, dtype=np.float32),
            indices=np.asarray(faces, dtype=np.int32),
            vertex_colors=_solid_vertex_rgba(len(vertices), color),
        ),
    )


def _log_hand_joint_correspondence_or_clear(rr, entity_path, gt_joints, pred_joints, color):
    if gt_joints is None or pred_joints is None:
        rr.log(entity_path, rr.Clear(recursive=False))
        return
    gt_joints = np.asarray(gt_joints, dtype=np.float32)
    pred_joints = np.asarray(pred_joints, dtype=np.float32)
    if gt_joints.shape != pred_joints.shape or gt_joints.ndim != 2 or gt_joints.shape[1] != 3:
        rr.log(entity_path, rr.Clear(recursive=False))
        return
    strips = np.stack([gt_joints, pred_joints], axis=1)
    rr.log(
        entity_path,
        rr.LineStrips3D(
            strips=strips,
            colors=np.repeat(np.array([[*color, 255]], dtype=np.uint8), strips.shape[0], axis=0),
        ),
    )


def _build_debug_gt_object_asset_transform(object_pose, display_scale):
    pose = object_pose.detach().cpu().float().clone()
    pose[:3, :3] /= float(display_scale)
    pose[:3, 3] /= float(display_scale)
    return pose


def _build_debug_pred_object_asset_transform(pred, sample_index, frame_idx, normalization_center, normalization_scale):
    rot6d = pred.get("pred_object_rot6d", None)
    trans = pred.get("pred_object_trans", None)
    scale = pred.get("pred_object_scale", None)
    object_valid = pred.get("object_valid", None)
    if rot6d is None or trans is None or scale is None:
        return None
    if object_valid is not None:
        if torch.is_tensor(object_valid):
            if object_valid.ndim >= 2 and not bool(object_valid[sample_index, frame_idx]):
                return None
        elif not bool(object_valid):
            return None

    pred_rot = rot6d_to_rotmat(rot6d[sample_index, frame_idx].detach().cpu().float().reshape(1, 6)).reshape(3, 3)
    pred_trans = trans[sample_index, frame_idx].detach().cpu().float().reshape(3)
    pred_scale = scale[sample_index, frame_idx].detach().cpu().float().reshape(-1)[:1]
    normalization_scale = torch.as_tensor(normalization_scale, dtype=pred_rot.dtype).reshape(-1)[:1].clamp_min(1e-6)
    pose = torch.eye(4, dtype=pred_rot.dtype)
    pose[:3, :3] = pred_rot
    pose[:3, 3] = pred_trans
    return _build_object_asset_transform(pose, normalization_center, pred_scale / normalization_scale)


def export_rrd_comparison(output_path, batch, pred, gt, sample_index, data_root):
    rr = _load_rerun()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init("pi3x_pred_vs_gt", spawn=False)
    rr.save(str(output_path))
    rr.log("world/camera", rr.ViewCoordinates.RDF)

    object_asset_by_path = {
        "world/gt_object_asset": None,
        "world/pred_object_asset": None,
    }
    if batch and bool(batch[0]["object_multiview"]["grasped_object_valid"][sample_index]):
        object_id = int(batch[0]["object_multiview"]["grasped_object_id"][sample_index])
        textured_object_mesh = _load_textured_object_mesh(str(Path(data_root).resolve()), object_id)
        for entity_path in object_asset_by_path:
            _log_textured_object_mesh_timeless(rr, entity_path, textured_object_mesh)
            object_asset_by_path[entity_path] = textured_object_mesh

    gt_pts = gt["local_points"][sample_index] if gt["local_points"].ndim == 4 else gt["local_points"][sample_index]
    # shapes are [B, N, H, W, 3]
    for frame_idx, view in enumerate(batch):
        _set_frame_time(rr, frame_idx)
        rgb = _tensor_rgb_to_uint8(view["img"][sample_index])
        valid_mask = gt["valid_masks"][sample_index, frame_idx]
        hand = view["hand"]
        object_multiview = view["object_multiview"]
        focus_mask = (
            hand["mask"][sample_index].bool()
            | object_multiview["grasped_object_mask"][sample_index].bool()
        )
        gt_z = gt["local_points"][sample_index, frame_idx, ..., 2]

        rr.log("frames/rgb", rr.Image(rgb))

        gt_points = gt["local_points"][sample_index, frame_idx].detach().cpu().numpy()
        mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        colors = rgb[mask_np]
        display_scale = float(gt["norm_factor"][sample_index].item())
        gt_align_factor = None

        if pred is not None:
            pred_z = pred["local_points"][sample_index, frame_idx, ..., 2]
            pred_lo, pred_hi = _compute_depth_vis_range(pred_z, valid_mask)
            pred_depth = _depth_to_uint8(pred_z, valid_mask, lo=pred_lo, hi=pred_hi)
            gt_align_factor = _compute_gt_depth_alignment_factor(
                gt_z,
                pred_z,
                valid_mask,
                focus_mask=focus_mask,
            )
            aligned_view = _apply_alignment_factor_to_view(view, gt_align_factor)
            display_scale *= float(gt_align_factor.item())
            gt_z_aligned = _apply_alignment_factor_to_depth(gt_z, gt_align_factor)
            gt_depth = _depth_to_uint8(gt_z_aligned, valid_mask, lo=pred_lo, hi=pred_hi)
            gt_points = _apply_alignment_factor_to_points(
                gt["local_points"][sample_index, frame_idx],
                gt_align_factor,
            ).detach().cpu().numpy()
            raw_depth_aligned = _depth_to_uint8(
                aligned_view["depthmap"][sample_index],
                valid_mask,
                lo=pred_lo,
                hi=pred_hi,
            )
            abs_error = (pred_z - gt_z_aligned).abs().detach().cpu().numpy()
            error_img = _depth_to_uint8(torch.from_numpy(abs_error), valid_mask)
            rr.log("frames/pred_depth", rr.Image(pred_depth))
            rr.log("frames/input_depth_aligned", rr.Image(raw_depth_aligned))
            rr.log("frames/depth_abs_error", rr.Image(error_img))
        else:
            gt_depth = _depth_to_uint8(gt_z, valid_mask)

        rr.log("frames/gt_depth", rr.Image(gt_depth))

        gt_object_asset_transform = None
        pred_object_asset_transform = None
        if bool(object_multiview["grasped_object_valid"][sample_index]):
            object_id = int(object_multiview["grasped_object_id"][sample_index])
            object_pose = object_multiview["grasped_object_pose_obj2cam"][sample_index].detach().cpu().float()
            if not torch.allclose(object_pose, torch.zeros_like(object_pose)):
                gt_object_asset_transform = _build_debug_gt_object_asset_transform(object_pose, display_scale)
                if pred is not None and "normalization_center" in object_multiview and "normalization_scale" in object_multiview:
                    pred_object_asset_transform = _build_debug_pred_object_asset_transform(
                        pred=pred,
                        sample_index=sample_index,
                        frame_idx=frame_idx,
                        normalization_center=object_multiview["normalization_center"],
                        normalization_scale=object_multiview["normalization_scale"],
                    )

        hand_vertices = None
        hand_faces = None
        pred_hand_vertices = None
        gt_hand_joints_np = None
        pred_hand_joints_np = None
        hand_pose_mano = torch.cat(
            [hand["pose_mano"][sample_index], hand["hand_transl"][sample_index]],
            dim=0,
        )
        if bool(hand["valid"][sample_index]) and not torch.allclose(
            hand_pose_mano, torch.zeros_like(hand_pose_mano)
        ):
            hand_vertices, hand_faces = _build_hand_mesh_vertices(
                hand_pose_mano=hand_pose_mano,
                mano_betas=hand["mano_betas"][sample_index],
                mano_side=hand["mano_side"][sample_index],
            )
            hand_vertices = hand_vertices / display_scale
        if pred is not None:
            pred_hand_vertices = _select_pred_hand_vertices(pred, sample_index, frame_idx)
            pred_hand_joints = _select_pred_hand_joints(pred, sample_index, frame_idx)
            if pred_hand_joints is not None:
                pred_hand_joints_np = _to_numpy(pred_hand_joints)

        if bool(hand["valid"][sample_index]) and "joints_3d_cam" in hand:
            gt_hand_joints_np = hand["joints_3d_cam"][sample_index].detach().cpu().numpy() / display_scale

        _log_textured_object_pose_or_clear(rr, "world/gt_object_asset", gt_object_asset_transform)
        _log_textured_object_pose_or_clear(rr, "world/pred_object_asset", pred_object_asset_transform)
        _log_mesh_or_clear(
            rr,
            "world/gt_hand_mesh",
            hand_vertices,
            hand_faces,
            _HAND_COLOR,
        )
        _log_mesh_or_clear(
            rr,
            "world/pred_hand_mesh",
            pred_hand_vertices,
            hand_faces,
            _HAND_COLOR,
        )
        if gt_hand_joints_np is not None:
            rr.log(
                "world/gt_hand_joints",
                rr.Points3D(
                    positions=np.asarray(gt_hand_joints_np, dtype=np.float32),
                    colors=_solid_vertex_rgba(len(gt_hand_joints_np), _HAND_COLOR),
                ),
            )
        else:
            rr.log("world/gt_hand_joints", rr.Clear(recursive=False))
        if pred_hand_joints_np is not None:
            rr.log(
                "world/pred_hand_joints",
                rr.Points3D(
                    positions=np.asarray(pred_hand_joints_np, dtype=np.float32),
                    colors=_solid_vertex_rgba(len(pred_hand_joints_np), _PRED_HAND_JOINT_COLOR),
                ),
            )
        else:
            rr.log("world/pred_hand_joints", rr.Clear(recursive=False))
        _log_hand_joint_correspondence_or_clear(
            rr,
            "world/hand_joint_correspondence",
            gt_hand_joints_np,
            pred_hand_joints_np,
            _HAND_JOINT_CORRESPONDENCE_COLOR,
        )

        if mask_np.any():
            rr.log("world/gt_points", rr.Points3D(positions=gt_points[mask_np], colors=colors))
            if pred is not None:
                pred_points = pred["local_points"][sample_index, frame_idx].detach().cpu().numpy()
                rr.log("world/pred_points", rr.Points3D(positions=pred_points[mask_np], colors=colors))
            else:
                rr.log("world/pred_points", rr.Clear(recursive=False))
        else:
            rr.log("world/gt_points", rr.Clear(recursive=False))
            rr.log("world/pred_points", rr.Clear(recursive=False))


def write_vis_upload_bundle(
    output_rrd_path,
    args,
    batch,
    gt,
    pred_summary,
):
    output_rrd_path = Path(output_rrd_path)
    output_rrd_path.parent.mkdir(parents=True, exist_ok=True)

    track_label = batch[0]["label"][args.sample_index]
    instances = [view["instance"][args.sample_index] for view in batch]
    item_id = args.item_id or f"{track_label}/batch{args.batch_index:04d}/sample{args.sample_index:02d}"

    metadata = {
        "data_root": str(Path(args.data_root).resolve()),
        "mode": args.mode,
        "subject": args.subject,
        "batch_index": args.batch_index,
        "sample_index": args.sample_index,
        "batch_size": args.batch_size,
        "frame_num": args.frame_num,
        "resolution": list(args.resolution),
        "track_label": track_label,
        "instances": instances,
        "gt_norm_factor": gt["norm_factor"].detach().cpu().tolist(),
        "valid_ratio": float(gt["valid_masks"].float().mean().item()),
        "pred_summary": pred_summary,
    }
    meta_path = _write_json(output_rrd_path.parent / "meta.json", metadata)
    manifest = {
        "schema_version": "1.0",
        "release": args.release,
        "dataset": args.dataset_name,
        "item_id": item_id,
        "metadata": metadata,
        "artifacts": [
            {
                "role": "interactive_rrd",
                "local_path": str(output_rrd_path.resolve()),
                "remote_name": output_rrd_path.name,
                "visibility": "private",
                "content_type": "application/octet-stream",
            },
            {
                "role": "metadata_json",
                "local_path": str(meta_path.resolve()),
                "remote_name": "meta.json",
                "visibility": "private",
                "content_type": "application/json",
            },
        ],
    }
    manifest_path = _write_json(output_rrd_path.parent / "manifest.json", manifest)
    return meta_path, manifest_path


def load_pi3x_model(device, ckpt=None):
    print(f"Loading model...")
    if ckpt is not None:
        model = Pi3X(use_multimodal=True).eval()
        if ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        try:
            model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        except Exception as exc:
            print(f"Failed to load pretrained Pi3X weights: {exc}")
            print("Falling back to randomly initialized weights. Provide --ckpt for meaningful predictions.")
            model = Pi3X(use_multimodal=True).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`
    model = model.to(device)
    return model


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)
    batch = load_batch(args)

    model_inputs = batch_to_model_inputs(batch, device)
    gt = prepare_gt_like_pi3x(batch)
    gt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in gt.items()}

    output = {
        "mode": args.mode,
        "subject": args.subject,
        "batch_index": args.batch_index,
        "sample_index": args.sample_index,
        "track_label": batch[0]["label"][args.sample_index],
        "gt_norm_factor": gt["norm_factor"].detach().cpu().tolist(),
        "valid_ratio": float(gt["valid_masks"].float().mean().item()),
    }

    pred = None
    if not args.gt_only:
        model = load_pi3x_model(device, ckpt=args.ckpt)
        with torch.no_grad():
            pred_raw = run_pi3x_prediction(model, model_inputs)
        pred = normalize_pred_like_pi3x(pred_raw, gt)
        output["pred_summary"] = summarize_pred_vs_gt(pred, gt)
    else:
        output["pred_summary"] = None

    if args.hand_debug:
        output["hand_debug"] = summarize_hand_mesh_debug(
            batch=batch,
            pred=pred,
            gt=gt,
            sample_index=args.sample_index,
            frame_idx=args.frame_index,
        )

    print(json.dumps(output, indent=2))

    if args.output_rrd is not None:
        export_rrd_comparison(args.output_rrd, batch, pred, gt, args.sample_index, args.data_root)
        meta_path, manifest_path = write_vis_upload_bundle(
            args.output_rrd,
            args,
            batch,
            gt,
            output["pred_summary"],
        )
        print(f"saved_rrd={args.output_rrd}")
        print(f"saved_meta={meta_path}")
        print(f"saved_manifest={manifest_path}")


if __name__ == "__main__":
    main()
