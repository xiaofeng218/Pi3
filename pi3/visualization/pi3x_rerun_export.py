from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import torch

from pi3.models.hamer.geometry import rot6d_to_rotmat
from pi3.models.hamer.config import get_config as get_hamer_config, resolve_mano_path_template
from pi3.models.hamer.mano_layer import build_mano_layer
from pi3.utils.geometry import homogenize_points, se3_inverse
from pi3.utils.vis_export import build_manifest, write_manifest


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


def _load_rerun():
    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:
        raise ImportError("rerun is required for 3D sample export") from exc
    return rr


def _load_trimesh():
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise ImportError("trimesh is required for object mesh export") from exc
    return trimesh


def _infer_module_device(module):
    if hasattr(module, "parameters"):
        try:
            first_param = next(module.parameters())
            return first_param.device
        except StopIteration:
            pass
        except TypeError:
            pass
    if hasattr(module, "buffers"):
        try:
            first_buffer = next(module.buffers())
            return first_buffer.device
        except StopIteration:
            pass
        except TypeError:
            pass
    return torch.device("cpu")


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _tensor_rgb_to_uint8(img):
    img = _to_numpy(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def _mask_to_uint8(mask):
    mask = _to_numpy(mask)
    return (mask > 0).astype(np.uint8) * 255


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


def _default_hamer_paths():
    repo_root = Path(__file__).resolve().parents[2]
    config_file = repo_root / "configs" / "hamer" / "model_config.yaml"
    cache_dir = repo_root / "data" / "model" / "hamer" / "_DATA"
    return config_file, cache_dir


@lru_cache(maxsize=2)
def _build_visualization_mano_layer(hand_side):
    config_file, cache_dir = _default_hamer_paths()
    if not config_file.exists():
        return None

    hamer_cfg = get_hamer_config(str(config_file), merge=True, cache_dir=str(cache_dir), update_cachedir=True)
    mano_cfg = {key.lower(): value for key, value in dict(hamer_cfg.MANO).items()}
    mano_data_dir = mano_cfg.get("data_dir", None)
    mano_cfg["model_path"] = resolve_mano_path_template(mano_cfg.get("model_path"), mano_data_dir)
    if "model_path" not in mano_cfg or not Path(mano_cfg["model_path"]).exists():
        return None
    mano_root = Path(mano_cfg["model_path"])
    if mano_root.is_file():
        mano_root = mano_root.parent
    right_model = mano_root / "MANO_RIGHT.pkl"
    left_model = mano_root / "MANO_LEFT.pkl"
    if not right_model.exists() or not left_model.exists():
        return None
    return build_mano_layer(
        side=hand_side,
        mano_root=str(mano_root),
        flat_hand_mean=mano_cfg.get("flat_hand_mean", False),
        ncomps=mano_cfg.get("ncomps", 45),
        use_pca=mano_cfg.get("use_pca", True),
        center_idx=mano_cfg.get("center_idx", None),
        root_rot_mode=mano_cfg.get("root_rot_mode", "axisang"),
        joint_rot_mode=mano_cfg.get("joint_rot_mode", "axisang"),
        robust_rot=mano_cfg.get("robust_rot", False),
    )


def _resolve_side_mano_layer(mano_layer, hand_side):
    if mano_layer is None:
        return None
    if isinstance(mano_layer, torch.nn.ModuleDict) or isinstance(mano_layer, dict):
        if hand_side in ("left", "right") and hand_side in mano_layer:
            return mano_layer[hand_side]
        raise ValueError(f"mano_layer does not provide side '{hand_side}'")
    side = getattr(mano_layer, "side", None)
    if side is None:
        raise ValueError("mano_layer must expose a side attribute")
    if side != hand_side:
        raise ValueError(f"mano_layer side '{side}' does not match requested side '{hand_side}'")
    return mano_layer


def _build_hand_mesh_vertices(hand_pose_mano, mano_betas, mano_layer, hand_side="right"):
    if mano_layer is None:
        return None, None, None
    mano_device = _infer_module_device(mano_layer)
    hand_pose_mano = hand_pose_mano.detach().to(device=mano_device, dtype=torch.float32).reshape(1, 51)
    mano_betas = mano_betas.detach().to(device=mano_device, dtype=torch.float32).reshape(1, 10)
    mano_pose = hand_pose_mano[:, :48]
    mano_trans = hand_pose_mano[:, 48:51]

    vis_mano_layer = _resolve_side_mano_layer(mano_layer, hand_side)
    if vis_mano_layer is None:
        return None, None, None

    with torch.no_grad():
        if not hasattr(vis_mano_layer, "forward"):
            raise TypeError("Visualization requires a ManoLayer with forward")
        mano_out = vis_mano_layer.forward(
            mano_pose.float(),
            mano_betas.float(),
            th_trans=mano_trans.float(),
        )
    verts = mano_out.vertices if hasattr(mano_out, "vertices") else mano_out[0]
    joints = mano_out.joints if hasattr(mano_out, "joints") else mano_out[1]
    verts_m = verts[0].detach().cpu().numpy()
    joints_m = joints[0].detach().cpu().numpy()
    faces = np.asarray(getattr(vis_mano_layer, "faces", getattr(vis_mano_layer, "th_faces", np.zeros((0, 3)))), dtype=np.int32)
    return verts_m, joints_m, faces


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
    device = normalized_vertices.device
    scale = scale.to(device=device, dtype=normalized_vertices.dtype)
    rot6d = rot6d.to(device=device, dtype=normalized_vertices.dtype)
    trans = trans.to(device=device, dtype=normalized_vertices.dtype)
    normalized_vertices = normalized_vertices * scale.reshape(1, 1)
    pose = torch.eye(4, dtype=normalized_vertices.dtype, device=device)
    pose[:3, :3] = rot6d_to_rotmat(rot6d.reshape(1, 6))[0].to(dtype=normalized_vertices.dtype, device=device)
    pose[:3, 3] = trans
    return _transform_vertices(normalized_vertices, pose)


def build_scene_gt_metric(batch):
    """Build scene geometry GT in first-view-aligned metric coordinates (no normalization)."""
    gt_pts = torch.stack([view["pts3d"] for view in batch], dim=1)
    masks = torch.stack([view["valid_mask"] for view in batch], dim=1)
    poses = torch.stack([view["camera_pose"] for view in batch], dim=1)
    imgs = torch.stack([view["img"] for view in batch], dim=1)

    device = imgs.device
    w2c_target = se3_inverse(poses[:, 0])

    gt_pts = torch.einsum("bij, bnhwj -> bnhwi", w2c_target, homogenize_points(gt_pts))[..., :3]
    poses = torch.einsum("bij, bnjk -> bnik", w2c_target, poses)

    extrinsics = se3_inverse(poses)
    gt_local_pts = torch.einsum("bnij, bnhwj -> bnhwi", extrinsics, homogenize_points(gt_pts))[..., :3]

    gt_intrs = torch.stack([view["camera_intrinsics"] for view in batch], dim=1)
    sparse_depth_masks = torch.stack([view["sparse_depth"] for view in batch], dim=1) > 0

    return dict(
        imgs=imgs,
        global_points=gt_pts,
        local_points=gt_local_pts,
        sparse_depth_masks=sparse_depth_masks,
        valid_masks=masks,
        camera_poses=poses,
        camera_intrinsics=gt_intrs,
        dataset_names=batch[0]["dataset"],
    )


def convert_scene_gt_to_pred_scale(scene_gt, scene_scale):
    """Convert scene geometry GT from metric to pred scale using scene_scale."""
    scene_gt = dict(scene_gt)
    B = scene_gt["local_points"].shape[0]
    scale = scene_scale.view(B, *([1] * (scene_gt["local_points"].ndim - 1)))
    scene_gt["local_points"] = scene_gt["local_points"] * scale
    scale_pose = scene_scale.view(B, 1, 1)
    scene_gt["camera_poses"] = scene_gt["camera_poses"].clone()
    scene_gt["camera_poses"][..., :3, 3] = scene_gt["camera_poses"][..., :3, 3] * scale_pose
    scene_gt["scene_scale"] = scene_scale
    return scene_gt


def _solid_vertex_rgba(vertex_count, color):
    rgba = np.array([*color, 255], dtype=np.uint8)
    return np.repeat(rgba[None, :], vertex_count, axis=0)


def _log_mesh_or_clear(rr, entity_path, vertices, faces, color):
    if vertices is None or faces is None:
        rr.log(entity_path, rr.Clear(recursive=False))
        return
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy()
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        rr.log(entity_path, rr.Clear(recursive=False))
        return
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertices,
            indices=faces,
            vertex_colors=_solid_vertex_rgba(len(vertices), color),
        ),
    )


def _build_gt_object_mesh_vertices(
    gt, sample_index, frame_idx, template_vertices,
    normalization_center, normalization_scale_metric,
):
    """Build GT object mesh vertices from unified GT (pred-scale pose, metric template)."""
    object_valid = gt.get("object_valid", None)
    object_pose = gt.get("object_pose_obj2cam", None)
    if object_pose is None:
        return None

    if object_valid is not None:
        if torch.is_tensor(object_valid):
            if object_valid.ndim >= 2 and not bool(object_valid[sample_index, frame_idx]):
                return None
        elif not bool(object_valid):
            return None

    pose = object_pose[sample_index, frame_idx].detach().cpu().float()
    if torch.allclose(pose, torch.zeros_like(pose)):
        return None

    object_normalization_scale = gt.get("object_normalization_scale", None)
    normalized_vertices = _normalize_object_vertices(template_vertices, normalization_center, normalization_scale_metric)
    device = normalized_vertices.device
    if object_normalization_scale is not None:
        scale = torch.as_tensor(object_normalization_scale, dtype=normalized_vertices.dtype, device=device).reshape(1)
        normalized_vertices = normalized_vertices * scale
    pose = pose.to(device=device, dtype=normalized_vertices.dtype)
    return _transform_vertices(normalized_vertices, pose)


def export_pi3x_rerun_sample(output_path, batch, pred, gt, sample_index, data_root, mano_layer=None, item_name="pi3x_train_sample"):
    rr = _load_rerun()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(item_name, spawn=False)
    rr.save(str(output_path))
    rr.log("world/camera", rr.ViewCoordinates.RDF)

    scene_scale = float(gt["scene_scale"][sample_index].item())
    hand_faces = None
    if mano_layer is not None:
        if isinstance(mano_layer, (dict, torch.nn.ModuleDict)):
            for side in ("right", "left"):
                side_layer = mano_layer.get(side) if isinstance(mano_layer, dict) else getattr(mano_layer, side, None)
                if side_layer is not None:
                    faces_attr = getattr(side_layer, "faces", getattr(side_layer, "th_faces", None))
                    if faces_attr is not None:
                        hand_faces = np.asarray(faces_attr, dtype=np.int32)
                        break
        else:
            hand_faces = np.asarray(
                getattr(mano_layer, "faces", getattr(mano_layer, "th_faces", np.zeros((0, 3)))),
                dtype=np.int32,
            )

    for frame_idx, view in enumerate(batch):
        rr.set_time_sequence("frame", frame_idx)
        rgb = _tensor_rgb_to_uint8(view["img"][sample_index])
        valid_mask = gt["valid_masks"][sample_index, frame_idx]
        rr.log("frames/rgb", rr.Image(rgb))

        pred_z = pred["local_points"][sample_index, frame_idx, ..., 2] if pred is not None else None
        gt_z = gt["local_points"][sample_index, frame_idx, ..., 2]

        if pred is not None:
            pred_lo, pred_hi = _compute_depth_vis_range(pred_z, valid_mask)
            pred_depth = _depth_to_uint8(pred_z, valid_mask, lo=pred_lo, hi=pred_hi)
            gt_depth = _depth_to_uint8(gt_z, valid_mask, lo=pred_lo, hi=pred_hi)
            raw_depth = view["depthmap"][sample_index] * scene_scale
            raw_depth_img = _depth_to_uint8(raw_depth, valid_mask, lo=pred_lo, hi=pred_hi)
            abs_error = (pred_z - gt_z).abs().detach().cpu().numpy()
            error_img = _depth_to_uint8(torch.from_numpy(abs_error), valid_mask)
            rr.log("frames/pred_depth", rr.Image(pred_depth))
            rr.log("frames/input_depth", rr.Image(raw_depth_img))
            rr.log("frames/depth_abs_error", rr.Image(error_img))
        else:
            gt_depth = _depth_to_uint8(gt_z, valid_mask)

        rr.log("frames/gt_depth", rr.Image(gt_depth))

        mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        colors = rgb[mask_np]

        gt_points = gt["local_points"][sample_index, frame_idx].detach().cpu().numpy()
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

        object_multiview = view["object_multiview"]
        object_vertices = None
        pred_object_vertices = None
        object_faces = None
        if bool(object_multiview["grasped_object_valid"][sample_index]):
            object_id = int(object_multiview["grasped_object_id"][sample_index])
            template_vertices, object_faces = _load_object_mesh_template(str(Path(data_root).resolve()), object_id)
            if "normalization_center" in object_multiview and "normalization_scale" in object_multiview:
                object_vertices = _build_gt_object_mesh_vertices(
                    gt=gt,
                    sample_index=sample_index,
                    frame_idx=frame_idx,
                    template_vertices=template_vertices,
                    normalization_center=object_multiview["normalization_center"],
                    normalization_scale_metric=object_multiview["normalization_scale"],
                )
            if pred is not None:
                pred_object_vertices = _build_pred_object_mesh_vertices(
                    pred=pred,
                    sample_index=sample_index,
                    frame_idx=frame_idx,
                    template_vertices=template_vertices,
                    normalization_center=object_multiview.get("normalization_center"),
                    normalization_scale=object_multiview.get("normalization_scale"),
                )

        _log_mesh_or_clear(
            rr, "world/gt_object_mesh", object_vertices, object_faces,
            _YCB_COLORS.get(int(object_multiview["grasped_object_id"][sample_index]), (180, 180, 180)),
        )
        _log_mesh_or_clear(
            rr, "world/pred_object_mesh", pred_object_vertices, object_faces,
            _YCB_COLORS.get(int(object_multiview["grasped_object_id"][sample_index]), (180, 180, 180)),
        )

        gt_hand_vertices_np = None
        gt_hand_joints_np = None
        if "hand_vertices" in gt and "hand_valid" in gt:
            owner = gt.get("hand_owner_index", None)
            if owner is not None and owner.numel() > 0:
                matches = (owner[:, 0] == sample_index) & (owner[:, 1] == frame_idx)
                if matches.any():
                    idx = matches.nonzero(as_tuple=False)[0, 0]
                    verts = gt["hand_vertices"][idx].detach().cpu().numpy() * scene_scale
                    joints = gt["hand_joints_3d"][idx].detach().cpu().numpy() * scene_scale
                    gt_hand_vertices_np = verts
                    gt_hand_joints_np = joints

        pred_hand_vertices = None
        if pred is not None:
            pred_hand_vertices = _select_pred_hand_vertices(pred, sample_index, frame_idx)

        _log_mesh_or_clear(rr, "world/gt_hand_mesh", gt_hand_vertices_np, hand_faces, _HAND_COLOR)
        _log_mesh_or_clear(rr, "world/pred_hand_mesh", pred_hand_vertices, hand_faces, _HAND_COLOR)
        if gt_hand_joints_np is not None:
            rr.log(
                "world/gt_hand_joints",
                rr.Points3D(
                    positions=np.asarray(gt_hand_joints_np, dtype=np.float32),
                    colors=_solid_vertex_rgba(len(gt_hand_joints_np), _HAND_COLOR),
                ),
            )

    meta = {
        "sample_index": sample_index,
        "data_root": str(Path(data_root).resolve()),
        "scene_scale": scene_scale,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    manifest = build_manifest(
        item_id=item_name,
        rrd_path=output_path,
        release="",
        dataset=str(batch[0].get("dataset", "")) if batch else "",
        metadata=meta,
    )
    write_manifest(output_path.with_name("manifest.json"), manifest)
    return output_path, meta_path
