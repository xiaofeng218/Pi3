from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pi3.models.hamer.config import get_config as get_hamer_config, resolve_mano_path_template
from pi3.models.hamer.mano_layer import build_mano_layer_pair
from pi3.utils.vis_export import build_manifest, write_manifest
from . import pi3x_rerun_export as dexycb_export


def _unwrap_singleton(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _default_hamer_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    config_file = repo_root / "configs" / "hamer" / "model_config.yaml"
    cache_dir = repo_root / "data" / "model" / "hamer" / "_DATA"
    return config_file, cache_dir


def _build_forehoi_visualization_mano_layers():
    config_file, cache_dir = _default_hamer_paths()
    if not config_file.exists():
        return None
    hamer_cfg = get_hamer_config(str(config_file), merge=True, cache_dir=str(cache_dir), update_cachedir=True)
    mano_cfg = {key.lower(): value for key, value in dict(hamer_cfg.MANO).items()}
    mano_data_dir = mano_cfg.get("data_dir", None)
    mano_cfg["model_path"] = resolve_mano_path_template(mano_cfg.get("model_path"), mano_data_dir)
    mano_root_value = mano_cfg.get("model_path")
    if not mano_root_value:
        return None
    mano_root = Path(mano_root_value)
    if mano_root.is_file():
        mano_root = mano_root.parent
    if not (mano_root / "MANO_RIGHT.pkl").exists() or not (mano_root / "MANO_LEFT.pkl").exists():
        return None
    return build_mano_layer_pair(
        mano_root=str(mano_root),
        flat_hand_mean=False,
        ncomps=45,
        use_pca=False,
        center_idx=mano_cfg.get("center_idx", None),
        root_rot_mode="axisang",
        joint_rot_mode="axisang",
        robust_rot=mano_cfg.get("robust_rot", False),
    )


def _resolve_forehoi_mano_layer(mano_layer):
    def _layer_is_compatible(layer) -> bool:
        if layer is None:
            return False
        return (not bool(getattr(layer, "use_pca", True))) and (not bool(getattr(layer, "flat_hand_mean", True)))

    if isinstance(mano_layer, (dict, torch.nn.ModuleDict)):
        right = mano_layer.get("right") if isinstance(mano_layer, dict) else getattr(mano_layer, "right", None)
        left = mano_layer.get("left") if isinstance(mano_layer, dict) else getattr(mano_layer, "left", None)
        if _layer_is_compatible(right) and _layer_is_compatible(left):
            return mano_layer
    elif _layer_is_compatible(mano_layer):
        return mano_layer
    return _build_forehoi_visualization_mano_layers()


def _load_object_meshes(parts_dir: Path) -> list[dict]:
    manifest_path = parts_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = manifest["parts"] if isinstance(manifest, dict) and "parts" in manifest else manifest
    meshes = []
    for part_meta in parts:
        part_dir = parts_dir / part_meta["key"]
        texture_path = part_dir / part_meta["texture_file"]
        uvs = np.load(part_dir / "vertex_texcoords.npy").astype(np.float32)
        uvs[:, 1] = 1.0 - uvs[:, 1]
        meshes.append(
            {
                "name": part_meta["name"],
                "vertex_positions_local": np.load(part_dir / "vertex_positions_local.npy").astype(np.float32),
                "triangle_indices": np.load(part_dir / "triangle_indices.npy").astype(np.int32),
                "vertex_texcoords": uvs,
                "albedo_texture": np.asarray(Image.open(texture_path).convert("RGBA")),
            }
        )
    return meshes


def export_forehoi_rerun_sample(output_path, batch, pred, gt, sample_index, data_root, mano_layer=None, item_name="pi3x_train_sample"):
    mano_layer = _resolve_forehoi_mano_layer(mano_layer)
    rr = dexycb_export._load_rerun()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init(item_name, spawn=False)
    rr.save(str(output_path))
    rr.log("world/camera", rr.ViewCoordinates.RDF)

    sample_view0 = batch[0]
    object_meta = sample_view0.get("object", {})
    asset_meta = object_meta.get("asset", {}) or sample_view0.get("object_multiview", {}).get("asset", {})
    asset_type = _unwrap_singleton(asset_meta.get("asset_type"))
    asset_path = _unwrap_singleton(asset_meta.get("asset_path"))
    object_meshes = []
    if asset_type == "object_parts_bundle" and asset_path:
        object_meshes = _load_object_meshes(Path(asset_path))
        for mesh in object_meshes:
            rr.log(
                f"world/object/{mesh['name']}",
                rr.Mesh3D(
                    vertex_positions=mesh["vertex_positions_local"],
                    triangle_indices=mesh["triangle_indices"],
                    vertex_texcoords=mesh["vertex_texcoords"],
                    albedo_texture=mesh["albedo_texture"],
                ),
                static=True,
            )

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
            hand_faces = np.asarray(getattr(mano_layer, "faces", getattr(mano_layer, "th_faces", np.zeros((0, 3)))), dtype=np.int32)

    scene_scale = float(gt["scene_scale"][sample_index].item()) if "scene_scale" in gt else 1.0
    for frame_idx, view in enumerate(batch):
        dexycb_export._set_frame_time(rr, frame_idx)
        rgb = dexycb_export._tensor_rgb_to_uint8(view["img"][sample_index])
        rr.log("frames/rgb", rr.Image(rgb))

        valid_mask = gt["valid_masks"][sample_index, frame_idx]
        mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        gt_points = gt["local_points"][sample_index, frame_idx].detach().cpu().numpy()
        if mask_np.any():
            rr.log("world/gt_points", rr.Points3D(positions=gt_points[mask_np], colors=rgb[mask_np]))
        else:
            rr.log("world/gt_points", rr.Clear(recursive=False))

        object_view = view.get("object", {})
        object_view_valid = object_view.get("valid", None)
        pose = None
        if object_view_valid is not None and bool(object_view_valid[sample_index]):
            pose = np.asarray(gt["object_pose_obj2cam"][sample_index, frame_idx].detach().cpu(), dtype=np.float32)
        for mesh in object_meshes:
            if pose is None:
                rr.log(f"world/object/{mesh['name']}", rr.Clear(recursive=False))
            else:
                rr.log(
                    f"world/object/{mesh['name']}",
                    rr.Transform3D(mat3x3=pose[:3, :3].tolist(), translation=pose[:3, 3].tolist()),
                )

        gt_hand_mesh_inputs = dexycb_export._select_gt_hand_mesh_inputs(gt, sample_index, frame_idx)
        gt_hand_vertices_np = None
        gt_hand_joints_np = None
        if gt_hand_mesh_inputs is not None:
            verts_m, joints_m, _ = dexycb_export._build_hand_mesh_vertices_from_rotmat(
                gt_hand_mesh_inputs["global_orient"],
                gt_hand_mesh_inputs["hand_pose"],
                gt_hand_mesh_inputs["betas"],
                mano_layer,
                hand_side=gt_hand_mesh_inputs["hand_side"],
                hand_transl=gt_hand_mesh_inputs["transl"],
                hand_scale=gt_hand_mesh_inputs["scale"],
            )
            if verts_m is not None:
                gt_hand_vertices_np = np.asarray(verts_m, dtype=np.float32)
            if joints_m is not None:
                gt_hand_joints_np = np.asarray(joints_m, dtype=np.float32)

        pred_hand_vertices = None
        pred_hand_joints_np = None
        if pred is not None:
            pred_hand_mesh_inputs = dexycb_export._select_pred_hand_mesh_inputs(pred, sample_index, frame_idx)
            if pred_hand_mesh_inputs is not None:
                verts_m, joints_m, _ = dexycb_export._build_hand_mesh_vertices_from_rotmat(
                    pred_hand_mesh_inputs["global_orient"],
                    pred_hand_mesh_inputs["hand_pose"],
                    pred_hand_mesh_inputs["betas"],
                    mano_layer,
                    hand_side=pred_hand_mesh_inputs["hand_side"],
                    hand_transl=pred_hand_mesh_inputs["transl"],
                    hand_scale=pred_hand_mesh_inputs["scale"],
                )
                pred_hand_vertices = None if verts_m is None else np.asarray(verts_m, dtype=np.float32)
                pred_hand_joints_np = None if joints_m is None else np.asarray(joints_m, dtype=np.float32)

        dexycb_export._log_mesh_or_clear(rr, "world/gt_hand_mesh", gt_hand_vertices_np, hand_faces, dexycb_export._HAND_COLOR)
        dexycb_export._log_mesh_or_clear(rr, "world/pred_hand_mesh", pred_hand_vertices, hand_faces, dexycb_export._HAND_COLOR)
        if gt_hand_joints_np is not None:
            rr.log("world/gt_hand_joints", rr.Points3D(positions=gt_hand_joints_np))
        else:
            rr.log("world/gt_hand_joints", rr.Clear(recursive=False))
        if pred_hand_joints_np is not None:
            rr.log("world/pred_hand_joints", rr.Points3D(positions=pred_hand_joints_np))
        else:
            rr.log("world/pred_hand_joints", rr.Clear(recursive=False))

    # ── OMV world-space point clouds (outside per-frame loop, static) ──
    _has_omv_pred = pred is not None and "omv_local_points" in pred and "omv_camera_poses" in pred
    _has_omv_gt = "omv_depth" in gt and "omv_intrinsics" in gt and "omv_camera_pose" in gt

    if _has_omv_pred:
        pred_omv_pts = pred["omv_local_points"][sample_index]        # (N_omv, H, W, 3)
        pred_omv_poses = pred["omv_camera_poses"][sample_index]       # (N_omv, 4, 4)
        N_omv = pred_omv_pts.shape[0]
        pred_world_pts = []
        for v in range(N_omv):
            pts = pred_omv_pts[v].reshape(-1, 3)                      # torch (H*W, 3)
            cam_to_world = torch.inverse(pred_omv_poses[v])            # torch (4, 4)
            pts_world = dexycb_export._transform_vertices(pts, cam_to_world).detach().cpu().numpy()
            pred_world_pts.append(pts_world)
        pred_omv_world = np.concatenate(pred_world_pts, axis=0)
        if pred_omv_world.shape[0] > 20000:
            idx = np.random.RandomState(42).choice(pred_omv_world.shape[0], 20000, replace=False)
            pred_omv_world = pred_omv_world[idx]
        rr.log("world/omv/pred_points", rr.Points3D(positions=pred_omv_world, colors=[100, 200, 100]), static=True)

        # Pred camera frustum axes
        for v in range(N_omv):
            cam_to_world = torch.inverse(pred_omv_poses[v]).detach().cpu().numpy()
            rr.log(
                f"world/omv/pred_camera_{v}",
                rr.Transform3D(
                    mat3x3=cam_to_world[:3, :3].tolist(),
                    translation=cam_to_world[:3, 3].tolist(),
                ),
                static=True,
            )

    if _has_omv_gt:
        gt_omv_depth = gt["omv_depth"][sample_index]                 # (N_omv, H, W)
        gt_omv_K = gt["omv_intrinsics"][sample_index]                 # (N_omv, 3, 3)
        gt_omv_pose = gt["omv_camera_pose"][sample_index]             # (N_omv, 4, 4)
        N_omv_gt = gt_omv_depth.shape[0]
        device = gt_omv_depth.device
        gt_world_pts = []
        for v in range(N_omv_gt):
            valid = gt_omv_depth[v] > 0
            if not valid.any():
                continue
            d = gt_omv_depth[v][valid].reshape(-1, 1)                  # torch (M, 1)
            Hv, Wv = gt_omv_depth[v].shape
            yy, xx = torch.meshgrid(
                torch.arange(Hv, device=device),
                torch.arange(Wv, device=device),
                indexing="ij",
            )
            uv = torch.stack([xx[valid].float(), yy[valid].float(), torch.ones_like(xx[valid].float())], dim=-1)
            K_inv = torch.inverse(gt_omv_K[v])
            rays = (K_inv @ uv.T).T   # torch (M, 3)
            pts_local = rays * d       # torch (M, 3)
            cam_to_world = torch.inverse(gt_omv_pose[v])
            pts_world = dexycb_export._transform_vertices(pts_local, cam_to_world).detach().cpu().numpy()
            gt_world_pts.append(pts_world)
        if gt_world_pts:
            gt_omv_world = np.concatenate(gt_world_pts, axis=0)
            if gt_omv_world.shape[0] > 20000:
                idx = np.random.RandomState(42).choice(gt_omv_world.shape[0], 20000, replace=False)
                gt_omv_world = gt_omv_world[idx]
            rr.log("world/omv/gt_points", rr.Points3D(positions=gt_omv_world, colors=[100, 100, 200]), static=True)
        else:
            rr.log("world/omv/gt_points", rr.Clear(recursive=False))

        # GT camera frustum axes
        for v in range(N_omv_gt):
            cam_to_world = torch.inverse(gt_omv_pose[v]).detach().cpu().numpy()
            rr.log(
                f"world/omv/gt_camera_{v}",
                rr.Transform3D(
                    mat3x3=cam_to_world[:3, :3].tolist(),
                    translation=cam_to_world[:3, 3].tolist(),
                ),
                static=True,
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
        dataset="ForeHOI",
        metadata=meta,
    )
    write_manifest(output_path.with_name("manifest.json"), manifest)
    return output_path, meta_path
