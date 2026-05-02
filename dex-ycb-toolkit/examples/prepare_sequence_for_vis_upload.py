#!/usr/bin/env python3
"""Prepare DexYCB sequence visualizations and a vis_upload-compatible manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
if "bool" not in np.__dict__:
    np.bool = bool
import trimesh


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dex_ycb_toolkit.headless_vis import save_sequence_videos
from dex_ycb_toolkit.sequence_loader import SequenceLoader


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


@dataclass(frozen=True)
class MeshTemplate:
    vertices: np.ndarray
    faces: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a DexYCB camera sequence bundle that can be uploaded by vis_upload.",
    )
    parser.add_argument("--camera-dir", required=True, help="Path to one DexYCB camera directory.")
    parser.add_argument("--output-root", required=True, help="Root directory for prepared output bundles.")
    parser.add_argument("--item-id", default=None, help="vis_upload item_id. Defaults to subject/sequence/camera.")
    parser.add_argument("--release", default="sequence-vis", help="Manifest release label.")
    parser.add_argument("--dataset", default="dexycb", help="Manifest dataset label.")
    parser.add_argument("--fps", default=15, type=int, help="FPS for the generated preview videos.")
    parser.add_argument("--num-frames", default=None, type=int, help="Optional frame count override in metadata.")
    parser.add_argument("--device", default="cpu", help="SequenceLoader device, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--sequence-name",
        default=None,
        help="DexYCB sequence name in subject/sequence form. Defaults to inferring from camera-dir.",
    )
    parser.add_argument(
        "--camera-serial",
        default=None,
        help="Camera serial override. Defaults to the last path component of camera-dir.",
    )
    return parser.parse_args()


def infer_sequence_identity(camera_dir: str | Path) -> tuple[str, str]:
    camera_path = Path(camera_dir).resolve()
    if len(camera_path.parents) < 2:
        raise ValueError(f"camera-dir does not contain subject/sequence/camera structure: {camera_path}")
    camera_serial = camera_path.name
    sequence_name = f"{camera_path.parent.parent.name}/{camera_path.parent.name}"
    return sequence_name, camera_serial


def default_item_id_for_camera_dir(camera_dir: str | Path) -> str:
    sequence_name, camera_serial = infer_sequence_identity(camera_dir)
    return f"{sequence_name}/{camera_serial}"


def _bundle_dir(output_root: str | Path, item_id: str) -> Path:
    return Path(output_root).joinpath(*item_id.split("/"))


def _write_json(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return target


def _normalize_video_outputs(video_outputs: dict[str, str | Path], bundle_dir: str | Path) -> dict[str, Path]:
    bundle_path = Path(bundle_dir)
    role_to_name = {
        "rgb": "rgb.mp4",
        "seg": "seg.mp4",
        "joints": "joints.mp4",
        "overlay": "overlay.mp4",
        "foreground_overlay": "foreground_overlay.mp4",
    }
    normalized: dict[str, Path] = {}
    for role, file_name in role_to_name.items():
        if role not in video_outputs:
            continue
        source = Path(video_outputs[role])
        target = bundle_path / file_name
        if source.resolve() != target.resolve():
            source.replace(target)
        normalized[role] = target
    return normalized


def _transform_vertices(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    hom = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)], axis=1)
    transformed = hom @ pose.T
    return transformed[:, :3]


def _color_to_linear_rgb(color: tuple[int, int, int]) -> list[float]:
    return [channel / 255.0 for channel in color]


def _colors_to_rgba(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.uint8)
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([colors, alpha], axis=1)


def _solid_vertex_rgba(vertex_count: int, color: tuple[int, int, int]) -> np.ndarray:
    rgba = np.array([*color, 255], dtype=np.uint8)
    return np.repeat(rgba[None, :], vertex_count, axis=0)


def _mesh3d(rr, vertex_positions: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray):
    try:
        return rr.Mesh3D(
            vertex_positions=vertex_positions,
            indices=faces,
            vertex_colors=vertex_colors,
        )
    except TypeError:
        return rr.Mesh3D(
            vertex_positions=vertex_positions,
            triangle_indices=faces,
        )


def _deproject_full_depth_pointcloud(
    loader: SequenceLoader,
    camera_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the full valid depth point cloud in the current camera coordinates."""
    if not hasattr(loader, "_load_frame_rgbd"):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    _, depth = loader._load_frame_rgbd(camera_index, loader._frame)
    depth_m = depth.astype(np.float32) / 1000.0

    rays = loader._p[camera_index].t().view(loader._h, loader._w, 3).cpu().numpy()
    points = depth_m[..., None] * rays

    valid = depth > 0
    valid &= np.isfinite(points).all(axis=2)

    rgb = np.asarray(loader.pcd_rgb[camera_index], dtype=np.uint8)
    return points[valid].astype(np.float32), rgb[valid]


def _load_rerun():
    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rerun is required to export sequence.rrd. Install the rerun-sdk package in the active environment."
        ) from exc
    return rr


def _load_object_mesh_templates(loader: SequenceLoader) -> dict[int, MeshTemplate]:
    templates: dict[int, MeshTemplate] = {}
    for obj_id, obj_file in zip(loader.ycb_ids, loader.ycb_group_layer.obj_file):
        mesh = trimesh.load(obj_file, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.dump()))
        templates[obj_id] = MeshTemplate(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
        )
    return templates


def export_rerun_sequence_rrd(
    sequence_name: str,
    camera_serial: str,
    output_path: str | Path,
    device: str = "cpu",
    rr_module=None,
    loader_factory=SequenceLoader,
    object_mesh_loader=None,
) -> Path:
    rr = rr_module or _load_rerun()
    loader = loader_factory(sequence_name, device=device, preload=False, app="renderer")

    if camera_serial not in loader.serials:
        raise ValueError(f"camera serial {camera_serial!r} not found in sequence {sequence_name!r}")
    camera_index = loader.serials.index(camera_serial)

    object_templates = (
        object_mesh_loader(loader) if object_mesh_loader is not None else _load_object_mesh_templates(loader)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init("dexycb_sequence_vis", spawn=False)
    rr.save(str(output_path))
    rr.log("world/camera", rr.ViewCoordinates.RDF, static=True)

    hand_layers = getattr(loader.mano_group_layer, "_layers", [])
    pointcloud_entity_path = "world/pointcloud"

    for frame_idx in range(loader.num_frames):
        loader.step()
        rr.set_time_sequence("frame", frame_idx)

        point_positions, point_colors = _deproject_full_depth_pointcloud(loader, camera_index)
        if point_positions.size == 0:
            rr.log(pointcloud_entity_path, rr.Clear(recursive=False))
        else:
            rr.log(
                pointcloud_entity_path,
                rr.Points3D(
                    positions=point_positions,
                    colors=_colors_to_rgba(point_colors),
                ),
            )

        for object_idx, obj_id in enumerate(loader.ycb_ids):
            entity_path = f"world/objects/{obj_id:02d}_{object_idx}"
            pose = loader.ycb_pose[camera_index][object_idx]
            if np.all(pose == 0.0):
                rr.log(entity_path, rr.Clear(recursive=False))
                continue

            template = object_templates[obj_id]
            vertex_positions = _transform_vertices(template.vertices, pose)
            rr.log(
                entity_path,
                _mesh3d(
                    rr,
                    vertex_positions=vertex_positions,
                    faces=template.faces,
                    vertex_colors=_solid_vertex_rgba(
                        vertex_positions.shape[0],
                        _YCB_COLORS.get(obj_id, (180, 180, 180)),
                    ),
                ),
            )

        hand_vertices = loader.mano_vert[camera_index]
        for hand_idx, vertices in enumerate(hand_vertices):
            entity_path = f"world/hands/{hand_idx}"
            if np.all(vertices == 0.0):
                rr.log(entity_path, rr.Clear(recursive=False))
                continue

            faces = hand_layers[hand_idx].f
            if hasattr(faces, "cpu"):
                faces = faces.cpu().numpy()
            faces = np.asarray(faces, dtype=np.int32)
            vertex_positions = np.asarray(vertices, dtype=np.float32)
            rr.log(
                entity_path,
                _mesh3d(
                    rr,
                    vertex_positions=vertex_positions,
                    faces=faces,
                    vertex_colors=_solid_vertex_rgba(vertex_positions.shape[0], _HAND_COLOR),
                ),
            )

    return output_path


def prepare_sequence_bundle_for_vis_upload(
    camera_dir: str | Path,
    output_root: str | Path,
    item_id: str | None = None,
    release: str = "sequence-vis",
    dataset: str = "dexycb",
    fps: int = 15,
    num_frames: int | None = None,
    device: str = "cpu",
    sequence_name: str | None = None,
    camera_serial: str | None = None,
    video_exporter=save_sequence_videos,
    rrd_exporter=export_rerun_sequence_rrd,
) -> Path:
    camera_dir = Path(camera_dir)
    inferred_sequence_name, inferred_camera_serial = infer_sequence_identity(camera_dir)
    sequence_name = sequence_name or inferred_sequence_name
    camera_serial = camera_serial or inferred_camera_serial
    item_id = item_id or default_item_id_for_camera_dir(camera_dir)

    bundle_dir = _bundle_dir(output_root, item_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rendered_videos = video_exporter(camera_dir=str(camera_dir), output_dir=str(bundle_dir), prefix="", fps=fps)
    normalized_videos = _normalize_video_outputs(rendered_videos, bundle_dir)

    rrd_path = bundle_dir / "sequence.rrd"
    rrd_exporter(sequence_name=sequence_name, camera_serial=camera_serial, output_path=rrd_path, device=device)

    metadata = {
        "source_camera_dir": str(camera_dir.resolve()),
        "sequence_name": sequence_name,
        "camera_serial": camera_serial,
        "fps": fps,
        "num_frames": num_frames,
    }
    meta_path = _write_json(bundle_dir / "meta.json", metadata)

    artifacts = []
    for role, remote_name in [
        ("preview_rgb", "rgb.mp4"),
        ("preview_seg", "seg.mp4"),
        ("preview_joints", "joints.mp4"),
        ("preview_overlay", "overlay.mp4"),
        ("preview_foreground_overlay", "foreground_overlay.mp4"),
    ]:
        file_path = bundle_dir / remote_name
        if file_path.exists():
            artifacts.append(
                {
                    "role": role,
                    "local_path": str(file_path.resolve()),
                    "remote_name": remote_name,
                    "visibility": "public",
                    "content_type": "video/mp4",
                }
            )

    artifacts.append(
        {
            "role": "interactive_rrd",
            "local_path": str(rrd_path.resolve()),
            "remote_name": "sequence.rrd",
            "visibility": "private",
            "content_type": "application/octet-stream",
        }
    )
    artifacts.append(
        {
            "role": "metadata_json",
            "local_path": str(meta_path.resolve()),
            "remote_name": "meta.json",
            "visibility": "private",
            "content_type": "application/json",
        }
    )

    manifest = {
        "schema_version": "1.0",
        "release": release,
        "dataset": dataset,
        "item_id": item_id,
        "metadata": metadata,
        "artifacts": artifacts,
    }
    return _write_json(bundle_dir / "manifest.json", manifest)


def main() -> int:
    args = parse_args()
    manifest_path = prepare_sequence_bundle_for_vis_upload(
        camera_dir=args.camera_dir,
        output_root=args.output_root,
        item_id=args.item_id,
        release=args.release,
        dataset=args.dataset,
        fps=args.fps,
        num_frames=args.num_frames,
        device=args.device,
        sequence_name=args.sequence_name,
        camera_serial=args.camera_serial,
    )
    print(f"prepared: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
