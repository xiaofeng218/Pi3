from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_rerun():
    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rerun is required to export canonical view point clouds") from exc
    return rr


def _load_depth_png(path: Path) -> np.ndarray:
    depth_mm = np.array(Image.open(path), dtype=np.uint16)
    return depth_mm.astype(np.float32) / 1000.0


def _load_color(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _depth_to_camera_points(depth_m: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    z = depth_m
    valid = z > 0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), valid

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    points_cam = np.stack([x, y, z], axis=-1)
    return points_cam[valid].astype(np.float32), valid


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = (transform @ homog.T).T
    return transformed[:, :3].astype(np.float32)


def export_canonical_views_rrd(model_dir: Path, output_path: Path, stride: int = 1) -> dict:
    rr = _load_rerun()
    canonical_dir = model_dir / "canonical_views_224"
    camera_params = np.load(canonical_dir / "camera_params.npz")
    meta = json.loads((canonical_dir / "meta.json").read_text(encoding="utf-8"))

    intrinsics = camera_params["K"].astype(np.float32)
    transforms_oc = camera_params["T_oc"].astype(np.float32)
    view_names = [str(v) for v in camera_params["view_names"]]

    merged_points = []
    merged_colors = []
    per_view_counts = []

    rr.init(f"canonical_views_{model_dir.name}", spawn=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(output_path))
    rr.log("world", rr.ViewCoordinates.RDF)

    reference_points_path = model_dir / "points.xyz"
    if reference_points_path.is_file():
        reference_points = np.loadtxt(reference_points_path, dtype=np.float32)
        rr.log(
            "world/reference_points_xyz",
            rr.Points3D(reference_points, colors=np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (reference_points.shape[0], 1))),
        )

    for view_idx, view_name in enumerate(view_names):
        color_path = canonical_dir / f"color_{view_idx:06d}.jpg"
        depth_path = canonical_dir / f"aligned_depth_to_color_{view_idx:06d}.png"
        color = _load_color(color_path)
        depth_m = _load_depth_png(depth_path)

        if stride > 1:
            color = color[::stride, ::stride]
            depth_m = depth_m[::stride, ::stride]
            K = intrinsics[view_idx].copy()
            K[0, 0] /= stride
            K[1, 1] /= stride
            K[0, 2] = (K[0, 2] + 0.5) / stride - 0.5
            K[1, 2] = (K[1, 2] + 0.5) / stride - 0.5
        else:
            K = intrinsics[view_idx]

        points_cam, valid = _depth_to_camera_points(depth_m, K)
        colors = color[valid].astype(np.uint8) if np.any(valid) else np.zeros((0, 3), dtype=np.uint8)
        points_obj = _transform_points(points_cam, transforms_oc[view_idx])

        merged_points.append(points_obj)
        merged_colors.append(colors)
        per_view_counts.append(int(points_obj.shape[0]))

        rr.set_time("view_idx", sequence=view_idx)
        rr.log(f"views/{view_name}/rgb", rr.Image(color))
        rr.log(f"views/{view_name}/depth_m", rr.DepthImage(depth_m))
        rr.log(f"world/views/{view_name}_camera_points", rr.Points3D(points_cam, colors=colors))
        rr.log(f"world/views/{view_name}_canonical_points", rr.Points3D(points_obj, colors=colors))

    merged_points_np = np.concatenate(merged_points, axis=0) if merged_points else np.zeros((0, 3), dtype=np.float32)
    merged_colors_np = np.concatenate(merged_colors, axis=0) if merged_colors else np.zeros((0, 3), dtype=np.uint8)
    rr.log("world/merged_canonical_points", rr.Points3D(merged_points_np, colors=merged_colors_np))

    summary = {
        "model_dir": str(model_dir.resolve()),
        "canonical_dir": str(canonical_dir.resolve()),
        "output_rrd": str(output_path.resolve()),
        "stride": int(stride),
        "object_name": meta.get("object_name", model_dir.name),
        "depth_encoding": meta.get("depth_encoding"),
        "coordinate_frame": meta.get("coordinate_frame"),
        "per_view_counts": per_view_counts,
        "merged_count": int(merged_points_np.shape[0]),
        "merged_bbox_min": merged_points_np.min(axis=0).tolist() if merged_points_np.size else [0.0, 0.0, 0.0],
        "merged_bbox_max": merged_points_np.max(axis=0).tolist() if merged_points_np.size else [0.0, 0.0, 0.0],
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DexYCB canonical_views depth point clouds to a rerun .rrd file")
    parser.add_argument(
        "--model-dir",
        default="data/dataset/dexycb/models/025_mug",
        help="DexYCB object directory containing canonical_views_224",
    )
    parser.add_argument(
        "--output",
        default="outputs/debug_canonical_views/025_mug/canonical_views_224_points.rrd",
        help="Output .rrd path",
    )
    parser.add_argument("--stride", type=int, default=1, help="Optional image/depth stride for lighter exports")
    args = parser.parse_args()

    summary = export_canonical_views_rrd(Path(args.model_dir), Path(args.output), stride=max(1, args.stride))
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
