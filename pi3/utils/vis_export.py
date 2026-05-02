from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SequenceOutputPlan:
    item_id: str
    item_dir: Path
    rrd_path: Path
    manifest_path: Path


def default_item_id(data_path: str | Path) -> str:
    path = Path(data_path)
    return path.stem if path.suffix else path.name


def plan_sequence_output(output_root: str | Path, job_name: str, item_id: str) -> SequenceOutputPlan:
    item_dir = Path(output_root) / job_name / item_id
    return SequenceOutputPlan(
        item_id=item_id,
        item_dir=item_dir,
        rrd_path=item_dir / "sequence.rrd",
        manifest_path=item_dir / "manifest.json",
    )


def build_manifest(item_id: str, rrd_path: str | Path, release: str, dataset: str, metadata: dict | None = None) -> dict:
    manifest = {
        "schema_version": "1.0",
        "item_id": item_id,
        "artifacts": [
            {
                "role": "interactive_rrd",
                "local_path": str(Path(rrd_path).resolve()),
                "remote_name": "sequence.rrd",
                "visibility": "private",
                "content_type": "application/octet-stream",
            }
        ],
    }
    if release:
        manifest["release"] = release
    if dataset:
        manifest["dataset"] = dataset
    if metadata:
        manifest["metadata"] = metadata
    return manifest


def write_manifest(manifest_path: str | Path, manifest: dict) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def collect_masked_points_and_colors(points, colors, masks):
    frame_points = []
    frame_colors = []
    color_array = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)

    for frame_points_src, frame_colors_src, frame_mask in zip(points, color_array, masks):
        frame_points.append(np.asarray(frame_points_src)[frame_mask])
        frame_colors.append(np.asarray(frame_colors_src)[frame_mask])

    return frame_points, frame_colors


def write_rrd_sequence(rrd_path: str | Path, points_per_frame, colors_per_frame, entity_path: str = "world/points") -> None:
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError("rerun is required to export .rrd sequences") from exc

    rrd_path = Path(rrd_path)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init("pi3_example_mm", spawn=False, default_enabled=True)
    for frame_idx, (frame_points, frame_colors) in enumerate(zip(points_per_frame, colors_per_frame)):
        rr.set_time_sequence("frame", frame_idx)
        rr.log(entity_path, rr.Points3D(frame_points, colors=frame_colors))

    rr.save(rrd_path)
