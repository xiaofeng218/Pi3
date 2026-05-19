"""Minimal contract test for the root DexYCB dataset adapter.

Run with:
    conda run -n pi3 python debug/test_dexycb_dataset_contract.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "prettytable" not in sys.modules:
    class _PrettyTable:
        def __init__(self, *args, **kwargs):
            self.rows = []

        def add_row(self, row):
            self.rows.append(row)

        def __str__(self):
            return "\n".join(str(row) for row in self.rows)

    sys.modules["prettytable"] = type(sys)("prettytable")
    sys.modules["prettytable"].PrettyTable = _PrettyTable

from datasets.base.utils import unified_collate_fn
from datasets.dexycb_dataset import DexYCBDataset


SERIALS = [
    "836212060125",
    "839512060362",
]


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_depth_png(path: Path, depth_mm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_mm.astype(np.uint16)).save(path)


def write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((480, 640, 3), value, dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def write_square_rgb(path: Path, value: int, size: int = 224) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((size, size, 3), value, dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def write_minimal_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0.0 0.0 0.5",
                "v 0.05 0.0 0.5",
                "v 0.0 0.05 0.5",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_label(
    path: Path,
    *,
    object_id: int,
    with_interaction: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seg = np.zeros((480, 640), dtype=np.uint8)
    pose_y = np.zeros((1, 3, 4), dtype=np.float32)
    pose_m = np.zeros((1, 51), dtype=np.float32)
    joint_3d = np.full((1, 21, 3), -1.0, dtype=np.float32)
    joint_2d = np.full((1, 21, 2), -1.0, dtype=np.float32)

    if with_interaction:
        seg[120:220, 120:220] = 255
        seg[200:320, 260:380] = object_id
        pose_y[0, :, :3] = np.eye(3, dtype=np.float32)
        pose_y[0, :, 3] = np.array([0.02, -0.01, 0.6], dtype=np.float32)
        pose_m[0, :10] = np.linspace(0.1, 1.0, 10, dtype=np.float32)
        pose_m[0, 48:51] = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        joint_3d[0, :, :] = np.linspace(0.0, 0.2, 63, dtype=np.float32).reshape(21, 3)
        joint_2d[0, :, :] = np.linspace(0.0, 200.0, 42, dtype=np.float32).reshape(21, 2)

    np.savez(
        path,
        seg=seg,
        pose_y=pose_y,
        pose_m=pose_m,
        joint_3d=joint_3d,
        joint_2d=joint_2d,
    )


def build_fixture(root: Path) -> None:
    object_id = 11

    intrinsics = {
        "color": {
            "fx": 600.0,
            "fy": 610.0,
            "ppx": 320.0,
            "ppy": 240.0,
        }
    }
    for serial in SERIALS:
        write_yaml(root / "calibration" / "intrinsics" / f"{serial}_640x480.yml", intrinsics)

    extrinsics = {
        "extrinsics": {
            SERIALS[0]: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            SERIALS[1]: [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }
    write_yaml(root / "calibration" / "extrinsics_fixture" / "extrinsics.yml", extrinsics)
    write_yaml(root / "calibration" / "mano_fixture_subject-01_right" / "mano.yml", {"betas": [0.01] * 10})
    write_yaml(root / "calibration" / "mano_fixture_subject-01_left" / "mano.yml", {"betas": [0.02] * 10})

    intrinsics_mv = np.repeat(
        np.array([[160.0, 0.0, 111.5], [0.0, 160.0, 111.5], [0.0, 0.0, 1.0]], dtype=np.float32)[None, :, :],
        8,
        axis=0,
    )
    camera_pose_mv = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 8, axis=0)
    object_pose_obj2cam_mv = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 8, axis=0)
    view_dirs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    for model_name in ("011_banana", "019_pitcher_base"):
        multiview_dir = root / "models" / model_name / "canonical_views_224"
        multiview_dir.mkdir(parents=True, exist_ok=True)
        write_minimal_obj(root / "models" / model_name / "textured_simple.obj")
        np.savez(
            multiview_dir / "camera_params.npz",
            K=intrinsics_mv,
            T_oc=camera_pose_mv,
            T_co=object_pose_obj2cam_mv,
            view_dirs=view_dirs,
            normalization_center=np.zeros(3, dtype=np.float32),
            normalization_scale=np.float32(1.0),
        )
        for view_idx in range(8):
            write_square_rgb(multiview_dir / f"color_{view_idx:06d}.jpg", value=80 + view_idx)
            write_depth_png(
                multiview_dir / f"aligned_depth_to_color_{view_idx:06d}.png",
                np.full((224, 224), 900 + view_idx, dtype=np.uint16),
            )

    subject_root = root / "20200709-subject-01"
    for seq_idx in range(5):
        sequence = subject_root / f"20200709_1418{seq_idx:02d}"
        mano_side = "left" if seq_idx == 2 else "right"
        mano_calib = f"mano_fixture_subject-01_{mano_side}"
        meta = {
            "serials": SERIALS,
            "num_frames": 4,
            "extrinsics": "fixture",
            "ycb_ids": [object_id],
            "ycb_grasp_ind": 0,
            "mano_sides": [mano_side],
            "mano_calib": [mano_calib],
        }
        write_yaml(sequence / "meta.yml", meta)
        np.savez(
            sequence / "pose.npz",
            pose_y=np.zeros((4, 1, 7), dtype=np.float32),
            pose_m=np.zeros((4, 1, 51), dtype=np.float32),
        )

        for serial in SERIALS:
            for frame_idx in range(4):
                camera_dir = sequence / serial
                write_rgb(camera_dir / f"color_{frame_idx:06d}.jpg", value=20 * (seq_idx + 1) + frame_idx)
                depth_mm = np.full((480, 640), 700 + frame_idx * 10, dtype=np.uint16)
                depth_mm[:20, :20] = 0
                write_depth_png(camera_dir / f"aligned_depth_to_color_{frame_idx:06d}.png", depth_mm)
                has_interaction = serial == SERIALS[0] and frame_idx in (1, 2)
                write_label(
                    camera_dir / f"labels_{frame_idx:06d}.npz",
                    object_id=object_id,
                    with_interaction=has_interaction,
                )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dexycb_mirror_") as tmpdir:
        root = Path(tmpdir)
        build_fixture(root)

        np_load = np.load

        def guarded_np_load(file, *args, **kwargs):
            path = str(file)
            if "labels_" in path:
                raise AssertionError("Index build should not read labels_*.npz")
            return np_load(file, *args, **kwargs)

        with mock.patch("numpy.load", side_effect=guarded_np_load):
            train_dataset = DexYCBDataset(
                data_root=str(root),
                mode="train",
                resolution=[[224, 224]],
                frame_num=3,
            )
            valid_dataset = DexYCBDataset(
                data_root=str(root),
                mode="valid",
                resolution=[[224, 224]],
                frame_num=3,
            )

        assert len(train_dataset) == 8, len(train_dataset)
        assert len(valid_dataset) == 2, len(valid_dataset)

        sample = train_dataset[0]
        assert set(sample.keys()) == {"views", "object_multiview_payload"}
        assert len(sample["views"]) == 3
        assert set(sample["object_multiview_payload"].keys()) == {"img", "depthmap", "camera_intrinsics", "camera_pose"}

        for view in sample["views"]:
            assert view["dataset"] == "DexYCB"
            assert view["img"].shape[0] == 3
            assert view["depthmap"].ndim == 2
            assert view["camera_intrinsics"].shape == (3, 3)
            assert view["camera_pose"].shape == (4, 4)
            assert "hand" in view
            assert view["hand"]["mask"].shape == view["depthmap"].shape
            assert view["hand"]["pose_mano"].shape == (48,)
            assert view["hand"]["hand_transl"].shape == (3,)
            assert view["hand"]["joints_3d_cam"].shape == (21, 3)
            assert view["hand"]["joints_2d"].shape == (21, 2)
            assert view["hand"]["mano_betas"].shape == (10,)
            assert view["hand"]["mano_side"] == "right"
            assert isinstance(bool(view["hand"]["valid"]), bool)
            if view["hand"]["valid"]:
                assert np.isfinite(view["hand"]["joints_2d"]).all()
            assert "object_multiview" in view
            assert "img" not in view["object_multiview"]
            assert "depthmap" not in view["object_multiview"]
            assert "camera_intrinsics" not in view["object_multiview"]
            assert "camera_pose" not in view["object_multiview"]
            assert "pts3d" not in view["object_multiview"]
            assert view["object_multiview"]["template_vertices"].shape[1] == 3
            assert view["object_multiview"]["normalization_center"].shape == (3,)
            assert "grasped_object_id" not in view["object_multiview"]
            assert "grasped_object_mask" not in view["object_multiview"]
            assert "grasped_object_pose_obj2cam" not in view["object_multiview"]
            assert "grasped_object_vertices_2d" not in view["object_multiview"]
            assert "grasped_object_vertices_2d_valid" not in view["object_multiview"]
            assert "grasped_object_valid" not in view["object_multiview"]
            assert view["object"]["grasped_object_id"] == 11
            assert view["object"]["mask"].shape == view["depthmap"].shape
            assert view["object"]["pose_obj2cam"].shape == (4, 4)
            assert isinstance(bool(view["object"]["valid"]), bool)

        loader = DataLoader(
            dataset=train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=unified_collate_fn,
        )
        batch = next(iter(loader))
        assert set(batch.keys()) == {"views", "object_multiview_payload"}
        assert len(batch["views"]) == 3
        assert batch["object_multiview_payload"]["img"].shape[0] == 2
        assert "img" not in batch["views"][0]["object_multiview"]
        assert "depthmap" not in batch["views"][0]["object_multiview"]
        assert "camera_intrinsics" not in batch["views"][0]["object_multiview"]
        assert "camera_pose" not in batch["views"][0]["object_multiview"]
        assert "pts3d" not in batch["views"][0]["object_multiview"]
        assert batch["views"][0]["object_multiview"]["template_vertices"].shape[0] == 2
        assert "grasped_object_mask" not in batch["views"][0]["object_multiview"]
        assert batch["views"][0]["object"]["mask"].shape == (2, 224, 224)

        print(json.dumps(
            {
                "train_tracks": len(train_dataset),
                "valid_tracks": len(valid_dataset),
                "batched_views": len(batch),
            },
            indent=2,
        ))

        payload_dataset = DexYCBDataset(
            data_root=str(root),
            mode="train",
            resolution=[[224, 224]],
            frame_num=3,
            include_object_multiview_payload=True,
        )
        payload_sample = payload_dataset[0]
        payload_mv = payload_sample["views"][0]["object_multiview"]
        assert payload_mv["img"].shape == (8, 3, 224, 224)
        assert payload_mv["depthmap"].shape == (8, 224, 224)
        assert payload_mv["camera_intrinsics"].shape == (8, 3, 3)
        assert payload_mv["camera_pose"].shape == (8, 4, 4)
        assert payload_mv["pts3d"].shape == (8, 224, 224, 3)
        train_mv = train_dataset.get_object_multiview_payload(11)
        assert "pts3d" not in train_mv

        selected_dataset = DexYCBDataset(
            data_root=str(root),
            mode="train",
            resolution=[[224, 224]],
            frame_num=3,
            selected_tracks={
                "right": {
                    "subject": "20200709-subject-01",
                    "sequence": "20200709_141800",
                    "camera": SERIALS[0],
                },
                "left": {
                    "subject": "20200709-subject-01",
                    "sequence": "20200709_141802",
                    "camera": SERIALS[1],
                },
            },
        )
        assert len(selected_dataset) == 2, len(selected_dataset)
        assert {track["mano_side"] for track in selected_dataset.tracks} == {"left", "right"}
        assert {track["sequence"] for track in selected_dataset.tracks} == {"20200709_141800", "20200709_141802"}


if __name__ == "__main__":
    main()
