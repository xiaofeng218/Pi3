from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.forehoi_dataset import ForeHOIDataset


def write_rgb(path: Path, value: int, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:28, 8:28] = 127
    mask[24:44, 24:44] = 255
    Image.fromarray(mask).save(path)


def write_depth_png_mm(path: Path, value_mm: int, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0]), value_mm, dtype=np.uint16)).save(path)


def build_fixture(root: Path) -> tuple[Path, Path]:
    raw_root = root / "forehoi"
    omv_root = root / "forehoi" / "object_multiview_pyrender"
    sequence_name = "seq_000"
    asset_key = "asset_000"
    seq_dir = raw_root / sequence_name
    (seq_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (seq_dir / "depth").mkdir(parents=True, exist_ok=True)
    (seq_dir / "mask").mkdir(parents=True, exist_ok=True)
    (seq_dir / "meta").mkdir(parents=True, exist_ok=True)

    for frame_idx in range(3):
        write_rgb(seq_dir / "rgb" / f"frame_{frame_idx:06d}.png", 40 + frame_idx)
        np.save(seq_dir / "depth" / f"frame_{frame_idx:06d}.npy", np.full((64, 64), 0.7 + 0.1 * frame_idx, dtype=np.float32))
        write_mask(seq_dir / "mask" / f"frame_{frame_idx:06d}.png")

    camera_intrinsics = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    camera_pose = np.eye(4, dtype=np.float32)
    hand_pose_mano = np.zeros((3, 48), dtype=np.float32)
    hand_pose_mano[:, 0] = 0.1
    hand_transl_cam = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.55], [0.0, 0.0, 0.6]], dtype=np.float32)
    hand_joints_3d_cam = np.linspace(0.0, 0.2, 3 * 21 * 3, dtype=np.float32).reshape(3, 21, 3)
    hand_mano_betas = np.full((10,), 0.03, dtype=np.float32)
    object_pose = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    object_pose[:, 2, 3] = np.array([0.7, 0.8, 0.9], dtype=np.float32)
    hand_valid = np.array([True, True, True], dtype=bool)
    object_valid = np.array([True, True, True], dtype=bool)
    np.savez(
        seq_dir / "meta" / "meta.npz",
        camera_intrinsics=camera_intrinsics,
        camera_pose=camera_pose,
        sampled_frame_indices=np.arange(3, dtype=np.int32),
        hand_pose_mano=hand_pose_mano,
        hand_transl_cam=hand_transl_cam,
        hand_joints_3d_cam=hand_joints_3d_cam,
        hand_mano_betas=hand_mano_betas,
        object_pose_obj2cam=object_pose,
        hand_valid=hand_valid,
        object_valid=object_valid,
    )
    (raw_root / "split.json").write_text(
        json.dumps(
            {
                "train": {
                    "shards": [sequence_name],
                },
                "val": {
                    "shards": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (seq_dir / "meta" / "sequence_meta.json").write_text(
        json.dumps(
            {
                "sequence_name": sequence_name,
                "frame_count": 3,
                "mano_side": "right",
                "objaverse_asset_key": asset_key,
                "object_parts_root": str(root / "object_parts"),
                "object_parts_key": asset_key,
                "object_scale_info": {
                    "source_extent": [2.0, 2.0, 2.0],
                    "target_extent": [1.0, 1.0, 1.0],
                    "alignment_scale": 0.5,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    parts_dir = root / "object_parts" / asset_key
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "manifest.json").write_text(
        json.dumps(
            {
                "parts": [],
                "scale_info": {
                    "source_center": [0.0, 0.0, 0.0],
                    "source_extent": 2.0,
                    "target_center": [0.0, 0.0, 0.0],
                    "target_extent": 1.0,
                    "alignment_scale": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle_dir = omv_root / asset_key
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for view_idx in range(10):
        write_rgb(bundle_dir / f"color_{view_idx:06d}.jpg", 100 + view_idx, size=(32, 32))
        write_depth_png_mm(bundle_dir / f"aligned_depth_to_color_{view_idx:06d}.png", 900 + view_idx, size=(32, 32))
        Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(bundle_dir / f"mask_{view_idx:06d}.png")
    np.savez(
        bundle_dir / "camera_params.npz",
        K=np.repeat(np.array([[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)[None, :, :], 10, axis=0),
        T_oc=np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 10, axis=0),
        normalization_center=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        normalization_scale=np.float32(2.5),
    )
    (bundle_dir / "meta.json").write_text(json.dumps({"normalization": {"center": [0.1, 0.2, 0.3], "scale": 2.5}}), encoding="utf-8")
    return raw_root, omv_root


class ForeHOIDatasetContractTests(unittest.TestCase):
    def test_forehoi_dataset_emits_required_contract(self):
        with tempfile.TemporaryDirectory(prefix="forehoi_contract_") as tmpdir:
            root = Path(tmpdir)
            raw_root, omv_root = build_fixture(root)
            dataset = ForeHOIDataset(
                data_root=str(raw_root),
                object_multiview_root=str(omv_root),
                mode="train",
                resolution=[[64, 64]],
                frame_num=2,
            )
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(set(sample.keys()), {"views", "object_multiview_payload"})
            self.assertEqual(len(sample["views"]), 2)
            self.assertTrue({"img", "depthmap", "camera_intrinsics", "camera_pose"}.issubset(sample["object_multiview_payload"].keys()))
            for view in sample["views"]:
                self.assertEqual(view["dataset"], "ForeHOI")
                self.assertEqual(view["hand"]["pose_repr"], "mano_full_aa")
                self.assertEqual(view["hand"]["global_orient_rotmat_gt"].shape, (1, 3, 3))
                self.assertEqual(view["hand"]["pose_rotmat_gt"].shape, (15, 3, 3))
                self.assertGreater(view["object"]["scale_meta"]["canonical_to_target_scale"], 0)
                self.assertEqual(view["object_multiview"]["normalization_center"].shape, (3,))
                self.assertIn(np.asarray(view["object_multiview"]["normalization_scale"]).shape, {(), (1,)})


if __name__ == "__main__":
    unittest.main()
