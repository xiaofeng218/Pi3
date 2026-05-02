from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
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
from debug.test_dexycb_dataset_contract import build_fixture
from pi3.models.pi3x import Pi3X


def _find_batch_with_valid_object(loader):
    for batch in loader:
        has_valid = False
        for view in batch:
            valid = view["object_multiview"]["grasped_object_valid"]
            if torch.is_tensor(valid) and bool(valid.any()):
                has_valid = True
                break
        if has_valid:
            return batch
    raise AssertionError("No DexYCB batch with valid object found")


class Pi3XDexYCBObjectIntegrationTests(unittest.TestCase):
    def test_dexycb_batch_runs_pi3x_object_dual_stream_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dexycb_pi3x_object_") as tmpdir:
            root = Path(tmpdir)
            build_fixture(root)

            dataset = DexYCBDataset(
                data_root=str(root),
                mode="train",
                resolution=[[28, 28]],
                frame_num=2,
                shuffle=False,
                seed=2024,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=unified_collate_fn,
            )
            batch = _find_batch_with_valid_object(loader)

            imgs = torch.stack([view["img"] for view in batch], dim=1)
            object_multiview = {
                "img": batch[0]["object_multiview"]["img"],
                "depthmap": batch[0]["object_multiview"]["depthmap"],
                "camera_intrinsics": batch[0]["object_multiview"]["camera_intrinsics"],
                "camera_pose": batch[0]["object_multiview"]["camera_pose"],
                "grasped_object_mask": torch.stack(
                    [view["object_multiview"]["grasped_object_mask"] for view in batch],
                    dim=1,
                ),
                "grasped_object_valid": torch.stack(
                    [view["object_multiview"]["grasped_object_valid"] for view in batch],
                    dim=1,
                ),
            }

            model = Pi3X(use_multimodal=False).eval()
            with torch.no_grad():
                out = model(imgs, object_multiview=object_multiview)

            self.assertIn("pred_object_rot6d", out)
            self.assertIn("pred_object_transl_dir", out)
            self.assertIn("pred_object_transl_log_scale", out)
            self.assertIn("pred_object_transl_scale", out)
            self.assertIn("pred_object_trans", out)
            self.assertIn("pred_object_log_scale", out)
            self.assertIn("pred_object_scale", out)
            self.assertIn("object_valid", out)
            self.assertEqual(tuple(out["pred_object_rot6d"].shape), (1, len(batch), 6))
            self.assertEqual(tuple(out["pred_object_transl_dir"].shape), (1, len(batch), 3))
            self.assertEqual(tuple(out["pred_object_transl_log_scale"].shape), (1, len(batch), 1))
            self.assertEqual(tuple(out["pred_object_transl_scale"].shape), (1, len(batch), 1))
            self.assertEqual(tuple(out["pred_object_trans"].shape), (1, len(batch), 3))
            self.assertEqual(tuple(out["pred_object_log_scale"].shape), (1, len(batch), 1))
            self.assertEqual(tuple(out["pred_object_scale"].shape), (1, len(batch), 1))
            self.assertTrue(torch.equal(out["object_valid"], object_multiview["grasped_object_valid"]))
            self.assertTrue(bool(out["object_valid"].any()))
            self.assertTrue(torch.isfinite(out["pred_object_rot6d"]).all())
            self.assertTrue(torch.allclose(out["pred_object_trans"], out["pred_object_transl_dir"] * out["pred_object_transl_scale"]))
            self.assertTrue(torch.allclose(out["pred_object_transl_scale"], torch.exp(out["pred_object_transl_log_scale"])))
            self.assertTrue(torch.isfinite(out["pred_object_trans"]).all())
            self.assertTrue(torch.isfinite(out["pred_object_scale"]).all())


if __name__ == "__main__":
    unittest.main()
