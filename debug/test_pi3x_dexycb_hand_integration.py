from __future__ import annotations

import sys
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest import mock

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
from pi3.models.hamer import HaMeREncoder, get_config as get_hamer_config
from pi3.models.pi3x import Pi3X


class _FakeHaMeRBackbone(torch.nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.output_dim = dim

    def forward(self, crops, hand_is_right=None):
        del hand_is_right
        pooled = crops.mean(dim=(-1, -2))
        if pooled.shape[1] < self.output_dim:
            repeat = (self.output_dim + pooled.shape[1] - 1) // pooled.shape[1]
            pooled = pooled.repeat(1, repeat)
        return pooled[:, : self.output_dim]


class _FakeMANO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.faces = torch.zeros((0, 3), dtype=torch.long)

    def forward(self, th_pose_coeffs, betas, th_trans=None, **kwargs):
        del th_pose_coeffs, betas, kwargs
        batch = th_trans.shape[0] if th_trans is not None else 1
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=torch.float32)
        joints = torch.zeros((batch, 21, 3), dtype=th_trans.dtype, device=th_trans.device) + th_trans.unsqueeze(1)
        vertices = torch.zeros((batch, 778, 3), dtype=th_trans.dtype, device=th_trans.device) + th_trans.unsqueeze(1)
        return type("FakeMANOOutput", (), {"joints": joints, "vertices": vertices})()

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del hand_pose, betas, kwargs
        batch = global_orient.shape[0]
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=global_orient.dtype, device=global_orient.device)
        return self.forward(torch.zeros((batch, 48), dtype=global_orient.dtype, device=global_orient.device), torch.zeros((batch, 10), dtype=global_orient.dtype, device=global_orient.device), th_trans=th_trans)


def _find_batch_with_valid_hand(loader):
    for batch in loader:
        has_valid = False
        for view in batch:
            valid = view["hand"]["valid"]
            if torch.is_tensor(valid) and bool(valid.any()):
                has_valid = True
                break
        if has_valid:
            return batch
    raise AssertionError("No DexYCB batch with valid hand found")


def _dexycb_batch_to_hand_inputs(batch):
    num_views = len(batch)
    batch_size = batch[0]["img"].shape[0]
    imgs = torch.stack([view["img"] for view in batch], dim=1)

    hand_masks = []
    owner_index = []
    hand_is_right = []
    gt_pose_mano = []
    gt_hand_transl = []
    gt_mano_betas = []
    gt_joints_3d_cam = []
    gt_joints_2d = []

    for view_idx, view in enumerate(batch):
        masks = view["hand"]["mask"]
        valid = view["hand"]["valid"]
        sides = view["hand"]["mano_side"]
        pose_mano = view["hand"]["pose_mano"]
        hand_transl = view["hand"]["hand_transl"]
        mano_betas = view["hand"]["mano_betas"]
        joints_3d_cam = view["hand"]["joints_3d_cam"]
        joints_2d = view["hand"]["joints_2d"]
        for batch_idx in range(batch_size):
            is_valid = bool(valid[batch_idx].item()) if torch.is_tensor(valid) else bool(valid[batch_idx])
            mask = masks[batch_idx]
            if not is_valid or not bool(mask.any()):
                continue
            hand_masks.append(mask.float())
            owner_index.append([batch_idx, view_idx, 0])
            side_value = sides[batch_idx]
            hand_is_right.append(side_value == "right")
            gt_pose_mano.append(pose_mano[batch_idx].float())
            gt_hand_transl.append(hand_transl[batch_idx].float())
            gt_mano_betas.append(mano_betas[batch_idx].float())
            gt_joints_3d_cam.append(joints_3d_cam[batch_idx].float())
            gt_joints_2d.append(joints_2d[batch_idx].float())

    if not hand_masks:
        raise AssertionError("DexYCB batch did not contain any valid hand after conversion")

    return {
        "imgs": imgs,
        "hand_masks": torch.stack(hand_masks, dim=0),
        "owner_index": torch.tensor(owner_index, dtype=torch.long),
        "hand_is_right": torch.tensor(hand_is_right, dtype=torch.bool),
        "gt_pose_mano": torch.stack(gt_pose_mano, dim=0),
        "gt_hand_transl": torch.stack(gt_hand_transl, dim=0),
        "gt_mano_betas": torch.stack(gt_mano_betas, dim=0),
        "gt_joints_3d_cam": torch.stack(gt_joints_3d_cam, dim=0),
        "gt_joints_2d": torch.stack(gt_joints_2d, dim=0),
        "num_views": num_views,
        "batch_size": batch_size,
    }


class Pi3XDexYCBHandIntegrationTests(unittest.TestCase):
    def test_dexycb_batch_runs_hamerencoder_and_pi3x_hand_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dexycb_pi3x_hand_") as tmpdir:
            root = Path(tmpdir)
            build_fixture(root)
            banana_dir = root / "models" / "011_banana"
            pitcher_dir = root / "models" / "019_pitcher_base"
            if banana_dir.is_dir() and not pitcher_dir.exists():
                shutil.copytree(banana_dir, pitcher_dir)

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
            batch = _find_batch_with_valid_hand(loader)
            hand_inputs = _dexycb_batch_to_hand_inputs(batch)

            hamer_cfg = get_hamer_config(
                str(REPO_ROOT / "third_party" / "hamer" / "_DATA" / "hamer_ckpts" / "model_config.yaml"),
                merge=True,
                cache_dir=str(REPO_ROOT / "third_party" / "hamer" / "_DATA"),
                update_cachedir=True,
            )
            encoder = HaMeREncoder(
                cfg=hamer_cfg,
                backbone=_FakeHaMeRBackbone(dim=1024),
                min_mask_area=1,
                rescale_factor=1.0,
            ).eval()

            with torch.no_grad():
                hamer_out = encoder(
                    hand_inputs["imgs"],
                    hand_inputs["hand_masks"],
                    hand_inputs["owner_index"],
                    hand_inputs["hand_is_right"],
                )

            self.assertGreater(hamer_out["hand_queries"].shape[0], 0)
            self.assertEqual(tuple(hamer_out["owner_index"].shape[1:]), (3,))
            self.assertTrue(torch.equal(hamer_out["owner_index"], hand_inputs["owner_index"]))
            self.assertEqual(tuple(hand_inputs["gt_pose_mano"].shape[1:]), (48,))
            self.assertEqual(tuple(hand_inputs["gt_hand_transl"].shape[1:]), (3,))
            self.assertEqual(tuple(hand_inputs["gt_mano_betas"].shape[1:]), (10,))
            self.assertEqual(tuple(hand_inputs["gt_joints_3d_cam"].shape[1:]), (21, 3))
            self.assertEqual(tuple(hand_inputs["gt_joints_2d"].shape[1:]), (21, 2))
            self.assertEqual(hand_inputs["gt_pose_mano"].shape[0], hamer_out["hand_queries"].shape[0])

            fake_mano = {"right": _FakeMANO(), "left": _FakeMANO()}
            with mock.patch.object(Pi3X, "_build_hand_mano_layer", return_value=fake_mano), mock.patch.object(Pi3X, "_build_hand_encoder", return_value=encoder):
                model = Pi3X(use_multimodal=False).eval()
            with torch.no_grad():
                out = model(
                    hand_inputs["imgs"],
                    hand_masks=hand_inputs["hand_masks"],
                    hand_owner_index=hand_inputs["owner_index"],
                    hand_is_right=hand_inputs["hand_is_right"],
                )

            self.assertIn("pred_hand_mano_params", out)
            self.assertIn("pred_hand_transl_dir", out)
            self.assertIn("pred_hand_transl_log_scale", out)
            self.assertIn("pred_hand_transl_scale", out)
            self.assertIn("pred_hand_transl", out)
            self.assertIn("pred_hand_log_scale", out)
            self.assertIn("pred_hand_scale", out)
            self.assertIn("pred_hand_mano_betas", out)
            self.assertIn("pred_hand_vertices", out)
            self.assertIn("pred_hand_joints_3d", out)
            self.assertEqual(tuple(out["pred_hand_transl_dir"].shape), hand_inputs["gt_hand_transl"].shape)
            self.assertEqual(tuple(out["pred_hand_transl_log_scale"].shape), (hamer_out["hand_queries"].shape[0], 1))
            self.assertEqual(tuple(out["pred_hand_transl_scale"].shape), (hamer_out["hand_queries"].shape[0], 1))
            self.assertEqual(tuple(out["pred_hand_transl"].shape), hand_inputs["gt_hand_transl"].shape)
            self.assertEqual(tuple(out["pred_hand_log_scale"].shape), (hamer_out["hand_queries"].shape[0], 1))
            self.assertEqual(tuple(out["pred_hand_scale"].shape), (hamer_out["hand_queries"].shape[0], 1))
            self.assertTrue(torch.allclose(out["pred_hand_transl"], out["pred_hand_transl_dir"] * out["pred_hand_transl_scale"]))
            self.assertTrue(torch.allclose(out["pred_hand_transl_scale"], torch.exp(out["pred_hand_transl_log_scale"])))
            self.assertTrue(torch.allclose(out["pred_hand_scale"], torch.exp(out["pred_hand_log_scale"])))
            self.assertEqual(out["pred_hand_mano_betas"].shape, hand_inputs["gt_mano_betas"].shape)
            self.assertEqual(tuple(out["pred_hand_vertices"].shape), (hamer_out["hand_queries"].shape[0], 778, 3))
            self.assertEqual(tuple(out["pred_hand_joints_3d"].shape), (hamer_out["hand_queries"].shape[0], 21, 3))
            self.assertEqual(tuple(out["pred_hand_mano_params"]["global_orient"].shape[1:]), (1, 3, 3))
            self.assertEqual(out["pred_hand_mano_params"]["hand_pose"].shape[0], hamer_out["hand_queries"].shape[0])
            self.assertEqual(tuple(out["pred_hand_mano_params"]["hand_pose"].shape[1:]), (15, 3, 3))
            self.assertEqual(out["pred_hand_mano_params"]["betas"].shape, hand_inputs["gt_mano_betas"].shape)
            self.assertEqual(tuple(out["points"].shape[:2]), (1, hand_inputs["num_views"]))
            self.assertTrue(torch.equal(out["hand_owner_index"], hamer_out["owner_index"]))
            self.assertTrue(torch.equal(out["hand_is_right"], hamer_out["hand_is_right"]))


if __name__ == "__main__":
    unittest.main()
