from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from debug.check_pi3x_pred_mesh_selfcheck import build_reference_payloads, run_selfcheck


class _FakeMANO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.th_faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        self.side = "right"

    def forward(self, th_pose_coeffs, betas, th_trans=None, **kwargs):
        del th_pose_coeffs, betas, kwargs
        batch = th_trans.shape[0] if th_trans is not None else 1
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=torch.float32)
        joints = torch.tensor(
            [[[0.0, 0.0, 0.0]] + [[0.02, 0.00, 0.00]] * 20],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]] + [[0.02, 0.02, 0.0]] * 775],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        return type("FakeMANOOutput", (), {"joints": joints, "vertices": vertices})()

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del global_orient, hand_pose, betas, kwargs
        return self.forward(torch.zeros((th_trans.shape[0] if th_trans is not None else 1, 48)), torch.zeros((th_trans.shape[0] if th_trans is not None else 1, 10)), th_trans=th_trans)


class MeshSelfcheckTests(unittest.TestCase):
    def test_reference_pred_meshes_match_aligned_gt_and_loss_is_zeroish(self) -> None:
        sample_batch = [
            {
                "img": torch.zeros(1, 3, 4, 4),
                "depthmap": torch.ones(1, 4, 4),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "dataset": ["dexycb"],
                "label": ["sample"],
                "instance": ["instance0"],
                "pts3d": torch.zeros(1, 4, 4, 3),
                "valid_mask": torch.ones(1, 4, 4, dtype=torch.bool),
                "sparse_depth": torch.ones(1, 4, 4),
                "hand": {
                    "mask": torch.ones(1, 4, 4),
                    "valid": torch.tensor([True]),
                    "pose_mano": torch.cat([torch.tensor([[0.1, 0.0, 0.0]]), torch.zeros(1, 45)], dim=1),
                    "hand_transl": torch.tensor([[0.02, 0.00, 0.00]]),
                    "joints_3d_cam": torch.tensor([[[0.0, 0.0, 0.0]] + [[0.02, 0.00, 0.00]] * 20]),
                    "mano_betas": torch.zeros(1, 10),
                    "mano_side": ["right"],
                },
                "object_multiview": {
                    "grasped_object_id": torch.tensor([1]),
                    "grasped_object_mask": torch.ones(1, 4, 4),
                    "grasped_object_valid": torch.tensor([True]),
                    "grasped_object_pose_obj2cam": torch.tensor(
                        [[[1.0, 0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.tensor([1.0, 1.0, 0.0]),
                    "normalization_scale": torch.tensor(2.0),
                },
            }
        ]
        scene_gt = {
            "local_points": torch.zeros(1, 1, 4, 4, 3),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([2.0]),
        }
        scene_gt["local_points"][0, 0, ..., 2] = 4.0

        fake_template = (
            torch.tensor([[1.0, 1.0, 0.0], [3.0, 1.0, 0.0], [1.0, 3.0, 0.0]], dtype=torch.float32),
            np.array([[0, 1, 2]], dtype=np.int32),
        )

        with patch("debug.check_pi3x_pred_mesh_selfcheck._load_object_mesh_template", return_value=fake_template):
            pred_vis, pred_loss, gt_loss, display_scale = build_reference_payloads(
                sample_batch=sample_batch,
                scene_gt=scene_gt,
                synthetic_depth_scale=2.0,
                data_root="/tmp/dummy",
                mano_layer=_FakeMANO(),
            )

            summary = run_selfcheck(
                sample_batch=sample_batch,
                scene_gt=scene_gt,
                pred_vis=pred_vis,
                pred_loss=pred_loss,
                gt_loss=gt_loss,
                data_root="/tmp/dummy",
                sample_index=0,
                output_rrd=None,
                mano_layer=_FakeMANO(),
            )

        self.assertAlmostEqual(display_scale.item(), 4.0, places=6)
        self.assertLess(summary["hand_max_abs_err"], 1e-6)
        self.assertLess(summary["object_max_abs_err"], 1e-6)
        self.assertLess(summary["loss"], 1e-2)


if __name__ == "__main__":
    unittest.main()
