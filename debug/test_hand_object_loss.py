from __future__ import annotations

import unittest

import torch

from pi3.models.hand_object_loss import (
    HandObjectLoss,
    estimate_scene_scale_from_depth,
)


class _FakeMANO(torch.nn.Module):
    def forward(self, th_pose_coeffs, betas, th_trans=None, **kwargs):
        del th_pose_coeffs, betas, kwargs
        batch = th_trans.shape[0] if th_trans is not None else 1
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=torch.float32)
        joints = torch.tensor(
            [[[0.0, 0.0, 0.0]] + [[1.0, 2.0, 3.0]] * 20],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0]] + [[1.0, 2.0, 3.0]] * 777],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        return type("FakeMANOOutput", (), {"joints": joints, "vertices": vertices})()

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del global_orient, hand_pose, betas, kwargs
        batch = th_trans.shape[0] if th_trans is not None else 1
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=torch.float32)
        joints = torch.tensor(
            [[[0.0, 0.0, 0.0]] + [[1.0, 2.0, 3.0]] * 20],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0]] + [[1.0, 2.0, 3.0]] * 777],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        return type("FakeMANOOutput", (), {"joints": joints, "vertices": vertices})()


class _SideMANO(torch.nn.Module):
    def __init__(self, sign: float):
        super().__init__()
        self.sign = float(sign)

    def forward(self, th_pose_coeffs, betas, th_trans=None, **kwargs):
        del th_pose_coeffs, betas, kwargs
        batch = th_trans.shape[0] if th_trans is not None else 1
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=torch.float32)
        joints = torch.tensor(
            [[[self.sign, 0.0, 0.0]] + [[2.0 * self.sign, 0.0, 0.0]] * 20],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        vertices = torch.tensor(
            [[[self.sign, 0.0, 0.0]] + [[2.0 * self.sign, 0.0, 0.0]] * 777],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        return type("FakeMANOOutput", (), {"joints": joints, "vertices": vertices})()

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del global_orient, hand_pose, betas, kwargs
        batch = th_trans.shape[0] if th_trans is not None else 1
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=torch.float32)
        joints = torch.tensor(
            [[[self.sign, 0.0, 0.0]] + [[2.0 * self.sign, 0.0, 0.0]] * 20],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        vertices = torch.tensor(
            [[[self.sign, 0.0, 0.0]] + [[2.0 * self.sign, 0.0, 0.0]] * 777],
            dtype=torch.float32,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        return type("FakeMANOOutput", (), {"joints": joints, "vertices": vertices})()


class HandObjectLossTests(unittest.TestCase):
    def test_loss_details_always_include_fixed_metric_keys(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "points": torch.zeros((1, 1, 1, 1, 3)),
        }
        targets = {
            "scene_scale": torch.tensor([1.0]),
        }

        total, details = loss(pred, targets)

        self.assertTrue(torch.isfinite(total))
        expected = {
            "hand_transl_loss",
            "hand_full_scale_loss",
            "hand_global_orient_loss",
            "hand_pose_loss",
            "hand_adversarial_prior_loss",
            "hand_joints_3d_loss",
            "hand_2d_loss",
            "object_rot_loss",
            "object_transl_loss",
            "object_scale_loss",
            "_weighted_loss_details",
        }
        self.assertTrue(expected.issubset(details.keys()))
        for key in expected - {"_weighted_loss_details"}:
            self.assertEqual(float(details[key]), 0.0)
        self.assertEqual(
            set(details["_weighted_loss_details"].keys()),
            expected - {"_weighted_loss_details"},
        )

    def test_estimate_scene_scale_from_depth_returns_pred_over_gt_ratio(self) -> None:
        gt_depth = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
        pred_depth = gt_depth * 3.0
        valid_mask = torch.ones_like(gt_depth, dtype=torch.bool)

        scale = estimate_scene_scale_from_depth(pred_depth, gt_depth, valid_mask)

        self.assertTrue(torch.allclose(scale, torch.tensor([3.0])))

    def test_hand_geometry_loss_is_root_relative(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 2.0]]),
            "pred_hand_transl_scale": torch.tensor([[2.0]]),
            "pred_hand_scale": torch.tensor([[4.0]]),
            "pred_hand_mano_params": {
                "global_orient": torch.eye(3).view(1, 1, 3, 3),
                "hand_pose": torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": torch.zeros(1, 10),
            },
            "pred_hand_joints_3d": torch.tensor([[[5.0, 5.0, 5.0]] + [[6.0, 7.0, 8.0]] * 20]),
            "pred_hand_vertices": torch.tensor([[[5.0, 5.0, 5.0]] + [[6.0, 7.0, 8.0]] * 777]),
        }
        targets = {
            "hand_valid": torch.tensor([True]),
            "scene_scale": torch.tensor([3.0]),
            "hand_global_orient_rotmat": torch.eye(3).view(1, 1, 3, 3),
            "hand_transl": torch.tensor([[0.0, 0.0, 2.0]]),
            "hand_pose_rotmat": torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_joints_3d": torch.tensor([[[4.0, 4.0, 4.0]] + [[5.0, 6.0, 7.0]] * 20]),
            "hand_mano_betas": torch.zeros(1, 10),
        }

        total, details = loss(pred, targets)

        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(float(details["hand_joints_3d_loss"]), 0.0, places=6)

    def test_hand_geometry_loss_uses_dense_gt_without_legacy_fallback(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 2.0]]),
            "pred_hand_transl_scale": torch.tensor([[2.0]]),
            "pred_hand_scale": torch.tensor([[4.0]]),
            "pred_hand_mano_params": {
                "global_orient": torch.eye(3).view(1, 1, 3, 3),
                "hand_pose": torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": torch.zeros(1, 10),
            },
            "pred_hand_joints_3d": torch.tensor([[[0.0, 0.0, 2.0]] + [[1.0, 2.0, 5.0]] * 20]),
            "pred_hand_vertices": torch.tensor([[[0.0, 0.0, 2.0]] + [[1.0, 2.0, 5.0]] * 777]),
        }
        targets = {
            "hand_valid": torch.tensor([True]),
            "scene_scale": torch.tensor([3.0]),
            "hand_global_orient_rotmat": torch.eye(3).view(1, 1, 3, 3),
            "hand_transl": torch.tensor([[0.0, 0.0, 2.0]]),
            "hand_pose_rotmat": torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_joints_3d": torch.tensor([[[0.0, 0.0, 2.0]] + [[1.0, 2.0, 5.0]] * 20]),
            "hand_mano_betas": torch.zeros(1, 10),
        }

        total, details = loss(pred, targets)

        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(float(details["hand_joints_3d_loss"]), 0.0, places=6)

    def test_hand_geometry_loss_respects_dense_hand_valid_mask(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
            "pred_hand_transl": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "pred_hand_transl_scale": torch.tensor([[[2.0], [2.0]]]),
            "pred_hand_scale": torch.tensor([[[4.0], [4.0]]]),
            "pred_hand_mano_params": {
                "global_orient": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 1, 1, 1),
                "hand_pose": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 15, 1, 1),
                "betas": torch.zeros(1, 2, 10),
            },
            "pred_hand_joints_3d": torch.tensor([[[[-1.0, 0.0, 0.0]] + [[-2.0, 0.0, 0.0]] * 20, [[99.0, 0.0, 0.0]] + [[99.0, 0.0, 0.0]] * 20]]),
            "pred_hand_vertices": torch.tensor([[[[-1.0, 0.0, 0.0]] + [[-2.0, 0.0, 0.0]] * 777, [[99.0, 0.0, 0.0]] + [[99.0, 0.0, 0.0]] * 777]]),
        }
        targets = {
            "hand_valid": torch.tensor([[True, False]]),
            "scene_scale": torch.tensor([3.0]),
            "hand_is_right": torch.tensor([[False, True]]),
            "hand_global_orient_rotmat": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 1, 1, 1),
            "hand_transl": torch.tensor([[[0.0, 0.0, 0.0], [123.0, 0.0, 0.0]]]),
            "hand_pose_rotmat": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 15, 1, 1),
            "hand_joints_3d": torch.tensor([[[[-1.0, 0.0, 0.0]] + [[-2.0, 0.0, 0.0]] * 20, [[123.0, 0.0, 0.0]] + [[123.0, 0.0, 0.0]] * 20]]),
            "hand_mano_betas": torch.zeros(1, 2, 10),
        }

        total, details = loss(pred, targets)

        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(float(details["hand_joints_3d_loss"]), 0.0, places=6)

    def test_object_losses_use_normalization_scale_and_scene_scale(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_object_rot6d": torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
            "pred_object_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_object_transl_scale": torch.tensor([[4.0]]),
            "pred_object_trans": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_object_scale": torch.tensor([[2.0]]),
        }
        targets = {
            "object_valid": torch.tensor([True]),
            "scene_scale": torch.tensor([4.0]),
            "object_pose_obj2cam": torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0, 0.0],
                                                  [0.0, 0.0, 1.0, 1.0],
                                                  [0.0, 0.0, 0.0, 1.0]]]),
            "object_normalization_scale": torch.tensor([2.0]),
        }

        total, details = loss(pred, targets)

        self.assertTrue(torch.isfinite(total))
        self.assertLess(float(details["object_rot_loss"]), 1e-2)
        self.assertAlmostEqual(float(details["object_scale_loss"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
