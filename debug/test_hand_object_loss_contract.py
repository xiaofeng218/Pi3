from __future__ import annotations

import unittest

import torch

from pi3.models.hand_object_loss import HandObjectLoss
from pi3.models.hamer.geometry import aa_to_rotmat


class _ConstantHandDiscriminator(torch.nn.Module):
    def __init__(self, output_value: float = 0.0) -> None:
        super().__init__()
        self.output_value = float(output_value)
        self.last_poses = None
        self.last_betas = None

    def forward(self, poses: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        self.last_poses = poses.detach().clone()
        self.last_betas = betas.detach().clone()
        batch = poses.shape[0]
        return poses.new_full((batch, 17), self.output_value)


class HandObjectLossContractTests(unittest.TestCase):
    def test_hand_geometry_losses_are_invariant_to_global_translation_rotation_and_scale(self) -> None:
        loss = HandObjectLoss(
            hand_global_orient_weight=0.0,
            hand_pose_weight=0.0,
            hand_beta_weight=0.0,
            hand_adversarial_weight=0.0,
        )
        gt_global_orient = aa_to_rotmat(torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3)
        pred_global_orient = aa_to_rotmat(torch.tensor([[0.0, 0.0, 1.5707964]], dtype=torch.float32)).view(1, 1, 3, 3)

        gt_local_joints = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
            dtype=torch.float32,
        )
        pred_local_joints = gt_local_joints * 3.0
        pred_local_joints = torch.matmul(pred_local_joints, pred_global_orient[:, 0].transpose(-1, -2))
        pred_local_joints = pred_local_joints + torch.tensor([[[5.0, -2.0, 1.0]]], dtype=torch.float32)

        gt_local_vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 2.0, 0.5], [1.0, 1.0, 1.0]]],
            dtype=torch.float32,
        )
        pred_local_vertices = gt_local_vertices * 3.0
        pred_local_vertices = torch.matmul(pred_local_vertices, pred_global_orient[:, 0].transpose(-1, -2))
        pred_local_vertices = pred_local_vertices + torch.tensor([[[5.0, -2.0, 1.0]]], dtype=torch.float32)

        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_scale": torch.tensor([[1.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_hand_joints_3d": pred_local_joints,
            "pred_hand_vertices": pred_local_vertices,
            "pred_hand_mano_params": {
                "global_orient": pred_global_orient,
                "hand_pose": torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": torch.zeros((1, 10), dtype=torch.float32),
            },
            "pred_hand_mano_betas": torch.zeros((1, 10), dtype=torch.float32),
        }
        gt = {
            "hand_valid": torch.tensor([True]),
            "hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "scene_scale": torch.tensor([1.0]),
            "hand_global_orient_rotmat": gt_global_orient,
            "hand_pose_rotmat": torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros((1, 10), dtype=torch.float32),
            "hand_joints_3d": gt_local_joints,
            "hand_vertices": gt_local_vertices,
        }

        _, details = loss(pred, gt)

        self.assertAlmostEqual(float(details["hand_joints_3d_loss"]), 0.0, places=5)
        self.assertNotIn("hand_vertices_loss", details)

    def test_hand_scale_gt_uses_hand_owner_index(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]] * 4),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]),
            "pred_hand_transl_scale": torch.ones(4, 1),
            "pred_hand_scale": torch.ones(4, 1),
        }
        gt = {
            "hand_valid": torch.tensor([True, True, True, True]),
            "hand_transl": torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 3.0],
                    [0.0, 0.0, 4.0],
                ]
            ),
            "scene_scale": torch.tensor([2.0, 5.0]),
            "hand_owner_index": torch.tensor(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                ],
                dtype=torch.long,
            ),
        }

        total, details = loss(pred, gt)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("hand_transl_loss", details)
        self.assertIn("hand_full_scale_loss", details)

    def test_log_scale_losses_use_log_predictions_directly(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_log_scale": torch.tensor([[50.0]]),
            "pred_hand_transl_scale": torch.tensor([[float("inf")]]),
            "pred_hand_log_scale": torch.tensor([[0.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_object_rot6d": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]),
            "pred_object_transl_dir": torch.tensor([[[0.0, 0.0, 1.0]]]),
            "pred_object_transl_log_scale": torch.tensor([[[0.0]]]),
            "pred_object_transl_scale": torch.tensor([[[1.0]]]),
            "pred_object_trans": torch.tensor([[[0.0, 0.0, 1.0]]]),
            "pred_object_log_scale": torch.tensor([[[50.0]]]),
            "pred_object_scale": torch.tensor([[[float("inf")]]]),
        }
        gt = {
            "hand_valid": torch.tensor([True]),
            "hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "scene_scale": torch.tensor([1.0]),
            "object_valid": torch.tensor([[True]]),
            "object_pose_obj2cam": torch.eye(4).view(1, 1, 4, 4),
            "object_normalization_scale": torch.tensor([[1.0]]),
        }

        total, details = loss(pred, gt)

        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(float(details["object_scale_loss"]), 50.0, places=5)

    def test_object_rotation_uses_matrix_mse_not_geodesic(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_object_rot6d": torch.tensor([[[1.0, 0.0, 0.0, 0.0, -1.0, 0.0]]]),
            "pred_object_transl_dir": torch.tensor([[[0.0, 0.0, 1.0]]]),
            "pred_object_transl_scale": torch.tensor([[[1.0]]]),
            "pred_object_trans": torch.tensor([[[0.0, 0.0, 1.0]]]),
            "pred_object_scale": torch.tensor([[[1.0]]]),
        }
        object_pose = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4)
        object_pose[0, 0, 2, 3] = 1.0
        gt = {
            "object_valid": torch.tensor([[True]]),
            "scene_scale": torch.tensor([1.0]),
            "object_normalization_scale": torch.tensor([[1.0]]),
            "object_pose_obj2cam": object_pose,
        }

        total, details = loss(pred, gt)

        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(float(details["object_rot_loss"]), 8.0 / 9.0, places=5)
        self.assertAlmostEqual(float(details["object_transl_loss"]), 0.0, places=5)

    def test_hand_parameter_losses_ignore_beta_and_vertex_supervision(self) -> None:
        loss = HandObjectLoss()
        pred_global_orient = aa_to_rotmat(torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3)
        pred_hand_mano_betas = torch.zeros((1, 10), dtype=torch.float32)
        gt_hand_pose_mano = torch.tensor([[1.0, 0.0, 0.0] + [2.0] * 45], dtype=torch.float32)
        gt_hand_mano_betas = torch.ones((1, 10), dtype=torch.float32)
        gt_global_orient = aa_to_rotmat(gt_hand_pose_mano[:, :3]).view(1, 1, 3, 3)
        gt_pose_rotmat = aa_to_rotmat(gt_hand_pose_mano[:, 3:].reshape(-1, 3)).view(1, 15, 3, 3)
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_scale": torch.tensor([[1.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_hand_joints_3d": torch.zeros((1, 21, 3), dtype=torch.float32),
            "pred_hand_vertices": torch.zeros((1, 778, 3), dtype=torch.float32),
            "pred_hand_mano_params": {
                "global_orient": pred_global_orient,
                "hand_pose": torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": pred_hand_mano_betas,
            },
            "pred_hand_mano_betas": pred_hand_mano_betas,
        }
        gt = {
            "hand_valid": torch.tensor([True]),
            "hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "scene_scale": torch.tensor([1.0]),
            "hand_global_orient_rotmat": gt_global_orient,
            "hand_pose_rotmat": gt_pose_rotmat,
            "hand_mano_betas": gt_hand_mano_betas,
            "hand_joints_3d": torch.zeros((1, 21, 3), dtype=torch.float32),
            "hand_vertices": torch.cat(
                [
                    torch.zeros((1, 1, 3), dtype=torch.float32),
                    torch.ones((1, 777, 3), dtype=torch.float32),
                ],
                dim=1,
            ),
        }

        total, details = loss(pred, gt)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("hand_global_orient_loss", details)
        self.assertIn("hand_pose_loss", details)
        self.assertIn("hand_transl_loss", details)
        self.assertIn("hand_full_scale_loss", details)
        expected_global_mse = torch.nn.functional.mse_loss(pred_global_orient, gt_global_orient).item()
        self.assertAlmostEqual(float(details["hand_global_orient_loss"]), expected_global_mse, places=5)
        self.assertAlmostEqual(float(details["hand_pose_loss"]), 0.8660, places=4)
        self.assertAlmostEqual(float(details["hand_transl_loss"]), 0.0, places=5)
        self.assertAlmostEqual(float(details["hand_full_scale_loss"]), 0.0, places=5)
        self.assertNotIn("hand_beta_loss", details)
        self.assertNotIn("hand_vertices_loss", details)

    def test_hand_2d_loss_uses_processed_image_intrinsics(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_scale": torch.tensor([[1.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_hand_joints_3d": torch.tensor(
                [[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]] + [[0.0, 0.0, 2.0]] * 19],
                dtype=torch.float32,
            ),
        }
        gt = {
            "hand_valid": torch.tensor([True]),
            "hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "scene_scale": torch.tensor([1.0]),
            "hand_joints_3d": torch.zeros((1, 21, 3), dtype=torch.float32),
            "hand_joints_2d": torch.tensor(
                [[[0.0, 0.0], [9.0, 0.0]] + [[0.0, 0.0]] * 19],
                dtype=torch.float32,
            ),
            "hand_camera_intrinsics": torch.tensor(
                [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]],
                dtype=torch.float32,
            ),
        }

        total, details = loss(pred, gt)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("hand_2d_loss", details)
        self.assertAlmostEqual(float(details["hand_2d_loss"]), 0.0, places=6)

    def test_dummy_invalid_hand_row_keeps_loss_keys_but_zeroes_values(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_scale": torch.tensor([[1.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_hand_mano_params": {
                "global_orient": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3),
                "hand_pose": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": torch.zeros((1, 10), dtype=torch.float32),
            },
            "pred_hand_mano_betas": torch.zeros((1, 10), dtype=torch.float32),
            "pred_hand_joints_3d": torch.zeros((1, 21, 3), dtype=torch.float32),
            "pred_hand_vertices": torch.zeros((1, 778, 3), dtype=torch.float32),
        }
        gt = {
            "hand_valid": torch.tensor([False]),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_transl": torch.zeros((1, 3), dtype=torch.float32),
            "scene_scale": torch.tensor([1.0]),
            "hand_global_orient_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros((1, 10), dtype=torch.float32),
            "hand_joints_3d": torch.zeros((1, 21, 3), dtype=torch.float32),
            "hand_vertices": torch.zeros((1, 778, 3), dtype=torch.float32),
            "hand_joints_2d": torch.zeros((1, 21, 2), dtype=torch.float32),
            "hand_camera_intrinsics": torch.eye(3, dtype=torch.float32).unsqueeze(0),
        }

        total, details = loss(pred, gt)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("hand_transl_loss", details)
        self.assertIn("hand_full_scale_loss", details)
        self.assertIn("hand_pose_loss", details)
        self.assertIn("hand_joints_3d_loss", details)
        self.assertNotIn("hand_beta_loss", details)
        self.assertNotIn("hand_vertices_loss", details)
        self.assertEqual(float(details["hand_transl_loss"]), 0.0)
        self.assertEqual(float(details["hand_full_scale_loss"]), 0.0)
        self.assertEqual(float(details["hand_pose_loss"]), 0.0)
        self.assertEqual(float(details["hand_joints_3d_loss"]), 0.0)

    def test_frozen_hand_discriminator_adds_adversarial_prior_loss(self) -> None:
        discriminator = _ConstantHandDiscriminator(output_value=0.0)
        loss = HandObjectLoss(
            hand_adversarial_weight=0.5,
            hand_discriminator=discriminator,
        )
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_scale": torch.tensor([[1.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_hand_mano_params": {
                "global_orient": aa_to_rotmat(torch.zeros((1, 3), dtype=torch.float32)).view(1, 1, 3, 3),
                "hand_pose": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": torch.zeros((1, 10), dtype=torch.float32),
            },
        }
        gt = {
            "hand_valid": torch.tensor([True]),
            "hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "scene_scale": torch.tensor([1.0]),
        }

        total, details = loss(pred, gt)

        self.assertTrue(torch.isfinite(total))
        self.assertIn("hand_adversarial_prior_loss", details)
        self.assertAlmostEqual(float(details["hand_adversarial_prior_loss"]), 17.0, places=5)
        self.assertIsNotNone(discriminator.last_poses)
        self.assertIsNotNone(discriminator.last_betas)
        self.assertEqual(discriminator.last_poses.shape, (1, 15, 3, 3))
        self.assertEqual(discriminator.last_betas.shape, (1, 10))
        self.assertAlmostEqual(float(total), 0.5 * 17.0, places=5)

    def test_hand_pose_loss_uses_predicted_rotmats_directly(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_scale": torch.tensor([[1.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_hand_mano_params": {
                "global_orient": aa_to_rotmat(torch.zeros((1, 3), dtype=torch.float32)).view(1, 1, 3, 3),
                "hand_pose": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": torch.zeros((1, 10), dtype=torch.float32),
            },
        }
        gt = {
            "hand_valid": torch.tensor([True]),
            "hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "scene_scale": torch.tensor([1.0]),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
        }

        _, details = loss(pred, gt)

        self.assertIn("hand_pose_loss", details)
        self.assertAlmostEqual(float(details["hand_pose_loss"]), 0.0, places=6)

    def test_hand_pose_loss_prefers_predecoded_gt_rotmats(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_scale": torch.tensor([[1.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_hand_mano_params": {
                "global_orient": aa_to_rotmat(torch.zeros((1, 3), dtype=torch.float32)).view(1, 1, 3, 3),
                "hand_pose": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": torch.zeros((1, 10), dtype=torch.float32),
            },
        }
        gt = {
            "hand_valid": torch.tensor([True]),
            "hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "scene_scale": torch.tensor([1.0]),
            "hand_global_orient_rotmat": aa_to_rotmat(torch.zeros((1, 3), dtype=torch.float32)).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
        }

        _, details = loss(pred, gt)

        self.assertIn("hand_pose_loss", details)
        self.assertAlmostEqual(float(details["hand_pose_loss"]), 0.0, places=6)

    def test_pose_weight_controls_total_without_beta_loss_details(self) -> None:
        pred_global_orient = aa_to_rotmat(torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3)
        pred_hand_mano_betas = torch.zeros((1, 10), dtype=torch.float32)
        gt_hand_pose_mano = torch.tensor([[0.0, 0.0, 0.0] + [2.0] * 45], dtype=torch.float32)
        gt_hand_mano_betas = torch.ones((1, 10), dtype=torch.float32)
        gt_global_orient = aa_to_rotmat(gt_hand_pose_mano[:, :3]).view(1, 1, 3, 3)
        gt_pose_rotmat = aa_to_rotmat(gt_hand_pose_mano[:, 3:].reshape(-1, 3)).view(1, 15, 3, 3)
        pred = {
            "pred_hand_transl_dir": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "pred_hand_transl_scale": torch.tensor([[1.0]]),
            "pred_hand_scale": torch.tensor([[1.0]]),
            "pred_hand_mano_params": {
                "global_orient": pred_global_orient,
                "hand_pose": torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": pred_hand_mano_betas,
            },
            "pred_hand_mano_betas": pred_hand_mano_betas,
        }
        gt = {
            "hand_valid": torch.tensor([True]),
            "hand_transl": torch.tensor([[0.0, 0.0, 1.0]]),
            "scene_scale": torch.tensor([1.0]),
            "hand_global_orient_rotmat": gt_global_orient,
            "hand_pose_rotmat": gt_pose_rotmat,
            "hand_mano_betas": gt_hand_mano_betas,
        }

        loss_off = HandObjectLoss(hand_pose_weight=0.0, hand_beta_weight=0.0)
        total_off, details_off = loss_off(pred, gt)
        loss_on = HandObjectLoss(hand_pose_weight=1.0, hand_beta_weight=1.0)
        total_on, details_on = loss_on(pred, gt)

        self.assertGreater(float(details_off["hand_pose_loss"]), 0.0)
        self.assertAlmostEqual(float(total_off), 0.0, places=6)
        self.assertAlmostEqual(
            float(total_on),
            float(details_on["hand_pose_loss"]),
            places=6,
        )
        self.assertNotIn("hand_beta_loss", details_off)
        self.assertNotIn("hand_beta_loss", details_on)

    def test_object_weights_control_total_without_hiding_details(self) -> None:
        pred = {
            "pred_object_rot6d": torch.tensor([[[1.0, 0.0, 0.0, 0.0, -1.0, 0.0]]]),
            "pred_object_transl_dir": torch.tensor([[[0.0, 0.0, 1.0]]]),
            "pred_object_transl_scale": torch.tensor([[[1.0]]]),
            "pred_object_trans": torch.tensor([[[0.0, 0.0, 1.0]]]),
            "pred_object_scale": torch.tensor([[[1.0]]]),
        }
        object_pose = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4)
        object_pose[0, 0, 2, 3] = 1.0
        gt = {
            "object_valid": torch.tensor([[True]]),
            "scene_scale": torch.tensor([1.0]),
            "object_normalization_scale": torch.tensor([[2.0]]),
            "object_pose_obj2cam": object_pose,
        }

        loss_off = HandObjectLoss(
            object_rot_weight=0.0,
            object_transl_weight=0.0,
            object_scale_weight=0.0,
        )
        total_off, details_off = loss_off(pred, gt)
        loss_on = HandObjectLoss(
            object_rot_weight=1.0,
            object_transl_weight=1.0,
            object_scale_weight=0.1,
        )
        total_on, details_on = loss_on(pred, gt)

        self.assertAlmostEqual(float(details_off["object_rot_loss"]), 8.0 / 9.0, places=5)
        self.assertAlmostEqual(float(details_on["object_rot_loss"]), 8.0 / 9.0, places=5)
        self.assertAlmostEqual(float(total_off), 0.0, places=6)
        self.assertGreater(float(total_on), float(total_off))

    def test_object_losses_ignore_2d_supervision(self) -> None:
        loss = HandObjectLoss()
        pred = {
            "pred_object_rot6d": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
            "pred_object_transl_dir": torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32),
            "pred_object_transl_scale": torch.tensor([[[2.0]]], dtype=torch.float32),
            "pred_object_trans": torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32),
            "pred_object_scale": torch.tensor([[[2.0]]], dtype=torch.float32),
        }
        gt = {
            "object_valid": torch.tensor([[True]]),
            "scene_scale": torch.tensor([1.0]),
            "object_pose_obj2cam": torch.eye(4).view(1, 1, 4, 4),
            "object_normalization_scale": torch.tensor([[2.0]], dtype=torch.float32),
            "object_normalization_center": torch.zeros((1, 3), dtype=torch.float32),
            "object_template_vertices": torch.tensor([[[0.0, 0.0, 0.5], [0.2, 0.0, 0.5]]], dtype=torch.float32),
            "object_vertices_2d": torch.tensor([[[[0.0, 0.0], [0.8, 0.0]]]], dtype=torch.float32),
            "object_vertices_2d_valid": torch.tensor([[[True, True]]]),
            "object_camera_intrinsics": torch.tensor(
                [[[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]]],
                dtype=torch.float32,
            ),
        }

        total, details = loss(pred, gt)

        self.assertTrue(torch.isfinite(total))
        self.assertNotIn("object_2d_loss", details)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
