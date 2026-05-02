from __future__ import annotations

import unittest

import torch

from debug.inspect_pi3x_pred_vs_gt import (
    _apply_alignment_factor_to_view,
    _apply_alignment_factor_to_points,
    _compute_gt_depth_alignment_factor,
    _transform_vertices,
    run_pi3x_prediction,
    summarize_hand_mesh_debug,
)


class InspectPi3XPredVsGtTests(unittest.TestCase):
    def test_run_pi3x_prediction_passes_gt_depth_pose_and_intrinsics(self) -> None:
        class RecordingModel:
            def __init__(self) -> None:
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return {"ok": True}

        model = RecordingModel()
        model_inputs = {
            "imgs": torch.randn(2, 3, 3, 224, 224),
            "depths": torch.randn(2, 3, 224, 224),
            "intrinsics": torch.randn(2, 3, 3, 3),
            "poses": torch.randn(2, 3, 4, 4),
        }

        pred_raw = run_pi3x_prediction(model, model_inputs)

        self.assertEqual(pred_raw, {"ok": True})
        self.assertEqual(len(model.calls), 1)
        self.assertIs(model.calls[0]["imgs"], model_inputs["imgs"])
        self.assertIs(model.calls[0]["depths"], model_inputs["depths"])
        self.assertIs(model.calls[0]["intrinsics"], model_inputs["intrinsics"])
        self.assertIs(model.calls[0]["poses"], model_inputs["poses"])

    def test_alignment_factor_prefers_hand_and_object_focus_mask(self) -> None:
        gt_depth = torch.tensor([[[[1.0, 2.0, 100.0, 200.0]]]], dtype=torch.float32)
        pred_depth = torch.tensor([[[[10.0, 20.0, 100.0, 100.0]]]], dtype=torch.float32)
        valid_mask = torch.ones_like(gt_depth, dtype=torch.bool)
        focus_mask = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.bool)

        factor = _compute_gt_depth_alignment_factor(
            gt_depth,
            pred_depth,
            valid_mask,
            focus_mask=focus_mask,
        )

        self.assertAlmostEqual(float(factor.item()), 0.1, places=6)

    def test_alignment_factor_uses_median_ratio_over_front_half_gt_pixels(self) -> None:
        gt_depth = torch.tensor([[[[1.0, 2.0, 100.0, 200.0, 300.0]]]], dtype=torch.float32)
        pred_depth = torch.tensor([[[[1.0, 1.0, 100.0, 100.0, 100.0]]]], dtype=torch.float32)
        valid_mask = torch.ones_like(gt_depth, dtype=torch.bool)

        factor = _compute_gt_depth_alignment_factor(gt_depth, pred_depth, valid_mask)

        self.assertAlmostEqual(float(factor.item()), 1.0, places=6)

    def test_alignment_factor_falls_back_when_focus_mask_has_no_valid_overlap(self) -> None:
        gt_depth = torch.tensor([[[[1.0, 2.0, 100.0, 200.0, 300.0]]]], dtype=torch.float32)
        pred_depth = torch.tensor([[[[1.0, 1.0, 100.0, 100.0, 100.0]]]], dtype=torch.float32)
        valid_mask = torch.ones_like(gt_depth, dtype=torch.bool)
        focus_mask = torch.zeros_like(gt_depth, dtype=torch.bool)

        factor = _compute_gt_depth_alignment_factor(
            gt_depth,
            pred_depth,
            valid_mask,
            focus_mask=focus_mask,
        )

        self.assertAlmostEqual(float(factor.item()), 1.0, places=6)

    def test_alignment_factor_falls_back_to_one_without_valid_overlap(self) -> None:
        gt_depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        pred_depth = torch.zeros_like(gt_depth)
        valid_mask = torch.ones_like(gt_depth, dtype=torch.bool)

        factor = _compute_gt_depth_alignment_factor(gt_depth, pred_depth, valid_mask)

        self.assertAlmostEqual(float(factor.item()), 1.0, places=6)

    def test_apply_alignment_factor_to_points_scales_positions(self) -> None:
        points = torch.tensor(
            [[[[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]]]],
            dtype=torch.float32,
        )

        aligned = _apply_alignment_factor_to_points(points, torch.tensor(2.0))

        expected = torch.tensor(
            [[[[0.5, 1.0, 2.0], [1.0, 2.0, 4.0]]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(aligned, expected))

    def test_transform_vertices_applies_pose_in_camera_frame(self) -> None:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        pose = torch.tensor(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        transformed = _transform_vertices(vertices, pose)

        expected = torch.tensor(
            [[2.0, 3.0, 4.0], [3.0, 3.0, 4.0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(transformed, expected))

    def test_apply_alignment_factor_to_view_scales_all_scale_sensitive_fields(self) -> None:
        view = {
            "depthmap": torch.tensor([[2.0, 0.0], [4.0, 8.0]], dtype=torch.float32),
            "camera_pose": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 1.0, 0.0, 4.0],
                    [0.0, 0.0, 1.0, 8.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            "pts3d": torch.tensor(
                [[[2.0, 4.0, 8.0], [1.0, 2.0, 4.0]]],
                dtype=torch.float32,
            ),
            "hand": {
                "joints_3d_cam": torch.tensor(
                    [[2.0, 4.0, 8.0], [-1.0, -1.0, -1.0]],
                    dtype=torch.float32,
                ),
                "hand_transl": torch.tensor([6.0, 8.0, 10.0], dtype=torch.float32),
                "pose_mano": torch.tensor([0.0] * 48, dtype=torch.float32),
            },
            "object_multiview": {
                "grasped_object_pose_obj2cam": torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 6.0],
                        [0.0, 1.0, 0.0, 8.0],
                        [0.0, 0.0, 1.0, 10.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
            },
            "z_far": 12.0,
        }

        aligned = _apply_alignment_factor_to_view(view, torch.tensor(2.0))

        self.assertTrue(torch.allclose(aligned["depthmap"], torch.tensor([[1.0, 0.0], [2.0, 4.0]])))
        self.assertTrue(
            torch.allclose(
                aligned["camera_pose"][:3, 3],
                torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                aligned["pts3d"],
                torch.tensor([[[1.0, 2.0, 4.0], [0.5, 1.0, 2.0]]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                aligned["hand"]["joints_3d_cam"][0],
                torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                aligned["hand"]["joints_3d_cam"][1],
                torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                aligned["object_multiview"]["grasped_object_pose_obj2cam"][:3, 3],
                torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                aligned["hand"]["pose_mano_full"][48:51],
                torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32),
            )
        )
        self.assertEqual(aligned["z_far"], 12.0)

    def test_apply_alignment_factor_to_view_handles_tensor_z_far(self) -> None:
        view = {
            "depthmap": torch.tensor([[2.0, 4.0]], dtype=torch.float32),
            "z_far": torch.tensor([12.0, 0.0], dtype=torch.float32),
        }

        aligned = _apply_alignment_factor_to_view(view, torch.tensor(2.0))

        self.assertTrue(torch.allclose(aligned["z_far"], torch.tensor([12.0, 0.0], dtype=torch.float32)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for device-mismatch regression test")
    def test_apply_alignment_factor_to_view_accepts_cuda_factor_for_cpu_view(self) -> None:
        view = {
            "depthmap": torch.tensor([[2.0, 4.0]], dtype=torch.float32),
            "camera_pose": torch.eye(4, dtype=torch.float32),
            "pts3d": torch.tensor([[[2.0, 4.0, 8.0]]], dtype=torch.float32),
            "hand_joints_3d_cam": torch.tensor([[2.0, 4.0, 8.0]], dtype=torch.float32),
            "grasped_object_pose_obj2cam": torch.eye(4, dtype=torch.float32),
            "hand_pose_mano": torch.tensor([0.0] * 48 + [6.0, 8.0, 10.0], dtype=torch.float32),
        }

        aligned = _apply_alignment_factor_to_view(view, torch.tensor(2.0, device="cuda"))

        self.assertEqual(aligned["depthmap"].device.type, "cpu")
        self.assertTrue(torch.allclose(aligned["depthmap"], torch.tensor([[1.0, 2.0]], dtype=torch.float32)))

    def test_summarize_hand_mesh_debug_reports_sources_and_aligned_deltas(self) -> None:
        class _DummyHandLayer:
            def __init__(self):
                self.th_faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
                self.side = "right"

            def forward(self, th_pose_coeffs, betas, th_trans=None, **kwargs):
                del th_pose_coeffs, betas, kwargs
                if th_trans is None:
                    th_trans = torch.zeros((1, 3), dtype=torch.float32)
                vertices = torch.tensor(
                    [[[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]]],
                    dtype=torch.float32,
                ) + th_trans.unsqueeze(1)
                joints = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32) + th_trans.unsqueeze(1)
                return type("FakeMANOOutput", (), {"vertices": vertices, "joints": joints})()

        batch = [
            {
                "hand": {
                    "pose_mano": torch.cat([torch.tensor([[0.1, 0.2, 0.3]]), torch.zeros(1, 45)], dim=1),
                    "hand_transl": torch.tensor([[0.10, 0.20, 0.30]], dtype=torch.float32),
                    "mano_betas": torch.zeros(1, 10, dtype=torch.float32),
                    "mano_side": ["right"],
                    "valid": torch.tensor([True]),
                }
            }
        ]
        pred = {
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "pred_hand_vertices": torch.tensor(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                dtype=torch.float32,
            ),
        }
        gt = {"norm_factor": torch.tensor([2.0], dtype=torch.float32)}

        with unittest.mock.patch("debug.inspect_pi3x_pred_vs_gt._get_mano_layer", return_value=_DummyHandLayer()):
            summary = summarize_hand_mesh_debug(batch=batch, pred=pred, gt=gt, sample_index=0, frame_idx=0)

        self.assertEqual(summary["hand_side"], "right")
        self.assertEqual(
            summary["gt_source"],
            "batch[frame].hand.pose_mano + hand_transl -> ManoLayer -> /display_scale",
        )
        self.assertEqual(summary["pred_source"], "pred.pred_hand_vertices")
        self.assertAlmostEqual(summary["display_scale"], 2.0, places=6)
        self.assertEqual(summary["gt_vertex0_raw"], [0.00010000000474974513, 0.00020000000949949026, 0.0003000000142492354])
        self.assertEqual(summary["gt_vertex0_aligned"], [5.0000002374872565e-05, 0.00010000000474974513, 0.0001500000071246177])
        self.assertEqual(summary["pred_vertex0"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(summary["aligned_centroid_l1"], 0.1111111119, places=6)


if __name__ == "__main__":
    unittest.main()
