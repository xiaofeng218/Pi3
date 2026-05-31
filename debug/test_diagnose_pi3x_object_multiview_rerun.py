from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug.diagnose_pi3x_object_multiview_rerun import (  # noqa: E402
    align_local_points_with_camera_poses,
    extract_object_multiview_inputs_from_views_batch,
    flatten_points_with_colors,
    gt_depth_to_world_points,
    run_pi3x_inference,
)


class DiagnosePi3XObjectMultiviewRerunTests(unittest.TestCase):
    def test_run_pi3x_inference_passes_multimodal_inputs_when_requested(self) -> None:
        scene_inputs = {
            "imgs": torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32),
            "depths": torch.ones((1, 2, 4, 4), dtype=torch.float32),
            "intrinsics": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
            "camera_poses": torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
        }

        calls = {}

        class _FakeModel:
            def __call__(self, imgs, **kwargs):
                calls["imgs"] = imgs
                calls["kwargs"] = kwargs
                return {"points": imgs}

        out = run_pi3x_inference(_FakeModel(), scene_inputs, device=torch.device("cpu"), inference_mode="multimodal")

        self.assertIn("depths", calls["kwargs"])
        self.assertIn("intrinsics", calls["kwargs"])
        self.assertIn("poses", calls["kwargs"])
        self.assertEqual(out["points"].shape, scene_inputs["imgs"].shape)

    def test_run_pi3x_inference_omits_multimodal_kwargs_in_image_mode(self) -> None:
        scene_inputs = {
            "imgs": torch.zeros((1, 2, 3, 4, 4), dtype=torch.float32),
            "depths": torch.ones((1, 2, 4, 4), dtype=torch.float32),
            "intrinsics": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
            "camera_poses": torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
        }

        calls = {}

        class _FakeModel:
            def __call__(self, imgs, **kwargs):
                calls["kwargs"] = kwargs
                return {"points": imgs}

        run_pi3x_inference(_FakeModel(), scene_inputs, device=torch.device("cpu"), inference_mode="image")

        self.assertEqual(calls["kwargs"], {})

    def test_extract_object_multiview_inputs_from_views_batch_returns_expected_shapes(self) -> None:
        batch = {
            "views": [
                {"object_multiview": {"normalization_scale": torch.tensor(1.0)}},
                {
                    "object_multiview": {
                        "img": torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 2, 3, 2, 2),
                        "depthmap": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                        "camera_intrinsics": torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
                        "camera_pose": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
                    }
                }
            ]
        }

        scene_inputs = extract_object_multiview_inputs_from_views_batch(batch)

        self.assertEqual(tuple(scene_inputs["imgs"].shape), (1, 2, 3, 2, 2))
        self.assertEqual(tuple(scene_inputs["depths"].shape), (1, 2, 2, 2))
        self.assertEqual(tuple(scene_inputs["intrinsics"].shape), (1, 2, 3, 3))
        self.assertEqual(tuple(scene_inputs["camera_poses"].shape), (1, 2, 4, 4))

    def test_gt_depth_to_world_points_projects_valid_pixels(self) -> None:
        depths = torch.tensor([[[[1.0, 0.0], [2.0, 3.0]]]], dtype=torch.float32)
        intrinsics = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3)
        poses = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4)
        imgs = torch.tensor(
            [[[[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [0.5, 0.5]], [[0.0, 1.0], [1.0, 0.0]]]]],
            dtype=torch.float32,
        )

        points, colors = gt_depth_to_world_points(depths, intrinsics, poses, imgs)

        expected_points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(points, expected_points))
        self.assertEqual(tuple(colors.shape), (3, 3))

    def test_align_local_points_with_camera_poses_applies_translation(self) -> None:
        local_points = torch.tensor(
            [[[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]]],
            dtype=torch.float32,
        )
        poses = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4)
        poses[0, 0, 0, 3] = 2.0

        aligned = align_local_points_with_camera_poses(local_points, poses)

        expected = torch.tensor(
            [[[[[2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(aligned, expected))

    def test_flatten_points_with_colors_filters_non_finite_rows(self) -> None:
        points = torch.tensor(
            [[[[[0.0, 0.0, 1.0], [float("nan"), 1.0, 1.0]]]]],
            dtype=torch.float32,
        )
        imgs = torch.ones((1, 1, 3, 1, 2), dtype=torch.float32)

        flat_points, flat_colors = flatten_points_with_colors(points, imgs)

        self.assertEqual(tuple(flat_points.shape), (1, 3))
        self.assertEqual(tuple(flat_colors.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
