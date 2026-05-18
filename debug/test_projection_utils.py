from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi3.utils.projection import (
    compute_2d_point_errors,
    map_points_between_intrinsics,
    project_points_cam_to_image,
    project_points_cam_to_image_torch,
)


class ProjectionUtilsTests(unittest.TestCase):
    def test_project_points_cam_to_image(self) -> None:
        points_cam = np.array(
            [
                [0.0, 0.0, 2.0],
                [1.0, -1.0, 2.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        intrinsics = np.array(
            [
                [100.0, 0.0, 10.0],
                [0.0, 120.0, 20.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        uv, valid = project_points_cam_to_image(points_cam, intrinsics)

        np.testing.assert_allclose(uv[0], np.array([10.0, 20.0], dtype=np.float32))
        np.testing.assert_allclose(uv[1], np.array([60.0, -40.0], dtype=np.float32))
        self.assertFalse(valid[2])
        self.assertTrue(np.isnan(uv[2]).all())

    def test_map_points_between_intrinsics(self) -> None:
        src = np.array(
            [
                [100.0, 0.0, 10.0],
                [0.0, 100.0, 20.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dst = np.array(
            [
                [200.0, 0.0, 5.0],
                [0.0, 200.0, 15.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        mapped = map_points_between_intrinsics(
            np.array([[10.0, 20.0], [60.0, 70.0]], dtype=np.float32),
            src,
            dst,
        )

        np.testing.assert_allclose(
            mapped,
            np.array([[5.0, 15.0], [105.0, 115.0]], dtype=np.float32),
        )

    def test_compute_2d_point_errors(self) -> None:
        errors, valid = compute_2d_point_errors(
            np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 0.0]], dtype=np.float32),
            np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        )

        np.testing.assert_allclose(errors[:2], np.array([5.0, 0.0], dtype=np.float32))
        self.assertTrue(valid[0])
        self.assertTrue(valid[1])
        self.assertFalse(valid[2])
        self.assertTrue(np.isnan(errors[2]))

    def test_project_points_cam_to_image_torch_supports_batched_intrinsics(self) -> None:
        points_cam = torch.tensor(
            [
                [[0.0, 0.0, 2.0], [1.0, -1.0, 2.0]],
                [[0.0, 1.0, 1.0], [2.0, 0.0, 2.0]],
            ],
            dtype=torch.float32,
        )
        intrinsics = torch.tensor(
            [
                [[100.0, 0.0, 10.0], [0.0, 120.0, 20.0], [0.0, 0.0, 1.0]],
                [[80.0, 0.0, 5.0], [0.0, 90.0, 15.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )

        uv, valid = project_points_cam_to_image_torch(points_cam, intrinsics)

        self.assertTrue(valid.all())
        torch.testing.assert_close(uv[0, 0], torch.tensor([10.0, 20.0]))
        torch.testing.assert_close(uv[0, 1], torch.tensor([60.0, -40.0]))
        torch.testing.assert_close(uv[1, 0], torch.tensor([5.0, 105.0]))
        torch.testing.assert_close(uv[1, 1], torch.tensor([85.0, 15.0]))


if __name__ == "__main__":
    unittest.main()
