import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from dex_ycb_toolkit.headless_vis import (
    HAND_SEG_COLOR,
    colorize_segmentation,
    compose_foreground_overlay,
    draw_hand_joints,
)


class HeadlessVisualizationTests(unittest.TestCase):
    def test_colorize_segmentation_maps_background_object_and_hand(self):
        seg = np.array([[0, 1], [255, 21]], dtype=np.uint8)

        colored = colorize_segmentation(seg)

        self.assertEqual(colored.shape, (2, 2, 3))
        np.testing.assert_array_equal(colored[0, 0], np.array([0, 0, 0], dtype=np.uint8))
        np.testing.assert_array_equal(colored[1, 0], np.array(HAND_SEG_COLOR, dtype=np.uint8))
        self.assertFalse(np.array_equal(colored[0, 1], colored[0, 0]))
        self.assertFalse(np.array_equal(colored[1, 1], colored[0, 0]))

    def test_draw_hand_joints_draws_only_valid_points(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        joints = np.full((21, 2), -1, dtype=np.float32)
        joints[0] = [10, 10]
        joints[1] = [20, 10]

        drawn = draw_hand_joints(image, joints)

        self.assertGreater(int(drawn.sum()), 0)
        self.assertGreater(int(drawn[10, 10].sum()), 0)
        self.assertGreater(int(drawn[10, 20].sum()), 0)

    def test_compose_foreground_overlay_blacks_background(self):
        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        seg_rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
        seg = np.zeros((4, 4), dtype=np.uint8)
        seg[1:3, 1:3] = 1
        joints = np.full((21, 2), -1, dtype=np.float32)

        overlay = compose_foreground_overlay(image, seg_rgb, seg, joints)

        np.testing.assert_array_equal(overlay[0, 0], np.array([0, 0, 0], dtype=np.uint8))
        self.assertGreater(int(overlay[1, 1].sum()), 0)

    def test_headless_script_saves_four_images(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "examples" / "visualize_sample_headless.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            color_file = tmp_path / "color_000000.jpg"
            label_file = tmp_path / "labels_000000.npz"
            output_dir = tmp_path / "out"

            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[8:24, 8:24] = 127
            cv2.imwrite(str(color_file), image)

            seg = np.zeros((32, 32), dtype=np.uint8)
            seg[8:24, 8:24] = 1
            joints = np.full((1, 21, 2), -1, dtype=np.float32)
            joints[0, 0] = [10, 10]
            joints[0, 1] = [20, 10]
            np.savez(label_file, seg=seg, joint_2d=joints)

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--color_file",
                    str(color_file),
                    "--label_file",
                    str(label_file),
                    "--output_dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            for name in ("rgb", "seg", "joints", "overlay"):
                self.assertTrue((output_dir / f"color_000000_{name}.png").is_file())

    def test_sequence_script_saves_five_videos(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "examples" / "visualize_sequence_headless.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            camera_dir = tmp_path / "932122061900"
            output_dir = tmp_path / "out"
            camera_dir.mkdir()

            for idx in range(2):
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                image[8:24, 8:24] = 80 + idx
                cv2.imwrite(str(camera_dir / f"color_{idx:06d}.jpg"), image)

                seg = np.zeros((32, 32), dtype=np.uint8)
                seg[8:24, 8:24] = 1
                joints = np.full((1, 21, 2), -1, dtype=np.float32)
                joints[0, 0] = [10, 10]
                joints[0, 1] = [20, 10]
                np.savez(camera_dir / f"labels_{idx:06d}.npz", seg=seg, joint_2d=joints)

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--camera_dir",
                    str(camera_dir),
                    "--output_dir",
                    str(output_dir),
                    "--fps",
                    "5",
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            for name in ("rgb", "seg", "joints", "overlay", "foreground_overlay"):
                self.assertTrue((output_dir / f"{camera_dir.name}_{name}.mp4").is_file())


if __name__ == "__main__":
    unittest.main()
