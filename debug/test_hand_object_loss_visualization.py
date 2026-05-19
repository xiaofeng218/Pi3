from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pi3.models.hand_object_loss import HandObjectLoss


class _FakeRerun:
    class ViewCoordinates:
        RDF = "RDF"

    class Clear:
        def __init__(self, recursive: bool = False):
            self.recursive = recursive

    class Points3D:
        def __init__(self, positions, colors=None):
            self.positions = np.asarray(positions)
            self.colors = None if colors is None else np.asarray(colors)

    class LineStrips3D:
        def __init__(self, strips, colors=None, **_kwargs):
            self.strips = np.asarray(strips)
            self.colors = None if colors is None else np.asarray(colors)

    def __init__(self):
        self.saved_path = None
        self.logged = []

    def init(self, *_args, **_kwargs):
        return None

    def save(self, path):
        self.saved_path = Path(path)
        self.saved_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_path.touch()

    def log(self, entity_path, payload, **_kwargs):
        self.logged.append((entity_path, payload))


class HandObjectLossVisualizationTests(unittest.TestCase):
    def test_debug_joint_visualization_exports_rerun_with_correspondence(self) -> None:
        fake_rr = _FakeRerun()
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "pi3.models.hand_object_loss._load_rerun",
            return_value=fake_rr,
        ):
            loss_fn = HandObjectLoss(
                hand_joints_3d_weight=1.0,
                debug_hand_joints_vis_enabled=True,
                debug_hand_joints_vis_dir=tmpdir,
                debug_hand_joints_vis_max_exports=1,
            )
            pred = {
                "pred_hand_transl": torch.zeros((1, 3), dtype=torch.float32),
                "pred_hand_scale": torch.ones((1, 1), dtype=torch.float32),
                "pred_hand_mano_params": {
                    "global_orient": torch.eye(3).view(1, 1, 3, 3),
                },
                "pred_hand_joints_3d": torch.linspace(0.0, 0.2, 63, dtype=torch.float32).view(1, 21, 3),
            }
            gt = {
                "scene_scale": torch.ones((1,), dtype=torch.float32),
                "hand_valid": torch.tensor([True]),
                "hand_is_right": torch.tensor([True]),
                "hand_global_orient_rotmat": torch.eye(3).view(1, 1, 3, 3),
                "hand_transl": torch.zeros((1, 3), dtype=torch.float32),
                "hand_scale": torch.ones((1, 1), dtype=torch.float32),
                "hand_joints_3d": torch.linspace(0.01, 0.21, 63, dtype=torch.float32).view(1, 21, 3),
            }

            loss, details = loss_fn(pred, gt)

            self.assertTrue(torch.is_tensor(loss))
            self.assertIn("hand_joints_3d_loss", details)
            self.assertIsNotNone(fake_rr.saved_path)
            self.assertTrue(fake_rr.saved_path.is_file())

            logged_entities = [entity for entity, _payload in fake_rr.logged]
            self.assertIn("world/gt_hand_joints", logged_entities)
            self.assertIn("world/gt_hand_skeleton", logged_entities)
            self.assertIn("world/pred_hand_joints", logged_entities)
            self.assertIn("world/pred_hand_skeleton", logged_entities)
            self.assertIn("world/hand_joint_correspondence", logged_entities)

            gt_skeleton_payload = next(
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/gt_hand_skeleton"
            )
            self.assertEqual(gt_skeleton_payload.strips.shape, (5, 5, 3))

            corr_payload = next(
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/hand_joint_correspondence"
            )
            self.assertEqual(corr_payload.strips.shape, (21, 2, 3))


if __name__ == "__main__":
    unittest.main()
