from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pi3.visualization.pi3x_rerun_export import export_pi3x_rerun_sample


class _FakeRerun:
    class ViewCoordinates:
        RDF = "RDF"

    def __init__(self):
        self.logged = []
        self.saved_path = None
        self.init_args = None
        self.time_sequences = []

    def init(self, name, spawn=False, default_enabled=True):
        self.init_args = (name, spawn, default_enabled)

    def save(self, path):
        self.saved_path = Path(path)
        self.saved_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_path.touch()

    def set_time_sequence(self, key, value):
        self.time_sequences.append((key, value))

    def log(self, entity_path, payload):
        self.logged.append((entity_path, payload))

    class Clear:
        def __init__(self, recursive=False):
            self.recursive = recursive

    class Image:
        def __init__(self, array):
            self.array = np.asarray(array)

    class Points3D:
        def __init__(self, positions, colors=None):
            self.positions = np.asarray(positions)
            self.colors = None if colors is None else np.asarray(colors)

    class Mesh3D:
        def __init__(self, vertex_positions, indices, vertex_colors=None):
            self.vertex_positions = np.asarray(vertex_positions)
            self.indices = np.asarray(indices)
            self.vertex_colors = None if vertex_colors is None else np.asarray(vertex_colors)


class _DummyManoLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.is_rhand = True
        self.side = "right"

    def forward(self, th_pose_coeffs, betas, th_trans=None, **kwargs):
        del th_pose_coeffs, betas, kwargs
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]]],
            dtype=torch.float32,
        )
        if th_trans is not None:
            vertices = vertices + th_trans.unsqueeze(1)
        joints = torch.tensor([[[0.02, 0.0, 0.0]]], dtype=torch.float32)
        if th_trans is not None:
            joints = joints + th_trans.unsqueeze(1)
        return SimpleNamespace(vertices=vertices, joints=joints)

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del global_orient, hand_pose, betas, kwargs
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]]],
            dtype=torch.float32,
        )
        if th_trans is not None:
            vertices = vertices + th_trans.unsqueeze(1)
        joints = torch.tensor([[[0.02, 0.0, 0.0]]], dtype=torch.float32)
        if th_trans is not None:
            joints = joints + th_trans.unsqueeze(1)
        return SimpleNamespace(vertices=vertices, joints=joints)


class _DummyLeftManoLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.is_rhand = False
        self.side = "left"

    def forward(self, th_pose_coeffs, betas, th_trans=None, **kwargs):
        del th_pose_coeffs, betas, kwargs
        vertices = torch.tensor(
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        if th_trans is not None:
            vertices = vertices + th_trans.unsqueeze(1)
        joints = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)
        if th_trans is not None:
            joints = joints + th_trans.unsqueeze(1)
        return SimpleNamespace(vertices=vertices, joints=joints)

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del global_orient, hand_pose, betas, kwargs
        vertices = torch.tensor(
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
        if th_trans is not None:
            vertices = vertices + th_trans.unsqueeze(1)
        joints = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)
        if th_trans is not None:
            joints = joints + th_trans.unsqueeze(1)
        return SimpleNamespace(vertices=vertices, joints=joints)


class Pi3XRerunExportTests(unittest.TestCase):
    def test_export_writes_meshes_and_point_clouds(self) -> None:
        fake_rr = _FakeRerun()

        batch = [
            {
                "img": torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
                "depthmap": torch.ones(1, 4, 4, dtype=torch.float32),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "hand": {
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_mano": torch.cat([torch.tensor([[0.1, 0.0, 0.0]]), torch.zeros(1, 45)], dim=1),
                    "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
                    "mano_betas": torch.zeros(1, 10, dtype=torch.float32),
                    "mano_side": ["right"],
                },
                "object_multiview": {
                    "grasped_object_id": torch.tensor([1]),
                    "grasped_object_mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "grasped_object_valid": torch.tensor([True]),
                    "grasped_object_pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "normalization_scale": torch.tensor([1.5], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 2.0
        pred = {
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
        }
        pred["local_points"][0, 0, ..., 2] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_object_mesh_template",
                return_value=(torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=torch.float32), np.array([[0, 1, 2]], dtype=np.int32)),
            ):
                export_pi3x_rerun_sample(
                    output_path=output_path,
                    batch=batch,
                    pred=pred,
                    gt=gt,
                    sample_index=0,
                    data_root="/tmp/dummy",
                    mano_layer=_DummyManoLayer(),
                    item_name="pi3x_train_sample",
                )

            self.assertTrue(output_path.exists())
            meta_path = output_path.with_suffix(".meta.json")
            self.assertTrue(meta_path.exists())
            manifest_path = output_path.with_name("manifest.json")
            self.assertTrue(manifest_path.exists())
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["sample_index"], 0)
            self.assertIn("gt_norm_factor", meta)

        logged_entities = [entity for entity, _ in fake_rr.logged]
        self.assertIn("world/gt_hand_mesh", logged_entities)
        self.assertIn("world/gt_hand_joints", logged_entities)
        self.assertIn("world/gt_object_mesh", logged_entities)
        self.assertIn("world/gt_points", logged_entities)
        self.assertIn("world/pred_points", logged_entities)

    def test_export_logs_predicted_hand_and_object_meshes_without_display_scale(self) -> None:
        fake_rr = _FakeRerun()

        batch = [
            {
                "img": torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
                "depthmap": torch.ones(1, 4, 4, dtype=torch.float32),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "hand": {
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_mano": torch.cat([torch.tensor([[0.1, 0.0, 0.0]]), torch.zeros(1, 45)], dim=1),
                    "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
                    "mano_betas": torch.zeros(1, 10, dtype=torch.float32),
                    "mano_side": ["right"],
                },
                "object_multiview": {
                    "grasped_object_id": torch.tensor([1]),
                    "grasped_object_mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "grasped_object_valid": torch.tensor([True]),
                    "grasped_object_pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "normalization_center": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
                    "normalization_scale": torch.tensor(2.0, dtype=torch.float32),
                },
            }
        ]
        gt = {
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 2.0
        pred = {
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "pred_hand_vertices": torch.tensor(
                [[[0.20, 0.00, 0.00], [0.00, 0.20, 0.00], [0.00, 0.00, 0.20]]],
                dtype=torch.float32,
            ),
            "pred_object_rot6d": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
            "pred_object_trans": torch.tensor([[[0.10, 0.20, 0.30]]], dtype=torch.float32),
            "pred_object_scale": torch.tensor([[[2.0]]], dtype=torch.float32),
            "object_valid": torch.tensor([[True]], dtype=torch.bool),
        }
        pred["local_points"][0, 0, ..., 2] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_object_mesh_template",
                return_value=(torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32), np.array([[0, 1, 2]], dtype=np.int32)),
            ):
                export_pi3x_rerun_sample(
                    output_path=output_path,
                    batch=batch,
                    pred=pred,
                    gt=gt,
                    sample_index=0,
                    data_root="/tmp/dummy",
                    mano_layer=_DummyManoLayer(),
                    item_name="pi3x_train_sample",
                )

            pred_object_meshes = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/pred_object_mesh" and payload.__class__.__name__ == "Mesh3D"
            ]
            pred_hand_meshes = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/pred_hand_mesh" and payload.__class__.__name__ == "Mesh3D"
            ]
            self.assertEqual(len(pred_object_meshes), 1)
            self.assertEqual(len(pred_hand_meshes), 1)

            np.testing.assert_allclose(
                pred_hand_meshes[0].vertex_positions,
                np.array(
                    [
                        [0.20, 0.00, 0.00],
                        [0.00, 0.20, 0.00],
                        [0.00, 0.00, 0.20],
                    ],
                    dtype=np.float32,
                ),
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                pred_object_meshes[0].vertex_positions,
                np.array(
                    [
                        [-0.90, -0.80, 0.30],
                        [1.10, -0.80, 0.30],
                        [-0.90, 1.20, 0.30],
                    ],
                    dtype=np.float32,
                ),
                rtol=0,
                atol=1e-6,
            )

    def test_export_logs_left_hand_vertices_without_mirroring(self) -> None:
        fake_rr = _FakeRerun()
        mano_layer = _DummyLeftManoLayer()

        batch = [
            {
                "img": torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
                "depthmap": torch.ones(1, 4, 4, dtype=torch.float32),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "hand": {
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_mano": torch.cat([torch.tensor([[0.1, 0.0, 0.0]]), torch.zeros(1, 45)], dim=1),
                    "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
                    "mano_betas": torch.zeros(1, 10, dtype=torch.float32),
                    "mano_side": ["left"],
                },
                "object_multiview": {
                    "grasped_object_id": torch.tensor([1]),
                    "grasped_object_mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "grasped_object_valid": torch.tensor([True]),
                    "grasped_object_pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "normalization_scale": torch.tensor([1.5], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_object_mesh_template",
                return_value=(torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=torch.float32), np.array([[0, 1, 2]], dtype=np.int32)),
            ):
                export_pi3x_rerun_sample(
                    output_path=output_path,
                    batch=batch,
                    pred=None,
                    gt=gt,
                    sample_index=0,
                    data_root="/tmp/dummy",
                    mano_layer=mano_layer,
                    item_name="pi3x_train_sample",
                )

            hand_meshes = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/gt_hand_mesh" and payload.__class__.__name__ == "Mesh3D"
            ]
            hand_joints = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/gt_hand_joints" and payload.__class__.__name__ == "Points3D"
            ]
            self.assertEqual(len(hand_meshes), 1)
            self.assertEqual(len(hand_joints), 1)
            vertices = hand_meshes[0].vertex_positions
            expected = np.array(
                [
                    [1.01, 0.02, 0.03],
                    [2.01, 0.02, 0.03],
                    [1.01, 1.02, 0.03],
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(vertices, expected, rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                hand_joints[0].positions,
                np.array([[2.01, 0.02, 0.03]], dtype=np.float32),
                rtol=0,
                atol=1e-6,
            )

    def test_export_uses_side_specific_left_mano_layer_when_available(self) -> None:
        fake_rr = _FakeRerun()

        batch = [
            {
                "img": torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
                "depthmap": torch.ones(1, 4, 4, dtype=torch.float32),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "hand": {
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_mano": torch.cat([torch.tensor([[0.1, 0.0, 0.0]]), torch.zeros(1, 45)], dim=1),
                    "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
                    "mano_betas": torch.zeros(1, 10, dtype=torch.float32),
                    "mano_side": ["left"],
                },
                "object_multiview": {
                    "grasped_object_id": torch.tensor([1]),
                    "grasped_object_mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "grasped_object_valid": torch.tensor([True]),
                    "grasped_object_pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "normalization_scale": torch.tensor([1.5], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._build_visualization_mano_layer",
                return_value=_DummyLeftManoLayer(),
            ), patch(
                "pi3.visualization.pi3x_rerun_export._load_object_mesh_template",
                return_value=(torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=torch.float32), np.array([[0, 1, 2]], dtype=np.int32)),
            ):
                export_pi3x_rerun_sample(
                    output_path=output_path,
                    batch=batch,
                    pred=None,
                    gt=gt,
                    sample_index=0,
                    data_root="/tmp/dummy",
                    mano_layer=torch.nn.ModuleDict({"right": _DummyManoLayer(), "left": _DummyLeftManoLayer()}),
                    item_name="pi3x_train_sample",
                )

            hand_meshes = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/gt_hand_mesh" and payload.__class__.__name__ == "Mesh3D"
            ]
            self.assertEqual(len(hand_meshes), 1)
            np.testing.assert_allclose(
                hand_meshes[0].vertex_positions,
                np.array(
                    [
                        [1.01, 0.02, 0.03],
                        [2.01, 0.02, 0.03],
                        [1.01, 1.02, 0.03],
                    ],
                    dtype=np.float32,
                ),
                rtol=0,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
