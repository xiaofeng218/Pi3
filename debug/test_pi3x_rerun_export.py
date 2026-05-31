from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pi3.models.hamer.geometry import aa_to_rotmat
from pi3.visualization.pi3x_rerun_export import (
    export_pi3x_rerun_sample,
    _build_gt_object_asset_transform,
    _strip_invalid_gltf_semantics,
)


class _FakeRerun:
    class ViewCoordinates:
        RDF = "RDF"

    def __init__(self):
        self.logged = []
        self.log_calls = []
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

    def set_time(self, timeline, *, sequence=None, recording=None, duration=None, timestamp=None):
        del recording, duration, timestamp
        self.time_sequences.append((timeline, sequence))

    def log(self, entity_path, payload, **kwargs):
        self.logged.append((entity_path, payload))
        self.log_calls.append((entity_path, payload, kwargs))

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
        def __init__(
            self,
            vertex_positions,
            triangle_indices=None,
            indices=None,
            vertex_colors=None,
            vertex_texcoords=None,
            albedo_texture=None,
            albedo_factor=None,
            face_rendering=None,
            vertex_normals=None,
            class_ids=None,
        ):
            self.vertex_positions = np.asarray(vertex_positions)
            triangle_indices = triangle_indices if triangle_indices is not None else indices
            self.indices = None if triangle_indices is None else np.asarray(triangle_indices)
            self.vertex_colors = None if vertex_colors is None else np.asarray(vertex_colors)
            self.vertex_texcoords = None if vertex_texcoords is None else np.asarray(vertex_texcoords)
            self.albedo_texture = albedo_texture
            self.albedo_factor = albedo_factor
            self.face_rendering = face_rendering
            self.vertex_normals = None if vertex_normals is None else np.asarray(vertex_normals)
            self.class_ids = class_ids

    class Asset3D:
        def __init__(self, path=None, contents=None, media_type=None, transform=None):
            self.path = None if path is None else str(path)
            self.contents = contents
            self.media_type = media_type
            self.transform = transform

    class Transform3D:
        def __init__(self, mat3x3=None, translation=None, rotation=None, scale=None, **kwargs):
            del rotation, scale, kwargs
            self.mat3x3 = None if mat3x3 is None else np.asarray(mat3x3)
            self.translation = None if translation is None else np.asarray(translation)

    class LineStrips3D:
        def __init__(self, strips, radii=None, colors=None, labels=None, class_ids=None, instance_keys=None):
            del radii, labels, class_ids, instance_keys
            self.strips = np.asarray(strips)
            self.colors = None if colors is None else np.asarray(colors)


def _fake_export_textured_object_glb(glb_path, data_root_str, obj_id, transform):
    del data_root_str, obj_id, transform
    glb_path = Path(glb_path)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(b"glb")
    return glb_path


def _fake_load_textured_object_mesh(data_root_str, obj_id):
    del data_root_str, obj_id
    return {
        "vertices": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "faces": np.array([[0, 1, 2]], dtype=np.uint32),
        "vertex_texcoords": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "albedo_texture": np.full((2, 2, 3), 127, dtype=np.uint8),
    }


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


class _DummyMillimeterManoLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.is_rhand = True
        self.side = "right"

    def forward(self, th_pose_coeffs, betas, th_trans=None, **kwargs):
        del th_pose_coeffs, betas, kwargs
        vertices = torch.tensor(
            [[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0]]],
            dtype=torch.float32,
        )
        if th_trans is not None:
            vertices = vertices + th_trans.unsqueeze(1)
        joints = torch.tensor([[[20.0, 0.0, 0.0]]], dtype=torch.float32)
        if th_trans is not None:
            joints = joints + th_trans.unsqueeze(1)
        return SimpleNamespace(vertices=vertices, joints=joints)

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del global_orient, hand_pose, betas, kwargs
        return self.forward(None, None, th_trans=th_trans)


class Pi3XRerunExportTests(unittest.TestCase):
    def test_strip_invalid_gltf_semantics_removes_private_attributes(self) -> None:
        gltf = {
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "TEXCOORD_0": 1,
                                "_color": 2,
                                "_CUSTOM": 3,
                            }
                        }
                    ]
                }
            ]
        }

        changed = _strip_invalid_gltf_semantics(gltf)

        self.assertTrue(changed)
        attributes = gltf["meshes"][0]["primitives"][0]["attributes"]
        self.assertEqual(attributes, {"POSITION": 0, "TEXCOORD_0": 1})

    def test_gt_object_asset_transform_uses_scene_scale_not_pred_scale_normalization(self) -> None:
        gt = {
            "object_valid": torch.tensor([[True]], dtype=torch.bool),
            "object_pose_obj2cam": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
            "object_normalization_scale": torch.tensor([1.0], dtype=torch.float32),
        }

        transform = _build_gt_object_asset_transform(
            gt=gt,
            sample_index=0,
            frame_idx=0,
            normalization_center=torch.zeros(3, dtype=torch.float32),
            normalization_scale_metric=torch.tensor([2.0], dtype=torch.float32),
        )

        np.testing.assert_allclose(
            transform[:3, :3].numpy(),
            np.eye(3, dtype=np.float32) * 0.5,
            rtol=0,
            atol=1e-6,
        )

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
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "vertices_2d": torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]], dtype=torch.float32),
                    "vertices_2d_valid": torch.tensor([[True, True, True]], dtype=torch.bool),
                },
                "object_multiview": {
                    "template_vertices": torch.tensor(
                        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.zeros(3, dtype=torch.float32),
                    "normalization_scale": torch.tensor([1.5], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
            "object_valid": torch.tensor([[True]], dtype=torch.bool),
            "object_pose_obj2cam": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
            "object_normalization_scale": torch.tensor([1.5], dtype=torch.float32),
            "hand_valid": torch.tensor([True]),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_global_orient_rotmat": aa_to_rotmat(torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
            "hand_scale": torch.tensor([[1.0]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor([[[0.03, 0.02, 0.03]]], dtype=torch.float32),
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
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
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
            self.assertIn("scene_scale", meta)

        logged_entities = [entity for entity, _ in fake_rr.logged]
        self.assertIn("world/gt_hand_mesh", logged_entities)
        self.assertIn("world/gt_hand_joints", logged_entities)
        self.assertIn("world/gt_object_asset", logged_entities)
        self.assertIn("world/gt_points", logged_entities)
        self.assertIn("world/pred_points", logged_entities)

    def test_export_gt_hand_mesh_does_not_apply_scene_scale_twice(self) -> None:
        fake_rr = _FakeRerun()

        batch = [
            {
                "img": torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
                "depthmap": torch.ones(1, 4, 4, dtype=torch.float32),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([False]),
                },
                "object_multiview": {
                    "template_vertices": torch.tensor(
                        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.zeros(3, dtype=torch.float32),
                    "normalization_scale": torch.tensor([1.0], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([2.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "hand_valid": torch.tensor([True]),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_global_orient_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            "hand_scale": torch.tensor([[2.0]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor([[[1.02, 2.0, 3.0]]], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 1.0
        pred = {"local_points": gt["local_points"].clone()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
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

        gt_hand_joints = [payload for entity, payload in fake_rr.logged if entity == "world/gt_hand_joints"][-1]
        np.testing.assert_allclose(
            gt_hand_joints.positions,
            np.array([[0.00104, 0.002, 0.003]], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )

    def test_export_gt_hand_mesh_converts_mano_millimeters_to_scene_units(self) -> None:
        fake_rr = _FakeRerun()

        batch = [
            {
                "img": torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
                "depthmap": torch.ones(1, 4, 4, dtype=torch.float32),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([False]),
                },
                "object_multiview": {
                    "template_vertices": torch.tensor(
                        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.zeros(3, dtype=torch.float32),
                    "normalization_scale": torch.tensor([1.0], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "hand_valid": torch.tensor([True]),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_global_orient_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            "hand_scale": torch.tensor([[1.0]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor([[[0.02, 0.0, 0.0]]], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 1.0
        pred = {"local_points": gt["local_points"].clone()}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
            ):
                export_pi3x_rerun_sample(
                    output_path=output_path,
                    batch=batch,
                    pred=pred,
                    gt=gt,
                    sample_index=0,
                    data_root="/tmp/dummy",
                    mano_layer=_DummyMillimeterManoLayer(),
                    item_name="pi3x_train_sample",
                )

        gt_hand_joints = [payload for entity, payload in fake_rr.logged if entity == "world/gt_hand_joints"][-1]
        np.testing.assert_allclose(
            gt_hand_joints.positions,
            np.array([[0.02, 0.0, 0.0]], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )

    def test_export_gt_hand_mesh_supports_dense_bn_gt_without_owner_index(self) -> None:
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
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([False]),
                },
                "object_multiview": {
                    "template_vertices": torch.tensor(
                        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.zeros(3, dtype=torch.float32),
                    "normalization_scale": torch.tensor([1.0], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "hand_valid": torch.tensor([[True]], dtype=torch.bool),
            "hand_is_right": torch.tensor([[True]], dtype=torch.bool),
            "hand_global_orient_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3).repeat(1, 1, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 1, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[[0.01, 0.02, 0.03]]], dtype=torch.float32),
            "hand_scale": torch.tensor([[[1.0]]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor([[[[0.01, 0.02, 0.03]]]], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
            ):
                export_pi3x_rerun_sample(
                    output_path=output_path,
                    batch=batch,
                    pred=None,
                    gt=gt,
                    sample_index=0,
                    data_root="/tmp/dummy",
                    mano_layer=_DummyManoLayer(),
                    item_name="pi3x_train_sample",
                )

        logged_entities = [entity for entity, _ in fake_rr.logged]
        self.assertIn("world/gt_hand_mesh", logged_entities)
        self.assertIn("world/gt_hand_joints", logged_entities)

    def test_export_gt_hand_mesh_supports_dense_bn_gt_with_sample_level_hand_scale(self) -> None:
        fake_rr = _FakeRerun()

        batch = []
        for _ in range(2):
            batch.append(
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
                    "object": {
                        "grasped_object_id": torch.tensor([1]),
                        "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                        "valid": torch.tensor([False]),
                    },
                    "object_multiview": {
                        "template_vertices": torch.tensor(
                            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                            dtype=torch.float32,
                        ),
                        "normalization_center": torch.zeros(3, dtype=torch.float32),
                        "normalization_scale": torch.tensor([1.0], dtype=torch.float32),
                    },
                }
            )
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 2, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 2, 4, 4, dtype=torch.bool),
            "hand_valid": torch.tensor([[True, True]], dtype=torch.bool),
            "hand_is_right": torch.tensor([[True, True]], dtype=torch.bool),
            "hand_global_orient_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3).repeat(1, 2, 1, 1, 1),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3).repeat(1, 2, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 2, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]], dtype=torch.float32),
            "hand_scale": torch.tensor([[[1.0]]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor([[[[0.01, 0.02, 0.03]], [[0.04, 0.05, 0.06]]]], dtype=torch.float32),
        }
        gt["local_points"][0, :, ..., 2] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
            ):
                export_pi3x_rerun_sample(
                    output_path=output_path,
                    batch=batch,
                    pred=None,
                    gt=gt,
                    sample_index=0,
                    data_root="/tmp/dummy",
                    mano_layer=_DummyManoLayer(),
                    item_name="pi3x_train_sample",
                )

        gt_hand_meshes = [
            payload
            for entity, payload in fake_rr.logged
            if entity == "world/gt_hand_mesh" and payload.__class__.__name__ == "Mesh3D"
        ]
        self.assertEqual(len(gt_hand_meshes), 2)

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
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "vertices_2d": torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]], dtype=torch.float32),
                    "vertices_2d_valid": torch.tensor([[True, True, True]], dtype=torch.bool),
                },
                "object_multiview": {
                    "template_vertices": torch.tensor(
                        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
                    "normalization_scale": torch.tensor(2.0, dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
            "object_valid": torch.tensor([[True]], dtype=torch.bool),
            "object_pose_obj2cam": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
            "object_normalization_scale": torch.tensor([2.0], dtype=torch.float32),
            "hand_valid": torch.tensor([True]),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_global_orient_rotmat": aa_to_rotmat(torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
            "hand_scale": torch.tensor([[1.0]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor(
                [[[0.01, 0.02, 0.03], [0.02, 0.03, 0.04], [0.03, 0.04, 0.05]]],
                dtype=torch.float32,
            ),
        }
        gt["local_points"][0, 0, ..., 2] = 2.0
        pred = {
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_is_right": torch.tensor([True], dtype=torch.bool),
            "pred_hand_vertices": torch.tensor(
                [[[0.20, 0.00, 0.00], [0.00, 0.20, 0.00], [0.00, 0.00, 0.20]]],
                dtype=torch.float32,
            ),
            "pred_hand_joints_3d": torch.tensor(
                [[[0.11, 0.12, 0.13], [0.21, 0.22, 0.23], [0.31, 0.32, 0.33]]],
                dtype=torch.float32,
            ),
            "pred_hand_mano_params": {
                "global_orient": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3),
                "hand_pose": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
                "betas": torch.zeros(1, 10, dtype=torch.float32),
            },
            "pred_hand_mano_betas": torch.zeros(1, 10, dtype=torch.float32),
            "pred_hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
            "pred_hand_scale": torch.tensor([[1.0]], dtype=torch.float32),
            "pred_object_rot6d": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
            "pred_object_trans": torch.tensor([[[0.10, 0.20, 0.30]]], dtype=torch.float32),
            "pred_object_scale": torch.tensor([[[2.0]]], dtype=torch.float32),
            "object_valid": torch.tensor([[True]], dtype=torch.bool),
        }
        pred["local_points"][0, 0, ..., 2] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
                create=True,
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
            pred_object_assets = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/pred_object_asset" and payload.__class__.__name__ == "Mesh3D"
            ]
            pred_hand_meshes = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/pred_hand_mesh" and payload.__class__.__name__ == "Mesh3D"
            ]
            pred_hand_joints = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/pred_hand_joints" and payload.__class__.__name__ == "Points3D"
            ]
            pred_object_transforms = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/pred_object_asset" and payload.__class__.__name__ == "Transform3D"
            ]
            hand_joint_correspondence = [
                payload
                for entity, payload in fake_rr.logged
                if entity == "world/hand_joint_correspondence" and payload.__class__.__name__ == "LineStrips3D"
            ]
            self.assertEqual(len(pred_object_assets), 1)
            self.assertEqual(len(pred_object_transforms), 1)
            self.assertEqual(len(pred_hand_meshes), 1)
            self.assertEqual(len(pred_hand_joints), 1)
            self.assertLessEqual(len(hand_joint_correspondence), 1)

            np.testing.assert_allclose(
                pred_hand_meshes[0].vertex_positions,
                np.array(
                    [
                        0.00001, 0.00002, 0.00003,
                        0.00003, 0.00002, 0.00003,
                        0.00001, 0.00004, 0.00003,
                    ],
                    dtype=np.float32,
                ),
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                pred_object_assets[0].vertex_texcoords,
                np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                pred_object_assets[0].vertex_positions.reshape(-1, 3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                rtol=0,
                atol=1e-6,
            )
            self.assertIsNotNone(pred_object_assets[0].albedo_texture)
            np.testing.assert_allclose(
                pred_object_transforms[0].mat3x3,
                np.eye(3, dtype=np.float32),
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                pred_object_transforms[0].translation,
                np.array([-0.90, -0.80, 0.30], dtype=np.float32),
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                pred_hand_joints[0].positions,
                np.array(
                    [
                        [0.00003, 0.00002, 0.00003],
                    ],
                    dtype=np.float32,
                ),
                rtol=0,
                atol=1e-6,
            )
            self.assertLessEqual(len(hand_joint_correspondence), 1)

    def test_export_logs_static_object_mesh_with_static_flag(self) -> None:
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
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "vertices_2d": torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]], dtype=torch.float32),
                    "vertices_2d_valid": torch.tensor([[True, True, True]], dtype=torch.bool),
                },
                "object_multiview": {
                    "template_vertices": torch.tensor(
                        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.zeros(3, dtype=torch.float32),
                    "normalization_scale": torch.tensor([1.5], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
            "object_valid": torch.tensor([[True]], dtype=torch.bool),
            "object_pose_obj2cam": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
            "object_normalization_scale": torch.tensor([1.5], dtype=torch.float32),
            "hand_valid": torch.tensor([True]),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_global_orient_rotmat": aa_to_rotmat(torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
            "hand_scale": torch.tensor([[1.0]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor([[[0.03, 0.02, 0.03]]], dtype=torch.float32),
        }
        pred = {"local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32)}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
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

        static_mesh_logs = [
            kwargs
            for entity, payload, kwargs in fake_rr.log_calls
            if entity == "world/gt_object_asset" and payload.__class__.__name__ == "Mesh3D"
        ]
        self.assertEqual(len(static_mesh_logs), 1)
        self.assertTrue(static_mesh_logs[0].get("static", False))

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
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "vertices_2d": torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]], dtype=torch.float32),
                    "vertices_2d_valid": torch.tensor([[True, True, True]], dtype=torch.bool),
                },
                "object_multiview": {
                    "template_vertices": torch.tensor(
                        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.zeros(3, dtype=torch.float32),
                    "normalization_scale": torch.tensor([1.5], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
            "object_valid": torch.tensor([[True]], dtype=torch.bool),
            "object_pose_obj2cam": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
            "object_normalization_scale": torch.tensor([1.5], dtype=torch.float32),
            "hand_valid": torch.tensor([True]),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_is_right": torch.tensor([False], dtype=torch.bool),
            "hand_global_orient_rotmat": aa_to_rotmat(torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
            "hand_scale": torch.tensor([[1.0]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor([[[2.01, 0.02, 0.03]]], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
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
                    0.00101, 0.00002, 0.00003,
                    0.00201, 0.00002, 0.00003,
                    0.00101, 0.00102, 0.00003,
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(vertices, expected, rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                hand_joints[0].positions,
                np.array([[0.00201, 0.00002, 0.00003]], dtype=np.float32),
                rtol=0,
                atol=1e-6,
            )

    def test_export_clears_pred_hand_mesh_when_owner_index_has_no_view_match(self) -> None:
        fake_rr = _FakeRerun()
        batch = [
            {
                "img": torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
                "depthmap": torch.ones(1, 4, 4, dtype=torch.float32),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "camera_intrinsics": torch.tensor([[[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
                "hand": {
                    "mask": torch.zeros(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([False]),
                    "pose_mano": torch.zeros(1, 48, dtype=torch.float32),
                    "hand_transl": torch.zeros(1, 3, dtype=torch.float32),
                    "joints_2d": torch.zeros(1, 21, 2, dtype=torch.float32),
                    "mano_betas": torch.zeros(1, 10, dtype=torch.float32),
                    "mano_side": ["right"],
                },
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.zeros(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([False]),
                    "pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "vertices_2d": torch.zeros(1, 3, 2, dtype=torch.float32),
                    "vertices_2d_valid": torch.zeros(1, 3, dtype=torch.bool),
                },
                "object_multiview": {
                    "normalization_scale": torch.tensor([1.0], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "hand_valid": torch.zeros((0,), dtype=torch.bool),
            "hand_owner_index": torch.zeros((0, 3), dtype=torch.long),
            "object_multiview": batch[0]["object_multiview"],
        }
        gt["local_points"][0, 0, ..., 2] = 2.0
        pred = {
            "local_points": gt["local_points"].clone(),
            "pred_hand_vertices": torch.tensor(
                [[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]],
                dtype=torch.float32,
            ),
            "hand_owner_index": torch.tensor([[0, 1, 0]], dtype=torch.long),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
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

        clear_logs = [
            payload
            for entity, payload in fake_rr.logged
            if entity == "world/pred_hand_mesh" and payload.__class__.__name__ == "Clear"
        ]
        self.assertTrue(clear_logs)

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
                "object": {
                    "grasped_object_id": torch.tensor([1]),
                    "mask": torch.ones(1, 4, 4, dtype=torch.float32),
                    "valid": torch.tensor([True]),
                    "pose_obj2cam": torch.eye(4).unsqueeze(0),
                    "vertices_2d": torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]], dtype=torch.float32),
                    "vertices_2d_valid": torch.tensor([[True, True, True]], dtype=torch.bool),
                },
                "object_multiview": {
                    "template_vertices": torch.tensor(
                        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "normalization_center": torch.zeros(3, dtype=torch.float32),
                    "normalization_scale": torch.tensor([1.5], dtype=torch.float32),
                },
            }
        ]
        gt = {
            "scene_scale": torch.tensor([1.0], dtype=torch.float32),
            "local_points": torch.zeros(1, 1, 4, 4, 3, dtype=torch.float32),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "norm_factor": torch.tensor([1.0], dtype=torch.float32),
            "object_valid": torch.tensor([[True]], dtype=torch.bool),
            "object_pose_obj2cam": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
            "object_normalization_scale": torch.tensor([1.5], dtype=torch.float32),
            "hand_valid": torch.tensor([True]),
            "hand_owner_index": torch.tensor([[0, 0, 0]], dtype=torch.long),
            "hand_is_right": torch.tensor([False], dtype=torch.bool),
            "hand_global_orient_rotmat": aa_to_rotmat(torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)).view(1, 1, 3, 3),
            "hand_pose_rotmat": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 15, 1, 1),
            "hand_mano_betas": torch.zeros(1, 10, dtype=torch.float32),
            "hand_transl": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float32),
            "hand_scale": torch.tensor([[1.0]], dtype=torch.float32),
            "hand_joints_3d": torch.tensor([[[2.01, 0.02, 0.03]]], dtype=torch.float32),
        }
        gt["local_points"][0, 0, ..., 2] = 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sample.rrd"
            with patch("pi3.visualization.pi3x_rerun_export._load_rerun", return_value=fake_rr), patch(
                "pi3.visualization.pi3x_rerun_export._build_visualization_mano_layer",
                return_value=_DummyLeftManoLayer(),
            ), patch(
                "pi3.visualization.pi3x_rerun_export._load_textured_object_mesh",
                side_effect=_fake_load_textured_object_mesh,
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
                        0.00101, 0.00002, 0.00003,
                        0.00201, 0.00002, 0.00003,
                        0.00101, 0.00102, 0.00003,
                    ],
                    dtype=np.float32,
                ),
                rtol=0,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
