from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pi3.models.hamer.hand_mano_head import HandMANOHead
from pi3.models.layers.hand_pose_head import HandPoseHead


class _FakeMANO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.faces = torch.zeros((0, 3), dtype=torch.long)

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del hand_pose, betas, kwargs
        batch = global_orient.shape[0]
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=global_orient.dtype, device=global_orient.device)
        joints = torch.ones((batch, 21, 3), dtype=global_orient.dtype, device=global_orient.device) + th_trans.unsqueeze(1)
        vertices = torch.ones((batch, 778, 3), dtype=global_orient.dtype, device=global_orient.device) + th_trans.unsqueeze(1)
        return type("FakeMANOOutput", (), {"joints": joints, "vertices": vertices})()


class _AsymmetricMANO(torch.nn.Module):
    def __init__(self, sign: float = 1.0):
        super().__init__()
        self.faces = torch.zeros((0, 3), dtype=torch.long)
        self.sign = float(sign)

    def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
        del global_orient, hand_pose, betas, kwargs
        batch = th_trans.shape[0]
        joints = torch.tensor(
            [[[self.sign, 0.0, 0.0]] + [[2.0 * self.sign, 0.0, 0.0]] * 20],
            dtype=th_trans.dtype,
            device=th_trans.device,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        vertices = torch.tensor(
            [[[self.sign, 0.0, 0.0]] + [[2.0 * self.sign, 0.0, 0.0]] * 777],
            dtype=th_trans.dtype,
            device=th_trans.device,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        return type("FakeMANOOutput", (), {"joints": joints, "vertices": vertices})()


class Pi3XHandModuleTests(unittest.TestCase):
    def _make_cfg(self):
        class _Node(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        cfg = _Node()
        cfg.MODEL = _Node()
        cfg.MODEL.IMAGE_SIZE = 224
        cfg.MODEL.MANO_HEAD = _Node()
        cfg.MODEL.MANO_HEAD.JOINT_REP = "6d"
        cfg.EXTRA = _Node()
        cfg.EXTRA.FOCAL_LENGTH = 5000
        cfg.MANO = _Node()
        cfg.MANO.NUM_HAND_JOINTS = 15
        mean_params = Path(tempfile.gettempdir()) / "pi3x_hand_mano_mean_params.npz"
        if not mean_params.is_file():
            np.savez(
                mean_params,
                pose=np.zeros((96,), dtype=np.float32),
                shape=np.zeros((10,), dtype=np.float32),
                cam=np.zeros((3,), dtype=np.float32),
            )
        cfg.MANO.MEAN_PARAMS = str(mean_params)
        return cfg

    def test_hand_pose_head_outputs_expected_shapes(self) -> None:
        head = HandPoseHead(self._make_cfg(), in_dim=16, hidden_dim=8, patch_h=1, patch_w=3)
        hand_tokens = torch.randn(3, 3, 16)

        out = head(hand_tokens)

        self.assertEqual(tuple(out["pred_hand_pose_6d"].shape), (3, 96))
        self.assertEqual(tuple(out["pred_hand_pose"].shape), (3, 15, 3, 3))
        self.assertEqual(tuple(out["pred_hand_global_orient"].shape), (3, 1, 3, 3))

    def test_hand_pose_head_starts_from_zero_residual(self) -> None:
        head = HandPoseHead(self._make_cfg(), in_dim=16, hidden_dim=8, patch_h=1, patch_w=3)

        self.assertTrue(torch.allclose(head.decpose.weight, torch.zeros_like(head.decpose.weight)))
        self.assertTrue(torch.allclose(head.decpose.bias, torch.zeros_like(head.decpose.bias)))

    def test_hand_mano_head_outputs_expected_shapes(self) -> None:
        head = HandMANOHead(self._make_cfg())
        batch = 3
        hand_pose = torch.eye(3).view(1, 1, 3, 3).repeat(batch, 15, 1, 1)
        global_orient = torch.eye(3).view(1, 1, 3, 3).repeat(batch, 1, 1, 1)
        hand_betas = torch.zeros((batch, 10))
        hand_is_right = torch.tensor([True, False, True], dtype=torch.bool)
        transl = torch.randn(batch, 3)
        scale = torch.ones(batch, 1)

        out = head(
            hand_pose=hand_pose,
            global_orient=global_orient,
            hand_betas=hand_betas,
            hand_is_right=hand_is_right,
            transl=transl,
            scale=scale,
        )

        self.assertEqual(tuple(out["pred_mano_params"]["global_orient"].shape), (3, 1, 3, 3))
        self.assertEqual(tuple(out["pred_mano_params"]["hand_pose"].shape), (3, 15, 3, 3))
        self.assertEqual(tuple(out["pred_mano_params"]["betas"].shape), (3, 10))
        self.assertTrue(torch.allclose(out["pred_hand_mano_betas"], hand_betas))

    def test_hand_mano_head_emits_geometry_when_mano_layer_is_attached(self) -> None:
        head = HandMANOHead(self._make_cfg(), mano_layer=_FakeMANO())
        batch = 3
        hand_pose = torch.eye(3).view(1, 1, 3, 3).repeat(batch, 15, 1, 1)
        global_orient = torch.eye(3).view(1, 1, 3, 3).repeat(batch, 1, 1, 1)
        hand_betas = torch.zeros((batch, 10))
        transl = torch.randn(batch, 3)
        scale = torch.ones(batch, 1)

        out = head(
            hand_pose=hand_pose,
            global_orient=global_orient,
            hand_betas=hand_betas,
            hand_is_right=torch.tensor([True, True, True], dtype=torch.bool),
            transl=transl,
            scale=scale,
        )

        self.assertIn("pred_hand_vertices", out)
        self.assertIn("pred_hand_joints_3d", out)
        self.assertIn("pred_hand_vertices_local", out)
        self.assertIn("pred_hand_joints_local", out)
        self.assertEqual(tuple(out["pred_hand_vertices"].shape), (3, 778, 3))
        self.assertEqual(tuple(out["pred_hand_joints_3d"].shape), (3, 21, 3))

    def test_hand_mano_head_local_geometry_ignores_global_translation_and_scale_heads(self) -> None:
        head = HandMANOHead(self._make_cfg(), mano_layer=_AsymmetricMANO(sign=1.0))
        hand_pose = torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1)
        global_orient = torch.eye(3).view(1, 1, 3, 3)
        hand_betas = torch.zeros((1, 10))
        transl = torch.tensor([[3.0, 0.0, 0.0]])
        scale = torch.exp(torch.ones((1, 1)))

        out = head(
            hand_pose=hand_pose,
            global_orient=global_orient,
            hand_betas=hand_betas,
            hand_is_right=torch.tensor([True], dtype=torch.bool),
            transl=transl,
            scale=scale,
        )

        self.assertTrue(torch.allclose(out["pred_hand_joints_local"][0, 0], torch.tensor([0.001, 0.0, 0.0])))
        self.assertTrue(torch.allclose(out["pred_hand_vertices_local"][0, 1], torch.tensor([0.002, 0.0, 0.0])))

    def test_hand_mano_head_uses_left_mano_layer_when_hand_is_left(self) -> None:
        mano_layers = torch.nn.ModuleDict({
            "right": _AsymmetricMANO(sign=1.0),
            "left": _AsymmetricMANO(sign=-1.0),
        })
        head = HandMANOHead(self._make_cfg(), mano_layer=mano_layers)
        hand_pose = torch.eye(3).view(1, 1, 3, 3).repeat(1, 15, 1, 1)
        global_orient = torch.eye(3).view(1, 1, 3, 3)
        hand_betas = torch.zeros((1, 10))
        transl = torch.zeros((1, 3))
        scale = torch.ones((1, 1))

        out = head(
            hand_pose=hand_pose,
            global_orient=global_orient,
            hand_betas=hand_betas,
            hand_is_right=torch.tensor([False], dtype=torch.bool),
            transl=transl,
            scale=scale,
        )

        self.assertTrue(torch.allclose(out["pred_hand_vertices"][0, 0], torch.tensor([-0.001, 0.0, 0.0])))
        self.assertTrue(torch.allclose(out["pred_hand_joints_3d"][0, 0], torch.tensor([-0.001, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
