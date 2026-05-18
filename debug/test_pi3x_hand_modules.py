from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pi3.models.hamer.hand_mano_head import HandMANOHead
from pi3.models.layers.hand_token_adapter import HandTokenAdapter


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
        del hand_pose, betas, kwargs
        batch = global_orient.shape[0]
        if th_trans is None:
            th_trans = torch.zeros((batch, 3), dtype=torch.float32, device=global_orient.device)
        joints = torch.tensor(
            [[[self.sign, 0.0, 0.0]] + [[2.0 * self.sign, 0.0, 0.0]] * 20],
            dtype=global_orient.dtype,
            device=global_orient.device,
        ).repeat(batch, 1, 1) + th_trans.unsqueeze(1)
        vertices = torch.tensor(
            [[[self.sign, 0.0, 0.0]] + [[2.0 * self.sign, 0.0, 0.0]] * 777],
            dtype=global_orient.dtype,
            device=global_orient.device,
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

    def test_hand_token_adapter_returns_single_dummy_dense_slot_without_hands(self) -> None:
        adapter = HandTokenAdapter(token_dim=8, patch_size=4)
        rgb_patch_tokens = torch.randn(2, 3, 4, 8)
        hand_queries = torch.zeros(0, 8)
        hand_masks = torch.zeros(0, 8, 8)
        owner_index = torch.zeros(0, 3, dtype=torch.long)
        hand_is_right = torch.zeros(0, dtype=torch.bool)

        out = adapter(
            rgb_patch_tokens,
            hand_queries,
            hand_masks,
            owner_index,
            hand_is_right,
            image_hw=(8, 8),
        )

        self.assertEqual(tuple(out["dense_tokens"].shape), (2, 3, 1, 8))
        self.assertEqual(tuple(out["dense_pos"].shape), (2, 3, 1, 2))
        self.assertEqual(tuple(out["dense_valid_mask"].shape), (2, 3, 1))
        self.assertFalse(bool(out["dense_valid_mask"].any()))
        self.assertEqual(out["num_hand_tokens"], 1)
        self.assertEqual(tuple(out["sparse_tokens"].shape), (0, 8))

    def test_hand_token_adapter_builds_single_slot_dense_tokens(self) -> None:
        adapter = HandTokenAdapter(token_dim=8, patch_size=4)
        rgb_patch_tokens = torch.randn(1, 2, 4, 8)
        hand_queries = torch.randn(1, 8)
        hand_masks = torch.zeros(1, 8, 8)
        hand_masks[0, :4, :4] = 1.0
        owner_index = torch.tensor([[0, 1, 0]], dtype=torch.long)
        hand_is_right = torch.tensor([True], dtype=torch.bool)

        out = adapter(
            rgb_patch_tokens,
            hand_queries,
            hand_masks,
            owner_index,
            hand_is_right,
            image_hw=(8, 8),
        )

        self.assertEqual(tuple(out["dense_tokens"].shape), (1, 2, 1, 8))
        self.assertEqual(tuple(out["dense_pos"].shape), (1, 2, 1, 2))
        self.assertEqual(tuple(out["dense_valid_mask"].shape), (1, 2, 1))
        self.assertEqual(out["num_hand_tokens"], 1)
        self.assertTrue(bool(out["dense_valid_mask"][0, 1, 0]))

    def test_hand_token_adapter_builds_two_slots_and_preserves_empty_token(self) -> None:
        adapter = HandTokenAdapter(token_dim=8, patch_size=4)
        rgb_patch_tokens = torch.randn(1, 2, 4, 8)
        hand_queries = torch.randn(2, 8)
        hand_masks = torch.zeros(2, 8, 8)
        hand_masks[0, :4, :4] = 1.0
        hand_masks[1, 4:, 4:] = 1.0
        owner_index = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
        hand_is_right = torch.tensor([True, False], dtype=torch.bool)

        out = adapter(
            rgb_patch_tokens,
            hand_queries,
            hand_masks,
            owner_index,
            hand_is_right,
            image_hw=(8, 8),
        )

        self.assertEqual(tuple(out["dense_tokens"].shape), (1, 2, 2, 8))
        self.assertEqual(out["num_hand_tokens"], 2)
        self.assertTrue(torch.equal(out["dense_valid_mask"][0, 0], torch.tensor([True, True])))
        self.assertTrue(torch.equal(out["dense_valid_mask"][0, 1], torch.tensor([False, False])))
        empty = adapter.empty_hand_token.detach().view(8)
        self.assertTrue(torch.allclose(out["dense_tokens"][0, 1, 0], empty))
        self.assertTrue(torch.allclose(out["dense_tokens"][0, 1, 1], empty))

    def test_hand_token_adapter_rejects_owner_slot_above_one(self) -> None:
        adapter = HandTokenAdapter(token_dim=8, patch_size=4)
        rgb_patch_tokens = torch.randn(1, 1, 4, 8)
        hand_queries = torch.randn(1, 8)
        hand_masks = torch.ones(1, 8, 8)
        owner_index = torch.tensor([[0, 0, 2]], dtype=torch.long)
        hand_is_right = torch.tensor([True], dtype=torch.bool)

        with self.assertRaises(ValueError):
            adapter(
                rgb_patch_tokens,
                hand_queries,
                hand_masks,
                owner_index,
                hand_is_right,
                image_hw=(8, 8),
            )

    def test_hand_mano_head_outputs_expected_shapes(self) -> None:
        head = HandMANOHead(self._make_cfg(), in_dim=16, hidden_dim=8)
        hand_tokens = torch.randn(3, 16)

        out = head(hand_tokens)

        self.assertEqual(tuple(out["pred_hand_transl_dir"].shape), (3, 3))
        self.assertEqual(tuple(out["pred_hand_transl_log_scale"].shape), (3, 1))
        self.assertEqual(tuple(out["pred_hand_transl_scale"].shape), (3, 1))
        self.assertEqual(tuple(out["pred_hand_transl"].shape), (3, 3))
        self.assertEqual(tuple(out["pred_hand_log_scale"].shape), (3, 1))
        self.assertEqual(tuple(out["pred_hand_scale"].shape), (3, 1))
        self.assertTrue(torch.allclose(out["pred_hand_transl_scale"], torch.exp(out["pred_hand_transl_log_scale"])))
        self.assertTrue(torch.allclose(out["pred_hand_scale"], torch.exp(out["pred_hand_log_scale"])))
        self.assertTrue(torch.allclose(
            out["pred_hand_transl"],
            out["pred_hand_transl_dir"] * out["pred_hand_transl_scale"],
        ))
        self.assertEqual(tuple(out["pred_hand_mano_betas"].shape), (3, 10))
        self.assertEqual(tuple(out["pred_mano_params"]["global_orient"].shape), (3, 1, 3, 3))
        self.assertEqual(tuple(out["pred_mano_params"]["hand_pose"].shape), (3, 15, 3, 3))
        self.assertEqual(tuple(out["pred_mano_params"]["betas"].shape), (3, 10))

    def test_hand_mano_head_pose_and_shape_heads_start_from_zero_residual(self) -> None:
        head = HandMANOHead(self._make_cfg(), in_dim=16, hidden_dim=8)

        self.assertTrue(torch.allclose(head.decpose.weight, torch.zeros_like(head.decpose.weight)))
        self.assertTrue(torch.allclose(head.decpose.bias, torch.zeros_like(head.decpose.bias)))
        self.assertTrue(torch.allclose(head.decshape.weight, torch.zeros_like(head.decshape.weight)))
        self.assertTrue(torch.allclose(head.decshape.bias, torch.zeros_like(head.decshape.bias)))

    def test_hand_mano_head_emits_geometry_when_mano_layer_is_attached(self) -> None:
        head = HandMANOHead(self._make_cfg(), in_dim=16, hidden_dim=8, mano_layer=_FakeMANO())
        hand_tokens = torch.randn(3, 16)

        out = head(hand_tokens)

        self.assertIn("pred_hand_vertices", out)
        self.assertIn("pred_hand_joints_3d", out)
        self.assertIn("pred_hand_vertices_local", out)
        self.assertIn("pred_hand_joints_local", out)
        self.assertEqual(tuple(out["pred_hand_vertices"].shape), (3, 778, 3))
        self.assertEqual(tuple(out["pred_hand_joints_3d"].shape), (3, 21, 3))
        self.assertEqual(tuple(out["pred_hand_vertices_local"].shape), (3, 778, 3))
        self.assertEqual(tuple(out["pred_hand_joints_local"].shape), (3, 21, 3))
        self.assertTrue(torch.allclose(
            out["pred_hand_vertices"][:, 0],
            out["pred_hand_joints_3d"][:, 0],
        ))
        self.assertTrue(torch.allclose(
            out["pred_hand_vertices"][:, 0],
            out["pred_hand_vertices"][:, 1],
        ))
        self.assertTrue(torch.allclose(
            out["pred_hand_vertices_local"][:, 0],
            out["pred_hand_joints_local"][:, 0],
        ))

    def test_hand_mano_head_local_geometry_ignores_global_translation_and_scale_heads(self) -> None:
        head = HandMANOHead(self._make_cfg(), in_dim=16, hidden_dim=8, mano_layer=_AsymmetricMANO(sign=1.0))
        with torch.no_grad():
            for module in (head.project, head.decpose, head.decshape, head.dectransl_dir, head.dectransl_scale, head.decscale):
                for param in module.parameters():
                    param.zero_()
            head.dectransl_scale.bias.fill_(2.0)
            head.decscale.bias.fill_(1.0)

        hand_tokens = torch.zeros(1, 16)
        out = head(hand_tokens, hand_is_right=torch.tensor([True], dtype=torch.bool))

        self.assertTrue(torch.allclose(out["pred_hand_joints_local"][0, 0], torch.tensor([0.001, 0.0, 0.0])))
        self.assertTrue(torch.allclose(out["pred_hand_vertices_local"][0, 1], torch.tensor([0.002, 0.0, 0.0])))

    def test_hand_mano_head_uses_left_mano_layer_when_hand_is_left(self) -> None:
        mano_layers = torch.nn.ModuleDict({
            "right": _AsymmetricMANO(sign=1.0),
            "left": _AsymmetricMANO(sign=-1.0),
        })
        head = HandMANOHead(self._make_cfg(), in_dim=16, hidden_dim=8, mano_layer=mano_layers)
        with torch.no_grad():
            for module in (head.project, head.decpose, head.decshape, head.dectransl_dir, head.dectransl_scale, head.decscale):
                for param in module.parameters():
                    param.zero_()

        hand_tokens = torch.zeros(1, 16)
        out = head(hand_tokens, hand_is_right=torch.tensor([False], dtype=torch.bool))

        self.assertTrue(torch.allclose(out["pred_hand_vertices"][0, 0], torch.tensor([-0.001, 0.0, 0.0])))
        self.assertTrue(torch.allclose(out["pred_hand_joints_3d"][0, 0], torch.tensor([-0.001, 0.0, 0.0])))
        self.assertTrue(torch.allclose(out["pred_hand_vertices"][0, 1], torch.tensor([-0.002, 0.0, 0.0])))
        self.assertTrue(torch.allclose(out["pred_hand_joints_3d"][0, 1], torch.tensor([-0.002, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
