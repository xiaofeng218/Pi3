from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from pi3.models.layers.object_pose_head import ObjectPoseHead
from pi3.models.layers.object_query_adapter import ObjectQueryAdapter


class _SpyEncoder(nn.Module):
    def __init__(self, token_dim: int = 8, patch_tokens: int = 3) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.patch_tokens = patch_tokens
        self.calls: list[torch.Tensor] = []

    def forward(self, imgs, is_training=True):
        del is_training
        self.calls.append(imgs.detach().clone())
        batch = imgs.shape[0]
        tokens = torch.arange(batch * self.patch_tokens * self.token_dim, dtype=imgs.dtype, device=imgs.device)
        tokens = tokens.reshape(batch, self.patch_tokens, self.token_dim)
        return {"x_norm_patchtokens": tokens}


class Pi3XObjectModuleTests(unittest.TestCase):
    def test_object_query_adapter_outputs_dense_local_tokens(self) -> None:
        adapter = ObjectQueryAdapter(token_dim=8, patch_size=4, crop_hw=(4, 12))
        encoder = _SpyEncoder(token_dim=8, patch_tokens=3)
        imgs = torch.randn(1, 2, 3, 8, 8)
        grasped_object_mask = torch.zeros(1, 2, 8, 8)
        grasped_object_mask[0, 0, 2:6, 1:7] = 1.0
        grasped_object_valid = torch.tensor([[True, False]], dtype=torch.bool)

        out = adapter(imgs, grasped_object_mask, grasped_object_valid, encoder)

        self.assertEqual(tuple(out["object_tokens"].shape), (1, 2, 3, 8))
        self.assertEqual(tuple(out["object_valid_mask"].shape), (1, 2))
        self.assertEqual(tuple(out["crop_boxes"].shape), (1, 2, 4))
        self.assertEqual(len(encoder.calls), 1)
        self.assertTrue(torch.equal(out["object_valid_mask"], grasped_object_valid))

    def test_object_query_adapter_rejects_mismatched_batch_view_shapes(self) -> None:
        adapter = ObjectQueryAdapter(token_dim=8, patch_size=4)
        encoder = _SpyEncoder(token_dim=8, patch_tokens=3)
        imgs = torch.randn(1, 2, 3, 8, 8)
        grasped_object_mask = torch.zeros(1, 1, 8, 8)
        grasped_object_valid = torch.tensor([[True, False]], dtype=torch.bool)

        with self.assertRaises(ValueError):
            adapter(imgs, grasped_object_mask, grasped_object_valid, encoder)

    def test_object_pose_head_outputs_rot_trans_and_scale(self) -> None:
        head = ObjectPoseHead(in_dim=16, hidden_dim=8, patch_h=1, patch_w=3)
        object_query_feat = torch.tensor(
            [
                [
                    [[0.0] * 16, [1.0] * 16, [2.0] * 16],
                    [[1.0] * 16, [2.0] * 16, [3.0] * 16],
                    [[2.0] * 16, [3.0] * 16, [4.0] * 16],
                ],
                [
                    [[-1.0] * 16, [0.5] * 16, [3.0] * 16],
                    [[0.5] * 16, [3.0] * 16, [1.5] * 16],
                    [[3.0] * 16, [1.5] * 16, [0.0] * 16],
                ],
            ]
        )

        out = head(object_query_feat)

        self.assertEqual(tuple(out["rot6d"].shape), (2, 3, 6))
        self.assertEqual(tuple(out["trans_dir"].shape), (2, 3, 3))
        self.assertEqual(tuple(out["trans_log_scale"].shape), (2, 3, 1))
        self.assertEqual(tuple(out["trans_scale"].shape), (2, 3, 1))
        self.assertEqual(tuple(out["trans"].shape), (2, 3, 3))
        self.assertEqual(tuple(out["log_scale"].shape), (2, 3, 1))
        self.assertEqual(tuple(out["scale"].shape), (2, 3, 1))
        self.assertTrue(torch.all(out["scale"] > 0))
        self.assertTrue(torch.allclose(out["scale"], torch.exp(out["log_scale"]), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(out["trans_scale"], torch.exp(out["trans_log_scale"]), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(out["trans"], out["trans_dir"] * out["trans_scale"], atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
