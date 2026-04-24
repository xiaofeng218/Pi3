from __future__ import annotations

import unittest

import torch

from pi3.models.layers.object_pose_head import ObjectPoseHead
from pi3.models.layers.object_query_adapter import ObjectQueryAdapter


class Pi3XObjectModuleTests(unittest.TestCase):
    def test_object_query_adapter_pools_valid_masks_and_emits_query_pos(self) -> None:
        adapter = ObjectQueryAdapter(token_dim=8, patch_size=4)
        rgb_patch_tokens = torch.randn(1, 2, 4, 8)
        grasped_object_mask = torch.zeros(1, 2, 8, 8)
        grasped_object_mask[0, 1, 4:, 4:] = 1.0
        grasped_object_valid = torch.tensor([[False, True]], dtype=torch.bool)

        out = adapter(rgb_patch_tokens, grasped_object_mask, grasped_object_valid, image_hw=(8, 8))

        self.assertEqual(tuple(out["object_query"].shape), (1, 2, 1, 8))
        self.assertEqual(tuple(out["object_query_pos"].shape), (1, 2, 1, 2))
        self.assertTrue(torch.equal(out["object_valid"], grasped_object_valid))
        self.assertFalse(torch.allclose(out["object_query"][0, 0, 0], out["object_query"][0, 1, 0]))

    def test_object_query_adapter_uses_empty_query_for_invalid_or_empty_masks(self) -> None:
        adapter = ObjectQueryAdapter(token_dim=8, patch_size=4)
        rgb_patch_tokens = torch.randn(1, 1, 4, 8)
        grasped_object_mask = torch.zeros(1, 1, 8, 8)
        grasped_object_valid = torch.tensor([[False]], dtype=torch.bool)

        out = adapter(rgb_patch_tokens, grasped_object_mask, grasped_object_valid, image_hw=(8, 8))

        self.assertEqual(tuple(out["object_query"].shape), (1, 1, 1, 8))
        self.assertTrue(torch.allclose(out["object_query"][0, 0, 0], adapter.empty_object_query.detach().view(8)))

    def test_object_pose_head_outputs_rot_trans_and_scale(self) -> None:
        head = ObjectPoseHead(in_dim=16, hidden_dim=8)
        object_query_feat = torch.randn(2, 3, 16)

        out = head(object_query_feat)

        self.assertEqual(tuple(out["rot6d"].shape), (2, 3, 6))
        self.assertEqual(tuple(out["trans"].shape), (2, 3, 3))
        self.assertEqual(tuple(out["log_scale"].shape), (2, 3, 1))
        self.assertEqual(tuple(out["scale"].shape), (2, 3, 1))
        self.assertTrue(torch.all(out["scale"] > 0))


if __name__ == "__main__":
    unittest.main()
