from __future__ import annotations

import unittest

import torch

from pi3.models.layers.object_pose_head import ObjectPoseHead
from pi3.models.layers.object_query_adapter import ObjectQueryAdapter


class Pi3XObjectModuleTests(unittest.TestCase):
    def test_object_query_adapter_pools_valid_masked_patch_and_emits_patch_center(self) -> None:
        adapter = ObjectQueryAdapter(token_dim=8, patch_size=4)
        rgb_patch_tokens = torch.tensor(
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        grasped_object_mask = torch.zeros(1, 1, 8, 8)
        grasped_object_mask[0, 0, 4:, 4:] = 1.0
        grasped_object_valid = torch.tensor([[True]], dtype=torch.bool)

        out = adapter(rgb_patch_tokens, grasped_object_mask, grasped_object_valid, image_hw=(8, 8))

        self.assertEqual(tuple(out["object_query"].shape), (1, 1, 1, 8))
        self.assertEqual(tuple(out["object_query_pos"].shape), (1, 1, 1, 2))
        self.assertTrue(torch.equal(out["object_valid"], grasped_object_valid))
        self.assertTrue(
            torch.equal(
                out["object_query"][0, 0, 0],
                torch.tensor([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]),
            )
        )
        self.assertTrue(torch.equal(out["object_query_pos"][0, 0, 0], torch.tensor([1, 1], dtype=torch.long)))

    def test_object_query_adapter_falls_back_consistently_for_invalid_or_empty_masks(self) -> None:
        adapter = ObjectQueryAdapter(token_dim=8, patch_size=4)
        rgb_patch_tokens = torch.tensor(
            [
                [
                    [
                        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        empty_mask = torch.zeros(1, 1, 8, 8)
        invalid = torch.tensor([[False]], dtype=torch.bool)
        valid_but_empty = torch.tensor([[True]], dtype=torch.bool)

        invalid_out = adapter(rgb_patch_tokens, empty_mask, invalid, image_hw=(8, 8))
        empty_valid_out = adapter(rgb_patch_tokens, empty_mask, valid_but_empty, image_hw=(8, 8))

        self.assertEqual(tuple(invalid_out["object_query"].shape), (1, 1, 1, 8))
        self.assertTrue(torch.equal(invalid_out["object_query"], empty_valid_out["object_query"]))
        self.assertTrue(torch.equal(invalid_out["object_query_pos"], empty_valid_out["object_query_pos"]))
        self.assertTrue(torch.equal(invalid_out["object_valid"], invalid))
        self.assertTrue(torch.equal(empty_valid_out["object_valid"], valid_but_empty))

    def test_object_pose_head_outputs_rot_trans_and_scale(self) -> None:
        head = ObjectPoseHead(in_dim=16, hidden_dim=8)
        object_query_feat = torch.tensor(
            [
                [
                    [0.0] * 16,
                    [1.0] * 16,
                    [2.0] * 16,
                ],
                [
                    [-1.0] * 16,
                    [0.5] * 16,
                    [3.0] * 16,
                ],
            ]
        )

        out = head(object_query_feat)

        self.assertEqual(tuple(out["rot6d"].shape), (2, 3, 6))
        self.assertEqual(tuple(out["trans"].shape), (2, 3, 3))
        self.assertEqual(tuple(out["log_scale"].shape), (2, 3, 1))
        self.assertEqual(tuple(out["scale"].shape), (2, 3, 1))
        self.assertTrue(torch.all(out["scale"] > 0))
        self.assertTrue(torch.allclose(out["scale"], torch.exp(out["log_scale"])))


if __name__ == "__main__":
    unittest.main()
