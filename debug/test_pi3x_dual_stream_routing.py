from __future__ import annotations

import unittest

import torch

from pi3.models.layers.dual_stream_routing import (
    build_scene_image_attn_mask,
    flatten_object_global_memory,
    flatten_valid_hand_queries,
)


class Pi3XDualStreamRoutingTests(unittest.TestCase):
    def test_build_scene_image_attn_mask_blocks_scene_queries_from_reading_hand_and_object_queries(self) -> None:
        mask = build_scene_image_attn_mask(num_scene_tokens=5, num_hand_queries=2, num_object_queries=1)

        self.assertEqual(tuple(mask.shape), (8, 8))
        self.assertTrue(mask[:5, :5].all())
        self.assertFalse(mask[:5, 5:].any())
        self.assertTrue(mask[5:, :5].all())
        self.assertTrue(mask[5:, 5:].all())

    def test_flatten_valid_hand_queries_drops_invalid_slots_and_returns_owner_index(self) -> None:
        hand_queries = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
        hand_valid = torch.tensor(
            [
                [[True, False], [False, False], [True, True]],
                [[False, True], [False, False], [False, False]],
            ],
            dtype=torch.bool,
        )

        flat, owner = flatten_valid_hand_queries(hand_queries, hand_valid)

        self.assertEqual(tuple(flat.shape), (4, 4))
        self.assertEqual(tuple(owner.shape), (4, 3))
        self.assertTrue(torch.equal(owner[0], torch.tensor([0, 0, 0])))
        self.assertTrue(torch.equal(flat[0], hand_queries[0, 0, 0]))
        self.assertTrue(torch.equal(flat[-1], hand_queries[1, 0, 1]))

    def test_flatten_object_global_memory_skips_register_tokens(self) -> None:
        object_tokens = torch.randn(1, 8, 261, 16)

        memory = flatten_object_global_memory(object_tokens, patch_start_idx=5)

        self.assertEqual(tuple(memory.shape), (1, 8 * 256, 16))
        self.assertTrue(torch.equal(memory[:, :256], object_tokens[:, 0, 5:, :]))


if __name__ == "__main__":
    unittest.main()
