from __future__ import annotations

import unittest

import torch

from pi3.models.layers.hand_token_adapter import HandTokenAdapter
from pi3.models.layers.object_query_adapter import ObjectQueryAdapter


class _FakePatchEncoder(torch.nn.Module):
    def __init__(self, token_dim: int = 8, patch_hw: tuple[int, int] = (16, 12)):
        super().__init__()
        self.token_dim = token_dim
        self.patch_hw = patch_hw
        self.last_imgs = None

    def forward(self, imgs, is_training=True):
        del is_training
        self.last_imgs = imgs.detach().cpu()
        batch = imgs.shape[0]
        num_tokens = self.patch_hw[0] * self.patch_hw[1]
        tokens = torch.arange(batch * num_tokens * self.token_dim, dtype=imgs.dtype, device=imgs.device)
        tokens = tokens.reshape(batch, num_tokens, self.token_dim)
        return {"x_norm_patchtokens": tokens}


class LocalTokenAdapterTests(unittest.TestCase):
    def test_hand_token_adapter_outputs_dense_local_tokens(self) -> None:
        encoder = _FakePatchEncoder(token_dim=8, patch_hw=(16, 12))
        adapter = HandTokenAdapter(token_dim=8, patch_size=14)
        imgs = torch.rand(2, 3, 3, 32, 32)
        hand_masks = torch.zeros(2, 3, 32, 32)
        hand_masks[0, 1, 4:12, 5:13] = 1.0
        hand_is_right = torch.tensor([[True, False, False], [False, False, False]], dtype=torch.bool)

        out = adapter(imgs, hand_masks, hand_is_right, encoder)

        self.assertEqual(tuple(out["hand_tokens"].shape), (2, 3, 192, 8))
        self.assertEqual(tuple(out["hand_valid_mask"].shape), (2, 3))
        self.assertTrue(torch.equal(out["hand_valid_mask"], torch.tensor([[False, True, False], [False, False, False]])))
        self.assertEqual(tuple(out["crop_boxes"].shape), (2, 3, 4))

    def test_object_query_adapter_outputs_dense_local_tokens(self) -> None:
        encoder = _FakePatchEncoder(token_dim=8, patch_hw=(16, 12))
        adapter = ObjectQueryAdapter(token_dim=8, patch_size=14)
        imgs = torch.rand(1, 2, 3, 32, 32)
        object_masks = torch.zeros(1, 2, 32, 32)
        object_masks[0, 0, 2:18, 6:20] = 1.0
        object_valid = torch.tensor([[True, False]], dtype=torch.bool)

        out = adapter(imgs, object_masks, object_valid, encoder)

        self.assertEqual(tuple(out["object_tokens"].shape), (1, 2, 192, 8))
        self.assertEqual(tuple(out["object_valid_mask"].shape), (1, 2))
        self.assertTrue(torch.equal(out["object_valid_mask"], object_valid))
        self.assertEqual(tuple(out["crop_boxes"].shape), (1, 2, 4))


if __name__ == "__main__":
    unittest.main()
