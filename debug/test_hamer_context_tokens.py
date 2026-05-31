from __future__ import annotations

import unittest

import torch

from pi3.models.hamer.encoder import HaMeREncoder


class _FakeContextBackbone(torch.nn.Module):
    def __init__(self, dim: int = 8, num_tokens: int = 6, query_dim: int = 1024):
        super().__init__()
        self.context_dim = dim
        self.output_dim = query_dim
        self.num_tokens = num_tokens
        self.last_crops = None
        self.last_hand_is_right = None

    def forward(self, crops, hand_is_right=None):
        self.last_crops = crops.detach().cpu()
        self.last_hand_is_right = None if hand_is_right is None else hand_is_right.detach().cpu()
        batch = crops.shape[0]
        tokens = torch.arange(batch * self.num_tokens * self.context_dim, dtype=crops.dtype, device=crops.device)
        tokens = tokens.reshape(batch, self.num_tokens, self.context_dim)
        betas = torch.arange(batch * 10, dtype=crops.dtype, device=crops.device).reshape(batch, 10)
        return tokens, betas


class HaMeRContextTokenTests(unittest.TestCase):
    def _make_cfg(self):
        class _Node(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        cfg = _Node()
        cfg.MODEL = _Node()
        cfg.MODEL.IMAGE_SIZE = 224
        cfg.MODEL.BACKBONE = _Node()
        cfg.MODEL.BACKBONE.TYPE = "vit"
        cfg.MODEL.MANO_HEAD = _Node()
        cfg.MODEL.MANO_HEAD.TRANSFORMER_DECODER = {}
        cfg.EXTRA = _Node()
        cfg.EXTRA.FOCAL_LENGTH = 5000
        cfg.MANO = _Node()
        cfg.MANO.NUM_HAND_JOINTS = 15
        cfg.MANO.MEAN_PARAMS = "/tmp/unused_mean_params.npz"
        return cfg

    def test_hamer_encoder_returns_dense_context_tokens_and_betas(self) -> None:
        backbone = _FakeContextBackbone(dim=8, num_tokens=6)
        encoder = HaMeREncoder(cfg=self._make_cfg(), backbone=backbone, min_mask_area=4, rescale_factor=1.0)
        imgs = torch.rand(2, 3, 3, 32, 32)
        hand_masks = torch.zeros(2, 3, 32, 32)
        hand_masks[0, 0, 4:12, 5:13] = 1.0
        hand_masks[1, 2, 8:18, 3:17] = 1.0
        hand_is_right = torch.tensor([[True, False, False], [False, False, True]], dtype=torch.bool)

        out = encoder(imgs, hand_masks, hand_is_right)

        self.assertEqual(tuple(out["hand_tokens"].shape), (2, 3, 6, 8))
        self.assertEqual(tuple(out["hand_betas"].shape), (2, 3, 10))
        self.assertEqual(tuple(out["hand_valid_mask"].shape), (2, 3))
        self.assertEqual(tuple(out["crop_boxes"].shape), (2, 3, 4))
        self.assertTrue(torch.equal(out["hand_valid_mask"], torch.tensor([[True, False, False], [False, False, True]])))
        self.assertTrue(torch.equal(backbone.last_hand_is_right, torch.tensor([True, True], dtype=torch.bool)))

    def test_hamer_encoder_fills_invalid_views_with_empty_tokens(self) -> None:
        backbone = _FakeContextBackbone(dim=4, num_tokens=3)
        encoder = HaMeREncoder(cfg=self._make_cfg(), backbone=backbone, min_mask_area=4, rescale_factor=1.0)
        imgs = torch.rand(1, 2, 3, 16, 16)
        hand_masks = torch.zeros(1, 2, 16, 16)
        hand_is_right = torch.tensor([[True, False]], dtype=torch.bool)

        out = encoder(imgs, hand_masks, hand_is_right)

        self.assertEqual(tuple(out["hand_tokens"].shape), (1, 2, 3, 4))
        self.assertFalse(bool(out["hand_valid_mask"].any()))
        self.assertTrue(torch.allclose(out["hand_tokens"], torch.zeros_like(out["hand_tokens"])))
        self.assertTrue(torch.allclose(out["hand_betas"], torch.zeros_like(out["hand_betas"])))

    def test_hamer_encoder_uses_context_dim_not_query_dim_for_empty_tokens(self) -> None:
        backbone = _FakeContextBackbone(dim=8, num_tokens=6, query_dim=1024)
        encoder = HaMeREncoder(cfg=self._make_cfg(), backbone=backbone, min_mask_area=4, rescale_factor=1.0)
        imgs = torch.rand(1, 2, 3, 16, 16)
        hand_masks = torch.zeros(1, 2, 16, 16)
        hand_masks[0, 1, 2:10, 3:11] = 1.0
        hand_is_right = torch.tensor([[True, False]], dtype=torch.bool)

        out = encoder(imgs, hand_masks, hand_is_right)

        self.assertEqual(tuple(out["hand_tokens"].shape), (1, 2, 6, 8))
        self.assertEqual(encoder.output_dim, 8)
        self.assertEqual(tuple(encoder.empty_hand_token.shape), (1, 1, 8))


if __name__ == "__main__":
    unittest.main()
