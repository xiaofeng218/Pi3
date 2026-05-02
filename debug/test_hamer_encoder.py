from __future__ import annotations

import unittest

import torch

from pi3.models.hamer.encoder import HaMeREncoder


class _FakeBackbone(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim
        self.last_crops = None
        self.last_hand_is_right = None

    def forward(self, crops, hand_is_right=None):
        self.last_crops = crops.detach().cpu()
        self.last_hand_is_right = None if hand_is_right is None else hand_is_right.detach().cpu()
        batch = crops.shape[0]
        values = torch.arange(batch, dtype=crops.dtype, device=crops.device).unsqueeze(1)
        return values.repeat(1, self.dim)


class HaMeREncoderTests(unittest.TestCase):
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

    def test_forward_filters_invalid_masks_and_preserves_order(self) -> None:
        backbone = _FakeBackbone(dim=8)
        encoder = HaMeREncoder(
            cfg=self._make_cfg(),
            backbone=backbone,
            min_mask_area=4,
            rescale_factor=1.0,
        )
        imgs = torch.rand(2, 2, 3, 32, 32)
        hand_masks = torch.zeros(4, 32, 32)
        hand_masks[0, 4:10, 5:11] = 1.0
        hand_masks[1, 8:9, 8:9] = 1.0
        hand_masks[2, 12:20, 2:8] = 1.0
        hand_masks[3, :, :] = 0.0
        owner_index = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=torch.long,
        )
        hand_is_right = torch.tensor([True, False, True, False], dtype=torch.bool)

        out = encoder(imgs, hand_masks, owner_index, hand_is_right)

        self.assertEqual(tuple(out["hand_queries"].shape), (2, 8))
        self.assertTrue(torch.equal(out["owner_index"], owner_index[[0, 2]]))
        self.assertTrue(torch.equal(out["hand_is_right"], hand_is_right[[0, 2]]))
        self.assertEqual(tuple(out["crop_boxes"].shape), (2, 4))

    def test_forward_returns_empty_sparse_outputs_when_no_valid_hands(self) -> None:
        encoder = HaMeREncoder(
            cfg=self._make_cfg(),
            backbone=_FakeBackbone(dim=8),
            min_mask_area=4,
            rescale_factor=1.0,
        )
        imgs = torch.rand(1, 1, 3, 16, 16)
        hand_masks = torch.zeros(2, 16, 16)
        owner_index = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.long)
        hand_is_right = torch.tensor([True, False], dtype=torch.bool)

        out = encoder(imgs, hand_masks, owner_index, hand_is_right)

        self.assertEqual(tuple(out["hand_queries"].shape), (0, 8))
        self.assertEqual(tuple(out["owner_index"].shape), (0, 3))
        self.assertEqual(tuple(out["hand_is_right"].shape), (0,))
        self.assertEqual(tuple(out["crop_boxes"].shape), (0, 4))

    def test_forward_flips_left_hand_crop_before_backbone(self) -> None:
        backbone = _FakeBackbone(dim=4)
        encoder = HaMeREncoder(
            cfg=self._make_cfg(),
            backbone=backbone,
            min_mask_area=1,
            rescale_factor=1.0,
        )
        imgs = torch.zeros(1, 2, 1, 4, 4)
        imgs[0, 0, 0] = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
        imgs[0, 1, 0] = imgs[0, 0, 0]
        hand_masks = torch.zeros(2, 4, 4)
        hand_masks[:, 1:3, 1:4] = 1.0
        owner_index = torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.long)
        hand_is_right = torch.tensor([True, False], dtype=torch.bool)

        out = encoder(imgs, hand_masks, owner_index, hand_is_right)

        self.assertEqual(out["hand_queries"].shape[0], 2)
        self.assertTrue(torch.equal(backbone.last_hand_is_right, hand_is_right))
        self.assertTrue(
            torch.allclose(
                backbone.last_crops[1],
                torch.flip(backbone.last_crops[0], dims=[2]),
            )
        )

    def test_random_sparse_inputs_smoke(self) -> None:
        backbone = _FakeBackbone(dim=16)
        encoder = HaMeREncoder(
            cfg=self._make_cfg(),
            backbone=backbone,
            min_mask_area=8,
            rescale_factor=1.5,
        )
        imgs = torch.rand(2, 3, 3, 96, 96)
        hand_masks = torch.zeros(6, 96, 96)
        hand_masks[0, 5:20, 7:18] = 1.0
        hand_masks[1, 12:30, 12:33] = 1.0
        hand_masks[2, 50:70, 40:60] = 1.0
        hand_masks[3, 3:4, 3:4] = 1.0
        owner_index = torch.tensor(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [1, 2, 0],
                [0, 2, 0],
            ],
            dtype=torch.long,
        )
        hand_is_right = torch.tensor([True, False, True, False, True, False], dtype=torch.bool)

        out = encoder(imgs, hand_masks, owner_index, hand_is_right)

        self.assertEqual(out["hand_queries"].shape[1], 16)
        self.assertEqual(out["owner_index"].shape[0], out["hand_queries"].shape[0])
        self.assertEqual(out["hand_is_right"].shape[0], out["hand_queries"].shape[0])
        self.assertEqual(out["crop_boxes"].shape[0], out["hand_queries"].shape[0])
        self.assertTrue(torch.equal(out["owner_index"], owner_index[[0, 1, 2]]))
        self.assertTrue(torch.equal(out["hand_is_right"], hand_is_right[[0, 1, 2]]))


if __name__ == "__main__":
    unittest.main()
