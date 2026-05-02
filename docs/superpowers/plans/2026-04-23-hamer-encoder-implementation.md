# HaMeREncoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sparse `HaMeREncoder` that converts full-frame Pi3X images plus sparse hand masks into sparse HaMeR query tokens, then verify it with deterministic unit tests and a random-data smoke test.

**Architecture:** Add a new `HaMeREncoder` wrapper that owns sparse mask filtering, bbox extraction, crop/preprocess, and handedness canonicalization. Add a `HaMeRBackbone` query-only path that reuses the local HaMeR ViT and transformer decoder but stops at the query token instead of regressing MANO parameters.

**Tech Stack:** PyTorch, local HaMeR modules under `pi3/models/hamer`, Python `unittest`, existing debug test layout.

---

### Task 1: Add Failing Tests for Sparse Encoder Contract

**Files:**
- Create: `debug/test_hamer_encoder.py`
- Test: `debug/test_hamer_encoder.py`

- [ ] **Step 1: Write the failing contract tests**

```python
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from pi3.models.hamer.encoder import HaMeREncoder


class _FakeBackbone(torch.nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim

    def forward(self, crops, hand_is_right=None):
        batch = crops.shape[0]
        base = torch.arange(batch, dtype=crops.dtype, device=crops.device).unsqueeze(1)
        return base.repeat(1, self.dim)


class HaMeREncoderTests(unittest.TestCase):
    def _make_cfg(self):
        return SimpleNamespace(
            MODEL=SimpleNamespace(IMAGE_SIZE=224),
            EXTRA=SimpleNamespace(FOCAL_LENGTH=5000),
        )

    def test_forward_filters_invalid_masks_and_preserves_order(self):
        encoder = HaMeREncoder(cfg=self._make_cfg(), backbone=_FakeBackbone(dim=8), min_mask_area=4, rescale_factor=2.0)
        imgs = torch.rand(2, 2, 3, 64, 64)
        masks = torch.zeros(4, 64, 64)
        masks[0, 10:20, 10:20] = 1
        masks[1, 5:6, 5:6] = 1
        masks[2, 30:40, 30:45] = 1
        owner_index = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        hand_is_right = torch.tensor([1, 0, 1, 0], dtype=torch.bool)

        out = encoder(imgs, masks, owner_index, hand_is_right)

        self.assertEqual(tuple(out["hand_queries"].shape), (2, 8))
        self.assertTrue(torch.equal(out["owner_index"], torch.tensor([[0, 0, 0], [1, 0, 0]])))
        self.assertTrue(torch.equal(out["hand_is_right"], torch.tensor([True, True])))
        self.assertEqual(tuple(out["crop_boxes"].shape), (2, 4))

    def test_forward_returns_empty_sparse_outputs_when_no_valid_hands(self):
        encoder = HaMeREncoder(cfg=self._make_cfg(), backbone=_FakeBackbone(dim=8), min_mask_area=4, rescale_factor=2.0)
        imgs = torch.rand(1, 1, 3, 32, 32)
        masks = torch.zeros(2, 32, 32)
        owner_index = torch.tensor([[0, 0, 0], [0, 0, 1]])
        hand_is_right = torch.tensor([1, 0], dtype=torch.bool)

        out = encoder(imgs, masks, owner_index, hand_is_right)

        self.assertEqual(out["hand_queries"].shape[0], 0)
        self.assertEqual(out["owner_index"].shape, (0, 3))
        self.assertEqual(out["hand_is_right"].shape, (0,))
        self.assertEqual(out["crop_boxes"].shape, (0, 4))

    def test_forward_flips_left_hands_before_backbone_call(self):
        recorder = mock.Mock()

        class _RecordingBackbone(torch.nn.Module):
            def forward(self, crops, hand_is_right=None):
                recorder(crops.detach().cpu(), hand_is_right.detach().cpu())
                return torch.ones(crops.shape[0], 4)

        encoder = HaMeREncoder(cfg=self._make_cfg(), backbone=_RecordingBackbone(), min_mask_area=1, rescale_factor=1.0)
        imgs = torch.zeros(1, 1, 3, 8, 8)
        imgs[0, 0, 0] = torch.arange(8).repeat(8, 1)
        masks = torch.zeros(1, 8, 8)
        masks[0, 2:6, 1:5] = 1
        owner_index = torch.tensor([[0, 0, 0]])
        hand_is_right = torch.tensor([0], dtype=torch.bool)

        encoder(imgs, masks, owner_index, hand_is_right)

        crops, flags = recorder.call_args.args
        self.assertFalse(flags.item())
        self.assertTrue(torch.allclose(crops[0, 0, :, 0], torch.flip(crops[0, 0, :, -1], dims=[0])))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest debug.test_hamer_encoder -v`
Expected: FAIL with `ModuleNotFoundError` or missing `HaMeREncoder`

- [ ] **Step 3: Commit the failing tests**

```bash
git add debug/test_hamer_encoder.py
git commit -m "test: add failing HaMeREncoder contract tests"
```

### Task 2: Implement Query-Only HaMeR Backbone

**Files:**
- Create: `pi3/models/hamer/backbone_query.py`
- Modify: `pi3/models/hamer/__init__.py`
- Test: `debug/test_hamer_encoder.py`

- [ ] **Step 1: Add a failing import test for `HaMeRBackbone` wiring**

```python
def test_hamer_backbone_returns_query_tokens():
    from pi3.models.hamer.backbone_query import HaMeRBackbone

    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(
            IMAGE_SIZE=224,
            BACKBONE=SimpleNamespace(TYPE="vit"),
            MANO_HEAD={
                "JOINT_REP": "6d",
                "TRANSFORMER_INPUT": "zero",
                "IEF_ITERS": 1,
                "TRANSFORMER_DECODER": {
                    "depth": 1,
                    "heads": 2,
                    "mlp_dim": 128,
                    "dim_head": 64,
                    "dropout": 0.0,
                    "emb_dropout": 0.0,
                    "emb_dropout_type": "normal",
                    "context_dim": 1280,
                },
            },
        ),
        MANO=SimpleNamespace(NUM_HAND_JOINTS=15, MEAN_PARAMS="dummy.npz"),
    )
    model = HaMeRBackbone(cfg)
    with mock.patch.object(model, "vit", return_value=torch.rand(2, 1280, 16, 12)):
        with mock.patch.object(model, "_get_init_token", return_value=torch.zeros(2, 1, 1)):
            queries = model(torch.rand(2, 3, 256, 192), torch.tensor([True, False]))
    assert queries.shape[0] == 2
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m unittest debug.test_hamer_encoder.HaMeREncoderTests.test_forward_filters_invalid_masks_and_preserves_order -v`
Expected: FAIL because `HaMeRBackbone` and `HaMeREncoder` do not exist yet

- [ ] **Step 3: Implement `HaMeRBackbone`**

```python
import einops
import numpy as np
import torch
import torch.nn as nn

from .backbones import create_backbone
from .components.pose_transformer import TransformerDecoder


class HaMeRBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vit = create_backbone(cfg)
        transformer_args = dict(
            num_tokens=1,
            token_dim=1,
            dim=1024,
        )
        transformer_args.update(dict(cfg.MODEL.MANO_HEAD["TRANSFORMER_DECODER"]))
        self.transformer = TransformerDecoder(**transformer_args)

    def _get_init_token(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)

    def forward(self, crops, hand_is_right=None):
        feats = self.vit(crops)
        context = einops.rearrange(feats, "b c h w -> b (h w) c")
        token = self._get_init_token(crops.shape[0], crops.device, context.dtype)
        token_out = self.transformer(token, context=context)
        return token_out.squeeze(1)
```

- [ ] **Step 4: Export the new symbol**

```python
from .config import get_config
from .load import load_hamer
from .model import HAMER
from .backbone_query import HaMeRBackbone

__all__ = ["HAMER", "HaMeRBackbone", "get_config", "load_hamer"]
```

- [ ] **Step 5: Run tests to verify the new backbone is wired**

Run: `python -m unittest debug.test_hamer_encoder -v`
Expected: still FAIL, but now inside `HaMeREncoder` behavior instead of missing query-backbone symbols

- [ ] **Step 6: Commit the backbone implementation**

```bash
git add pi3/models/hamer/backbone_query.py pi3/models/hamer/__init__.py debug/test_hamer_encoder.py
git commit -m "feat: add query-only HaMeR backbone"
```

### Task 3: Implement Sparse `HaMeREncoder`

**Files:**
- Create: `pi3/models/hamer/encoder.py`
- Modify: `pi3/models/hamer/__init__.py`
- Test: `debug/test_hamer_encoder.py`

- [ ] **Step 1: Implement minimal sparse encoder structure**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_query import HaMeRBackbone


class HaMeREncoder(nn.Module):
    def __init__(self, cfg, backbone=None, min_mask_area=16, rescale_factor=2.0):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone if backbone is not None else HaMeRBackbone(cfg)
        self.min_mask_area = int(min_mask_area)
        self.rescale_factor = float(rescale_factor)

    def _normalize_mask_shape(self, hand_masks):
        if hand_masks.ndim == 4 and hand_masks.shape[1] == 1:
            return hand_masks[:, 0]
        if hand_masks.ndim == 3:
            return hand_masks
        raise ValueError("hand_masks must be KxHxW or Kx1xHxW")

    def _compute_bbox_from_mask(self, mask):
        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        if xs.numel() == 0:
            raise ValueError("empty mask")
        return torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=mask.device, dtype=torch.float32)

    def _expand_box(self, box, height, width):
        x1, y1, x2, y2 = box.unbind()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = (x2 - x1 + 1) * self.rescale_factor
        bh = (y2 - y1 + 1) * self.rescale_factor
        x1 = torch.clamp(cx - bw / 2, min=0, max=width - 1)
        x2 = torch.clamp(cx + bw / 2, min=0, max=width - 1)
        y1 = torch.clamp(cy - bh / 2, min=0, max=height - 1)
        y2 = torch.clamp(cy + bh / 2, min=0, max=height - 1)
        return torch.stack([x1, y1, x2, y2])

    def _crop_and_resize(self, img, box):
        x1, y1, x2, y2 = box.round().to(torch.int64)
        crop = img[:, y1 : y2 + 1, x1 : x2 + 1]
        crop = F.interpolate(crop.unsqueeze(0), size=(256, 192), mode="bilinear", align_corners=False).squeeze(0)
        return crop

    def _canonicalize_handedness(self, crop, is_right):
        return crop if bool(is_right) else torch.flip(crop, dims=[2])

    def forward(self, imgs, hand_masks, owner_index, hand_is_right):
        hand_masks = self._normalize_mask_shape(hand_masks)
        if hand_masks.shape[0] != owner_index.shape[0] or hand_masks.shape[0] != hand_is_right.shape[0]:
            raise ValueError("sparse inputs must align row-wise")

        if hand_masks.shape[0] == 0:
            return {
                "hand_queries": imgs.new_zeros((0, 1024)),
                "owner_index": owner_index.new_zeros((0, 3)),
                "hand_is_right": hand_is_right.new_zeros((0,), dtype=hand_is_right.dtype),
                "crop_boxes": imgs.new_zeros((0, 4)),
            }

        valid_boxes = []
        valid_owner = []
        valid_right = []
        crops = []
        _, _, _, H, W = imgs.shape

        for idx in range(hand_masks.shape[0]):
            mask = hand_masks[idx]
            if int((mask > 0).sum().item()) < self.min_mask_area:
                continue
            b, n, m = owner_index[idx].tolist()
            if not (0 <= b < imgs.shape[0] and 0 <= n < imgs.shape[1]):
                raise ValueError("owner_index points outside imgs")
            box = self._expand_box(self._compute_bbox_from_mask(mask), H, W)
            crop = self._crop_and_resize(imgs[b, n], box)
            crop = self._canonicalize_handedness(crop, hand_is_right[idx])
            valid_boxes.append(box)
            valid_owner.append(owner_index[idx])
            valid_right.append(hand_is_right[idx])
            crops.append(crop)

        if not crops:
            return {
                "hand_queries": imgs.new_zeros((0, 1024)),
                "owner_index": owner_index.new_zeros((0, 3)),
                "hand_is_right": hand_is_right.new_zeros((0,), dtype=hand_is_right.dtype),
                "crop_boxes": imgs.new_zeros((0, 4)),
            }

        crop_batch = torch.stack(crops, dim=0)
        queries = self.backbone(crop_batch, torch.stack(valid_right))
        return {
            "hand_queries": queries,
            "owner_index": torch.stack(valid_owner),
            "hand_is_right": torch.stack(valid_right),
            "crop_boxes": torch.stack(valid_boxes),
        }
```

- [ ] **Step 2: Export `HaMeREncoder`**

```python
from .config import get_config
from .load import load_hamer
from .model import HAMER
from .backbone_query import HaMeRBackbone
from .encoder import HaMeREncoder

__all__ = ["HAMER", "HaMeRBackbone", "HaMeREncoder", "get_config", "load_hamer"]
```

- [ ] **Step 3: Run the unit tests and make them pass**

Run: `python -m unittest debug.test_hamer_encoder -v`
Expected: PASS for the sparse contract tests

- [ ] **Step 4: Commit the sparse encoder**

```bash
git add pi3/models/hamer/encoder.py pi3/models/hamer/__init__.py debug/test_hamer_encoder.py
git commit -m "feat: add sparse HaMeREncoder"
```

### Task 4: Add Random-Data Smoke Test and Final Verification

**Files:**
- Modify: `debug/test_hamer_encoder.py`
- Test: `debug/test_hamer_encoder.py`

- [ ] **Step 1: Add random-data smoke coverage**

```python
    def test_random_sparse_inputs_smoke(self):
        encoder = HaMeREncoder(cfg=self._make_cfg(), backbone=_FakeBackbone(dim=16), min_mask_area=8, rescale_factor=1.5)
        imgs = torch.rand(2, 3, 3, 96, 96)
        masks = torch.zeros(6, 96, 96)
        masks[0, 5:20, 7:18] = 1
        masks[1, 12:30, 12:33] = 1
        masks[2, 50:70, 40:60] = 1
        masks[3, 3:4, 3:4] = 1
        owner_index = torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 2, 0],
            [0, 2, 0],
        ])
        hand_is_right = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)

        out = encoder(imgs, masks, owner_index, hand_is_right)

        self.assertEqual(out["hand_queries"].shape[1], 16)
        self.assertEqual(out["owner_index"].shape[0], out["hand_queries"].shape[0])
        self.assertEqual(out["hand_is_right"].shape[0], out["hand_queries"].shape[0])
        self.assertEqual(out["crop_boxes"].shape[0], out["hand_queries"].shape[0])
```

- [ ] **Step 2: Run the random-data smoke test**

Run: `python -m unittest debug.test_hamer_encoder.HaMeREncoderTests.test_random_sparse_inputs_smoke -v`
Expected: PASS

- [ ] **Step 3: Run the full verification command**

Run: `python -m unittest debug.test_hamer_encoder -v`
Expected: PASS with all `HaMeREncoder` tests green

- [ ] **Step 4: Commit the smoke test coverage**

```bash
git add debug/test_hamer_encoder.py
git commit -m "test: add HaMeREncoder random-data smoke coverage"
```
