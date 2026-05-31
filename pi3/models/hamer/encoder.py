from __future__ import annotations

import torch
import torch.nn as nn

from .backbone_query import HaMeRBackbone
from ..layers.local_crop_utils import (
    canonicalize_hand_crop,
    compute_bbox_from_mask,
    crop_and_resize,
    expand_box,
    normalize_mask_shape,
)


class HaMeREncoder(nn.Module):
    def __init__(self, cfg, backbone: nn.Module | None = None, min_mask_area: int = 16, rescale_factor: float = 2.0):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone if backbone is not None else HaMeRBackbone(cfg)
        self.min_mask_area = int(min_mask_area)
        self.rescale_factor = float(rescale_factor)
        self.output_dim = int(
            getattr(
                self.backbone,
                "context_dim",
                getattr(self.backbone, "output_dim", getattr(self.backbone, "dim", 1280)),
            )
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        self.empty_hand_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))

    def _infer_num_tokens(self) -> int:
        if hasattr(self.backbone, "num_tokens"):
            return int(getattr(self.backbone, "num_tokens"))
        image_h, image_w = 256, 192
        patch_size = 16
        return (image_h // patch_size) * (image_w // patch_size)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            # ViT backbone: backbone.* → backbone.vit.*
            if key.startswith("backbone.") and not key.startswith("backbone.vit."):
                new_key = "backbone.vit." + key[len("backbone."):]
            # Transformer decoder: mano_head.transformer.* → backbone.transformer.*
            elif key.startswith("mano_head.transformer."):
                new_key = "backbone.transformer." + key[len("mano_head.transformer."):]
            # decshape head: mano_head.decshape.* → backbone.decshape.*
            elif key.startswith("mano_head.decshape."):
                new_key = "backbone.decshape." + key[len("mano_head.decshape."):]
            # init_betas buffer
            elif key == "mano_head.init_betas":
                new_key = "backbone.init_betas"
            # Skip unrelated keys: decpose, deccam, discriminator, mano buffers, etc.
            elif key.startswith("mano_head.") or key.startswith("discriminator.") or key.startswith("mano.") or key == "initialized":
                continue
            remapped[new_key] = value

        result = super().load_state_dict(remapped, strict=strict)

        for param in self.backbone.decshape.parameters():
            param.requires_grad = False

        return result

    def _empty_output(self, imgs: torch.Tensor, hand_masks: torch.Tensor, hand_is_right: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_views = imgs.shape[:2]
        mask_shape = hand_masks.shape[-2:]
        num_tokens = self._infer_num_tokens()
        return {
            "hand_tokens": self.empty_hand_token.view(1, 1, 1, self.output_dim).expand(batch_size, num_views, num_tokens, self.output_dim).clone(),
            "hand_betas": imgs.new_zeros((batch_size, num_views, 10)),
            "hand_valid_mask": torch.zeros((batch_size, num_views), dtype=torch.bool, device=imgs.device),
            "hand_is_right": hand_is_right,
            "crop_boxes": imgs.new_zeros((batch_size, num_views, 4)),
            "raw_hand_masks": hand_masks.new_zeros((batch_size, num_views, *mask_shape)),
        }

    def _normalize_mask_shape(self, hand_masks: torch.Tensor) -> torch.Tensor:
        return normalize_mask_shape(hand_masks)

    def _compute_bbox_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return compute_bbox_from_mask(mask)

    def _expand_box(self, box: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return expand_box(box, height=height, width=width, rescale_factor=self.rescale_factor)

    def _crop_and_resize(self, img: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return crop_and_resize(img, box, output_hw=(256, 192))

    def _normalize_input(self, crop: torch.Tensor) -> torch.Tensor:
        if crop.shape[0] == 3:
            return (crop - self.image_mean.to(device=crop.device, dtype=crop.dtype)) / self.image_std.to(device=crop.device, dtype=crop.dtype)
        return crop

    def _canonicalize_handedness(self, crop: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
        return canonicalize_hand_crop(crop, is_right)

    def forward(
        self,
        imgs: torch.Tensor,
        hand_masks: torch.Tensor,
        hand_is_right: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hand_masks = self._normalize_mask_shape(hand_masks)
        if hand_masks.ndim != 4:
            raise ValueError("hand_masks must have shape (B,N,H,W)")
        if hand_masks.shape[:2] != hand_is_right.shape:
            raise ValueError("hand_masks and hand_is_right must align on (B,N)")

        if hand_masks.numel() == 0:
            return self._empty_output(imgs, hand_masks, hand_is_right)

        batch_size, num_views, _, image_h, image_w = imgs.shape
        dense_tokens = None
        dense_betas = imgs.new_zeros((batch_size, num_views, 10))
        dense_valid_mask = torch.zeros((batch_size, num_views), dtype=torch.bool, device=imgs.device)
        dense_boxes = imgs.new_zeros((batch_size, num_views, 4))
        valid_right = []
        valid_index = []
        crops = []

        for b in range(batch_size):
            for n in range(num_views):
                mask = hand_masks[b, n]
                if int((mask > 0).sum().item()) < self.min_mask_area:
                    continue
                box = self._expand_box(self._compute_bbox_from_mask(mask), image_h, image_w)
                crop = self._crop_and_resize(imgs[b, n], box)
                crop = self._canonicalize_handedness(crop, hand_is_right[b, n])
                crop = self._normalize_input(crop)
                dense_boxes[b, n] = box
                valid_right.append(hand_is_right[b, n])
                valid_index.append((b, n))
                crops.append(crop)

        if not crops:
            return self._empty_output(imgs, hand_masks, hand_is_right)

        crop_batch = torch.stack(crops, dim=0)
        valid_right_tensor = torch.stack(valid_right)
        hand_tokens, hand_betas = self.backbone(crop_batch, valid_right_tensor)
        num_tokens, token_dim = hand_tokens.shape[1], hand_tokens.shape[2]
        dense_tokens = self.empty_hand_token.view(1, 1, 1, self.output_dim).expand(batch_size, num_views, num_tokens, token_dim).clone()

        for idx, (b, n) in enumerate(valid_index):
            dense_tokens[b, n] = hand_tokens[idx]
            dense_betas[b, n] = hand_betas[idx]
            dense_valid_mask[b, n] = True

        return {
            "hand_tokens": dense_tokens,
            "hand_betas": dense_betas,
            "hand_valid_mask": dense_valid_mask,
            "hand_is_right": hand_is_right,
            "crop_boxes": dense_boxes,
            "raw_hand_masks": hand_masks,
        }
