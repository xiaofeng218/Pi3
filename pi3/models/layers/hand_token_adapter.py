from __future__ import annotations

import torch
import torch.nn as nn

from .local_crop_utils import canonicalize_hand_crop, compute_bbox_from_mask, crop_and_resize, expand_box


class HandTokenAdapter(nn.Module):
    def __init__(self, token_dim: int = 1024, patch_size: int = 14, crop_hw: tuple[int, int] = (224, 168), rescale_factor: float = 2.0):
        super().__init__()
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.crop_hw = tuple(int(v) for v in crop_hw)
        self.rescale_factor = float(rescale_factor)

        self.side_embed = nn.Embedding(2, self.token_dim)
        self.empty_hand_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))

    def _normalize_mask_shape(self, hand_masks: torch.Tensor) -> torch.Tensor:
        if hand_masks.ndim != 4:
            raise ValueError("hand_masks must have shape (B,N,H,W)")
        return hand_masks

    def forward(
        self,
        imgs: torch.Tensor,
        hand_masks: torch.Tensor,
        hand_is_right: torch.Tensor,
        encoder: nn.Module,
    ) -> dict[str, torch.Tensor]:
        hand_masks = self._normalize_mask_shape(hand_masks)
        if hand_masks.shape[:2] != imgs.shape[:2] or hand_masks.shape[:2] != hand_is_right.shape:
            raise ValueError("imgs, hand_masks, and hand_is_right must align on (B,N)")

        batch_size, num_views, _, image_h, image_w = imgs.shape
        valid_mask = torch.zeros((batch_size, num_views), dtype=torch.bool, device=imgs.device)
        crop_boxes = imgs.new_zeros((batch_size, num_views, 4))
        crops = []
        valid_index = []

        for b in range(batch_size):
            for n in range(num_views):
                mask = hand_masks[b, n]
                if not bool((mask > 0).any()):
                    continue
                box = expand_box(compute_bbox_from_mask(mask), height=image_h, width=image_w, rescale_factor=self.rescale_factor)
                crop = crop_and_resize(imgs[b, n], box, output_hw=self.crop_hw)
                crop = canonicalize_hand_crop(crop, hand_is_right[b, n])
                crops.append(crop)
                valid_index.append((b, n))
                valid_mask[b, n] = True
                crop_boxes[b, n] = box

        if crops:
            crop_batch = torch.stack(crops, dim=0)
            token_batch = encoder(crop_batch, is_training=True)["x_norm_patchtokens"]
            num_tokens = token_batch.shape[1]
            dense_tokens = self.empty_hand_token.view(1, 1, 1, self.token_dim).expand(batch_size, num_views, num_tokens, self.token_dim).clone()
            for idx, (b, n) in enumerate(valid_index):
                dense_tokens[b, n] = token_batch[idx] + self.side_embed(hand_is_right[b, n].long()).view(1, -1)
        else:
            num_tokens = (self.crop_hw[0] // self.patch_size) * (self.crop_hw[1] // self.patch_size)
            dense_tokens = self.empty_hand_token.view(1, 1, 1, self.token_dim).expand(batch_size, num_views, num_tokens, self.token_dim).clone()

        return {
            "hand_tokens": dense_tokens,
            "hand_valid_mask": valid_mask,
            "crop_boxes": crop_boxes,
        }
