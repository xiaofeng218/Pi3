from __future__ import annotations

import torch
import torch.nn as nn

from .local_crop_utils import compute_bbox_from_mask, crop_and_resize, expand_box


class ObjectQueryAdapter(nn.Module):
    def __init__(self, token_dim: int = 1024, patch_size: int = 14, crop_hw: tuple[int, int] = (224, 168), rescale_factor: float = 2.0):
        super().__init__()
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.crop_hw = tuple(int(v) for v in crop_hw)
        self.rescale_factor = float(rescale_factor)
        self.empty_object_query = nn.Parameter(torch.zeros(1, 1, self.token_dim))

    def _validate_shapes(
        self,
        imgs: torch.Tensor,
        grasped_object_mask: torch.Tensor,
        grasped_object_valid: torch.Tensor,
    ) -> None:
        if imgs.ndim != 5:
            raise ValueError("imgs must have shape (B,N,C,H,W)")
        if grasped_object_mask.ndim != 4:
            raise ValueError("grasped_object_mask must have shape (B,N,H,W)")
        if grasped_object_valid.ndim != 2:
            raise ValueError("grasped_object_valid must have shape (B,N)")

        batch_shape = imgs.shape[:2]
        if grasped_object_mask.shape[:2] != batch_shape:
            raise ValueError("grasped_object_mask batch/view shape must match imgs")
        if grasped_object_valid.shape != batch_shape:
            raise ValueError("grasped_object_valid batch/view shape must match imgs")

    def forward(
        self,
        imgs: torch.Tensor,
        grasped_object_mask: torch.Tensor,
        grasped_object_valid: torch.Tensor,
        encoder: nn.Module,
    ) -> dict[str, torch.Tensor]:
        self._validate_shapes(imgs, grasped_object_mask, grasped_object_valid)
        batch, num_views, _, image_h, image_w = imgs.shape
        valid_index = []
        crops = []
        crop_boxes = imgs.new_zeros((batch, num_views, 4))

        for batch_idx in range(batch):
            for view_idx in range(num_views):
                if not bool(grasped_object_valid[batch_idx, view_idx]):
                    continue
                mask = grasped_object_mask[batch_idx, view_idx]
                if not bool((mask > 0).any()):
                    continue
                box = expand_box(compute_bbox_from_mask(mask), height=image_h, width=image_w, rescale_factor=self.rescale_factor)
                crop_boxes[batch_idx, view_idx] = box
                crops.append(crop_and_resize(imgs[batch_idx, view_idx], box, output_hw=self.crop_hw))
                valid_index.append((batch_idx, view_idx))

        if crops:
            crop_batch = torch.stack(crops, dim=0)
            token_batch = encoder(crop_batch, is_training=True)["x_norm_patchtokens"]
            num_tokens = token_batch.shape[1]
            object_tokens = self.empty_object_query.view(1, 1, 1, self.token_dim).expand(batch, num_views, num_tokens, self.token_dim).clone()
            for idx, (batch_idx, view_idx) in enumerate(valid_index):
                object_tokens[batch_idx, view_idx] = token_batch[idx]
        else:
            num_tokens = (self.crop_hw[0] // self.patch_size) * (self.crop_hw[1] // self.patch_size)
            object_tokens = self.empty_object_query.view(1, 1, 1, self.token_dim).expand(batch, num_views, num_tokens, self.token_dim).clone()

        return {
            "object_tokens": object_tokens,
            "object_valid_mask": grasped_object_valid,
            "crop_boxes": crop_boxes,
        }
