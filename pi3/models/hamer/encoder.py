from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_query import HaMeRBackbone


class HaMeREncoder(nn.Module):
    def __init__(self, cfg, backbone: nn.Module | None = None, min_mask_area: int = 16, rescale_factor: float = 2.0):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone if backbone is not None else HaMeRBackbone(cfg)
        self.min_mask_area = int(min_mask_area)
        self.rescale_factor = float(rescale_factor)
        self.output_dim = int(getattr(self.backbone, "output_dim", getattr(self.backbone, "dim", 1024)))

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

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

    def _empty_output(self, imgs: torch.Tensor, owner_index: torch.Tensor, hand_is_right: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hand_queries": imgs.new_zeros((0, self.output_dim)),
            "hand_betas": imgs.new_zeros((0, 10)),
            "owner_index": owner_index.new_zeros((0, 3)),
            "hand_is_right": hand_is_right.new_zeros((0,), dtype=hand_is_right.dtype),
            "crop_boxes": imgs.new_zeros((0, 4)),
            "source_index": owner_index.new_zeros((0,), dtype=torch.long),
        }

    def _normalize_mask_shape(self, hand_masks: torch.Tensor) -> torch.Tensor:
        if hand_masks.ndim == 4 and hand_masks.shape[1] == 1:
            return hand_masks[:, 0]
        if hand_masks.ndim == 3:
            return hand_masks
        raise ValueError("hand_masks must have shape (K,H,W) or (K,1,H,W)")

    def _compute_bbox_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        if xs.numel() == 0:
            raise ValueError("Hand mask is empty; cannot compute bounding box")
        return torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=mask.device, dtype=torch.float32)

    def _expand_box(self, box: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x1, y1, x2, y2 = box.unbind()
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1 + 1.0) * self.rescale_factor
        bh = (y2 - y1 + 1.0) * self.rescale_factor
        new_x1 = torch.clamp(cx - bw * 0.5, min=0.0, max=float(width - 1))
        new_x2 = torch.clamp(cx + bw * 0.5, min=0.0, max=float(width - 1))
        new_y1 = torch.clamp(cy - bh * 0.5, min=0.0, max=float(height - 1))
        new_y2 = torch.clamp(cy + bh * 0.5, min=0.0, max=float(height - 1))
        return torch.stack([new_x1, new_y1, new_x2, new_y2])

    def _crop_and_resize(self, img: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = box.round().to(torch.int64)
        x2 = torch.clamp(x2, min=x1)
        y2 = torch.clamp(y2, min=y1)
        crop = img[:, y1 : y2 + 1, x1 : x2 + 1]
        if crop.numel() == 0:
            raise ValueError("Expanded crop is empty")
        return F.interpolate(crop.unsqueeze(0), size=(256, 192), mode="bilinear", align_corners=False).squeeze(0)

    def _normalize_input(self, crop: torch.Tensor) -> torch.Tensor:
        if crop.shape[0] == 3:
            return (crop - self.image_mean.to(device=crop.device, dtype=crop.dtype)) / self.image_std.to(device=crop.device, dtype=crop.dtype)
        return crop

    def _canonicalize_handedness(self, crop: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
        return crop if bool(is_right) else torch.flip(crop, dims=[2])

    def forward(
        self,
        imgs: torch.Tensor,
        hand_masks: torch.Tensor,
        owner_index: torch.Tensor,
        hand_is_right: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hand_masks = self._normalize_mask_shape(hand_masks)
        if hand_masks.shape[0] != owner_index.shape[0] or hand_masks.shape[0] != hand_is_right.shape[0]:
            raise ValueError("hand_masks, owner_index, and hand_is_right must align row-wise")

        if hand_masks.shape[0] == 0:
            return self._empty_output(imgs, owner_index, hand_is_right)

        _, _, _, image_h, image_w = imgs.shape
        valid_boxes = []
        valid_owner = []
        valid_right = []
        valid_source_index = []
        crops = []

        for idx in range(hand_masks.shape[0]):
            mask = hand_masks[idx]
            if int((mask > 0).sum().item()) < self.min_mask_area:
                continue

            b, n, _ = owner_index[idx].tolist()
            if not (0 <= b < imgs.shape[0] and 0 <= n < imgs.shape[1]):
                raise ValueError("owner_index points outside imgs")

            box = self._expand_box(self._compute_bbox_from_mask(mask), image_h, image_w)
            crop = self._crop_and_resize(imgs[b, n], box)
            crop = self._canonicalize_handedness(crop, hand_is_right[idx])
            crop = self._normalize_input(crop)

            valid_boxes.append(box)
            valid_owner.append(owner_index[idx])
            valid_right.append(hand_is_right[idx])
            valid_source_index.append(idx)
            crops.append(crop)

        if not crops:
            return self._empty_output(imgs, owner_index, hand_is_right)

        crop_batch = torch.stack(crops, dim=0)
        valid_right_tensor = torch.stack(valid_right)
        hand_queries, hand_betas = self.backbone(crop_batch, valid_right_tensor)

        return {
            "hand_queries": hand_queries,
            "hand_betas": hand_betas,
            "owner_index": torch.stack(valid_owner),
            "hand_is_right": valid_right_tensor,
            "crop_boxes": torch.stack(valid_boxes),
            "source_index": torch.tensor(valid_source_index, device=imgs.device, dtype=torch.long),
        }
