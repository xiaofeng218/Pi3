from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask[:, 0]
    if mask.ndim == 4:
        return mask
    if mask.ndim == 3:
        return mask
    raise ValueError("mask tensor must have shape (K,H,W), (K,1,H,W), or (B,N,H,W)")


def compute_bbox_from_mask(mask: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.nonzero(mask > 0, as_tuple=True)
    if xs.numel() == 0:
        raise ValueError("mask is empty; cannot compute bounding box")
    return torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], device=mask.device, dtype=torch.float32)


def expand_box(
    box: torch.Tensor,
    height: int,
    width: int,
    rescale_factor: float,
    target_aspect_ratio: float = 3.0 / 4.0,
) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind()
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1 + 1.0) * float(rescale_factor)
    bh = (y2 - y1 + 1.0) * float(rescale_factor)

    if bw / max(bh, 1e-6) < target_aspect_ratio:
        bw = bh * float(target_aspect_ratio)
    else:
        bh = bw / float(target_aspect_ratio)

    bw = min(bw, float(width))
    bh = min(bh, float(height))

    new_x1 = cx - bw * 0.5
    new_x2 = cx + bw * 0.5
    new_y1 = cy - bh * 0.5
    new_y2 = cy + bh * 0.5

    if new_x1 < 0.0:
        shift = -new_x1
        new_x1 += shift
        new_x2 += shift
    if new_x2 > float(width - 1):
        shift = new_x2 - float(width - 1)
        new_x1 -= shift
        new_x2 -= shift
    if new_y1 < 0.0:
        shift = -new_y1
        new_y1 += shift
        new_y2 += shift
    if new_y2 > float(height - 1):
        shift = new_y2 - float(height - 1)
        new_y1 -= shift
        new_y2 -= shift

    new_x1 = torch.clamp(new_x1, min=0.0, max=float(width - 1))
    new_x2 = torch.clamp(new_x2, min=0.0, max=float(width - 1))
    new_y1 = torch.clamp(new_y1, min=0.0, max=float(height - 1))
    new_y2 = torch.clamp(new_y2, min=0.0, max=float(height - 1))
    return torch.stack([new_x1, new_y1, new_x2, new_y2])


def crop_and_resize(img: torch.Tensor, box: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    x1, y1, x2, y2 = box.round().to(torch.int64)
    x2 = torch.clamp(x2, min=x1)
    y2 = torch.clamp(y2, min=y1)
    crop = img[:, y1 : y2 + 1, x1 : x2 + 1]
    if crop.numel() == 0:
        raise ValueError("expanded crop is empty")
    return F.interpolate(crop.unsqueeze(0), size=output_hw, mode="bilinear", align_corners=False).squeeze(0)


def canonicalize_hand_crop(crop: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
    return crop if bool(is_right) else torch.flip(crop, dims=[2])


def compute_patch_grid(image_hw: tuple[int, int], patch_size: int) -> tuple[int, int]:
    image_h, image_w = image_hw
    return image_h // patch_size, image_w // patch_size
