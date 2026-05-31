from __future__ import annotations

import torch

from pi3.models.layers.local_crop_utils import (
    canonicalize_hand_crop,
    compute_bbox_from_mask,
    compute_patch_grid,
    crop_and_resize,
    expand_box,
)


def test_compute_bbox_from_mask_returns_xyxy_bounds() -> None:
    mask = torch.zeros((8, 10), dtype=torch.float32)
    mask[2:6, 3:8] = 1.0

    box = compute_bbox_from_mask(mask)

    assert torch.equal(box, torch.tensor([3.0, 2.0, 7.0, 5.0]))


def test_expand_box_clips_to_image_bounds() -> None:
    box = torch.tensor([1.0, 2.0, 4.0, 6.0])

    expanded = expand_box(box, height=8, width=6, rescale_factor=2.0)

    assert torch.all(expanded >= 0.0)
    assert expanded[2] <= 5.0
    assert expanded[3] <= 7.0


def test_expand_box_targets_three_four_width_height_ratio() -> None:
    box = torch.tensor([3.0, 2.0, 5.0, 10.0])

    expanded = expand_box(box, height=20, width=20, rescale_factor=1.5)

    expanded_w = expanded[2] - expanded[0] + 1.0
    expanded_h = expanded[3] - expanded[1] + 1.0
    assert torch.isclose(expanded_w / expanded_h, torch.tensor(3.0 / 4.0), atol=0.15)


def test_canonicalize_hand_crop_flips_left_hand_only() -> None:
    crop = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)

    right = canonicalize_hand_crop(crop, is_right=torch.tensor(True))
    left = canonicalize_hand_crop(crop, is_right=torch.tensor(False))

    assert torch.equal(right, crop)
    assert torch.equal(left, torch.flip(crop, dims=[2]))


def test_crop_and_resize_respects_requested_output_size() -> None:
    img = torch.arange(3 * 12 * 10, dtype=torch.float32).reshape(3, 12, 10)
    box = torch.tensor([2.0, 3.0, 7.0, 10.0])

    hamer_crop = crop_and_resize(img, box, output_hw=(256, 192))
    dino_crop = crop_and_resize(img, box, output_hw=(224, 168))

    assert tuple(hamer_crop.shape) == (3, 256, 192)
    assert tuple(dino_crop.shape) == (3, 224, 168)


def test_crop_and_resize_preserves_full_output_shape_for_wide_crop() -> None:
    img = torch.ones((1, 4, 10), dtype=torch.float32)
    box = torch.tensor([0.0, 1.0, 9.0, 2.0])

    crop = crop_and_resize(img, box, output_hw=(8, 8))

    assert tuple(crop.shape) == (1, 8, 8)
    assert torch.all(crop > 0)


def test_crop_and_resize_preserves_full_output_shape_for_tall_crop() -> None:
    img = torch.ones((1, 10, 4), dtype=torch.float32)
    box = torch.tensor([1.0, 0.0, 2.0, 9.0])

    crop = crop_and_resize(img, box, output_hw=(8, 8))

    assert tuple(crop.shape) == (1, 8, 8)
    assert torch.all(crop > 0)


def test_hamer_and_dino_resize_paths_produce_same_patch_grid() -> None:
    hamer_grid = compute_patch_grid(image_hw=(256, 192), patch_size=16)
    dino_grid = compute_patch_grid(image_hw=(224, 168), patch_size=14)

    assert hamer_grid == (16, 12)
    assert dino_grid == hamer_grid
