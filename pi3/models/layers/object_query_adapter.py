from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectQueryAdapter(nn.Module):
    def __init__(self, token_dim: int = 1024, patch_size: int = 14):
        super().__init__()
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.empty_object_query = nn.Parameter(torch.zeros(1, 1, 1, self.token_dim))

    def _compute_patch_grid(
        self,
        rgb_patch_tokens: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[int, int]:
        image_h, image_w = image_hw
        patch_h = image_h // self.patch_size
        patch_w = image_w // self.patch_size
        if patch_h * patch_w != rgb_patch_tokens.shape[2]:
            raise ValueError("rgb_patch_tokens shape does not match image_hw and patch_size")
        return patch_h, patch_w

    def _validate_shapes(
        self,
        rgb_patch_tokens: torch.Tensor,
        grasped_object_mask: torch.Tensor,
        grasped_object_valid: torch.Tensor,
    ) -> None:
        if rgb_patch_tokens.ndim != 4:
            raise ValueError("rgb_patch_tokens must have shape (B,N,P,C)")
        if grasped_object_mask.ndim != 4:
            raise ValueError("grasped_object_mask must have shape (B,N,H,W)")
        if grasped_object_valid.ndim != 2:
            raise ValueError("grasped_object_valid must have shape (B,N)")

        batch_shape = rgb_patch_tokens.shape[:2]
        if grasped_object_mask.shape[:2] != batch_shape:
            raise ValueError("grasped_object_mask batch/view shape must match rgb_patch_tokens")
        if grasped_object_valid.shape != batch_shape:
            raise ValueError("grasped_object_valid batch/view shape must match rgb_patch_tokens")

    def _pool_single(
        self,
        patch_tokens: torch.Tensor,
        mask: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        weights = F.interpolate(
            mask[None, None].float(),
            size=(patch_h, patch_w),
            mode="area",
        ).reshape(-1)
        if float(weights.sum().item()) <= 0:
            return (
                patch_tokens.new_zeros((self.token_dim,)),
                patch_tokens.new_zeros((2,), dtype=torch.long),
                False,
            )

        weights = weights / weights.sum().clamp_min(1e-6)
        pooled = (patch_tokens * weights.to(device=patch_tokens.device).unsqueeze(-1)).sum(dim=0)

        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        center_y = (ys.min().float() + ys.max().float()) * 0.5
        center_x = (xs.min().float() + xs.max().float()) * 0.5
        pos = torch.stack(
            [
                torch.clamp((center_y / self.patch_size).floor().long(), min=0, max=patch_h - 1),
                torch.clamp((center_x / self.patch_size).floor().long(), min=0, max=patch_w - 1),
            ]
        )
        return pooled, pos, True

    def forward(
        self,
        rgb_patch_tokens: torch.Tensor,
        grasped_object_mask: torch.Tensor,
        grasped_object_valid: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        self._validate_shapes(rgb_patch_tokens, grasped_object_mask, grasped_object_valid)
        patch_h, patch_w = self._compute_patch_grid(rgb_patch_tokens, image_hw)
        batch, num_views = grasped_object_valid.shape

        object_query = self.empty_object_query.expand(batch, num_views, 1, self.token_dim).clone()
        object_query_pos = grasped_object_valid.new_zeros((batch, num_views, 1, 2), dtype=torch.long)

        for batch_idx in range(batch):
            for view_idx in range(num_views):
                if not bool(grasped_object_valid[batch_idx, view_idx]):
                    continue
                pooled, pos, has_support = self._pool_single(
                    rgb_patch_tokens[batch_idx, view_idx],
                    grasped_object_mask[batch_idx, view_idx],
                    patch_h,
                    patch_w,
                )
                if not has_support:
                    continue
                object_query[batch_idx, view_idx, 0] = pooled
                object_query_pos[batch_idx, view_idx, 0] = pos

        return {
            "object_query": object_query,
            "object_query_pos": object_query_pos,
            "object_valid": grasped_object_valid,
        }
