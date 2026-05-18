from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandTokenAdapter(nn.Module):
    def __init__(self, token_dim: int = 1024, patch_size: int = 14, max_num_hands: int = 2):
        super().__init__()
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.max_num_hands = int(max_num_hands)

        self.side_embed = nn.Embedding(2, self.token_dim)
        self.empty_hand_token = nn.Parameter(torch.zeros(1, 1, 1, self.token_dim))
        self.fuse_mlp = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )

    def _normalize_mask_shape(self, hand_masks: torch.Tensor) -> torch.Tensor:
        if hand_masks.ndim == 4 and hand_masks.shape[1] == 1:
            return hand_masks[:, 0]
        if hand_masks.ndim == 3:
            return hand_masks
        raise ValueError("hand_masks must have shape (K,H,W) or (K,1,H,W)")

    def _compute_patch_grid(self, rgb_patch_tokens: torch.Tensor, image_hw: tuple[int, int]) -> tuple[int, int]:
        image_h, image_w = image_hw
        patch_h = image_h // self.patch_size
        patch_w = image_w // self.patch_size
        if patch_h * patch_w != rgb_patch_tokens.shape[2]:
            raise ValueError("rgb_patch_tokens shape does not match image_hw and patch_size")
        return patch_h, patch_w

    def _compute_mask_weights_and_pos(
        self,
        mask: torch.Tensor,
        patch_h: int,
        patch_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        downsampled = F.interpolate(
            mask[None, None].float(),
            size=(patch_h, patch_w),
            mode="area",
        ).reshape(-1)

        if float(downsampled.sum().item()) <= 0:
            ys, xs = torch.nonzero(mask > 0, as_tuple=True)
            if xs.numel() == 0:
                raise ValueError("hand mask is empty")
            center_y = (ys.min().float() + ys.max().float()) * 0.5
            center_x = (xs.min().float() + xs.max().float()) * 0.5
            patch_y = torch.clamp((center_y / self.patch_size).floor().long(), min=0, max=patch_h - 1)
            patch_x = torch.clamp((center_x / self.patch_size).floor().long(), min=0, max=patch_w - 1)
            downsampled = mask.new_zeros((patch_h * patch_w,), dtype=torch.float32)
            downsampled[patch_y * patch_w + patch_x] = 1.0
            pos = torch.stack([patch_y, patch_x])
            return downsampled, pos

        weights = downsampled / downsampled.sum().clamp_min(1e-6)

        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        center_y = (ys.min().float() + ys.max().float()) * 0.5
        center_x = (xs.min().float() + xs.max().float()) * 0.5
        patch_y = torch.clamp((center_y / self.patch_size).floor().long(), min=0, max=patch_h - 1)
        patch_x = torch.clamp((center_x / self.patch_size).floor().long(), min=0, max=patch_w - 1)
        pos = torch.stack([patch_y, patch_x])
        return weights, pos

    def _pool_rgb_features(
        self,
        rgb_patch_tokens: torch.Tensor,
        hand_masks: torch.Tensor,
        owner_index: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_h, patch_w = self._compute_patch_grid(rgb_patch_tokens, image_hw)
        pooled = []
        positions = []
        for idx in range(hand_masks.shape[0]):
            b, n, _ = owner_index[idx].tolist()
            weights, pos = self._compute_mask_weights_and_pos(hand_masks[idx], patch_h, patch_w)
            pooled.append((rgb_patch_tokens[b, n] * weights.to(rgb_patch_tokens.device).unsqueeze(-1)).sum(dim=0))
            positions.append(pos.to(device=rgb_patch_tokens.device, dtype=torch.long))
        if not pooled:
            return (
                rgb_patch_tokens.new_zeros((0, self.token_dim)),
                owner_index.new_zeros((0, 2)),
            )
        return torch.stack(pooled, dim=0), torch.stack(positions, dim=0)

    def _scatter_dense(
        self,
        sparse_tokens: torch.Tensor,
        sparse_pos: torch.Tensor,
        owner_index: torch.Tensor,
        batch_size: int,
        num_views: int,
        feature_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if owner_index.numel() == 0:
            return None, None, None, 0

        max_slot = int(owner_index[:, 2].max().item())
        if max_slot >= self.max_num_hands:
            raise ValueError(f"owner_index slot must be < {self.max_num_hands}, got {max_slot}")
        num_hand_tokens = max_slot + 1
        dim = feature_dim if feature_dim is not None else self.token_dim

        dense_tokens = sparse_tokens.new_zeros((batch_size, num_views, num_hand_tokens, dim))
        dense_pos = owner_index.new_zeros((batch_size, num_views, num_hand_tokens, 2))
        dense_valid_mask = torch.zeros((batch_size, num_views, num_hand_tokens), dtype=torch.bool, device=owner_index.device)

        for idx in range(owner_index.shape[0]):
            b, n, m = owner_index[idx].tolist()
            dense_tokens[b, n, m] = sparse_tokens[idx]
            dense_pos[b, n, m] = sparse_pos[idx]
            dense_valid_mask[b, n, m] = True

        return dense_tokens, dense_pos, dense_valid_mask, num_hand_tokens

    def forward(
        self,
        rgb_patch_tokens: torch.Tensor,
        hand_queries: torch.Tensor,
        hand_masks: torch.Tensor,
        owner_index: torch.Tensor,
        hand_is_right: torch.Tensor,
        image_hw: tuple[int, int],
        hand_betas: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int | None]:
        # TODO 1.2: 明显没有必要设置这个函数，mask形状规范一下就行
        hand_masks = self._normalize_mask_shape(hand_masks)
        if hand_masks.shape[0] != hand_queries.shape[0]:
            raise ValueError("hand_masks and hand_queries must align row-wise")
        if hand_masks.shape[0] != owner_index.shape[0] or hand_masks.shape[0] != hand_is_right.shape[0]:
            raise ValueError("hand_masks, owner_index, and hand_is_right must align row-wise")
        if hand_queries.shape[-1] != self.token_dim:
            raise ValueError("hand_queries last dimension must match token_dim")
        if hand_betas is not None and hand_betas.shape[0] != hand_queries.shape[0]:
            raise ValueError("hand_betas and hand_queries must align row-wise")

        # 处理完全没有手的情况，因为要保证所有的情况下手相关的损失都走一遍计算图，多卡运算时需要保证计算图一致，这样多卡计算梯度就不会空等。
        if hand_queries.shape[0] == 0:
            batch_size, num_views = rgb_patch_tokens.shape[:2]
            dummy_query = rgb_patch_tokens.new_zeros((1, self.token_dim))
            dummy_side = torch.zeros((1,), dtype=torch.long, device=rgb_patch_tokens.device)
            dummy_rgb = rgb_patch_tokens.new_zeros((1, self.token_dim))
            dummy_token = self.fuse_mlp(torch.cat([dummy_query + self.side_embed(dummy_side), dummy_rgb], dim=-1))
            dense_tokens = self.empty_hand_token.expand(batch_size, num_views, 1, self.token_dim).clone()
            dense_tokens = dense_tokens + dummy_token.view(1, 1, 1, self.token_dim)
            dense_pos = owner_index.new_zeros((batch_size, num_views, 1, 2))
            dense_valid_mask = torch.zeros((batch_size, num_views, 1), dtype=torch.bool, device=rgb_patch_tokens.device)
            dense_betas = rgb_patch_tokens.new_zeros((batch_size, num_views, 1, 10))
            return {
                "sparse_tokens": hand_queries.new_zeros((0, self.token_dim)),
                "sparse_pos": owner_index.new_zeros((0, 2)),
                "sparse_betas": rgb_patch_tokens.new_zeros((0, 10)),
                "dense_tokens": dense_tokens,
                "dense_pos": dense_pos,
                "dense_valid_mask": dense_valid_mask,
                "num_hand_tokens": 1,
                "dense_betas": dense_betas,
            }

        dino_hand_feat, sparse_pos = self._pool_rgb_features(rgb_patch_tokens, hand_masks, owner_index, image_hw)
        query_with_side = hand_queries + self.side_embed(hand_is_right.long())
        sparse_tokens = self.fuse_mlp(torch.cat([query_with_side, dino_hand_feat], dim=-1))
        sparse_betas = hand_betas if hand_betas is not None else rgb_patch_tokens.new_zeros((sparse_tokens.shape[0], 10))
        dense_tokens, dense_pos, dense_valid_mask, num_hand_tokens = self._scatter_dense(
            sparse_tokens,
            sparse_pos,
            owner_index,
            batch_size=rgb_patch_tokens.shape[0],
            num_views=rgb_patch_tokens.shape[1],
        )
        dense_betas, _, _, _ = self._scatter_dense(
            sparse_betas,
            torch.zeros_like(sparse_pos),
            owner_index,
            batch_size=rgb_patch_tokens.shape[0],
            num_views=rgb_patch_tokens.shape[1],
            feature_dim=10,
        )
        return {
            "sparse_tokens": sparse_tokens,
            "sparse_pos": sparse_pos,
            "sparse_betas": sparse_betas,
            "dense_tokens": dense_tokens,
            "dense_pos": dense_pos,
            "dense_valid_mask": dense_valid_mask,
            "num_hand_tokens": num_hand_tokens,
            "dense_betas": dense_betas,
        }
