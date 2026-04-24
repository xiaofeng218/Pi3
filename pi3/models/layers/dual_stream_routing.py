from __future__ import annotations

import torch


def build_scene_image_attn_mask(
    num_scene_tokens: int,
    num_hand_queries: int,
    num_object_queries: int,
) -> torch.Tensor:
    total_tokens = num_scene_tokens + num_hand_queries + num_object_queries
    keep_mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool)
    scene_end = num_scene_tokens
    keep_mask[:scene_end, :scene_end] = True
    keep_mask[scene_end:, :scene_end] = True
    keep_mask[scene_end:, scene_end:] = True
    return keep_mask


def flatten_valid_hand_queries(
    hand_queries: torch.Tensor,
    hand_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hand_queries.ndim != 4:
        raise ValueError("hand_queries must have shape (B, N, K, C)")
    if hand_valid.shape != hand_queries.shape[:-1]:
        raise ValueError("hand_valid must have shape (B, N, K) aligned with hand_queries")

    owner_index = torch.nonzero(hand_valid, as_tuple=False)
    if owner_index.numel() == 0:
        empty_flat = hand_queries.new_zeros((0, hand_queries.shape[-1]))
        empty_owner = owner_index.new_zeros((0, 3))
        return empty_flat, empty_owner

    flat_queries = hand_queries[owner_index[:, 0], owner_index[:, 1], owner_index[:, 2]]
    return flat_queries, owner_index


def scatter_valid_hand_queries(
    base: torch.Tensor,
    owner_index: torch.Tensor,
    flat_queries: torch.Tensor,
) -> torch.Tensor:
    if base.ndim != 4:
        raise ValueError("base must have shape (B, N, K, C)")
    if owner_index.ndim != 2 or owner_index.shape[-1] != 3:
        raise ValueError("owner_index must have shape (M, 3)")
    if flat_queries.ndim != 2 or flat_queries.shape[-1] != base.shape[-1]:
        raise ValueError("flat_queries must have shape (M, C) aligned with base")
    if owner_index.shape[0] != flat_queries.shape[0]:
        raise ValueError("owner_index and flat_queries must have the same leading dimension")

    scattered = base.clone()
    if owner_index.numel() == 0:
        return scattered

    scattered[owner_index[:, 0], owner_index[:, 1], owner_index[:, 2]] = flat_queries
    return scattered


def flatten_object_global_memory(
    object_tokens: torch.Tensor,
    patch_start_idx: int,
) -> torch.Tensor:
    if object_tokens.ndim != 4:
        raise ValueError("object_tokens must have shape (B, N, T, C)")
    if patch_start_idx < 0 or patch_start_idx > object_tokens.shape[2]:
        raise ValueError("patch_start_idx must fall within the token dimension")

    patch_tokens = object_tokens[:, :, patch_start_idx:, :]
    batch_size, num_views, num_patches, channels = patch_tokens.shape
    return patch_tokens.reshape(batch_size, num_views * num_patches, channels)
