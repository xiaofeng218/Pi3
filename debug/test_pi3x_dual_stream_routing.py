from __future__ import annotations

import unittest
from unittest import mock

import torch

import pi3.models.layers.attention as attention_mod
from pi3.models.layers.attention import (
    Attention,
    AttentionRope,
    CrossAttentionRope,
    FlashAttention,
    FlashCrossAttentionRope,
    FlashAttentionRope,
    MemEffAttention,
    MemEffAttentionRope,
)
from pi3.models.layers.dual_stream_routing import (
    build_scene_image_attn_mask,
    flatten_object_global_memory,
    scatter_valid_hand_queries,
    flatten_valid_hand_queries,
)


class _FakeRope(torch.nn.Module):
    def forward(self, x: torch.Tensor, xpos: torch.Tensor | None) -> torch.Tensor:
        if xpos is None:
            return x
        return x + xpos[..., :1].unsqueeze(1).to(dtype=x.dtype, device=x.device)


def _set_identity_attention_weights(attn: torch.nn.Module) -> None:
    with torch.no_grad():
        attn.qkv.weight.zero_()
        attn.qkv.weight[0:2] = torch.eye(2)
        attn.qkv.weight[2:4] = torch.eye(2)
        attn.qkv.weight[4:6] = torch.eye(2)
        attn.proj.weight.copy_(torch.eye(2))


def _set_identity_cross_attention_weights(attn: torch.nn.Module) -> None:
    with torch.no_grad():
        attn.q_proj.weight.copy_(torch.eye(2))
        attn.k_proj.weight.copy_(torch.eye(2))
        attn.v_proj.weight.copy_(torch.eye(2))
        attn.proj.weight.copy_(torch.eye(2))


class Pi3XDualStreamRoutingTests(unittest.TestCase):
    def test_attention_accepts_scene_image_keep_mask_and_preserves_scene_only_reads(self) -> None:
        attn = Attention(dim=2, num_heads=1, qkv_bias=False, proj_bias=False)
        _set_identity_attention_weights(attn)

        tokens = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [10.0, 10.0],
                ]
            ],
            dtype=torch.float32,
        )
        keep_mask = build_scene_image_attn_mask(num_scene_tokens=2, num_hand_queries=1, num_object_queries=0)

        masked = attn(tokens, attn_bias=keep_mask)
        scene_only = attn(tokens[:, :2], attn_bias=None)

        self.assertTrue(torch.allclose(masked[:, :2], scene_only, atol=1e-6, rtol=1e-6))

    def test_memeff_attention_fallback_preserves_scene_keep_mask(self) -> None:
        attn = MemEffAttention(dim=2, num_heads=1, qkv_bias=False, proj_bias=False)
        reference = Attention(dim=2, num_heads=1, qkv_bias=False, proj_bias=False)
        _set_identity_attention_weights(attn)
        _set_identity_attention_weights(reference)
        reference.load_state_dict(attn.state_dict())

        tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]], dtype=torch.float32)
        keep_mask = build_scene_image_attn_mask(num_scene_tokens=2, num_hand_queries=1, num_object_queries=0)

        with mock.patch.object(attention_mod, "XFORMERS_AVAILABLE", False):
            masked = attn(tokens, attn_bias=keep_mask)
        expected = reference(tokens, attn_bias=keep_mask)

        self.assertTrue(torch.allclose(masked, expected, atol=1e-6, rtol=1e-6))

    def test_memeff_attention_rope_fallback_preserves_mask_and_positions(self) -> None:
        rope = _FakeRope()
        attn = MemEffAttentionRope(dim=2, num_heads=1, qkv_bias=False, proj_bias=False, rope=rope)
        reference = AttentionRope(dim=2, num_heads=1, qkv_bias=False, proj_bias=False, rope=rope)
        _set_identity_attention_weights(attn)
        _set_identity_attention_weights(reference)
        reference.load_state_dict(attn.state_dict())

        tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]], dtype=torch.float32)
        xpos = torch.tensor([[[0.25, 0.0], [0.5, 0.0], [0.75, 0.0]]], dtype=torch.float32)
        keep_mask = build_scene_image_attn_mask(num_scene_tokens=2, num_hand_queries=1, num_object_queries=0)

        with mock.patch.object(attention_mod, "XFORMERS_AVAILABLE", False):
            masked = attn(tokens, attn_bias=keep_mask, xpos=xpos)
        expected = reference(tokens, attn_bias=keep_mask, xpos=xpos)

        self.assertTrue(torch.allclose(masked, expected, atol=1e-6, rtol=1e-6))

    def test_cross_attention_boolean_keep_mask_uses_true_as_keep_semantics(self) -> None:
        attn = CrossAttentionRope(dim=2, num_heads=1, qkv_bias=False, proj_bias=False)
        _set_identity_cross_attention_weights(attn)

        query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        key = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)
        value = torch.tensor([[[3.0, 0.0], [0.0, 5.0]]], dtype=torch.float32)
        keep_mask = torch.tensor([[True, False]], dtype=torch.bool)

        masked = attn(query, key, value, attn_bias=keep_mask)
        expected = value[:, :1, :]

        self.assertTrue(torch.allclose(masked, expected, atol=1e-6, rtol=1e-6))

    def test_flash_attention_keep_mask_matches_base_attention_contract(self) -> None:
        attn = FlashAttention(dim=2, num_heads=1, qkv_bias=False, proj_bias=False)
        reference = Attention(dim=2, num_heads=1, qkv_bias=False, proj_bias=False)
        _set_identity_attention_weights(attn)
        _set_identity_attention_weights(reference)
        reference.load_state_dict(attn.state_dict())

        tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]], dtype=torch.float32)
        keep_mask = build_scene_image_attn_mask(num_scene_tokens=2, num_hand_queries=1, num_object_queries=0)

        masked = attn(tokens, attn_bias=keep_mask)
        expected = reference(tokens, attn_bias=keep_mask)

        self.assertTrue(torch.allclose(masked, expected, atol=1e-6, rtol=1e-6))

    def test_flash_cross_attention_keep_mask_matches_base_cross_attention_contract(self) -> None:
        attn = FlashCrossAttentionRope(dim=2, num_heads=1, qkv_bias=False, proj_bias=False)
        reference = CrossAttentionRope(dim=2, num_heads=1, qkv_bias=False, proj_bias=False)
        _set_identity_cross_attention_weights(attn)
        _set_identity_cross_attention_weights(reference)
        reference.load_state_dict(attn.state_dict())

        query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        key = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)
        value = torch.tensor([[[3.0, 0.0], [0.0, 5.0]]], dtype=torch.float32)
        keep_mask = torch.tensor([[True, False]], dtype=torch.bool)

        masked = attn(query, key, value, attn_bias=keep_mask)
        expected = reference(query, key, value, attn_bias=keep_mask)

        self.assertTrue(torch.allclose(masked, expected, atol=1e-6, rtol=1e-6))

    def test_build_scene_image_attn_mask_blocks_scene_queries_from_reading_hand_and_object_queries(self) -> None:
        mask = build_scene_image_attn_mask(num_scene_tokens=5, num_hand_queries=2, num_object_queries=1)

        self.assertEqual(tuple(mask.shape), (8, 8))
        self.assertTrue(mask[:5, :5].all())
        self.assertFalse(mask[:5, 5:].any())
        self.assertTrue(mask[5:, :5].all())
        self.assertTrue(mask[5:, 5:].all())

    def test_flatten_valid_hand_queries_drops_invalid_slots_and_returns_owner_index(self) -> None:
        hand_queries = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
        hand_valid = torch.tensor(
            [
                [[True, False], [False, False], [True, True]],
                [[False, True], [False, False], [False, False]],
            ],
            dtype=torch.bool,
        )

        flat, owner = flatten_valid_hand_queries(hand_queries, hand_valid)

        self.assertEqual(tuple(flat.shape), (4, 4))
        self.assertEqual(tuple(owner.shape), (4, 3))
        expected_owner = torch.tensor(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 2, 1],
                [1, 0, 1],
            ],
            dtype=torch.long,
        )
        expected_flat = torch.stack(
            [
                hand_queries[0, 0, 0],
                hand_queries[0, 2, 0],
                hand_queries[0, 2, 1],
                hand_queries[1, 0, 1],
            ],
            dim=0,
        )
        self.assertTrue(torch.equal(owner, expected_owner))
        self.assertTrue(torch.equal(flat, expected_flat))

    def test_flatten_valid_hand_queries_returns_empty_outputs_for_all_invalid_slots(self) -> None:
        hand_queries = torch.arange(2 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 2, 3)
        hand_valid = torch.zeros(2, 2, 2, dtype=torch.bool)

        flat, owner = flatten_valid_hand_queries(hand_queries, hand_valid)

        self.assertEqual(tuple(flat.shape), (0, 3))
        self.assertEqual(tuple(owner.shape), (0, 3))
        self.assertEqual(flat.dtype, hand_queries.dtype)
        self.assertEqual(owner.dtype, torch.long)

    def test_scatter_valid_hand_queries_writes_back_only_valid_slots(self) -> None:
        base = torch.zeros(2, 2, 2, 3, dtype=torch.float32)
        owner = torch.tensor(
            [
                [0, 0, 1],
                [1, 1, 0],
            ],
            dtype=torch.long,
        )
        flat = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        )

        scattered = scatter_valid_hand_queries(base, owner, flat)

        expected = torch.zeros_like(base)
        expected[0, 0, 1] = flat[0]
        expected[1, 1, 0] = flat[1]
        self.assertTrue(torch.equal(scattered, expected))

    def test_flatten_object_global_memory_skips_register_tokens(self) -> None:
        object_tokens = torch.arange(1 * 3 * 7 * 2, dtype=torch.float32).reshape(1, 3, 7, 2)

        memory = flatten_object_global_memory(object_tokens, patch_start_idx=5)

        expected_memory = torch.cat(
            [
                object_tokens[:, 0, 5:, :],
                object_tokens[:, 1, 5:, :],
                object_tokens[:, 2, 5:, :],
            ],
            dim=1,
        )

        self.assertEqual(tuple(memory.shape), (1, 3 * 2, 2))
        self.assertTrue(torch.equal(memory, expected_memory))


if __name__ == "__main__":
    unittest.main()
