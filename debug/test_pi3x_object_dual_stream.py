from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from pi3.models.pi3x import Pi3X


class _SpyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, imgs, is_training=True):
        self.calls.append(imgs.detach().clone())
        batch = imgs.shape[0]
        return {"x_norm_patchtokens": torch.randn(batch, 4, 1024, device=imgs.device, dtype=imgs.dtype)}


class _IdentityBlock(nn.Module):
    def forward(self, x, xpos=None, attn_mask=None, attn_keep_mask=None):
        return x


class _SpyObjectQueryAdapter(nn.Module):
    def __init__(self, token_dim: int = 1024) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.last_call: dict[str, torch.Tensor | tuple[int, int]] | None = None
        self.valid_token = nn.Parameter(torch.full((1, 1, 1, token_dim), 2.0), requires_grad=False)
        self.empty_token = nn.Parameter(torch.full((1, 1, 1, token_dim), -3.0), requires_grad=False)

    def forward(self, rgb_patch_tokens, grasped_object_mask, grasped_object_valid, image_hw):
        self.last_call = {
            "rgb_patch_tokens": rgb_patch_tokens.detach().clone(),
            "grasped_object_mask": grasped_object_mask.detach().clone(),
            "grasped_object_valid": grasped_object_valid.detach().clone(),
            "image_hw": image_hw,
        }
        batch, num_views = grasped_object_valid.shape
        object_query = self.empty_token.expand(batch, num_views, 1, self.token_dim).clone()
        object_query[grasped_object_valid] = self.valid_token[0, 0, 0]
        object_query_pos = torch.zeros(batch, num_views, 1, 2, dtype=torch.long, device=grasped_object_valid.device)
        return {
            "object_query": object_query,
            "object_query_pos": object_query_pos,
            "object_valid": grasped_object_valid,
        }


class _SpyObjectPoseHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, object_query_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        self.last_input = object_query_feat.detach().clone()
        route_code = object_query_feat[..., :1]
        return {
            "rot6d": route_code.expand(-1, -1, 6).clone(),
            "trans": route_code.expand(-1, -1, 3).clone(),
            "log_scale": route_code.clone(),
            "scale": route_code.clone(),
        }


class Pi3XObjectDualStreamTests(unittest.TestCase):
    def test_forward_uses_shared_object_stream_encoder_and_routes_invalid_object_view_through_empty_query(self) -> None:
        model = Pi3X(use_multimodal=False).eval()
        encoder = _SpyEncoder()
        object_query_adapter = _SpyObjectQueryAdapter(token_dim=model.dec_embed_dim)
        object_pose_head = _SpyObjectPoseHead()

        model.encoder = encoder
        model.decoder = nn.ModuleList([_IdentityBlock(), _IdentityBlock()])
        model.patch_size = 4
        model.object_query_adapter = object_query_adapter
        model.object_pose_head = object_pose_head

        imgs = torch.rand(1, 2, 3, 8, 8)
        canonical_imgs = torch.rand(1, 8, 3, 8, 8)
        object_multiview = {
            "img": canonical_imgs,
            "depthmap": torch.zeros(1, 8, 8, 8),
            "camera_intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(1, 8, 1, 1),
            "camera_pose": torch.eye(4).view(1, 1, 4, 4).repeat(1, 8, 1, 1),
            "grasped_object_mask": torch.tensor(
                [
                    [
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        [[0, 0, 0, 0, 0, 0, 0, 0]] * 8,
                    ]
                ],
                dtype=torch.float32,
            ).reshape(1, 2, 8, 8),
            "grasped_object_valid": torch.tensor([[True, False]], dtype=torch.bool),
        }

        out = model(imgs, object_multiview=object_multiview)

        expected_scene_batch = ((imgs - model.image_mean) / model.image_std).reshape(2, 3, 8, 8)
        expected_object_batch = ((canonical_imgs - model.image_mean) / model.image_std).reshape(8, 3, 8, 8)
        self.assertEqual(len(encoder.calls), 2)
        self.assertTrue(torch.allclose(encoder.calls[0], expected_scene_batch))
        self.assertTrue(torch.allclose(encoder.calls[1], expected_object_batch))
        self.assertIsNotNone(object_query_adapter.last_call)
        self.assertTrue(torch.equal(object_query_adapter.last_call["grasped_object_valid"], object_multiview["grasped_object_valid"]))
        self.assertEqual(object_query_adapter.last_call["image_hw"], (8, 8))
        self.assertIsNotNone(object_pose_head.last_input)
        self.assertEqual(object_pose_head.last_input[0, 0, 0].item(), 2.0)
        self.assertEqual(object_pose_head.last_input[0, 1, 0].item(), -3.0)
        self.assertIn("pred_object_rot6d", out)
        self.assertIn("pred_object_trans", out)
        self.assertIn("pred_object_scale", out)
        self.assertEqual(tuple(out["pred_object_rot6d"].shape), (1, 2, 6))
        self.assertEqual(tuple(out["pred_object_trans"].shape), (1, 2, 3))
        self.assertEqual(tuple(out["pred_object_scale"].shape), (1, 2, 1))
        self.assertTrue(torch.equal(out["pred_object_rot6d"][0, 0], torch.full((6,), 2.0)))
        self.assertTrue(torch.equal(out["pred_object_rot6d"][0, 1], torch.full((6,), -3.0)))


if __name__ == "__main__":
    unittest.main()
