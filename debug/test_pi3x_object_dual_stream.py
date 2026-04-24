from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from pi3.models.pi3x import Pi3X


class _SpyEncoder(nn.Module):
    def __init__(self, token_dim: int = 1024, patch_tokens: int = 4) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.patch_tokens = patch_tokens
        self.seen_batches: list[torch.Tensor] = []

    def forward(self, imgs, is_training=True):
        self.seen_batches.append(imgs.detach().clone())
        batch = imgs.shape[0]
        image_signal = imgs.mean(dim=(1, 2, 3), keepdim=False).view(batch, 1, 1)
        patch_offset = torch.arange(self.patch_tokens, device=imgs.device, dtype=imgs.dtype).view(1, self.patch_tokens, 1)
        feat_offset = torch.linspace(0.0, 1.0, self.token_dim, device=imgs.device, dtype=imgs.dtype).view(1, 1, self.token_dim)
        tokens = image_signal + patch_offset + feat_offset
        return {"x_norm_patchtokens": tokens}

    def saw_all_normalized_samples(self, imgs: torch.Tensor, image_mean: torch.Tensor, image_std: torch.Tensor) -> bool:
        expected = ((imgs - image_mean) / image_std).reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
        if not self.seen_batches:
            return False
        observed = torch.cat(self.seen_batches, dim=0)
        return all(any(torch.allclose(sample, seen) for seen in observed) for sample in expected)


class _MixingBlock(nn.Module):
    def forward(self, x, xpos=None, attn_mask=None, attn_keep_mask=None):
        context = x.mean(dim=1, keepdim=True)
        return x + context


class _SpyObjectQueryAdapter(nn.Module):
    def __init__(self, token_dim: int = 1024) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.calls: list[dict[str, torch.Tensor | tuple[int, int]]] = []
        self.valid_token = nn.Parameter(torch.full((1, 1, 1, token_dim), 2.0), requires_grad=False)
        self.empty_token = nn.Parameter(torch.full((1, 1, 1, token_dim), -3.0), requires_grad=False)

    def forward(self, rgb_patch_tokens, grasped_object_mask, grasped_object_valid, image_hw):
        self.calls.append(
            {
                "rgb_patch_tokens": rgb_patch_tokens.detach().clone(),
                "grasped_object_mask": grasped_object_mask.detach().clone(),
                "grasped_object_valid": grasped_object_valid.detach().clone(),
                "image_hw": image_hw,
            }
        )
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
        self.inputs: list[torch.Tensor] = []

    def forward(self, object_query_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        self.inputs.append(object_query_feat.detach().clone())
        summary = object_query_feat.mean(dim=-1, keepdim=True)
        return {
            "rot6d": summary.expand(-1, -1, 6).clone(),
            "trans": summary.expand(-1, -1, 3).clone(),
            "log_scale": summary.clone(),
            "scale": torch.exp(summary),
        }


def _build_object_multiview(canonical_imgs: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
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


class Pi3XObjectDualStreamTests(unittest.TestCase):
    def test_forward_object_predictions_depend_on_canonical_stream_and_preserve_invalid_view_routing(self) -> None:
        model = Pi3X(use_multimodal=False).eval()
        encoder = _SpyEncoder(token_dim=model.dec_embed_dim, patch_tokens=4)
        object_query_adapter = _SpyObjectQueryAdapter(token_dim=model.dec_embed_dim)
        object_pose_head = _SpyObjectPoseHead()

        model.encoder = encoder
        model.decoder = nn.ModuleList([_MixingBlock(), _MixingBlock()])
        model.patch_size = 4
        model.object_query_adapter = object_query_adapter
        model.object_pose_head = object_pose_head

        imgs = torch.rand(1, 2, 3, 8, 8)
        canonical_imgs_a = torch.full((1, 8, 3, 8, 8), 0.2)
        canonical_imgs_b = torch.full((1, 8, 3, 8, 8), 0.8)
        object_multiview_a = _build_object_multiview(canonical_imgs_a)
        object_multiview_b = _build_object_multiview(canonical_imgs_b)

        out_a = model(imgs, object_multiview=object_multiview_a)
        out_b = model(imgs, object_multiview=object_multiview_b)

        self.assertTrue(encoder.saw_all_normalized_samples(canonical_imgs_a, model.image_mean, model.image_std))
        self.assertTrue(encoder.saw_all_normalized_samples(canonical_imgs_b, model.image_mean, model.image_std))
        self.assertEqual(len(object_query_adapter.calls), 2)
        self.assertTrue(
            torch.equal(
                object_query_adapter.calls[0]["grasped_object_valid"],
                object_multiview_a["grasped_object_valid"],
            )
        )
        self.assertTrue(
            torch.equal(
                object_query_adapter.calls[1]["grasped_object_valid"],
                object_multiview_b["grasped_object_valid"],
            )
        )
        self.assertEqual(object_query_adapter.calls[0]["image_hw"], (8, 8))
        self.assertEqual(object_query_adapter.calls[1]["image_hw"], (8, 8))
        self.assertEqual(len(object_pose_head.inputs), 2)
        self.assertFalse(torch.allclose(object_pose_head.inputs[0][0, 0], object_pose_head.inputs[1][0, 0]))
        self.assertFalse(torch.allclose(object_pose_head.inputs[0][0, 0], object_pose_head.inputs[0][0, 1]))
        self.assertIn("pred_object_rot6d", out_a)
        self.assertIn("pred_object_trans", out_a)
        self.assertIn("pred_object_scale", out_a)
        self.assertEqual(tuple(out_a["pred_object_rot6d"].shape), (1, 2, 6))
        self.assertEqual(tuple(out_a["pred_object_trans"].shape), (1, 2, 3))
        self.assertEqual(tuple(out_a["pred_object_scale"].shape), (1, 2, 1))
        self.assertTrue(torch.isfinite(out_a["pred_object_rot6d"]).all())
        self.assertTrue(torch.isfinite(out_a["pred_object_trans"]).all())
        self.assertTrue(torch.isfinite(out_a["pred_object_scale"]).all())
        self.assertFalse(torch.allclose(out_a["pred_object_rot6d"][0, 0], out_b["pred_object_rot6d"][0, 0]))


if __name__ == "__main__":
    unittest.main()
