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
        del is_training
        self.seen_batches.append(imgs.detach().clone())
        batch = imgs.shape[0]
        tokens = torch.arange(batch * self.patch_tokens * self.token_dim, dtype=imgs.dtype, device=imgs.device)
        tokens = tokens.reshape(batch, self.patch_tokens, self.token_dim)
        return {"x_norm_patchtokens": tokens}


class _SpyHandEncoder(nn.Module):
    def __init__(self, token_dim: int = 1024, num_tokens: int = 3) -> None:
        super().__init__()
        self.output_dim = token_dim
        self.num_tokens = num_tokens
        self.calls = []

    def forward(self, imgs, hand_masks, hand_is_right):
        self.calls.append((imgs.detach().clone(), hand_masks.detach().clone(), hand_is_right.detach().clone()))
        batch, views = imgs.shape[:2]
        tokens = imgs.new_zeros((batch, views, self.num_tokens, self.output_dim))
        tokens[:, :, :, 0] = 5.0
        betas = imgs.new_zeros((batch, views, 10))
        valid = (hand_masks.sum(dim=(-1, -2)) > 0)
        return {
            "hand_tokens": tokens,
            "hand_betas": betas,
            "hand_valid_mask": valid,
            "hand_is_right": hand_is_right,
            "crop_boxes": imgs.new_zeros((batch, views, 4)),
            "raw_hand_masks": hand_masks,
        }


class _SpyHandAdapter(nn.Module):
    def __init__(self, token_dim: int = 1024, num_tokens: int = 3) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.crop_hw = (14, 42)
        self.calls = []

    def forward(self, imgs, hand_masks, hand_is_right, encoder):
        self.calls.append((imgs.detach().clone(), hand_masks.detach().clone(), hand_is_right.detach().clone(), encoder))
        batch, views = imgs.shape[:2]
        tokens = imgs.new_zeros((batch, views, self.num_tokens, self.token_dim))
        tokens[:, :, :, 1] = 7.0
        valid = (hand_masks.sum(dim=(-1, -2)) > 0)
        return {
            "hand_tokens": tokens,
            "hand_valid_mask": valid,
            "crop_boxes": imgs.new_zeros((batch, views, 4)),
        }


class _SpyObjectAdapter(nn.Module):
    def __init__(self, token_dim: int = 1024, num_tokens: int = 3) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.crop_hw = (14, 42)
        self.calls = []

    def forward(self, imgs, object_masks, object_valid, encoder):
        self.calls.append((imgs.detach().clone(), object_masks.detach().clone(), object_valid.detach().clone(), encoder))
        batch, views = imgs.shape[:2]
        tokens = imgs.new_zeros((batch, views, self.num_tokens, self.token_dim))
        tokens[:, :, :, 2] = 11.0
        return {
            "object_tokens": tokens,
            "object_valid_mask": object_valid,
            "crop_boxes": imgs.new_zeros((batch, views, 4)),
        }


class _SpyHandMANOHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, hand_pose, hand_is_right=None, hand_betas=None, global_orient=None, transl=None, scale=None):
        self.inputs.append(
            (
                hand_pose.detach().clone(),
                hand_is_right.detach().clone(),
                hand_betas.detach().clone(),
                None if global_orient is None else global_orient.detach().clone(),
                None if transl is None else transl.detach().clone(),
                None if scale is None else scale.detach().clone(),
            )
        )
        batch = hand_pose.shape[0]
        eye = torch.eye(3, device=hand_pose.device).view(1, 1, 3, 3).repeat(batch, 1, 1, 1)
        return {
            "pred_mano_params": {
                "global_orient": global_orient,
                "hand_pose": hand_pose,
                "betas": hand_betas,
            },
            "pred_hand_mano_betas": hand_betas,
            "pred_hand_joints_3d": torch.zeros((batch, 21, 3), device=hand_pose.device),
            "pred_hand_vertices": torch.zeros((batch, 778, 3), device=hand_pose.device),
            "pred_hand_joints_local": torch.zeros((batch, 21, 3), device=hand_pose.device),
            "pred_hand_vertices_local": torch.zeros((batch, 778, 3), device=hand_pose.device),
        }


class _SpyHandPoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, hand_tokens, valid_mask=None):
        self.inputs.append((hand_tokens.detach().clone(), None if valid_mask is None else valid_mask.detach().clone()))
        batch, views = hand_tokens.shape[:2]
        eye = torch.eye(3, device=hand_tokens.device).view(1, 1, 1, 3, 3).repeat(batch, views, 15, 1, 1)
        return {
            "pred_hand_pose_6d": torch.zeros((batch, views, 96), device=hand_tokens.device),
            "pred_hand_global_orient": torch.eye(3, device=hand_tokens.device).view(1, 1, 1, 3, 3).repeat(batch, views, 1, 1, 1).reshape(batch, views, 1, 3, 3),
            "pred_hand_pose": eye,
        }


class _SpyTaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, tokens, xpos=None):
        self.inputs.append((tokens.detach().clone(), None if xpos is None else xpos.detach().clone()))
        pad_dim = 512 - tokens.shape[-1]
        if pad_dim <= 0:
            return tokens[..., :512]
        pad = torch.zeros((*tokens.shape[:-1], pad_dim), device=tokens.device, dtype=tokens.dtype)
        return torch.cat([tokens, pad], dim=-1)


class _SpyHandGlobalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, hand_tokens, valid_mask=None):
        self.inputs.append((hand_tokens.detach().clone(), None if valid_mask is None else valid_mask.detach().clone()))
        batch, views = hand_tokens.shape[:2]
        eye = torch.eye(3, device=hand_tokens.device).view(1, 1, 1, 3, 3).repeat(batch, views, 1, 1, 1).reshape(batch, views, 1, 3, 3)
        return {
            "pred_hand_global_orient_6d": torch.zeros((batch, views, 6), device=hand_tokens.device),
            "pred_hand_global_orient": eye,
            "pred_hand_transl_dir": torch.zeros((batch, views, 3), device=hand_tokens.device),
            "pred_hand_transl_log_scale": torch.zeros((batch, views, 1), device=hand_tokens.device),
            "pred_hand_transl_scale": torch.ones((batch, views, 1), device=hand_tokens.device),
            "pred_hand_transl": torch.zeros((batch, views, 3), device=hand_tokens.device),
            "pred_hand_log_scale": torch.zeros((batch, views, 1), device=hand_tokens.device),
            "pred_hand_scale": torch.ones((batch, views, 1), device=hand_tokens.device),
        }


class _SpyObjectPoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, object_tokens, valid_mask=None):
        self.inputs.append((object_tokens.detach().clone(), None if valid_mask is None else valid_mask.detach().clone()))
        batch, views = object_tokens.shape[:2]
        return {
            "rot6d": torch.zeros((batch, views, 6), device=object_tokens.device),
            "trans_dir": torch.zeros((batch, views, 3), device=object_tokens.device),
            "trans_log_scale": torch.zeros((batch, views, 1), device=object_tokens.device),
            "trans_scale": torch.ones((batch, views, 1), device=object_tokens.device),
            "trans": torch.zeros((batch, views, 3), device=object_tokens.device),
            "log_scale": torch.zeros((batch, views, 1), device=object_tokens.device),
            "scale": torch.ones((batch, views, 1), device=object_tokens.device),
        }


class Pi3XDenseLocalTokenTests(unittest.TestCase):
    def test_forward_uses_dense_local_token_interfaces(self) -> None:
        model = Pi3X(use_multimodal=False).eval()
        model.encoder = _SpyEncoder(token_dim=model.dec_embed_dim, patch_tokens=4)
        model.hand_encoder = _SpyHandEncoder(token_dim=model.dec_embed_dim, num_tokens=3)
        model.hand_token_adapter = _SpyHandAdapter(token_dim=model.dec_embed_dim, num_tokens=3)
        model.object_query_adapter = _SpyObjectAdapter(token_dim=model.dec_embed_dim, num_tokens=3)
        model.hand_global_decoder = _SpyTaskDecoder()
        model.hand_pose_decoder = _SpyTaskDecoder()
        model.object_pose_decoder = _SpyTaskDecoder()
        model.hand_global_head = _SpyHandGlobalHead()
        model.hand_pose_head = _SpyHandPoseHead()
        model.hand_mano_head = _SpyHandMANOHead()
        model.object_pose_head = _SpyObjectPoseHead()

        captured = {}

        def _fake_decode(scene_hidden, object_hidden=None, hand_hidden=None):
            captured["hand_hidden"] = hand_hidden
            captured["object_hidden"] = object_hidden
            batch = 1
            views = 2
            return (
                torch.zeros((batch * views, 9, model.dec_embed_dim * 2)),
                torch.zeros((batch * views, 9, 2)),
                hand_hidden[0].reshape(batch * views, hand_hidden[0].shape[2], hand_hidden[0].shape[3]) if hand_hidden is not None else None,
                object_hidden[0].reshape(batch * views, object_hidden[0].shape[2], object_hidden[0].shape[3]) if object_hidden is not None else None,
            )

        model.decode = _fake_decode
        object.__setattr__(
            model,
            "forward_head",
            lambda *args, **kwargs: {
                "points": torch.zeros((1, 2, 8, 8, 3)),
                "local_points": torch.zeros((1, 2, 8, 8, 3)),
                "rays": torch.zeros((1, 2, 8, 8, 3)),
                "conf": torch.zeros((1, 2, 8, 8, 1)),
                "camera_poses": torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
                "metric": torch.ones((1,)),
            },
        )

        imgs = torch.rand(1, 2, 3, 8, 8)
        hand_masks = torch.zeros(1, 2, 8, 8)
        hand_masks[0, 0, 1:5, 1:5] = 1.0
        hand_is_right = torch.tensor([[True, False]], dtype=torch.bool)
        object_masks = torch.zeros(1, 2, 8, 8)
        object_masks[0, 1, 2:6, 2:6] = 1.0
        object_valid = torch.tensor([[False, True]], dtype=torch.bool)
        object_multiview = {
            "img": torch.rand(1, 2, 3, 8, 8),
            "depthmap": torch.zeros(1, 2, 8, 8),
            "camera_intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
            "camera_pose": torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
        }

        out = model(
            imgs,
            hand_masks=hand_masks,
            hand_is_right=hand_is_right,
            object_masks=object_masks,
            object_valid=object_valid,
            object_multiview=object_multiview,
        )

        self.assertIn("hand_hidden", captured)
        self.assertEqual(tuple(captured["hand_hidden"][0].shape), (1, 2, 3 + model.patch_start_idx, model.dec_embed_dim))
        self.assertEqual(tuple(captured["hand_hidden"][2].shape), (1, 2))
        self.assertIn("object_hidden", captured)
        self.assertEqual(tuple(captured["object_hidden"][0].shape), (1, 2, 3 + model.patch_start_idx, model.dec_embed_dim))
        self.assertEqual(tuple(captured["object_hidden"][2].shape), (1, 2))
        self.assertEqual(tuple(model.hand_global_decoder.inputs[0][0].shape), (2, 3 + model.patch_start_idx, model.dec_embed_dim))
        self.assertEqual(tuple(model.object_pose_decoder.inputs[0][0].shape), (2, 3 + model.patch_start_idx, model.dec_embed_dim))
        self.assertIn("pred_hand_mano_params", out)
        self.assertIn("pred_object_rot6d", out)


if __name__ == "__main__":
    unittest.main()
