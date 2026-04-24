from __future__ import annotations

import unittest

import torch

from pi3.models.pi3x import Pi3X


class _FakeEncoder(torch.nn.Module):
    def forward(self, imgs, is_training=True):
        batch = imgs.shape[0]
        return {"x_norm_patchtokens": torch.randn(batch, 4, 1024, device=imgs.device, dtype=imgs.dtype)}


class _IdentityBlock(torch.nn.Module):
    def forward(self, x, xpos=None, attn_mask=None, attn_keep_mask=None):
        return x


class Pi3XObjectDualStreamTests(unittest.TestCase):
    def test_forward_emits_object_pose_predictions_for_valid_object_views(self) -> None:
        model = Pi3X(use_multimodal=False).eval()
        model.encoder = _FakeEncoder()
        model.decoder = torch.nn.ModuleList([_IdentityBlock(), _IdentityBlock()])
        model.patch_size = 4

        imgs = torch.rand(1, 2, 3, 8, 8)
        object_multiview = {
            "img": torch.rand(1, 8, 3, 8, 8),
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

        self.assertIn("pred_object_rot6d", out)
        self.assertIn("pred_object_trans", out)
        self.assertIn("pred_object_scale", out)
        self.assertEqual(tuple(out["pred_object_rot6d"].shape), (1, 2, 6))
        self.assertEqual(tuple(out["pred_object_trans"].shape), (1, 2, 3))
        self.assertEqual(tuple(out["pred_object_scale"].shape), (1, 2, 1))


if __name__ == "__main__":
    unittest.main()
