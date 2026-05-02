from __future__ import annotations

import unittest

import torch

from trainers.pi3x_trainer import Pi3XTrainer


class _DummyTrainableModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_token_adapter = torch.nn.Linear(4, 4)
        self.object_query_adapter = torch.nn.Linear(4, 4)
        self.hand_mano_head = torch.nn.Linear(4, 4)
        self.object_pose_head = torch.nn.Linear(4, 4)
        self.register_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.metric_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.ho_decoder = torch.nn.Module()
        self.ho_decoder.lora_a = torch.nn.Parameter(torch.ones(4, 4))
        self.ho_decoder.bias = torch.nn.Parameter(torch.zeros(4))
        self.ho_decoder.norm = torch.nn.LayerNorm(4)
        self.encoder = torch.nn.Linear(4, 4)


class _DummyPi3XModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_multimodal = True
        self.hand_token_adapter = torch.nn.Linear(4, 4)
        self.object_query_adapter = torch.nn.Linear(4, 4)
        self.hand_mano_head = torch.nn.Linear(4, 4)
        self.object_pose_head = torch.nn.Linear(4, 4)
        self.register_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.metric_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.ho_decoder = torch.nn.Module()
        self.ho_decoder.lora_a = torch.nn.Parameter(torch.ones(4, 4))
        self.ho_decoder.bias = torch.nn.Parameter(torch.zeros(4))
        self.ho_decoder.norm = torch.nn.LayerNorm(4)
        self.hand_mano_layer = None

    def forward(self, **kwargs):
        imgs = kwargs["imgs"]
        batch, views = imgs.shape[:2]
        height, width = imgs.shape[-2:]
        num_hands = 0 if kwargs.get("hand_queries") is None else kwargs["hand_queries"].shape[0]
        local_points = torch.ones((batch, views, height, width, 3), device=imgs.device)
        local_points[..., 2] = 2.0
        pred = {
            "local_points": local_points,
            "points": torch.ones_like(local_points),
            "camera_poses": torch.eye(4, device=imgs.device)[None, None].repeat(batch, views, 1, 1),
            "metric": torch.ones((batch,), device=imgs.device),
            "pred_hand_transl_dir": torch.tensor([[1.0, 0.0, 0.0]], device=imgs.device).repeat(num_hands, 1),
            "pred_hand_transl_log_scale": torch.zeros((num_hands, 1), device=imgs.device),
            "pred_hand_transl_scale": torch.ones((num_hands, 1), device=imgs.device),
            "pred_hand_transl": torch.tensor([[1.0, 0.0, 0.0]], device=imgs.device).repeat(num_hands, 1),
            "pred_hand_log_scale": torch.zeros((num_hands, 1), device=imgs.device),
            "pred_hand_scale": torch.ones((num_hands, 1), device=imgs.device),
            "pred_hand_mano_params": {
                "global_orient": torch.zeros((num_hands, 1, 3, 3), device=imgs.device),
                "hand_pose": torch.zeros((num_hands, 15, 3, 3), device=imgs.device),
                "betas": torch.zeros((num_hands, 10), device=imgs.device),
            },
            "pred_object_rot6d": torch.zeros((batch, views, 6), device=imgs.device),
            "pred_object_transl_dir": torch.tensor([[[0.0, 1.0, 0.0]]], device=imgs.device).repeat(batch, views, 1),
            "pred_object_transl_log_scale": torch.zeros((batch, views, 1), device=imgs.device),
            "pred_object_transl_scale": torch.ones((batch, views, 1), device=imgs.device),
            "pred_object_trans": torch.tensor([[[0.0, 1.0, 0.0]]], device=imgs.device).repeat(batch, views, 1),
            "pred_object_log_scale": torch.zeros((batch, views, 1), device=imgs.device),
            "pred_object_scale": torch.ones((batch, views, 1), device=imgs.device),
            "object_valid": torch.ones((batch, views), dtype=torch.bool, device=imgs.device),
        }
        return pred


class _DummyHandEncoder(torch.nn.Module):
    def forward(self, imgs, hand_masks, owner_index, hand_is_right):
        del imgs, hand_masks
        batch = owner_index.shape[0]
        return {
            "hand_queries": torch.arange(batch * 4, dtype=torch.float32, device=owner_index.device).view(batch, 4),
            "owner_index": owner_index,
            "hand_is_right": hand_is_right,
        }


class _Cfg(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Pi3XTrainerSmokeTests(unittest.TestCase):
    def test_build_optimizer_groups_trainable_modules_as_expected(self) -> None:
        cfg = _Cfg(type="AdamW", lr=1e-4, weight_decay=5e-2, betas=[0.9, 0.95], encoder_lr=1e-5)
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        model = _DummyTrainableModule()

        optimizer = trainer.build_optimizer(cfg, model)
        self.assertEqual(len(optimizer.param_groups), 3)
        group_lrs = sorted({group["lr"] for group in optimizer.param_groups})
        self.assertEqual(group_lrs, [1e-4])
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertIn("ho_decoder.lora_a", trainable)
        self.assertIn("hand_token_adapter.weight", trainable)

    def test_forward_batch_builds_hand_and_object_inputs(self) -> None:
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        trainer.model = _DummyPi3XModel()
        trainer.hand_encoder = _DummyHandEncoder()
        trainer.train_loss = torch.nn.Identity()
        trainer.test_loss = torch.nn.Identity()
        trainer.accelerator = type("A", (), {"device": torch.device("cpu")})()

        def _view():
            return {
                "img": torch.zeros(1, 3, 4, 4),
                "depthmap": torch.ones(1, 4, 4),
                "camera_intrinsics": torch.eye(3).unsqueeze(0),
                "camera_pose": torch.eye(4).unsqueeze(0),
                "hand": {
                    "mask": torch.ones(1, 4, 4),
                    "valid": torch.tensor([True]),
                    "pose_mano": torch.zeros(1, 48),
                    "hand_transl": torch.tensor([[1.0, 0.0, 0.0]]),
                    "joints_3d_cam": torch.zeros(1, 21, 3),
                    "joints_2d": torch.zeros(1, 21, 2),
                    "mano_betas": torch.zeros(1, 10),
                    "mano_side": ["right"],
                },
                "object_multiview": {
                    "img": torch.zeros(1, 2, 3, 4, 4),
                    "depthmap": torch.ones(1, 2, 4, 4),
                    "camera_intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
                    "camera_pose": torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
                    "pts3d": torch.zeros(1, 2, 4, 4, 3),
                    "normalization_center": torch.zeros(1, 3),
                    "normalization_scale": torch.tensor([1.5]),
                    "grasped_object_id": torch.tensor([1]),
                    "grasped_object_mask": torch.ones(1, 4, 4),
                    "grasped_object_valid": torch.tensor([True]),
                    "grasped_object_pose_obj2cam": torch.eye(4).unsqueeze(0),
                },
            }

        batch = [_view(), _view()]
        pred, gt = trainer.forward_batch(batch, mode="train")

        self.assertIn("scene_scale", gt)
        self.assertEqual(tuple(gt["object_pose_obj2cam"].shape), (1, 2, 4, 4))
        self.assertEqual(tuple(gt["object_valid"].shape), (1, 2))
        self.assertEqual(tuple(gt["hand_transl"].shape), (2, 3))
        self.assertTrue(torch.allclose(gt["scene_scale"], torch.full((1,), 2.0)))
        self.assertEqual(tuple(pred["pred_object_rot6d"].shape), (1, 2, 6))
        self.assertEqual(tuple(pred["pred_hand_transl"].shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
