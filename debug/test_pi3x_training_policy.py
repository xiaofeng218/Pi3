from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from trainers.pi3x_training_policy import apply_pi3x_training_policy


class _DummyPi3X(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_token = nn.Parameter(torch.ones(1))
        self.metric_token = nn.Parameter(torch.ones(1))
        self.decoder = nn.Linear(2, 2)
        self.point_decoder = nn.Linear(2, 2)
        self.metric_head = nn.Linear(2, 1)
        self.hand_token_adapter = nn.Linear(2, 2)
        self.object_query_adapter = nn.Linear(2, 2)
        self.hand_mano_head = nn.Linear(2, 2)
        self.object_pose_head = nn.Linear(2, 2)
        self.ho_decoder = nn.Module()
        self.ho_decoder.lora_proj = nn.Linear(2, 2)
        self.ho_decoder.norm1 = nn.LayerNorm(2)
        self.ho_decoder.ls1 = nn.Parameter(torch.ones(2))
        self.ho_decoder.proj = nn.Linear(2, 2)


class Pi3XTrainingPolicyTests(unittest.TestCase):
    def test_only_ho_related_parameters_remain_trainable(self) -> None:
        model = _DummyPi3X()

        apply_pi3x_training_policy(model)

        trainable = {name for name, param in model.named_parameters() if param.requires_grad}

        self.assertIn("hand_token_adapter.weight", trainable)
        self.assertIn("object_query_adapter.weight", trainable)
        self.assertIn("hand_mano_head.weight", trainable)
        self.assertIn("object_pose_head.weight", trainable)

        self.assertNotIn("register_token", trainable)
        self.assertNotIn("metric_token", trainable)
        self.assertNotIn("decoder.weight", trainable)
        self.assertNotIn("point_decoder.weight", trainable)
        self.assertNotIn("metric_head.weight", trainable)
        self.assertIn("ho_decoder.lora_proj.weight", trainable)
        self.assertIn("ho_decoder.norm1.weight", trainable)
        self.assertIn("ho_decoder.ls1", trainable)
        self.assertIn("ho_decoder.proj.bias", trainable)
        self.assertNotIn("ho_decoder.proj.weight", trainable)

    def test_register_and_metric_tokens_are_frozen(self) -> None:
        model = _DummyPi3X()

        apply_pi3x_training_policy(model)

        self.assertFalse(model.register_token.requires_grad)
        self.assertFalse(model.metric_token.requires_grad)

    def test_ho_decoder_only_trains_lora_norm_ls_and_bias(self) -> None:
        model = _DummyPi3X()

        apply_pi3x_training_policy(model)

        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertIn("ho_decoder.lora_proj.weight", trainable)
        self.assertIn("ho_decoder.norm1.weight", trainable)
        self.assertIn("ho_decoder.ls1", trainable)
        self.assertIn("ho_decoder.proj.bias", trainable)
        self.assertNotIn("ho_decoder.proj.weight", trainable)


if __name__ == "__main__":
    unittest.main()
