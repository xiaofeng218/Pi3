from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from trainers.pi3x_training_policy import apply_pi3x_training_policy


class _DummyHOBlock(nn.Module):
    """Minimal block that mimics HOBlockRope structure for training policy tests."""

    def __init__(self) -> None:
        super().__init__()
        # Self-attention
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(2, 6)  # dim * 3
        self.attn.proj = nn.Linear(2, 2)

        # Cross-attention
        self.cross_attn = nn.Module()
        self.cross_attn.q_proj = nn.Linear(2, 2)
        self.cross_attn.k_proj = nn.Linear(2, 2)
        self.cross_attn.v_proj = nn.Linear(2, 2)
        self.cross_attn.proj = nn.Linear(2, 2)

        # Norms
        self.norm1 = nn.LayerNorm(2)
        self.norm2 = nn.LayerNorm(2)
        self.norm3 = nn.LayerNorm(2)
        self.norm_y = nn.LayerNorm(2)

        # Layer scales
        self.ls1 = nn.Module()
        self.ls1.gamma = nn.Parameter(torch.ones(2))
        self.ls2 = nn.Module()
        self.ls2.gamma = nn.Parameter(torch.ones(2))
        self.ls_y = nn.Module()
        self.ls_y.gamma = nn.Parameter(torch.ones(2))

        # FFN (Mlp style)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(2, 8)
        self.mlp.fc2 = nn.Linear(8, 2)


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
        self.hand_global_decoder = nn.Linear(2, 2)
        self.hand_pose_decoder = nn.Linear(2, 2)
        self.object_pose_decoder = nn.Linear(2, 2)
        self.hand_global_head = nn.Linear(2, 2)
        self.hand_pose_head = nn.Linear(2, 2)
        self.hand_mano_head = nn.Linear(2, 2)
        self.object_pose_head = nn.Linear(2, 2)
        self.hand_token_fuse = nn.Linear(2, 2)
        self.ho_hand_scene_cross_alpha = nn.Parameter(torch.zeros(2))
        self.ho_object_scene_cross_alpha = nn.Parameter(torch.zeros(2))
        self.ho_decoder = nn.ModuleList([_DummyHOBlock(), _DummyHOBlock()])


class Pi3XTrainingPolicyTests(unittest.TestCase):
    def test_head_and_adapter_parameters_remain_trainable(self) -> None:
        model = _DummyPi3X()

        apply_pi3x_training_policy(model)

        trainable = {name for name, param in model.named_parameters() if param.requires_grad}

        self.assertIn("hand_token_adapter.weight", trainable)
        self.assertIn("object_query_adapter.weight", trainable)
        self.assertIn("hand_global_decoder.weight", trainable)
        self.assertIn("hand_pose_decoder.weight", trainable)
        self.assertIn("object_pose_decoder.weight", trainable)
        self.assertIn("hand_global_head.weight", trainable)
        self.assertIn("hand_pose_head.weight", trainable)
        self.assertIn("hand_mano_head.weight", trainable)
        self.assertIn("object_pose_head.weight", trainable)
        self.assertIn("hand_token_fuse.weight", trainable)
        self.assertIn("ho_hand_scene_cross_alpha", trainable)
        self.assertIn("ho_object_scene_cross_alpha", trainable)

    def test_core_model_parameters_are_frozen(self) -> None:
        model = _DummyPi3X()

        apply_pi3x_training_policy(model)

        trainable = {name for name, param in model.named_parameters() if param.requires_grad}

        self.assertNotIn("register_token", trainable)
        self.assertNotIn("metric_token", trainable)
        self.assertNotIn("decoder.weight", trainable)
        self.assertNotIn("point_decoder.weight", trainable)
        self.assertNotIn("metric_head.weight", trainable)

    def test_register_and_metric_tokens_are_frozen(self) -> None:
        model = _DummyPi3X()

        apply_pi3x_training_policy(model)

        self.assertFalse(model.register_token.requires_grad)
        self.assertFalse(model.metric_token.requires_grad)

    def test_ho_decoder_is_fully_trainable(self) -> None:
        model = _DummyPi3X()

        apply_pi3x_training_policy(model)

        trainable = {name for name, param in model.named_parameters() if param.requires_grad}

        # Cross-attention (full fine-tune)
        self.assertIn("ho_decoder.0.cross_attn.q_proj.weight", trainable)
        self.assertIn("ho_decoder.0.cross_attn.k_proj.weight", trainable)
        self.assertIn("ho_decoder.0.cross_attn.v_proj.weight", trainable)
        self.assertIn("ho_decoder.0.cross_attn.proj.weight", trainable)

        # Norms (full fine-tune)
        self.assertIn("ho_decoder.0.norm1.weight", trainable)
        self.assertIn("ho_decoder.0.norm2.weight", trainable)
        self.assertIn("ho_decoder.0.norm3.weight", trainable)
        self.assertIn("ho_decoder.0.norm_y.weight", trainable)

        # Layer scales (full fine-tune)
        self.assertIn("ho_decoder.0.ls1.gamma", trainable)
        self.assertIn("ho_decoder.0.ls2.gamma", trainable)
        self.assertIn("ho_decoder.0.ls_y.gamma", trainable)

        # Self-attention (LoRA: base weight frozen, lora_A/lora_B trainable)
        self.assertFalse(model.ho_decoder[0].attn.qkv.weight.requires_grad)
        self.assertTrue(model.ho_decoder[0].attn.qkv.lora_A.requires_grad)
        self.assertTrue(model.ho_decoder[0].attn.qkv.lora_B.requires_grad)

        # FFN (LoRA: base weight frozen, lora_A/lora_B trainable)
        self.assertFalse(model.ho_decoder[0].mlp.fc1.weight.requires_grad)
        self.assertTrue(model.ho_decoder[0].mlp.fc1.lora_A.requires_grad)
        self.assertTrue(model.ho_decoder[0].mlp.fc1.lora_B.requires_grad)

    def test_lora_replaces_linear_layers(self) -> None:
        from pi3.models.layers.lora import LoRALinear

        model = _DummyPi3X()

        apply_pi3x_training_policy(model)

        # After policy application, qkv should be replaced with LoRALinear
        self.assertIsInstance(model.ho_decoder[0].attn.qkv, LoRALinear)
        self.assertIsInstance(model.ho_decoder[0].attn.proj, LoRALinear)
        self.assertIsInstance(model.ho_decoder[0].mlp.fc1, LoRALinear)
        self.assertIsInstance(model.ho_decoder[0].mlp.fc2, LoRALinear)

        # Cross-attention linear layers should NOT be LoRA (full fine-tune)
        self.assertNotIsInstance(model.ho_decoder[0].cross_attn.q_proj, LoRALinear)
        self.assertNotIsInstance(model.ho_decoder[0].cross_attn.k_proj, LoRALinear)
        self.assertNotIsInstance(model.ho_decoder[0].cross_attn.v_proj, LoRALinear)
        self.assertNotIsInstance(model.ho_decoder[0].cross_attn.proj, LoRALinear)


if __name__ == "__main__":
    unittest.main()
