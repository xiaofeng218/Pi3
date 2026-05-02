from __future__ import annotations

import unittest

from pi3.models.pi3x import Pi3X
from trainers.pi3x_training_policy import apply_pi3x_training_policy


class Pi3XTrainingPolicyTests(unittest.TestCase):
    def test_pi3x_policy_leaves_only_expected_modules_trainable(self) -> None:
        model = Pi3X(
            use_multimodal=False,
            hand_mano_layer=None,
            ho_lora_cfg={"rank": 4, "alpha": 8, "targets": ["cross_attn"]},
        )
        apply_pi3x_training_policy(model)

        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertTrue(any(name.startswith("ho_decoder") and "lora_" in name for name in trainable))
        self.assertFalse(any(name.startswith("encoder.") for name in trainable))
        self.assertFalse(any(name.startswith("decoder.") for name in trainable))
        self.assertFalse("depth_emb" in trainable)
        self.assertFalse("ray_embed.proj.weight" in trainable)


if __name__ == "__main__":
    unittest.main()
