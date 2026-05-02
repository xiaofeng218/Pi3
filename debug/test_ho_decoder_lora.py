from __future__ import annotations

import unittest

import torch

from pi3.models.layers.attention import FlashAttentionRope, FlashCrossAttentionRope
from pi3.models.layers.block import HOBlockRope


class HODecoderLoRATests(unittest.TestCase):
    def test_ho_block_accepts_lora_cfg_and_keeps_output_shape(self) -> None:
        blk = HOBlockRope(
            dim=64,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            act_layer=torch.nn.GELU,
            norm_layer=torch.nn.LayerNorm,
            attn_class=FlashAttentionRope,
            cross_attn_class=FlashCrossAttentionRope,
            init_values=0.01,
            qk_norm=True,
            rope=None,
            lora_cfg={"rank": 4, "alpha": 8, "targets": ["cross_attn"]},
        )

        x = torch.randn(2, 5, 64)
        y = torch.randn(2, 7, 64)

        out = blk(x, y, xpos=None, ypos=None)

        self.assertEqual(tuple(out.shape), tuple(x.shape))
        trainable = [name for name, p in blk.named_parameters() if p.requires_grad]
        self.assertTrue(any("lora_" in name for name in trainable))
        self.assertTrue(
            all(
                ("lora_" in name)
                or ("norm" in name)
                or ("ls" in name)
                or name.endswith("bias")
                for name in trainable
            )
        )


if __name__ == "__main__":
    unittest.main()
