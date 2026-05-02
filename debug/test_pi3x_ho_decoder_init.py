from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from pi3.models.layers.block import HOBlockRope, init_ho_block_from_decoder_block
from pi3.models.pi3x import Pi3X


class HOBlockWarmStartTests(unittest.TestCase):
    def test_helper_copies_decoder_weights_into_ho_block(self) -> None:
        model = Pi3X(use_multimodal=False).eval()
        blk = model.decoder[0]
        ho_blk = HOBlockRope(
            dim=model.dec_embed_dim,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            act_layer=torch.nn.GELU,
            norm_layer=type(blk.norm1),
            ffn_layer=type(blk.mlp),
            init_values=0.01,
            qk_norm=True,
            attn_class=type(blk.attn),
            rope=model.rope,
        )

        init_ho_block_from_decoder_block(ho_blk, blk, cross_scale=1e-3)

        self.assertTrue(torch.equal(ho_blk.norm1.weight, blk.norm1.weight))
        self.assertTrue(torch.equal(ho_blk.norm1.bias, blk.norm1.bias))
        self.assertTrue(torch.equal(ho_blk.attn.qkv.weight, blk.attn.qkv.weight))
        self.assertTrue(torch.equal(ho_blk.attn.proj.weight, blk.attn.proj.weight))
        self.assertTrue(torch.equal(ho_blk.norm3.weight, blk.norm2.weight))
        self.assertTrue(torch.equal(ho_blk.mlp.fc1.weight, blk.mlp.fc1.weight))
        self.assertTrue(torch.equal(ho_blk.ls1.gamma, blk.ls1.gamma))
        self.assertTrue(torch.equal(ho_blk.ls2.gamma, blk.ls2.gamma))
        self.assertTrue(torch.equal(ho_blk.norm2.weight, blk.norm1.weight))
        self.assertTrue(torch.equal(ho_blk.norm_y.weight, blk.norm1.weight))

        dim = ho_blk.cross_attn.q_proj.weight.shape[0]
        self.assertTrue(torch.equal(ho_blk.cross_attn.q_proj.weight, blk.attn.qkv.weight[:dim]))
        self.assertTrue(torch.equal(ho_blk.cross_attn.k_proj.weight, blk.attn.qkv.weight[dim:2 * dim]))
        self.assertTrue(torch.equal(ho_blk.cross_attn.v_proj.weight, blk.attn.qkv.weight[2 * dim:3 * dim]))
        self.assertTrue(torch.equal(ho_blk.cross_attn.proj.weight, blk.attn.proj.weight))
        self.assertTrue(torch.equal(ho_blk.cross_attn.q_norm.weight, blk.attn.q_norm.weight))
        self.assertTrue(torch.equal(ho_blk.cross_attn.k_norm.weight, blk.attn.k_norm.weight))
        self.assertTrue(torch.allclose(ho_blk.ls_y.gamma, torch.full_like(ho_blk.ls_y.gamma, 1e-3)))

    def test_pi3x_initializes_every_ho_decoder_layer_from_decoder(self) -> None:
        model = Pi3X(use_multimodal=False).eval()

        for blk, ho_blk in zip(model.decoder, model.ho_decoder):
            self.assertTrue(torch.equal(ho_blk.attn.qkv.weight, blk.attn.qkv.weight))
            self.assertTrue(torch.equal(ho_blk.norm1.weight, blk.norm1.weight))
            self.assertTrue(torch.equal(ho_blk.norm3.weight, blk.norm2.weight))
            self.assertTrue(torch.equal(ho_blk.cross_attn.proj.weight, blk.attn.proj.weight))
            self.assertTrue(torch.allclose(ho_blk.ls_y.gamma, torch.full_like(ho_blk.ls_y.gamma, 1e-3)))

    def test_pi3x_ckpt_loads_ho_and_lazy_hand_head_weights(self) -> None:
        model = Pi3X(use_multimodal=False).eval()
        hand_head = model._get_hand_mano_head(torch.device("cpu"))
        with torch.no_grad():
            model.ho_decoder[0].ls_y.gamma.fill_(0.25)
            hand_head.dectransl_scale.bias.fill_(0.5)
            hand_head.decscale.bias.fill_(1.5)

        with tempfile.TemporaryDirectory(prefix="pi3x_ckpt_") as tmpdir:
            ckpt_path = Path(tmpdir) / "pi3x.pt"
            torch.save({"state_dict": model.state_dict()}, ckpt_path)

            loaded = Pi3X(use_multimodal=False, ckpt=str(ckpt_path)).eval()

        self.assertIsNotNone(loaded.hand_mano_head)
        self.assertTrue(torch.allclose(loaded.ho_decoder[0].ls_y.gamma, model.ho_decoder[0].ls_y.gamma))
        self.assertTrue(torch.allclose(loaded.hand_mano_head.dectransl_scale.bias, hand_head.dectransl_scale.bias))
        self.assertTrue(torch.allclose(loaded.hand_mano_head.decscale.bias, hand_head.decscale.bias))


if __name__ == "__main__":
    unittest.main()
