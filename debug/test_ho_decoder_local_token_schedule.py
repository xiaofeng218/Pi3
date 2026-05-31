from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from pi3.models.pi3x import Pi3X


class _PassSceneBlock(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = float(delta)
        self.calls = 0

    def forward(self, x, xpos=None, attn_mask=None):
        del xpos, attn_mask
        self.calls += 1
        return x + self.delta


class _PassCrossBlock(nn.Module):
    def __init__(self, delta: float = 0.0):
        super().__init__()
        self.delta = float(delta)
        self.calls = 0
        self.cross_calls = 0

    def forward(self, x, y=None, xpos=None, ypos=None, enable_self_attn=True, enable_cross_attn=True, cross_alpha=None):
        del xpos, ypos
        self.calls += 1
        out = x
        if enable_self_attn:
            out = out + 1.0
        if enable_cross_attn and y is not None:
            self.cross_calls += 1
            cross = y.mean(dim=1, keepdim=True)
            if cross_alpha is not None:
                cross = cross * torch.as_tensor(cross_alpha, device=x.device, dtype=x.dtype)
            out = out + cross
        return out + self.delta


class Pi3XHODecoderScheduleTests(unittest.TestCase):
    def test_decode_keeps_scene_and_omv_progression_while_updating_dense_local_tokens(self) -> None:
        model = Pi3X(use_multimodal=False).eval()
        model.decoder = nn.ModuleList([_PassSceneBlock(10.0), _PassSceneBlock(20.0)])
        model.ho_decoder = nn.ModuleList([_PassCrossBlock(), _PassCrossBlock()])
        model.ho_hand_scene_cross_alpha.data.zero_()
        model.ho_object_scene_cross_alpha.data.zero_()
        hidden = torch.zeros((2, 9, model.dec_embed_dim))
        pos = torch.zeros((2, 9, 2), dtype=torch.long)
        scene_hidden = (hidden, 2, pos, None, torch.ones((1, 2), dtype=torch.bool), None, 8, 8, 0)

        hand_tokens = torch.ones((1, 2, 3, model.dec_embed_dim))
        hand_pos = torch.zeros((1, 2, 3, 2), dtype=torch.long)
        hand_valid = torch.tensor([[True, False]], dtype=torch.bool)

        object_tokens = torch.full((1, 2, 3, model.dec_embed_dim), 2.0)
        object_pos = torch.zeros((1, 2, 3, 2), dtype=torch.long)
        object_valid = torch.tensor([[False, True]], dtype=torch.bool)
        object_state = (
            torch.full((2, 9, model.dec_embed_dim), 3.0),
            2,
            torch.zeros((2, 9, 2), dtype=torch.long),
            None,
            torch.ones((1, 2), dtype=torch.bool),
            None,
            8,
            8,
            0,
        )

        scene_out, _pos_out, hand_out, object_out, omv_out, omv_pos_out = model.decode(
            scene_hidden,
            object_hidden=(object_tokens, object_pos, object_valid, object_state),
            hand_hidden=(hand_tokens, hand_pos, hand_valid),
        )

        self.assertEqual(model.decoder[0].calls, 1)
        self.assertEqual(model.decoder[1].calls, 1)
        self.assertEqual(model.ho_decoder[0].calls, 2)
        self.assertEqual(model.ho_decoder[1].calls, 2)
        self.assertEqual(model.ho_decoder[0].cross_calls, 1)
        self.assertEqual(model.ho_decoder[1].cross_calls, 0)
        self.assertEqual(tuple(scene_out.shape), (2, 9, model.dec_embed_dim * 2))
        self.assertEqual(tuple(hand_out.shape), (2, 3, model.dec_embed_dim * 2))
        self.assertEqual(tuple(object_out.shape), (2, 3, model.dec_embed_dim * 2))
        self.assertFalse(torch.allclose(hand_out[0], hand_out[1]))
        self.assertFalse(torch.allclose(object_out[0], object_out[1]))


if __name__ == "__main__":
    unittest.main()
