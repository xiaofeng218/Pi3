from __future__ import annotations

import unittest
from unittest import mock

import torch

from pi3.models.curope.curope2d import cuRoPE2D


class CuRoPE2DContractTests(unittest.TestCase):
    def test_forward_accepts_non_contiguous_tokens(self) -> None:
        rope = cuRoPE2D(freq=100.0, F0=1.0)
        qkv = torch.randn(2, 5, 3, 4, 8).transpose(1, 3)
        tokens = qkv[:, :, 0]
        positions = torch.zeros(2, 5, 2, dtype=torch.int64).contiguous()

        self.assertFalse(tokens.is_contiguous())
        self.assertFalse(tokens.transpose(1, 2).is_contiguous())

        def _fake_rope_2d(tensor, pos, base_freq, f0):
            self.assertTrue(tensor.is_contiguous())
            self.assertTrue(pos.is_contiguous())
            self.assertEqual(tuple(tensor.shape), (2, 5, 4, 8))
            del base_freq, f0
            tensor.add_(1.0)

        with mock.patch("pi3.models.curope.curope2d._kernels.rope_2d", side_effect=_fake_rope_2d):
            out = rope(tokens, positions)

        self.assertEqual(tuple(out.shape), tuple(tokens.shape))
        self.assertTrue(torch.allclose(out, tokens + 1.0))


if __name__ == "__main__":
    unittest.main()
