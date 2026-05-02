from __future__ import annotations

import unittest

import torch

from debug.inspect_encoder_patch_specs import (
    normalize_patch_size,
    resolve_device,
    summarize_hamer_backbone_output,
    summarize_pi3_encoder_output,
)


class InspectEncoderPatchSpecsTests(unittest.TestCase):
    def test_normalize_patch_size_accepts_int_and_tuple(self) -> None:
        self.assertEqual(normalize_patch_size(14), (14, 14))
        self.assertEqual(normalize_patch_size((16, 12)), (16, 12))

    def test_summarize_pi3_encoder_output_reads_patch_tokens(self) -> None:
        output = {
            "x_norm_patchtokens": torch.zeros(2, 256, 1024),
            "x_norm_regtokens": torch.zeros(2, 4, 1024),
        }

        summary = summarize_pi3_encoder_output(output, image_hw=(224, 224), patch_hw=(14, 14))

        self.assertEqual(summary["token_shape"], [2, 256, 1024])
        self.assertEqual(summary["token_grid_hw"], [16, 16])
        self.assertEqual(summary["embed_dim"], 1024)

    def test_summarize_hamer_backbone_output_reads_feature_map(self) -> None:
        output = torch.zeros(2, 1280, 16, 12)

        summary = summarize_hamer_backbone_output(output, patch_hw=(16, 16))

        self.assertEqual(summary["feature_shape"], [2, 1280, 16, 12])
        self.assertEqual(summary["token_count"], 192)
        self.assertEqual(summary["token_grid_hw"], [16, 12])
        self.assertEqual(summary["embed_dim"], 1280)

    def test_resolve_device_rejects_unavailable_cuda(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No CUDA GPUs are available"):
            resolve_device("cuda", cuda_available=False)


if __name__ == "__main__":
    unittest.main()
