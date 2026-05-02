from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from debug.demo_hamer_encoder_mask_bbox import (
    build_encoder_inputs,
    prepare_encoder_crops,
    remap_hamer_state_dict_for_encoder,
    run_demo,
)


class _FakeEncoder(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim
        self.last_inputs = None
        self.min_mask_area = 1
        self.rescale_factor = 1.0
        self.output_dim = dim
        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _normalize_mask_shape(self, hand_masks):
        if hand_masks.ndim == 4 and hand_masks.shape[1] == 1:
            return hand_masks[:, 0]
        return hand_masks

    def _compute_bbox_from_mask(self, mask):
        ys, xs = torch.nonzero(mask > 0, as_tuple=True)
        return torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.float32, device=mask.device)

    def _expand_box(self, box, height, width):
        x1, y1, x2, y2 = box.unbind()
        return torch.stack(
            [
                torch.clamp(x1, min=0.0, max=float(width - 1)),
                torch.clamp(y1, min=0.0, max=float(height - 1)),
                torch.clamp(x2, min=0.0, max=float(width - 1)),
                torch.clamp(y2, min=0.0, max=float(height - 1)),
            ]
        )

    def _crop_and_resize(self, img, box):
        x1, y1, x2, y2 = box.round().to(torch.int64)
        return torch.nn.functional.interpolate(
            img[:, y1 : y2 + 1, x1 : x2 + 1].unsqueeze(0),
            size=(256, 192),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _canonicalize_handedness(self, crop, is_right):
        return crop if bool(is_right) else torch.flip(crop, dims=[2])

    def _normalize_input(self, crop):
        return (crop - self.image_mean.to(device=crop.device, dtype=crop.dtype)) / self.image_std.to(
            device=crop.device, dtype=crop.dtype
        )

    def forward(self, imgs, hand_masks, owner_index, hand_is_right):
        self.last_inputs = {
            "imgs": imgs.detach().cpu(),
            "hand_masks": hand_masks.detach().cpu(),
            "owner_index": owner_index.detach().cpu(),
            "hand_is_right": hand_is_right.detach().cpu(),
        }
        batch = hand_masks.shape[0]
        queries = torch.arange(batch * self.dim, dtype=imgs.dtype, device=imgs.device).view(batch, self.dim)
        return {
            "hand_queries": queries,
            "owner_index": owner_index,
            "hand_is_right": hand_is_right,
            "crop_boxes": torch.tensor([[3.0, 2.0, 8.0, 6.0]], dtype=imgs.dtype, device=imgs.device).repeat(batch, 1),
        }


class DemoHaMeREncoderMaskBBoxTests(unittest.TestCase):
    def test_build_encoder_inputs_creates_sparse_single_hand_batch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hamer_encoder_demo_") as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "image.jpg"
            mask_path = tmpdir / "mask.png"

            rgb = np.zeros((10, 12, 3), dtype=np.uint8)
            rgb[2:7, 3:9] = 200
            mask = np.zeros((10, 12), dtype=np.uint8)
            mask[2:7, 3:9] = 255
            Image.fromarray(rgb).save(image_path)
            Image.fromarray(mask).save(mask_path)

            batch = build_encoder_inputs(
                image_path=image_path,
                mask_path=mask_path,
                hand_side="left",
                device=torch.device("cpu"),
            )

            self.assertEqual(tuple(batch["imgs"].shape), (1, 1, 3, 10, 12))
            self.assertEqual(tuple(batch["hand_masks"].shape), (1, 10, 12))
            self.assertTrue(torch.equal(batch["owner_index"], torch.tensor([[0, 0, 0]], dtype=torch.long)))
            self.assertTrue(torch.equal(batch["hand_is_right"], torch.tensor([False], dtype=torch.bool)))
            self.assertEqual(batch["bbox_xyxy"], [3.0, 2.0, 8.0, 6.0])

    def test_remap_hamer_state_dict_for_encoder_keeps_backbone_and_query_decoder(self) -> None:
        state_dict = {
            "backbone.pos_embed": torch.ones(2, 3),
            "mano_head.transformer.pos_embedding": torch.zeros(1, 2, 3),
            "mano_head.init_hand_pose": torch.randn(1, 48),
            "mano_head.decpose.weight": torch.randn(2, 2),
        }

        mapped = remap_hamer_state_dict_for_encoder(state_dict)

        self.assertIn("backbone.vit.pos_embed", mapped)
        self.assertIn("backbone.transformer.pos_embedding", mapped)
        self.assertIn("backbone.init_hand_pose", mapped)
        self.assertNotIn("mano_head.decpose.weight", mapped)

    def test_prepare_encoder_crops_returns_visualizable_canonical_crop(self) -> None:
        encoder = _FakeEncoder(dim=4)
        imgs = torch.zeros(1, 1, 3, 10, 12)
        imgs[0, 0, :, 2:7, 3:9] = torch.tensor([0.2, 0.4, 0.8]).view(3, 1, 1)
        hand_masks = torch.zeros(1, 10, 12)
        hand_masks[0, 2:7, 3:9] = 1.0
        owner_index = torch.tensor([[0, 0, 0]], dtype=torch.long)
        hand_is_right = torch.tensor([False], dtype=torch.bool)

        prepared = prepare_encoder_crops(encoder, imgs, hand_masks, owner_index, hand_is_right)

        self.assertEqual(tuple(prepared["normalized_crops"].shape), (1, 3, 256, 192))
        self.assertEqual(tuple(prepared["visual_crops"].shape), (1, 3, 256, 192))
        self.assertEqual(tuple(prepared["crop_boxes"].shape), (1, 4))
        self.assertTrue(torch.equal(prepared["owner_index"], owner_index))
        self.assertTrue(torch.equal(prepared["hand_is_right"], hand_is_right))

    def test_run_demo_writes_queries_metadata_and_visualization(self) -> None:
        fake_encoder = _FakeEncoder(dim=6)
        with tempfile.TemporaryDirectory(prefix="hamer_encoder_demo_") as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "image.jpg"
            mask_path = tmpdir / "mask.png"
            out_dir = tmpdir / "out"

            rgb = np.zeros((10, 12, 3), dtype=np.uint8)
            rgb[2:7, 3:9] = [100, 150, 200]
            mask = np.zeros((10, 12), dtype=np.uint8)
            mask[2:7, 3:9] = 255
            Image.fromarray(rgb).save(image_path)
            Image.fromarray(mask).save(mask_path)

            args = Namespace(
                image_path=str(image_path),
                mask_path=str(mask_path),
                checkpoint=None,
                config_file=None,
                out_folder=str(out_dir),
                hand_side="right",
                device="cpu",
                prepare_example_data=False,
                example_dir=None,
                data_root=None,
            )

            def _fake_render(*args, **kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                crop_render = out_dir / "mesh_crop.png"
                full_render = out_dir / "mesh_full.png"
                Image.new("RGB", (64, 64), color=(255, 255, 255)).save(crop_render)
                Image.new("RGB", (64, 64), color=(200, 200, 200)).save(full_render)
                return {
                    "mesh_crop_path": crop_render,
                    "mesh_full_path": full_render,
                }

            fake_predictions = {
                "pred_cam": torch.zeros(1, 3),
                "pred_cam_t": torch.zeros(1, 3),
                "pred_vertices": torch.zeros(1, 778, 3),
                "pred_keypoints_2d": torch.zeros(1, 21, 2),
            }

            with (
                mock.patch("debug.demo_hamer_encoder_mask_bbox.load_encoder_for_demo", return_value=fake_encoder),
                mock.patch("debug.demo_hamer_encoder_mask_bbox.load_prediction_model_for_demo", return_value=object()),
                mock.patch("debug.demo_hamer_encoder_mask_bbox.predict_mano_from_queries", return_value=fake_predictions),
                mock.patch("debug.demo_hamer_encoder_mask_bbox.render_mesh_visualizations", side_effect=_fake_render),
            ):
                outputs = run_demo(args)

            self.assertTrue(outputs["query_path"].is_file())
            self.assertTrue(outputs["meta_path"].is_file())
            self.assertTrue(outputs["vis_path"].is_file())
            self.assertTrue(outputs["crop_path"].is_file())
            self.assertTrue(outputs["mesh_crop_path"].is_file())
            self.assertTrue(outputs["mesh_full_path"].is_file())

            saved = torch.load(outputs["query_path"], map_location="cpu", weights_only=False)
            self.assertEqual(tuple(saved["hand_queries"].shape), (1, 6))

            metadata = json.loads(outputs["meta_path"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["bbox_xyxy"], [3.0, 2.0, 8.0, 6.0])
            self.assertEqual(metadata["num_hands"], 1)


if __name__ == "__main__":
    unittest.main()
