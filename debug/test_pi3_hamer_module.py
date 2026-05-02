from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from yacs.config import CfgNode as CN


class Pi3HAMERModuleTests(unittest.TestCase):
    def test_pi3x_default_hamer_paths_use_repo_configs_not_third_party(self) -> None:
        from pi3.models.pi3x import Pi3X

        config_file, cache_dir = Pi3X.__new__(Pi3X)._default_hamer_paths()

        self.assertEqual(config_file.name, "model_config.yaml")
        self.assertIn("configs/hamer", config_file.as_posix())
        self.assertNotIn("third_party", config_file.as_posix())
        self.assertIn("data/model/hamer/_DATA", cache_dir.as_posix())
        self.assertNotIn("third_party", cache_dir.as_posix())

    def test_visualization_default_hamer_paths_use_repo_configs_not_third_party(self) -> None:
        from pi3.visualization.pi3x_rerun_export import _default_hamer_paths

        config_file, cache_dir = _default_hamer_paths()

        self.assertEqual(config_file.name, "model_config.yaml")
        self.assertIn("configs/hamer", config_file.as_posix())
        self.assertNotIn("third_party", config_file.as_posix())
        self.assertIn("data/model/hamer/_DATA", cache_dir.as_posix())
        self.assertNotIn("third_party", cache_dir.as_posix())

    def test_get_config_updates_relative_mano_paths(self) -> None:
        from pi3.models.hamer.config import get_config

        with tempfile.TemporaryDirectory(prefix="pi3_hamer_cfg_") as tmpdir:
            tmpdir = Path(tmpdir)
            config_path = tmpdir / "model_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "MODEL:",
                        "  IMAGE_SIZE: 224",
                        "MANO:",
                        "  MODEL_PATH: data/mano",
                        "  MEAN_PARAMS: data/mano_mean_params.npz",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = get_config(
                str(config_path),
                merge=True,
                cache_dir="/tmp/pi3_hamer_cache",
                update_cachedir=True,
            )

            self.assertEqual(cfg.MANO.MODEL_PATH, "/tmp/pi3_hamer_cache/data/mano")
            self.assertEqual(cfg.MANO.MEAN_PARAMS, "/tmp/pi3_hamer_cache/data/mano_mean_params.npz")

    def test_get_config_resolves_repo_relative_config_and_cache(self) -> None:
        from pi3.models.hamer.config import get_config, repo_root

        cfg = get_config(
            "configs/hamer/model_config.yaml",
            merge=True,
            cache_dir="data/model/hamer/_DATA",
            update_cachedir=True,
        )

        self.assertEqual(cfg.MANO.MODEL_PATH, str(repo_root() / "data" / "model" / "hamer" / "_DATA" / "data" / "mano"))
        self.assertEqual(
            cfg.MANO.MEAN_PARAMS,
            str(repo_root() / "data" / "model" / "hamer" / "_DATA" / "data" / "mano_mean_params.npz"),
        )

    def test_hamer_forward_returns_inference_outputs(self) -> None:
        from pi3.models.hamer.model import HAMER

        class FakeBackbone(torch.nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                return torch.ones((batch, 1280, 16, 12), dtype=x.dtype, device=x.device)

        class FakeHead(torch.nn.Module):
            def forward(self, feats):
                batch = feats.shape[0]
                eye = torch.eye(3, dtype=feats.dtype, device=feats.device).view(1, 1, 3, 3)
                pred_mano = {
                    "global_orient": eye.repeat(batch, 1, 1, 1),
                    "hand_pose": eye.repeat(batch, 15, 1, 1),
                    "betas": torch.zeros((batch, 10), dtype=feats.dtype, device=feats.device),
                }
                pred_cam = torch.tensor([[1.0, 0.1, 0.2]], dtype=feats.dtype, device=feats.device).repeat(batch, 1)
                return pred_mano, pred_cam, {}

        class FakeMANOOutput:
            def __init__(self, joints, vertices):
                self.joints = joints
                self.vertices = vertices

        class FakeMANO(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.faces = torch.zeros((0, 3), dtype=torch.long)

            def forward_rotmat(self, global_orient, hand_pose, betas, th_trans=None, **kwargs):
                del global_orient, hand_pose, th_trans, kwargs
                batch = betas.shape[0]
                joints = torch.zeros((batch, 21, 3), dtype=betas.dtype, device=betas.device)
                vertices = torch.zeros((batch, 778, 3), dtype=betas.dtype, device=betas.device)
                return FakeMANOOutput(joints=joints, vertices=vertices)

        cfg = CN(new_allowed=True)
        cfg.MODEL = CN(new_allowed=True)
        cfg.MODEL.IMAGE_SIZE = 224
        cfg.MODEL.BACKBONE = CN(new_allowed=True)
        cfg.MODEL.BACKBONE.TYPE = "vit"
        cfg.MODEL.MANO_HEAD = CN(new_allowed=True)
        cfg.EXTRA = CN(new_allowed=True)
        cfg.EXTRA.FOCAL_LENGTH = 5000
        cfg.MANO = CN(new_allowed=True)
        cfg.MANO.NUM_HAND_JOINTS = 15
        with tempfile.TemporaryDirectory(prefix="pi3_hamer_mano_") as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "MANO_RIGHT.pkl").write_bytes(b"placeholder")
            cfg.MANO.MODEL_PATH = str(tmpdir)
            cfg.MANO.MEAN_PARAMS = str(tmpdir / "mano_mean_params.npz")

            with mock.patch("pi3.models.hamer.model.create_backbone", return_value=FakeBackbone()), \
                 mock.patch("pi3.models.hamer.model.build_mano_head", return_value=FakeHead()), \
                 mock.patch("pi3.models.hamer.model.build_mano_layer", return_value=FakeMANO()):
                model = HAMER(cfg)
                batch = {"img": torch.randn(2, 3, 224, 224)}
                out = model(batch)

        self.assertEqual(out["pred_cam"].shape, (2, 3))
        self.assertEqual(out["pred_cam_t"].shape, (2, 3))
        self.assertEqual(out["focal_length"].shape, (2, 2))
        self.assertEqual(out["pred_keypoints_3d"].shape, (2, 21, 3))
        self.assertEqual(out["pred_vertices"].shape, (2, 778, 3))
        self.assertEqual(out["pred_keypoints_2d"].shape, (2, 21, 2))
        self.assertEqual(sorted(out["pred_mano_params"].keys()), ["betas", "global_orient", "hand_pose"])

    def test_make_hamer_model_without_pretrained_skips_checkpoint_load(self) -> None:
        from pi3.models.hamer.load import _make_hamer_model

        class FakeModel(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg

        cfg = CN(new_allowed=True)
        cfg.MODEL = CN(new_allowed=True)
        cfg.MODEL.BACKBONE = CN(new_allowed=True)
        cfg.MODEL.BACKBONE.TYPE = "vit"
        cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = "unused-backbone-weights"
        cfg.freeze()

        with mock.patch("pi3.models.hamer.load.get_config", return_value=cfg) as get_config_mock, \
             mock.patch("pi3.models.hamer.load.torch.load") as torch_load_mock:
            model = _make_hamer_model(
                config_file="/tmp/model_config.yaml",
                pretrained=False,
                model_cls=FakeModel,
            )

        self.assertIsInstance(model, FakeModel)
        get_config_mock.assert_called_once()
        torch_load_mock.assert_not_called()

    def test_make_hamer_model_loads_state_dict_and_returns_info(self) -> None:
        from pi3.models.hamer.load import _make_hamer_model

        class FakeModel(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg
                self.loaded = None

            def load_state_dict(self, state_dict, strict=False):
                self.loaded = (state_dict, strict)
                return "incompatible"

        cfg = CN(new_allowed=True)
        cfg.MODEL = CN(new_allowed=True)
        cfg.MODEL.BACKBONE = CN(new_allowed=True)
        cfg.MODEL.BACKBONE.TYPE = "vit"
        cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = "unused-backbone-weights"
        cfg.freeze()

        checkpoint = {"state_dict": {"weight": torch.tensor([1.0])}}
        with mock.patch("pi3.models.hamer.load.get_config", return_value=cfg), \
             mock.patch("pi3.models.hamer.load.torch.load", return_value=checkpoint) as torch_load_mock:
            model, returned_cfg, incompatible = _make_hamer_model(
                config_file="/tmp/model_config.yaml",
                checkpoint="/tmp/hamer.ckpt",
                pretrained=True,
                strict=True,
                model_cls=FakeModel,
                return_info=True,
            )

        self.assertIsInstance(model, FakeModel)
        self.assertIs(returned_cfg, cfg)
        self.assertEqual(incompatible, "incompatible")
        self.assertEqual(model.loaded, (checkpoint["state_dict"], True))
        torch_load_mock.assert_called_once()

    def test_load_hamer_keeps_legacy_tuple_contract(self) -> None:
        from pi3.models.hamer.load import load_hamer

        fake_model = object()
        fake_cfg = object()
        fake_incompatible = object()
        with mock.patch(
            "pi3.models.hamer.load._make_hamer_model",
            return_value=(fake_model, fake_cfg, fake_incompatible),
        ) as make_mock:
            model, cfg, incompatible = load_hamer("/tmp/checkpoints/hamer.ckpt")

        self.assertIs(model, fake_model)
        self.assertIs(cfg, fake_cfg)
        self.assertIs(incompatible, fake_incompatible)
        self.assertTrue(make_mock.called)

    def test_hamer_encoder_returns_direct_model(self) -> None:
        from pi3.models.hamer.load import hamer_encoder

        fake_model = object()
        with mock.patch("pi3.models.hamer.load._make_hamer_model", return_value=fake_model) as make_mock:
            model = hamer_encoder(config_file="/tmp/model_config.yaml", pretrained=False)

        self.assertIs(model, fake_model)
        self.assertTrue(make_mock.called)
        self.assertEqual(make_mock.call_args.kwargs["return_info"], False)

    def test_load_hamer_encoder_keeps_tuple_contract(self) -> None:
        from pi3.models.hamer.load import load_hamer_encoder

        fake_model = object()
        fake_cfg = object()
        fake_incompatible = object()
        with mock.patch(
            "pi3.models.hamer.load._make_hamer_model",
            return_value=(fake_model, fake_cfg, fake_incompatible),
        ) as make_mock:
            model, cfg, incompatible = load_hamer_encoder("/tmp/checkpoints/hamer_encoder.ckpt")

        self.assertIs(model, fake_model)
        self.assertIs(cfg, fake_cfg)
        self.assertIs(incompatible, fake_incompatible)
        self.assertTrue(make_mock.called)
        self.assertEqual(make_mock.call_args.kwargs["return_info"], True)


if __name__ == "__main__":
    unittest.main()
