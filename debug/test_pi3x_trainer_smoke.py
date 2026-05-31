from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import torch

from datasets.base.utils import unified_collate_fn
from trainers.pi3x_trainer import Pi3XTrainer
from trainers.pi3x_training_policy import apply_pi3x_training_policy


def _make_dummy_ho_block() -> torch.nn.Module:
    """Create a minimal block that mimics HOBlockRope structure for testing."""
    block = torch.nn.Module()
    # Self-attention
    block.attn = torch.nn.Module()
    block.attn.qkv = torch.nn.Linear(4, 12)
    block.attn.proj = torch.nn.Linear(4, 4)
    # Cross-attention
    block.cross_attn = torch.nn.Module()
    block.cross_attn.q_proj = torch.nn.Linear(4, 4)
    block.cross_attn.k_proj = torch.nn.Linear(4, 4)
    block.cross_attn.v_proj = torch.nn.Linear(4, 4)
    block.cross_attn.proj = torch.nn.Linear(4, 4)
    # Norms
    block.norm1 = torch.nn.LayerNorm(4)
    block.norm2 = torch.nn.LayerNorm(4)
    block.norm3 = torch.nn.LayerNorm(4)
    block.norm_y = torch.nn.LayerNorm(4)
    # Layer scales
    block.ls1 = torch.nn.Module()
    block.ls1.gamma = torch.nn.Parameter(torch.ones(4))
    block.ls2 = torch.nn.Module()
    block.ls2.gamma = torch.nn.Parameter(torch.ones(4))
    block.ls_y = torch.nn.Module()
    block.ls_y.gamma = torch.nn.Parameter(torch.ones(4))
    # FFN
    block.mlp = torch.nn.Module()
    block.mlp.fc1 = torch.nn.Linear(4, 16)
    block.mlp.fc2 = torch.nn.Linear(16, 4)
    return block


class _DummyTrainableModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_token_adapter = torch.nn.Linear(4, 4)
        self.object_query_adapter = torch.nn.Linear(4, 4)
        self.hand_global_decoder = torch.nn.Linear(4, 4)
        self.hand_pose_decoder = torch.nn.Linear(4, 4)
        self.object_pose_decoder = torch.nn.Linear(4, 4)
        self.hand_global_head = torch.nn.Linear(4, 4)
        self.hand_pose_head = torch.nn.Linear(4, 4)
        self.hand_mano_head = torch.nn.Linear(4, 4)
        self.object_pose_head = torch.nn.Linear(4, 4)
        self.hand_token_fuse = torch.nn.Linear(4, 4)
        self.ho_hand_scene_cross_alpha = torch.nn.Parameter(torch.zeros(36))
        self.ho_object_scene_cross_alpha = torch.nn.Parameter(torch.zeros(36))
        self.register_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.metric_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.ho_decoder = torch.nn.ModuleList([_make_dummy_ho_block()])
        self.encoder = torch.nn.Linear(4, 4)


class _DummyPi3XModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_multimodal = True
        self.hand_token_adapter = torch.nn.Linear(4, 4)
        self.object_query_adapter = torch.nn.Linear(4, 4)
        self.hand_global_decoder = torch.nn.Linear(4, 4)
        self.hand_pose_decoder = torch.nn.Linear(4, 4)
        self.object_pose_decoder = torch.nn.Linear(4, 4)
        self.hand_global_head = torch.nn.Linear(4, 4)
        self.hand_pose_head = torch.nn.Linear(4, 4)
        self.hand_mano_head = torch.nn.Linear(4, 4)
        self.object_pose_head = torch.nn.Linear(4, 4)
        self.hand_token_fuse = torch.nn.Linear(4, 4)
        self.ho_hand_scene_cross_alpha = torch.nn.Parameter(torch.zeros(36))
        self.ho_object_scene_cross_alpha = torch.nn.Parameter(torch.zeros(36))
        self.register_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.metric_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        self.ho_decoder = torch.nn.ModuleList([_make_dummy_ho_block()])
        class _DecodingMano(torch.nn.Module):
            side = "right"

            def decode_pose_coeffs_to_rotmat(self, pose_coeffs):
                batch = pose_coeffs.shape[0]
                global_orient = torch.eye(3, dtype=pose_coeffs.dtype, device=pose_coeffs.device).view(1, 1, 3, 3).repeat(batch, 1, 1, 1)
                hand_pose = torch.eye(3, dtype=pose_coeffs.dtype, device=pose_coeffs.device).view(1, 1, 3, 3).repeat(batch, 15, 1, 1)
                return global_orient, hand_pose

        self.hand_mano_layer = torch.nn.ModuleDict({"right": _DecodingMano(), "left": _DecodingMano()})

    def forward(self, **kwargs):
        imgs = kwargs["imgs"]
        assert torch.isfinite(imgs).all()
        batch, views = imgs.shape[:2]
        height, width = imgs.shape[-2:]
        hand_valid = (kwargs["hand_masks"].sum(dim=(-1, -2)) > 0)
        local_points = torch.ones((batch, views, height, width, 3), device=imgs.device)
        local_points[..., 2] = 2.0
        return {
            "local_points": local_points,
            "points": torch.ones_like(local_points),
            "camera_poses": torch.eye(4, device=imgs.device)[None, None].repeat(batch, views, 1, 1),
            "metric": torch.ones((batch,), device=imgs.device),
            "pred_hand_transl_dir": torch.tensor([[[1.0, 0.0, 0.0]]], device=imgs.device).repeat(batch, views, 1),
            "pred_hand_transl_log_scale": torch.zeros((batch, views, 1), device=imgs.device),
            "pred_hand_transl_scale": torch.ones((batch, views, 1), device=imgs.device),
            "pred_hand_transl": torch.tensor([[[1.0, 0.0, 0.0]]], device=imgs.device).repeat(batch, views, 1),
            "pred_hand_log_scale": torch.zeros((batch, views, 1), device=imgs.device),
            "pred_hand_scale": torch.ones((batch, views, 1), device=imgs.device),
            "pred_hand_mano_params": {
                "global_orient": torch.zeros((batch, views, 1, 3, 3), device=imgs.device),
                "hand_pose": torch.zeros((batch, views, 15, 3, 3), device=imgs.device),
                "betas": torch.zeros((batch, views, 10), device=imgs.device),
            },
            "hand_valid_mask": hand_valid,
            "pred_object_rot6d": torch.zeros((batch, views, 6), device=imgs.device),
            "pred_object_transl_dir": torch.tensor([[[0.0, 1.0, 0.0]]], device=imgs.device).repeat(batch, views, 1),
            "pred_object_transl_log_scale": torch.zeros((batch, views, 1), device=imgs.device),
            "pred_object_transl_scale": torch.ones((batch, views, 1), device=imgs.device),
            "pred_object_trans": torch.tensor([[[0.0, 1.0, 0.0]]], device=imgs.device).repeat(batch, views, 1),
            "pred_object_log_scale": torch.zeros((batch, views, 1), device=imgs.device),
            "pred_object_scale": torch.ones((batch, views, 1), device=imgs.device),
            "object_valid": torch.ones((batch, views), dtype=torch.bool, device=imgs.device),
        }


class _CapturingPi3XModel(_DummyPi3XModel):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return super().forward(**kwargs)


class _LocalJointPi3XModel(_DummyPi3XModel):
    def forward(self, **kwargs):
        out = super().forward(**kwargs)
        batch, views = kwargs["imgs"].shape[:2]
        out["pred_hand_joints_local"] = torch.ones((batch, views, 21, 3), device=kwargs["imgs"].device)
        out["pred_hand_vertices_local"] = torch.ones((batch, views, 778, 3), device=kwargs["imgs"].device)
        return out


class _Cfg(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _DeviceTrackingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class Pi3XTrainerSmokeTests(unittest.TestCase):
    def test_unified_collate_fn_normalizes_nested_singleton_samples(self) -> None:
        sample_a = [[{"img": torch.zeros(3, 4, 4), "hand": {"mask": torch.zeros(4, 4), "mano_side": "right"}, "object_multiview": {"grasped_object_mask": torch.zeros(4, 4)}}]]
        sample_b = [[{"img": torch.ones(3, 4, 4), "hand": {"mask": torch.ones(4, 4), "mano_side": "left"}, "object_multiview": {"grasped_object_mask": torch.ones(4, 4)}}]]

        collated = unified_collate_fn([sample_a, sample_b])

        self.assertIsInstance(collated, list)
        self.assertEqual(len(collated), 1)
        self.assertIsInstance(collated[0], dict)
        self.assertIsInstance(collated[0]["hand"], dict)
        self.assertEqual(collated[0]["img"].shape[0], 2)
        self.assertEqual(collated[0]["hand"]["mask"].shape[0], 2)
        self.assertEqual(collated[0]["hand"]["mano_side"], ["right", "left"])

    def test_forward_batch_rejects_legacy_view_list_batch(self) -> None:
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        trainer.model = _DummyPi3XModel()
        trainer.train_loss = torch.nn.Identity()
        trainer.test_loss = torch.nn.Identity()
        trainer.accelerator = type("A", (), {"device": torch.device("cpu")})()
        batch = [{"img": torch.zeros(1, 3, 4, 4)}]

        with self.assertRaisesRegex(TypeError, "batch dict containing `views`"):
            trainer.forward_batch(batch, mode="train")

    def test_pretrain_validation_export_uses_before_train_tag(self) -> None:
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        trainer.accelerator = type("A", (), {"is_main_process": True})()
        trainer._vis_cfg = {"enabled": True, "interval": 100, "sample_index": 0, "output_subdir": "rerun", "item_name": "pi3x_train_sample"}
        trainer.cfg = _Cfg(log=_Cfg(output_dir=""))
        trainer.model = type("B", (), {"hand_mano_layer": None})()
        trainer._resolve_data_root = lambda: None

        batch = {"views": [{"img": torch.zeros(1, 3, 4, 4)}], "scene_inputs": {}, "gt_metric": {}, "gt_scale_meta": {}}
        forward_outputs = ({"pred": torch.tensor(1.0)}, {"gt": torch.tensor(1.0)})

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.cfg.log.output_dir = tmpdir
            with mock.patch("trainers.pi3x_trainer.export_pi3x_rerun_sample") as export_mock:
                trainer.maybe_export_validation_sample(
                    epoch=-1,
                    batch_idx=0,
                    batch=batch,
                    forward_outputs=forward_outputs,
                    loss_outputs={},
                    mode="test",
                    global_step=None,
                )

        export_path = export_mock.call_args.kwargs["output_path"]
        self.assertIn("before_train", str(export_path))

    def test_pretrain_validation_export_accepts_views_only_sample(self) -> None:
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        trainer.accelerator = type("A", (), {"is_main_process": True})()
        trainer._vis_cfg = {"enabled": True, "interval": 100, "sample_index": 0, "output_subdir": "rerun", "item_name": "pi3x_train_sample"}
        trainer.cfg = _Cfg(log=_Cfg(output_dir=""))
        trainer.model = type("B", (), {"hand_mano_layer": None})()
        trainer._resolve_data_root = lambda: None

        batch = {
            "views": [{"img": torch.zeros(1, 3, 4, 4)}],
            "object_multiview_payload": {},
        }
        forward_outputs = ({"pred": torch.tensor(1.0)}, {"gt": torch.tensor(1.0)})

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.cfg.log.output_dir = tmpdir
            with mock.patch("trainers.pi3x_trainer.export_pi3x_rerun_sample") as export_mock:
                trainer.maybe_export_validation_sample(
                    epoch=-1,
                    batch_idx=0,
                    batch=batch,
                    forward_outputs=forward_outputs,
                    loss_outputs={},
                    mode="test",
                    global_step=None,
                )

        export_path = export_mock.call_args.kwargs["output_path"]
        self.assertIn("before_train", str(export_path))

    def test_init_moves_loss_modules_to_accelerator_device(self) -> None:
        trainer = None
        cfg = _Cfg(
            loss=_Cfg(train_loss=_Cfg(_target_="train_dummy"), test_loss=_Cfg(_target_="test_dummy")),
            hand_encoder=None,
            vis={},
        )
        train_loss = _DeviceTrackingLoss()
        test_loss = _DeviceTrackingLoss()

        def _fake_base_init(self, _cfg):
            self.cfg = _cfg
            self.accelerator = type("A", (), {"device": torch.device("cpu")})()

        with mock.patch("trainers.pi3x_trainer.BaseTrainer.__init__", new=_fake_base_init):
            with mock.patch("trainers.pi3x_trainer.hydra.utils.instantiate", side_effect=[train_loss, test_loss]):
                trainer = Pi3XTrainer(cfg)

        self.assertIsNotNone(trainer)
        self.assertTrue(train_loss.to_calls)
        self.assertTrue(test_loss.to_calls)
        self.assertEqual(train_loss.to_calls[-1][0][0], torch.device("cpu"))
        self.assertEqual(test_loss.to_calls[-1][0][0], torch.device("cpu"))

    def test_forward_batch_preserves_local_hand_joints_outputs(self) -> None:
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        trainer.model = _LocalJointPi3XModel()
        trainer.train_loss = torch.nn.Identity()
        trainer.test_loss = torch.nn.Identity()
        trainer.accelerator = type("A", (), {"device": torch.device("cpu")})()
        batch = {
            "views": [
                {
                    "img": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
                    "depthmap": torch.ones((1, 4, 4), dtype=torch.float32),
                    "pts3d": torch.ones((1, 4, 4, 3), dtype=torch.float32),
                    "valid_mask": torch.ones((1, 4, 4), dtype=torch.bool),
                    "sparse_depth": torch.ones((1, 4, 4), dtype=torch.float32),
                    "dataset": ["dexycb"],
                    "camera_intrinsics": torch.eye(3, dtype=torch.float32).unsqueeze(0),
                    "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                    "hand": {
                        "mask": torch.ones((1, 4, 4), dtype=torch.float32),
                        "valid": torch.tensor([True]),
                        "pose_mano": torch.zeros((1, 48), dtype=torch.float32),
                        "pose_repr": ["mano_full_aa"],
                        "mano_betas": torch.zeros((1, 10), dtype=torch.float32),
                        "hand_transl": torch.zeros((1, 3), dtype=torch.float32),
                        "joints_3d_cam": torch.zeros((1, 21, 3), dtype=torch.float32),
                        "joints_2d": torch.zeros((1, 21, 2), dtype=torch.float32),
                        "mano_side": ["right"],
                        "global_orient_rotmat_gt": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3),
                        "pose_rotmat_gt": torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3).repeat(1, 1, 15, 1, 1),
                    },
                    "object": {
                        "mask": torch.zeros((1, 4, 4), dtype=torch.float32),
                        "valid": torch.tensor([False]),
                        "grasped_object_id": torch.tensor([1], dtype=torch.long),
                        "pose_obj2cam": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                    },
                    "object_multiview": {
                        "img": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
                        "depthmap": torch.zeros((1, 4, 4), dtype=torch.float32),
                        "camera_intrinsics": torch.eye(3, dtype=torch.float32).unsqueeze(0),
                        "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                        "normalization_center": torch.zeros((3,), dtype=torch.float32),
                        "normalization_scale": torch.ones((1,), dtype=torch.float32),
                    },
                }
            ]
        }

        pred, _gt = trainer.forward_batch(batch, mode="train")

        self.assertIn("pred_hand_joints_local", pred)
        self.assertIn("pred_hand_vertices_local", pred)
        self.assertEqual(tuple(pred["pred_hand_joints_local"].shape), (1, 1, 21, 3))
        self.assertEqual(tuple(pred["pred_hand_vertices_local"].shape), (1, 1, 778, 3))

    def test_build_optimizer_groups_trainable_modules_as_expected(self) -> None:
        cfg = _Cfg(type="AdamW", lr=1e-4, weight_decay=5e-2, betas=[0.9, 0.95], encoder_lr=1e-5)
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        model = _DummyTrainableModule()
        apply_pi3x_training_policy(model)

        optimizer = trainer.build_optimizer(cfg, model)
        self.assertEqual(len(optimizer.param_groups), 3)
        group_lrs = sorted({group["lr"] for group in optimizer.param_groups})
        self.assertEqual(group_lrs, [1e-4])
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertIn("ho_decoder.0.cross_attn.q_proj.weight", trainable)
        self.assertIn("ho_decoder.0.norm2.weight", trainable)
        self.assertIn("ho_hand_scene_cross_alpha", trainable)
        self.assertIn("hand_token_adapter.weight", trainable)
        # Self-attn base weight is frozen (LoRA), only lora_A/lora_B are trainable
        self.assertNotIn("ho_decoder.0.attn.qkv.weight", trainable)

    def test_forward_batch_accepts_precomputed_dataset_sample(self) -> None:
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        trainer.model = _CapturingPi3XModel()
        trainer.train_loss = torch.nn.Identity()
        trainer.test_loss = torch.nn.Identity()
        trainer.accelerator = type("A", (), {"device": torch.device("cpu")})()

        batch = {
            "views": [
                {
                    "img": torch.zeros(1, 3, 4, 4),
                    "pts3d": torch.zeros(1, 4, 4, 3),
                    "valid_mask": torch.ones(1, 4, 4, dtype=torch.bool),
                    "camera_pose": torch.eye(4).unsqueeze(0),
                    "camera_intrinsics": torch.eye(3).unsqueeze(0),
                    "sparse_depth": torch.ones(1, 4, 4),
                    "depthmap": torch.ones(1, 4, 4),
                }
            ],
            "scene_inputs": {
                "imgs": torch.zeros(1, 1, 3, 4, 4),
                "depths": torch.ones(1, 1, 4, 4),
                "intrinsics": torch.eye(3).view(1, 1, 3, 3),
                "poses": torch.eye(4).view(1, 1, 4, 4),
                "hand_masks": torch.ones(1, 1, 4, 4),
                "hand_is_right": torch.tensor([[True]], dtype=torch.bool),
                "object_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
                "object_valid": torch.ones(1, 1, dtype=torch.bool),
                "object_multiview": {
                    "img": torch.zeros(1, 2, 3, 4, 4),
                    "depthmap": torch.ones(1, 2, 4, 4),
                    "camera_intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
                    "camera_pose": torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
                },
            },
            "gt_metric": {
                "hand_valid": torch.tensor([[True]], dtype=torch.bool),
                "hand_transl": torch.tensor([[[1.0, 0.0, 0.0]]]),
                "hand_mano_betas": torch.zeros(1, 1, 10),
                "hand_joints_3d": torch.full((1, 1, 21, 3), 0.5),
                "hand_joints_2d": torch.zeros(1, 1, 21, 2),
                "hand_camera_intrinsics": torch.eye(3).view(1, 1, 3, 3),
                "hand_is_right": torch.tensor([[True]], dtype=torch.bool),
                "hand_global_orient_rotmat_gt": torch.eye(3).view(1, 1, 1, 3, 3),
                "hand_pose_rotmat_gt": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 15, 1, 1),
                "object_valid": torch.tensor([[True]], dtype=torch.bool),
                "object_pose_obj2cam": torch.tensor(
                    [[[[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 1.0]]]]
                ),
                "object_camera_intrinsics": torch.eye(3).view(1, 1, 3, 3),
                "object_normalization_center": torch.tensor([[0.1, 0.2, 0.3]]),
                "object_scale_canonical_to_target": torch.tensor([1.25]),
            },
            "gt_scale_meta": {
                "scene_focus_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            },
        }

        scene_gt = {
            "local_points": torch.zeros(1, 1, 4, 4, 3),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        }
        with mock.patch("trainers.pi3x_trainer.estimate_scene_scale_from_depth", return_value=torch.tensor([2.0])):
            with mock.patch("trainers.pi3x_trainer.vis_export.build_scene_gt_metric", return_value=scene_gt):
                with mock.patch("trainers.pi3x_trainer.vis_export.convert_scene_gt_to_pred_scale", side_effect=lambda gt_metric, scene_scale: {**dict(gt_metric), "scene_scale": scene_scale}):
                    pred, gt = trainer.forward_batch(batch, mode="train")

        self.assertIn("object_masks", trainer.model.last_kwargs)
        self.assertIn("object_valid", trainer.model.last_kwargs)
        self.assertEqual(trainer.model.last_kwargs["object_masks"].shape, (1, 1, 4, 4))
        self.assertEqual(trainer.model.last_kwargs["object_valid"].shape, (1, 1))
        self.assertEqual(tuple(pred["pred_hand_transl"].shape), (1, 1, 3))
        self.assertEqual(tuple(gt["object_pose_obj2cam"].shape), (1, 1, 4, 4))
        self.assertNotIn("hand_owner_index", gt)
        self.assertTrue(torch.allclose(gt["hand_transl"], torch.tensor([[[2.0, 0.0, 0.0]]])))
        self.assertTrue(torch.allclose(gt["hand_scale"], torch.tensor([[[2.0]]])))
        self.assertTrue(torch.allclose(gt["hand_joints_3d"], torch.full((1, 1, 21, 3), 1.0)))
        self.assertTrue(torch.allclose(gt["object_pose_obj2cam"][0, 0, :3, 3], torch.tensor([3.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(gt["object_normalization_center"], torch.tensor([[0.1, 0.2, 0.3]])))
        self.assertTrue(torch.allclose(gt["object_normalization_scale"], torch.tensor([[[2.5]]])))
        self.assertNotIn("hand_vertices", gt)
        self.assertTrue(torch.allclose(gt["hand_global_orient_rotmat"], torch.eye(3).view(1, 1, 1, 3, 3)))
        self.assertEqual(tuple(gt["hand_pose_rotmat"].shape), (1, 1, 15, 3, 3))
        self.assertNotIn("hand_pose_mano", gt)
        self.assertNotIn("hand_pose_coeffs", gt)
        self.assertNotIn("object_vertices_2d", gt)
        self.assertNotIn("object_vertices_2d_valid", gt)

    def test_forward_batch_accepts_views_only_dataset_sample(self) -> None:
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        trainer.model = _CapturingPi3XModel()
        trainer.train_loss = torch.nn.Identity()
        trainer.test_loss = torch.nn.Identity()
        trainer.accelerator = type("A", (), {"device": torch.device("cpu")})()

        batch = {
            "views": [
                {
                    "img": torch.zeros(1, 3, 4, 4),
                    "pts3d": torch.zeros(1, 4, 4, 3),
                    "valid_mask": torch.ones(1, 4, 4, dtype=torch.bool),
                    "camera_pose": torch.eye(4).unsqueeze(0),
                    "camera_intrinsics": torch.eye(3).unsqueeze(0),
                    "sparse_depth": torch.ones(1, 4, 4),
                    "depthmap": torch.ones(1, 4, 4),
                    "hand": {
                        "mask": torch.ones(1, 4, 4, dtype=torch.bool),
                        "valid": torch.tensor([True]),
                        "pose_mano": torch.zeros(1, 48),
                        "pose_repr": ["mano_full_aa"],
                        "hand_transl": torch.tensor([[1.0, 0.0, 0.0]]),
                        "mano_betas": torch.zeros(1, 10),
                        "joints_3d_cam": torch.full((1, 21, 3), 0.5),
                        "joints_2d": torch.zeros(1, 21, 2),
                        "mano_side": ["right"],
                        "global_orient_rotmat_gt": torch.eye(3).view(1, 1, 3, 3),
                        "pose_rotmat_gt": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 15, 1, 1),
                    },
                    "object": {
                        "grasped_object_id": torch.tensor([11], dtype=torch.int32),
                        "mask": torch.ones(1, 4, 4, dtype=torch.bool),
                        "valid": torch.tensor([True]),
                        "pose_obj2cam": torch.tensor(
                            [[
                                [1.0, 0.0, 0.0, 1.5],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.5],
                                [0.0, 0.0, 0.0, 1.0],
                            ]]
                        ),
                    },
                    "object_multiview": {
                        "normalization_center": torch.tensor([[0.1, 0.2, 0.3]]),
                        "normalization_scale": torch.tensor([1.25]),
                    },
                }
            ]
        }

        scene_gt = {
            "local_points": torch.zeros(1, 1, 4, 4, 3),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        }
        with mock.patch("trainers.pi3x_trainer.estimate_scene_scale_from_depth", return_value=torch.tensor([2.0])):
            with mock.patch("trainers.pi3x_trainer.vis_export.build_scene_gt_metric", return_value=scene_gt):
                with mock.patch("trainers.pi3x_trainer.vis_export.convert_scene_gt_to_pred_scale", side_effect=lambda gt_metric, scene_scale: {**dict(gt_metric), "scene_scale": scene_scale}):
                    pred, gt = trainer.forward_batch(batch, mode="train")

        self.assertEqual(tuple(pred["pred_hand_transl"].shape), (1, 1, 3))
        self.assertEqual(tuple(gt["object_pose_obj2cam"].shape), (1, 1, 4, 4))
        self.assertTrue(torch.allclose(gt["hand_transl"], torch.tensor([[[2.0, 0.0, 0.0]]])))
        self.assertTrue(torch.allclose(gt["object_normalization_scale"], torch.tensor([[[2.5]]])))

    def test_forward_batch_sanitizes_non_finite_scene_images_in_precomputed_sample(self) -> None:
        trainer = Pi3XTrainer.__new__(Pi3XTrainer)
        trainer.model = _CapturingPi3XModel()
        trainer.train_loss = torch.nn.Identity()
        trainer.test_loss = torch.nn.Identity()
        trainer.accelerator = type("A", (), {"device": torch.device("cpu")})()

        imgs = torch.zeros(1, 1, 3, 4, 4)
        imgs[0, 0, 0, 0, 0] = float("nan")
        batch = {
            "views": [
                {
                    "img": torch.zeros(1, 3, 4, 4),
                    "pts3d": torch.zeros(1, 4, 4, 3),
                    "valid_mask": torch.ones(1, 4, 4, dtype=torch.bool),
                    "camera_pose": torch.eye(4).unsqueeze(0),
                    "camera_intrinsics": torch.eye(3).unsqueeze(0),
                    "sparse_depth": torch.ones(1, 4, 4),
                    "depthmap": torch.ones(1, 4, 4),
                }
            ],
            "scene_inputs": {
                "imgs": imgs,
                "depths": torch.ones(1, 1, 4, 4),
                "intrinsics": torch.eye(3).view(1, 1, 3, 3),
                "poses": torch.eye(4).view(1, 1, 4, 4),
                "hand_masks": torch.zeros(1, 1, 4, 4),
                "hand_is_right": torch.tensor([[True]], dtype=torch.bool),
                "object_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
                "object_valid": torch.ones(1, 1, dtype=torch.bool),
                "object_multiview": {
                    "img": imgs.repeat(1, 2, 1, 1, 1),
                    "depthmap": torch.ones(1, 2, 4, 4),
                    "camera_intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
                    "camera_pose": torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
                },
            },
            "gt_metric": {
                "hand_valid": torch.tensor([[False]], dtype=torch.bool),
                "hand_transl": torch.zeros(1, 1, 3),
                "hand_mano_betas": torch.zeros(1, 1, 10),
                "hand_joints_3d": torch.zeros(1, 1, 21, 3),
                "hand_joints_2d": torch.zeros(1, 1, 21, 2),
                "hand_camera_intrinsics": torch.eye(3).view(1, 1, 3, 3),
                "hand_is_right": torch.tensor([[True]], dtype=torch.bool),
                "hand_global_orient_rotmat_gt": torch.eye(3).view(1, 1, 1, 3, 3),
                "hand_pose_rotmat_gt": torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 15, 1, 1),
                "object_valid": torch.tensor([[True]], dtype=torch.bool),
                "object_pose_obj2cam": torch.eye(4).view(1, 1, 4, 4),
                "object_camera_intrinsics": torch.eye(3).view(1, 1, 3, 3),
                "object_normalization_center": torch.zeros(1, 3),
                "object_scale_canonical_to_target": torch.ones(1),
            },
            "gt_scale_meta": {
                "scene_focus_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            },
        }

        scene_gt = {
            "local_points": torch.zeros(1, 1, 4, 4, 3),
            "valid_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        }
        with mock.patch("trainers.pi3x_trainer.vis_export.build_scene_gt_metric", return_value=scene_gt):
            with mock.patch("trainers.pi3x_trainer.vis_export.convert_scene_gt_to_pred_scale", side_effect=lambda gt_metric, _scene_scale: dict(gt_metric)):
                trainer.forward_batch(batch, mode="train")

        self.assertTrue(torch.isfinite(trainer.model.last_kwargs["imgs"]).all())


if __name__ == "__main__":
    unittest.main()
