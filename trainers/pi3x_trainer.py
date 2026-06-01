from __future__ import annotations

from typing import Any

from pathlib import Path

import hydra
import torch
from pi3.models.hand_object_loss import estimate_scene_scale_from_depth
from pi3.visualization import pi3x_rerun_export as vis_export
from pi3.visualization import export_pi3x_rerun_sample
from trainers.checkpoint_utils import load_trainable_checkpoint, save_trainable_checkpoint
from trainers.base_trainer_accelerate import BaseTrainer
from trainers.pi3x_batch_utils import build_precomputed_pi3x_sample
from trainers.pi3x_training_policy import apply_pi3x_training_policy


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class _LossOutput(dict):
    @property
    def loss(self):
        return self["loss"]

    def __getattr__(self, item):
        return self[item]


class Pi3XTrainer(BaseTrainer):
    def __init__(self, cfg):
        self._loss_cfg = cfg.loss.train_loss
        self._test_loss_cfg = cfg.loss.test_loss
        self._vis_cfg = _cfg_get(cfg, "vis", {})
        super().__init__(cfg)
        self.train_loss = hydra.utils.instantiate(self._loss_cfg).to(self.accelerator.device)
        self.test_loss = hydra.utils.instantiate(self._test_loss_cfg).to(self.accelerator.device)

    @property
    def _base_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def prepare_model(self):
        model = super().prepare_model()
        load_report = getattr(model, "_last_load_report", None)
        if load_report is not None:
            self.log_info(
                "Loaded Pi3X base checkpoint: "
                f"missing={len(load_report.missing_keys)}, unexpected={len(load_report.unexpected_keys)}"
            )
        apply_pi3x_training_policy(model)
        return model

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            heads = []
            adapters = []
            ho_cross = []
            for name, param in model_.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith(
                    (
                        "ho_decoder",
                        "ho_hand_scene_cross_alpha",
                        "ho_object_scene_cross_alpha",
                    )
                ):
                    ho_cross.append(param)
                elif name.startswith(
                    (
                        "hand_token_adapter",
                        "object_query_adapter",
                        "hand_global_decoder",
                        "hand_pose_decoder",
                        "object_pose_decoder",
                        "hand_global_head",
                        "hand_pose_head",
                        "hand_mano_head",
                        "object_pose_head",
                        "hand_token_fuse",
                    )
                ):
                    adapters.append(param)
                else:
                    heads.append(param)
            lr = _cfg_get(cfg_optimizer, "lr", 1e-4)
            weight_decay = _cfg_get(cfg_optimizer, "weight_decay", 0.0)
            return [
                {"params": heads, "lr": lr, "weight_decay": weight_decay},
                {"params": adapters, "lr": lr, "weight_decay": 0.0},
                {"params": ho_cross, "lr": lr, "weight_decay": 0.0},
            ]

        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def save_training_state(self, save_path, epoch):
        extra = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "train_num_epoch": int(self.cfg.train.num_epoch),
            "iters_per_epoch": int(self.iters_per_epoch),
        }
        save_trainable_checkpoint(
            save_path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            extra=extra,
        )

    def load_training_state(self, load_path):
        state = load_trainable_checkpoint(
            load_path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            map_location=self.accelerator.device,
        )
        self.log_info(
            f"Loaded trainable checkpoint from {load_path} "
            f"(missing={len(state['missing_keys'])}, unexpected={len(state['unexpected_keys'])})"
        )

    def before_epoch(self, epoch):
        if hasattr(self.train_loader, "dataset") and hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if (
            hasattr(self.train_loader, "batch_sampler")
            and hasattr(self.train_loader.batch_sampler, "batch_sampler")
            and hasattr(self.train_loader.batch_sampler.batch_sampler, "sampler")
            and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, "set_epoch")
        ):
            self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, "batch_sampler") and hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

        if self.test_loader is self.train_loader:
            return

        if hasattr(self.test_loader, "dataset") and hasattr(self.test_loader.dataset, "set_epoch"):
            self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
        if (
            hasattr(self.test_loader, "batch_sampler")
            and hasattr(self.test_loader, "batch_sampler")
            and hasattr(self.test_loader.batch_sampler, "batch_sampler")
            and hasattr(self.test_loader.batch_sampler.batch_sampler, "sampler")
            and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, "set_epoch")
        ):
            self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, "batch_sampler") and hasattr(self.test_loader.batch_sampler, "set_epoch"):
            self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

    def _sanitize_tensor(self, name, tensor, nan=0.0, posinf=0.0, neginf=0.0):
        if not torch.is_tensor(tensor):
            return tensor
        if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
            return tensor
        return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)

    def _vis_enabled(self):
        return bool(_cfg_get(self._vis_cfg, "enabled", False))

    def _vis_interval(self):
        return max(1, int(_cfg_get(self._vis_cfg, "interval", 1)))

    def _vis_sample_index(self):
        return max(0, int(_cfg_get(self._vis_cfg, "sample_index", 0)))

    def _vis_output_subdir(self):
        return str(_cfg_get(self._vis_cfg, "output_subdir", "rerun"))

    def _vis_item_name(self):
        return str(_cfg_get(self._vis_cfg, "item_name", "pi3x_train_sample"))

    def _vis_output_root(self):
        return Path(self.cfg.log.output_dir) / self._vis_output_subdir()

    def _resolve_data_root(self):
        train_root = _cfg_get(self.cfg, "train_dataset", {})
        test_root = _cfg_get(self.cfg, "test_dataset", {})
        data_root = (
            _cfg_get(test_root, "data_root", None)
            or _cfg_get(train_root, "data_root", None)
            or _cfg_get(_cfg_get(self.cfg, "data", {}), "test_dataset", {}).get("data_root", None)
            or _cfg_get(_cfg_get(self.cfg, "data", {}), "train_dataset", {}).get("data_root", None)
        )
        if data_root is None:
            raise RuntimeError("vis export requires data_root in the train/test dataset config")
        return data_root

    def _has_precomputed_sample(self, batch) -> bool:
        return isinstance(batch, dict) and all(key in batch for key in ("views", "scene_inputs", "gt_metric", "gt_scale_meta"))

    def _has_views_sample(self, batch) -> bool:
        return isinstance(batch, dict) and "views" in batch

    def _decode_hand_pose_coeffs_to_rotmat(self, hand_pose_coeffs, hand_is_right):
        mano_layer = getattr(self._base_model, "hand_mano_layer", None)
        if mano_layer is None:
            raise RuntimeError("hand_mano_layer is required to decode DexYCB hand pose coefficients")
        prefix_shape = hand_pose_coeffs.shape[:-1]
        flat_pose_coeffs = hand_pose_coeffs.reshape(-1, hand_pose_coeffs.shape[-1])
        flat_is_right = hand_is_right.reshape(-1)
        batch_size = flat_pose_coeffs.shape[0]
        global_orient = torch.zeros((batch_size, 1, 3, 3), dtype=hand_pose_coeffs.dtype, device=hand_pose_coeffs.device)
        hand_pose = torch.zeros((batch_size, 15, 3, 3), dtype=hand_pose_coeffs.dtype, device=hand_pose_coeffs.device)

        def _decode_with_layer(layer, mask):
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            decoded_global, decoded_pose = layer.decode_pose_coeffs_to_rotmat(flat_pose_coeffs[idx])
            global_orient[idx] = decoded_global.to(dtype=hand_pose_coeffs.dtype, device=hand_pose_coeffs.device)
            hand_pose[idx] = decoded_pose.to(dtype=hand_pose_coeffs.dtype, device=hand_pose_coeffs.device)

        if isinstance(mano_layer, torch.nn.ModuleDict):
            for side_value, side_name in ((True, "right"), (False, "left")):
                side_mask = flat_is_right == side_value
                if side_mask.any():
                    _decode_with_layer(mano_layer[side_name], side_mask)
        else:
            _decode_with_layer(mano_layer, torch.ones((batch_size,), dtype=torch.bool, device=hand_pose_coeffs.device))
        return (
            global_orient.reshape(*prefix_shape, 1, 3, 3),
            hand_pose.reshape(*prefix_shape, 15, 3, 3),
        )

    def forward_batch(self, batch, mode="train"):
        del mode
        if not self._has_views_sample(batch):
            raise TypeError(
                "Pi3XTrainer now expects a batch dict containing `views`."
            )
        if not self._has_precomputed_sample(batch):
            batch = build_precomputed_pi3x_sample(batch)

        views_batch = batch["views"]
        scene_inputs = batch["scene_inputs"]
        gt_metric_dense = batch["gt_metric"]
        gt_scale_meta = batch["gt_scale_meta"]

        imgs = self._sanitize_tensor("scene_inputs.imgs", scene_inputs["imgs"], nan=0.0, posinf=1.0, neginf=0.0)
        depths = self._sanitize_tensor("scene_inputs.depths", scene_inputs["depths"], nan=0.0, posinf=0.0, neginf=0.0)
        intrinsics = self._sanitize_tensor("scene_inputs.intrinsics", scene_inputs["intrinsics"], nan=0.0, posinf=0.0, neginf=0.0)
        poses = self._sanitize_tensor("scene_inputs.poses", scene_inputs["poses"], nan=0.0, posinf=0.0, neginf=0.0)
        object_masks = scene_inputs["object_masks"]
        object_valid = scene_inputs["object_valid"]
        object_multiview = scene_inputs["object_multiview"]
        if object_multiview is not None:
            object_multiview = dict(object_multiview)
            object_multiview["img"] = self._sanitize_tensor(
                "scene_inputs.object_multiview.img",
                object_multiview["img"],
                nan=0.0, posinf=1.0, neginf=0.0,
            )
            object_multiview["depthmap"] = self._sanitize_tensor(
                "scene_inputs.object_multiview.depthmap",
                object_multiview["depthmap"],
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            object_multiview["camera_intrinsics"] = self._sanitize_tensor(
                "scene_inputs.object_multiview.camera_intrinsics",
                object_multiview["camera_intrinsics"],
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            object_multiview["camera_pose"] = self._sanitize_tensor(
                "scene_inputs.object_multiview.camera_pose",
                object_multiview["camera_pose"],
                nan=0.0, posinf=0.0, neginf=0.0,
            )
        scene_focus_masks = gt_scale_meta["scene_focus_masks"]

        hand_masks = scene_inputs["hand_masks"]
        hand_is_right = scene_inputs["hand_is_right"]

        model_kwargs = {
            "imgs": imgs,
            "depths": depths if self._base_model.use_multimodal else None,
            "intrinsics": intrinsics if self._base_model.use_multimodal else None,
            "poses": poses if self._base_model.use_multimodal else None,
            "with_prior": True,
            "hand_masks": hand_masks,
            "hand_is_right": hand_is_right,
            "object_masks": object_masks,
            "object_valid": object_valid,
            "object_multiview": object_multiview,
        }
        if self._base_model.use_multimodal:
            bsz, num_views = imgs.shape[:2]
            full_mask = torch.ones((bsz, num_views), dtype=torch.bool, device=imgs.device)
            model_kwargs["mask_add_depth"] = full_mask
            model_kwargs["mask_add_ray"] = full_mask
            model_kwargs["mask_add_pose"] = full_mask

        pred = self.model(**model_kwargs)

        with torch.no_grad():
            scene_scale = estimate_scene_scale_from_depth(
                pred["local_points"][..., 2].detach(),
                depths,
                valid_mask=depths > 0,
                focus_mask=scene_focus_masks,
            )

            scene_gt_metric = vis_export.build_scene_gt_metric(views_batch)
            scene_gt_metric["imgs"] = imgs
            gt = vis_export.convert_scene_gt_to_pred_scale(scene_gt_metric, scene_scale)
            del scene_gt_metric

            scale_per_hand = scene_scale.view(-1, 1, 1)
            gt_global_orient_rotmat = gt_metric_dense.get("hand_global_orient_rotmat_gt", None)
            gt_hand_pose_rotmat = gt_metric_dense.get("hand_pose_rotmat_gt", None)
            if gt_global_orient_rotmat is None or gt_hand_pose_rotmat is None:
                gt_global_orient_rotmat, gt_hand_pose_rotmat = self._decode_hand_pose_coeffs_to_rotmat(
                    gt_metric_dense["hand_pose_coeffs"],
                    gt_metric_dense["hand_is_right"],
                )
            gt["hand_valid"] = gt_metric_dense["hand_valid"]
            gt["hand_global_orient_rotmat"] = gt_global_orient_rotmat
            gt["hand_pose_rotmat"] = gt_hand_pose_rotmat
            gt["hand_mano_betas"] = gt_metric_dense["hand_mano_betas"]
            gt["hand_transl"] = gt_metric_dense["hand_transl"] * scale_per_hand
            gt["hand_scale"] = scale_per_hand
            gt["hand_joints_3d"] = gt_metric_dense["hand_joints_3d"] * scale_per_hand.unsqueeze(-1)
            gt["hand_joints_2d"] = gt_metric_dense["hand_joints_2d"]
            gt["hand_camera_intrinsics"] = gt_metric_dense["hand_camera_intrinsics"]
            gt["hand_is_right"] = gt_metric_dense["hand_is_right"]

            obj_pose = gt_metric_dense["object_pose_obj2cam"].clone()
            obj_pose[..., :3, 3] = obj_pose[..., :3, 3] * scene_scale.view(-1, 1, 1)
            gt["object_valid"] = gt_metric_dense["object_valid"]
            gt["object_pose_obj2cam"] = obj_pose
            gt["object_normalization_center"] = gt_metric_dense["object_normalization_center"]
            object_scale_metric = gt_metric_dense.get(
                "object_scale_canonical_to_target",
                gt_metric_dense.get(
                "object_scale_canonical_to_scene_metric",
                gt_metric_dense.get("object_normalization_scale"),
                ),
            )
            if object_scale_metric.ndim == 1:
                object_scale_metric = object_scale_metric.view(-1, 1, 1)
            gt["object_normalization_scale"] = object_scale_metric * scene_scale.view(-1, 1, 1)

            # OMV GT data for point and camera supervision
            omv_data = scene_inputs.get("object_multiview", None)
            if omv_data is not None:
                gt["omv_depth"] = omv_data["depthmap"]
                gt["omv_intrinsics"] = omv_data["camera_intrinsics"]
                # Use absolute object-frame poses so GT matches the encoder input,
                # which also receives absolute poses (normalize_poses_to_view0=False).
                gt["omv_camera_pose"] = omv_data["camera_pose"]

        return [pred, gt]

    def calculate_loss(self, output, batch, mode="train"):
        pred, gt = output
        if mode == "train":
            loss, details = self.train_loss(pred, gt)
        else:
            loss, details = self.test_loss(pred, gt)
        return _LossOutput(loss=loss, **details)

    def maybe_export_validation_sample(self, epoch, batch_idx, batch, forward_outputs, loss_outputs, mode="test", global_step=None):
        del loss_outputs
        if not self.accelerator.is_main_process:
            return
        if not self._vis_enabled():
            return

        views_batch = batch["views"] if self._has_views_sample(batch) else batch

        # For validation mode, export the first batch every epoch.
        if mode == "test" and global_step is None:
            if epoch >= 0 and epoch % self._vis_interval() != 0:
                return
            if batch_idx != 0:
                return

        pred, gt = forward_outputs
        sample_index = min(self._vis_sample_index(), views_batch[0]["img"].shape[0] - 1)
        if sample_index < 0:
            return

        # Naming: step-based uses step number, epoch-based uses epoch number
        if global_step is not None:
            tag = f"step_{global_step:07d}"
            log_prefix = "train_vis"
            log_step = global_step
        else:
            tag = "before_train" if epoch < 0 else f"epoch_{epoch:04d}"
            log_prefix = "val_vis"
            log_step = epoch

        output_root = self._vis_output_root() / tag
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"sample_{sample_index:03d}.rrd"
        export_pi3x_rerun_sample(
            output_path=output_path,
            batch=views_batch,
            pred=pred,
            gt=gt,
            sample_index=sample_index,
            data_root=self._resolve_data_root(),
            mano_layer=self._base_model.hand_mano_layer,
            item_name=self._vis_item_name(),
        )
        del log_prefix, log_step
