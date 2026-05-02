from __future__ import annotations

from typing import Any

from pathlib import Path

import hydra
import torch
from PIL import Image as PILImage

from pi3.models.hand_object_loss import estimate_scene_scale_from_depth
from pi3.visualization import pi3x_rerun_export as vis_export
from pi3.visualization import export_pi3x_rerun_sample
from trainers.checkpoint_utils import load_trainable_checkpoint, save_trainable_checkpoint
from trainers.base_trainer_accelerate import BaseTrainer
from trainers.pi3x_training_policy import apply_pi3x_training_policy


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(value.item())
    return bool(value)


class _LossOutput(dict):
    @property
    def loss(self):
        return self["loss"]

    def __getattr__(self, item):
        return self[item]


class Pi3XTrainer(BaseTrainer):
    def __init__(self, cfg):
        self._loss_cfg = cfg.loss.train_loss
        self._hand_encoder_cfg = cfg.get("hand_encoder", None)
        self._vis_cfg = _cfg_get(cfg, "vis", {})
        super().__init__(cfg)
        self.train_loss = hydra.utils.instantiate(self._loss_cfg)
        self.test_loss = hydra.utils.instantiate(self._loss_cfg)
        self.hand_encoder = self._build_hand_encoder()

    def _build_hand_encoder(self):
        if self._hand_encoder_cfg is None:
            return None
        hand_encoder = hydra.utils.instantiate(self._hand_encoder_cfg)
        hand_encoder.eval()
        for param in hand_encoder.parameters():
            param.requires_grad = False
        return hand_encoder.to(self.accelerator.device)

    def prepare_model(self):
        model = super().prepare_model()
        # Materialize the lazy hand head before parameter counting / optimizer setup.
        model._get_hand_mano_head(torch.device("cpu"))
        load_report = getattr(model, "_last_load_report", None)
        if load_report is not None:
            self.log_info(
                "Loaded Pi3X base checkpoint: "
                f"missing={len(load_report.missing_keys)}, unexpected={len(load_report.unexpected_keys)}"
            )
        apply_pi3x_training_policy(model)
        if hasattr(model, "hand_mano_head") and model.hand_mano_head is not None and hasattr(model.hand_mano_head, "mano"):
            for param in model.hand_mano_head.mano.parameters():
                param.requires_grad = False
        return model

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            heads = []
            adapters = []
            lora = []
            for name, param in model_.named_parameters():
                if not param.requires_grad:
                    continue
                if "ho_decoder" in name and "lora_" in name:
                    lora.append(param)
                elif name.startswith(
                    (
                        "hand_token_adapter",
                        "object_query_adapter",
                        "hand_mano_head",
                        "object_pose_head",
                        "register_token",
                        "metric_token",
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
                {"params": lora, "lr": lr, "weight_decay": 0.0},
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

    def _stack_scene_inputs(self, batch):
        imgs = torch.stack([view["img"] for view in batch], dim=1)
        depths = torch.stack([view["depthmap"] for view in batch], dim=1)
        intrinsics = torch.stack([view["camera_intrinsics"] for view in batch], dim=1)
        poses = torch.stack([view["camera_pose"] for view in batch], dim=1)
        scene_focus_masks = torch.stack(
            [view["hand"]["mask"].bool() | view["object_multiview"]["grasped_object_mask"].bool() for view in batch],
            dim=1,
        )
        object_masks = torch.stack([view["object_multiview"]["grasped_object_mask"].bool() for view in batch], dim=1)
        object_valid = torch.stack([view["object_multiview"]["grasped_object_valid"].bool() for view in batch], dim=1)
        object_multiview = dict(batch[0]["object_multiview"])
        object_multiview["grasped_object_mask"] = object_masks
        object_multiview["grasped_object_valid"] = object_valid
        return imgs, depths, intrinsics, poses, scene_focus_masks, object_multiview

    def _build_hand_inputs(self, batch, imgs):
        batch_size = batch[0]["img"].shape[0]
        hand_masks = []
        owner_index = []
        hand_is_right = []

        for view_idx, view in enumerate(batch):
            masks = view["hand"]["mask"]
            valid = view["hand"]["valid"]
            sides = view["hand"]["mano_side"]
            for batch_idx in range(batch_size):
                if not _as_bool(valid[batch_idx]) or not bool(masks[batch_idx].any()):
                    continue
                hand_masks.append(masks[batch_idx].float())
                owner_index.append([batch_idx, view_idx, 0])
                hand_is_right.append(sides[batch_idx] == "right")

        device = imgs.device
        height, width = imgs.shape[-2:]
        if hand_masks:
            hand_masks_tensor = torch.stack(hand_masks, dim=0).to(device=device)
            owner_index_tensor = torch.tensor(owner_index, dtype=torch.long, device=device)
            hand_is_right_tensor = torch.tensor(hand_is_right, dtype=torch.bool, device=device)
        else:
            hand_masks_tensor = imgs.new_zeros((0, height, width))
            owner_index_tensor = torch.zeros((0, 3), dtype=torch.long, device=device)
            hand_is_right_tensor = torch.zeros((0,), dtype=torch.bool, device=device)

        return {
            "hand_masks": hand_masks_tensor,
            "owner_index": owner_index_tensor,
            "hand_is_right": hand_is_right_tensor,
        }

    def _gather_hand_gt(self, batch, owner_index):
        device = batch[0]["img"].device
        gt_pose_mano = []
        gt_hand_transl = []
        gt_mano_betas = []
        gt_joints_3d_cam = []
        gt_hand_is_right = []

        for hand_owner in owner_index.tolist():
            batch_idx, view_idx, _ = hand_owner
            view = batch[view_idx]
            gt_pose_mano.append(view["hand"]["pose_mano"][batch_idx].float())
            gt_hand_transl.append(view["hand"]["hand_transl"][batch_idx].float())
            gt_mano_betas.append(view["hand"]["mano_betas"][batch_idx].float())
            gt_joints_3d_cam.append(view["hand"]["joints_3d_cam"][batch_idx].float())
            gt_hand_is_right.append(torch.tensor(view["hand"]["mano_side"][batch_idx] == "right", dtype=torch.bool))


        if gt_pose_mano:
            return {
                "hand_valid": torch.ones((len(gt_pose_mano),), dtype=torch.bool, device=device),
                "hand_pose_mano": torch.stack(gt_pose_mano, dim=0).to(device=device),
                "hand_transl": torch.stack(gt_hand_transl, dim=0).to(device=device),
                "hand_mano_betas": torch.stack(gt_mano_betas, dim=0).to(device=device),
                "hand_joints_3d_cam": torch.stack(gt_joints_3d_cam, dim=0).to(device=device),
                "hand_is_right": torch.stack(gt_hand_is_right, dim=0).to(device=device),
            }

        return {
            "hand_valid": torch.zeros((0,), dtype=torch.bool, device=device),
            "hand_pose_mano": torch.zeros((0, 48), dtype=torch.float32, device=device),
            "hand_transl": torch.zeros((0, 3), dtype=torch.float32, device=device),
            "hand_mano_betas": torch.zeros((0, 10), dtype=torch.float32, device=device),
            "hand_joints_3d_cam": torch.zeros((0, 21, 3), dtype=torch.float32, device=device),
            "hand_is_right": torch.zeros((0,), dtype=torch.bool, device=device)
        }

    def _compute_hand_gt_mesh(self, hand_gt):
        """Compute GT hand vertices and joints by calling MANO (metric meters output)."""
        device = hand_gt["hand_pose_mano"].device
        N = hand_gt["hand_pose_mano"].shape[0]
        if N == 0 or self.model.hand_mano_layer is None:
            return (
                hand_gt.get("hand_joints_3d_cam", torch.zeros((0, 21, 3), device=device)),
                torch.zeros((0, 778, 3), device=device),
            )

        hand_pose = hand_gt["hand_pose_mano"].float()
        hand_betas = hand_gt["hand_mano_betas"].float()
        hand_transl = hand_gt["hand_transl"].float()  # m → mm for MANO
        hand_is_right = hand_gt.get("hand_is_right", None)
        if hand_is_right is None:
            default_side = bool(getattr(self.model.hand_mano_layer, "is_rhand", True))
            hand_is_right = torch.full((N,), default_side, dtype=torch.bool, device=device)

        vertices = torch.zeros((N, 778, 3), dtype=hand_pose.dtype, device=device)
        joints = torch.zeros((N, 21, 3), dtype=hand_pose.dtype, device=device)

        for side_value in (True, False):
            side_mask = hand_is_right == side_value
            if not side_mask.any():
                continue
            side = "right" if side_value else "left"
            if side not in self.model.hand_mano_layer:
                continue
            mano = self.model.hand_mano_layer[side]
            idx = side_mask.nonzero(as_tuple=False).squeeze(-1)
            with torch.no_grad():
                mano_out = mano.forward(
                    hand_pose[idx],
                    hand_betas[idx],
                    th_trans=hand_transl[idx],
                )
            vertices[idx] = mano_out.vertices.reshape(len(idx), -1, 3).to(dtype=hand_pose.dtype) / 1000.0  # mm → m
            joints[idx] = mano_out.joints.reshape(len(idx), -1, 3).to(dtype=hand_pose.dtype) / 1000.0    # mm → m

        return joints, vertices

    def _build_object_gt(self, batch):
        device = batch[0]["img"].device
        object_pose = torch.stack([view["object_multiview"]["grasped_object_pose_obj2cam"] for view in batch], dim=1).to(device=device)
        object_valid = torch.stack([view["object_multiview"]["grasped_object_valid"].bool() for view in batch], dim=1).to(device=device)
        object_normalization_scale = batch[0]["object_multiview"]["normalization_scale"]
        if not torch.is_tensor(object_normalization_scale):
            object_normalization_scale = torch.as_tensor(object_normalization_scale)
        object_normalization_scale = object_normalization_scale.to(device=device)
        return {
            "object_valid": object_valid,
            "object_pose_obj2cam": object_pose,
            "object_normalization_scale": object_normalization_scale,
        }

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

    def forward_batch(self, batch, mode="train"):
        if isinstance(batch, dict):
            pred = self.model(**batch)
            return [pred, batch]

        imgs, depths, intrinsics, poses, scene_focus_masks, object_multiview = self._stack_scene_inputs(batch)
        hand_inputs = self._build_hand_inputs(batch, imgs)

        if self.hand_encoder is None or hand_inputs["hand_masks"].shape[0] == 0:
            hand_queries = None
            hand_masks = None
            hand_owner_index = None
            hand_is_right = None
        else:
            with torch.no_grad():
                hand_encoder_out = self.hand_encoder(
                    imgs,
                    hand_inputs["hand_masks"],
                    hand_inputs["owner_index"],
                    hand_inputs["hand_is_right"],
                )
            valid_source_index = hand_encoder_out.get("source_index", None)
            if valid_source_index is not None:
                hand_masks = hand_inputs["hand_masks"].index_select(0, valid_source_index)
            else:
                hand_masks = hand_inputs["hand_masks"]
            hand_queries = hand_encoder_out["hand_queries"]
            hand_owner_index = hand_encoder_out["owner_index"]
            hand_is_right = hand_encoder_out["hand_is_right"]

        resolved_owner = hand_owner_index if hand_owner_index is not None else hand_inputs["owner_index"]

        # Step 1: Build GT in metric scale
        hand_gt_metric = self._gather_hand_gt(batch, resolved_owner)
        gt_hand_joints_metric, gt_hand_vertices_metric = self._compute_hand_gt_mesh(hand_gt_metric)
        object_gt_metric = self._build_object_gt(batch)

        # Step 2: Forward model
        model_kwargs = {
            "imgs": imgs,
            "depths": depths if self.model.use_multimodal else None,
            "intrinsics": intrinsics if self.model.use_multimodal else None,
            "poses": poses if self.model.use_multimodal else None,
            "with_prior": True,
            "hand_queries": hand_queries,
            "hand_masks": hand_masks,
            "hand_owner_index": hand_owner_index,
            "hand_is_right": hand_is_right,
            "object_multiview": object_multiview,
        }
        if self.model.use_multimodal:
            bsz, num_views = imgs.shape[:2]
            full_mask = torch.ones((bsz, num_views), dtype=torch.bool, device=imgs.device)
            model_kwargs["mask_add_depth"] = full_mask
            model_kwargs["mask_add_ray"] = full_mask
            model_kwargs["mask_add_pose"] = full_mask

        pred = self.model(**model_kwargs)

        # Step 3: Compute scene_scale (pred units / metric)
        with torch.no_grad():
            scene_scale = estimate_scene_scale_from_depth(
                pred["local_points"][..., 2].detach(),
                depths,
                valid_mask=depths > 0,
                focus_mask=scene_focus_masks,
            )

        # Step 4: Build scene geometry GT in metric, then convert to pred scale
        scene_gt_metric = vis_export.build_scene_gt_metric(batch)
        gt = vis_export.convert_scene_gt_to_pred_scale(scene_gt_metric, scene_scale)
        del scene_gt_metric

        # Step 5: Merge hand GT — joints/vertices stay metric, transl converted to pred scale
        scale_per_hand = scene_scale[resolved_owner[:, 0]].view(-1, 1)
        gt["hand_valid"] = hand_gt_metric["hand_valid"]
        gt["hand_pose_mano"] = hand_gt_metric["hand_pose_mano"]
        gt["hand_mano_betas"] = hand_gt_metric["hand_mano_betas"]
        gt["hand_transl"] = hand_gt_metric["hand_transl"] * scale_per_hand
        gt["hand_joints_3d"] = gt_hand_joints_metric
        gt["hand_vertices"] = gt_hand_vertices_metric
        gt["hand_is_right"] = hand_gt_metric["hand_is_right"]
        gt["hand_owner_index"] = resolved_owner

        # Step 6: Merge object GT — transl and normalization_scale converted to pred scale
        scale_obj = scene_scale.view(-1, 1, 1)
        obj_pose = object_gt_metric["object_pose_obj2cam"].clone()
        obj_pose[..., :3, 3] = obj_pose[..., :3, 3] * scale_obj
        gt["object_valid"] = object_gt_metric["object_valid"]
        gt["object_pose_obj2cam"] = obj_pose
        gt["object_normalization_scale"] = object_gt_metric["object_normalization_scale"] * scene_scale

        del hand_gt_metric, gt_hand_joints_metric, gt_hand_vertices_metric, object_gt_metric

        # Clear per-step intermediates to help GC
        del model_kwargs, scene_focus_masks, hand_inputs

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

        # For validation mode, check epoch interval. For training mode, caller controls timing.
        if mode == "test" and global_step is None:
            if epoch % self._vis_interval() != 0:
                return
            if batch_idx != 0:
                return

        pred, gt = forward_outputs
        sample_index = min(self._vis_sample_index(), batch[0]["img"].shape[0] - 1)
        if sample_index < 0:
            return

        # Naming: step-based uses step number, epoch-based uses epoch number
        if global_step is not None:
            tag = f"step_{global_step:07d}"
            log_prefix = "train_vis"
            log_step = global_step
        else:
            tag = f"epoch_{epoch:04d}"
            log_prefix = "val_vis"
            log_step = epoch

        output_root = self._vis_output_root() / tag
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"sample_{sample_index:03d}.rrd"
        export_pi3x_rerun_sample(
            output_path=output_path,
            batch=batch,
            pred=pred,
            gt=gt,
            sample_index=sample_index,
            data_root=self._resolve_data_root(),
            mano_layer=self.model.hand_mano_layer,
            item_name=self._vis_item_name(),
        )

        first_view = batch[0]
        rgb = PILImage.fromarray(vis_export._tensor_rgb_to_uint8(first_view["img"][sample_index]))
        valid_mask = gt["valid_masks"][sample_index, 0]
        pred_z = pred["local_points"][sample_index, 0, ..., 2]
        gt_z = gt["local_points"][sample_index, 0, ..., 2]
        pred_lo, pred_hi = vis_export._compute_depth_vis_range(pred_z, valid_mask)
        pred_depth = PILImage.fromarray(vis_export._depth_to_uint8(pred_z, valid_mask, lo=pred_lo, hi=pred_hi))
        gt_depth = PILImage.fromarray(vis_export._depth_to_uint8(gt_z, valid_mask, lo=pred_lo, hi=pred_hi))
        depth_error = PILImage.fromarray(
            vis_export._depth_to_uint8(
                (pred_z - gt_z).abs(),
                valid_mask,
            )
        )
        self.log_all(
            {
                "rgb": rgb,
                "pred_depth": pred_depth,
                "gt_depth": gt_depth,
                "depth_abs_error": depth_error,
            },
            step=log_step,
            prefix=log_prefix,
        )
