from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .base.base_dataset import BaseDataset
import pi3.utils.cropping as cropping
from pi3.models.hamer.geometry import aa_to_rotmat
from pi3.models.hamer.config import get_config as get_hamer_config, resolve_mano_path_template
from pi3.models.hamer.mano_layer import build_mano_layer_pair

_IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)
from pi3.utils.projection import project_points_cam_to_image


class ForeHOIDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        object_multiview_root=None,
        max_sequences=None,
        include_object_multiview_payload=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if data_root is None:
            raise ValueError("data_root is required")
        self.data_root = Path(data_root)
        self.object_multiview_root = (
            Path(object_multiview_root)
            if object_multiview_root is not None
            else self.data_root.parent / "object_multiview_pyrender"
        )
        self.dataset_label = "ForeHOI"
        self.include_object_multiview_payload = bool(include_object_multiview_payload)
        self.sequence_meta_cache: dict[str, dict] = {}
        self.meta_cache: dict[str, dict] = {}
        self.object_multiview_cache: dict[str, dict] = {}
        self.object_parts_scale_cache: dict[str, dict] = {}
        self.hand_pose_decoder = None
        self.sequences = self._build_sequences(max_sequences=max_sequences)

    def __len__(self):
        return len(self.sequences)

    def _build_sequences(self, max_sequences=None):
        split_path = self.data_root / "split.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing ForeHOI split file: {split_path}")
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        split_key = "train" if self.mode == "train" else "val"
        shards = split_data[split_key]["shards"]
        sequences = []
        for shard_name in shards:
            if (self.data_root / shard_name).is_dir():
                sequences.append({"sequence_name": shard_name})
        if max_sequences is not None:
            sequences = sequences[: int(max_sequences)]
        return sequences

    def _load_sequence_bundle(self, sequence_name: str) -> dict:
        if sequence_name in self.meta_cache:
            return self.meta_cache[sequence_name]
        sequence_dir = self.data_root / sequence_name
        meta_dir = sequence_dir / "meta"
        sequence_meta = json.loads((meta_dir / "sequence_meta.json").read_text(encoding="utf-8"))
        meta = np.load(meta_dir / "meta.npz", allow_pickle=False)
        bundle = {
            "sequence_dir": sequence_dir,
            "sequence_meta": sequence_meta,
            "camera_intrinsics": meta["camera_intrinsics"].astype(np.float32),
            "camera_pose": meta["camera_pose"].astype(np.float32),
            "sampled_frame_indices": meta["sampled_frame_indices"].astype(np.int32),
            "hand_pose_mano": meta["hand_pose_mano"].astype(np.float32),
            "hand_transl_cam": meta["hand_transl_cam"].astype(np.float32),
            "hand_joints_3d_cam": meta["hand_joints_3d_cam"].astype(np.float32),
            "hand_mano_betas": meta["hand_mano_betas"].astype(np.float32),
            "object_pose_obj2cam": meta["object_pose_obj2cam"].astype(np.float32),
            "hand_valid": meta["hand_valid"].astype(bool),
            "object_valid": meta["object_valid"].astype(bool),
        }
        self.meta_cache[sequence_name] = bundle
        self.sequence_meta_cache[sequence_name] = sequence_meta
        return bundle

    def _load_rgb_array(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))

    def _load_rgb_pil(self, path: Path):
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def _default_hamer_paths(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_file = repo_root / "configs" / "hamer" / "model_config.yaml"
        cache_dir = repo_root / "data" / "model" / "hamer" / "_DATA"
        return config_file, cache_dir

    def _get_hand_pose_decoder(self):
        if self.hand_pose_decoder is not None:
            return self.hand_pose_decoder
        config_file, cache_dir = self._default_hamer_paths()
        hamer_cfg = get_hamer_config(str(config_file), merge=True, cache_dir=str(cache_dir), update_cachedir=True)
        mano_cfg = {key.lower(): value for key, value in dict(hamer_cfg.MANO).items()}
        mano_data_dir = mano_cfg.get("data_dir", None)
        mano_cfg["model_path"] = resolve_mano_path_template(mano_cfg.get("model_path"), mano_data_dir)
        mano_root = Path(mano_cfg["model_path"])
        if mano_root.is_file():
            mano_root = mano_root.parent
        self.hand_pose_decoder = build_mano_layer_pair(
            mano_root=str(mano_root),
            flat_hand_mean=False,
            ncomps=45,
            use_pca=False,
            center_idx=mano_cfg.get("center_idx", None),
            root_rot_mode="axisang",
            joint_rot_mode="axisang",
            robust_rot=mano_cfg.get("robust_rot", False),
        )
        return self.hand_pose_decoder

    def _decode_hand_pose_gt_rotmats(self, pose_mano: np.ndarray, mano_side: str) -> tuple[np.ndarray, np.ndarray]:
        pose_mano = np.asarray(pose_mano, dtype=np.float32).reshape(48)
        decoder = self._get_hand_pose_decoder()
        layer = decoder["right" if mano_side == "right" else "left"]
        with torch.no_grad():
            pose_coeffs = torch.as_tensor(pose_mano, dtype=torch.float32).reshape(1, -1)
            decoded_global, decoded_pose = layer.decode_pose_coeffs_to_rotmat(pose_coeffs)
        return (
            decoded_global[0].detach().cpu().numpy().astype(np.float32),
            decoded_pose[0].detach().cpu().numpy().astype(np.float32),
        )

    def _crop_resize_with_masks(self, image, depthmap, intrinsics, resolution, rng=None, info=None, masks=None):
        del rng, info
        if masks is None:
            masks = {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        mask_arrays = {key: np.asarray(value).astype(np.uint8) for key, value in masks.items()}
        if not self.use_crop:
            src_w, src_h = image.size
            dst_w, dst_h = resolution
            if (src_w, src_h) != (dst_w, dst_h):
                image = image.resize((dst_w, dst_h), resample=Image.BICUBIC)
                depthmap = cv2.resize(depthmap, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
                mask_arrays = {
                    key: cv2.resize(value, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
                    for key, value in mask_arrays.items()
                }
                intrinsics = intrinsics.copy()
                sx = float(dst_w) / float(src_w)
                sy = float(dst_h) / float(src_h)
                intrinsics[0, 0] *= sx
                intrinsics[1, 1] *= sy
                intrinsics[0, 2] *= sx
                intrinsics[1, 2] *= sy
            return image, depthmap, intrinsics, {key: value.astype(bool) for key, value in mask_arrays.items()}

        width, height = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, width - cx)
        min_margin_y = min(cy, height - cy)
        left, top = cx - min_margin_x, cy - min_margin_y
        right, bottom = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (left, top, right, bottom)
        image, depthmap, intrinsics, _, _ = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        mask_arrays = {key: value[top:bottom, left:right] for key, value in mask_arrays.items()}
        return image, depthmap, intrinsics, {key: value.astype(bool) for key, value in mask_arrays.items()}

    def _sample_frame_indices(self, frame_count: int, rng) -> list[int]:
        del rng
        if frame_count <= 1:
            return [0] * self.frame_num
        return list(np.linspace(0, frame_count - 1, self.frame_num, dtype=int))

    def _load_object_multiview_bundle(self, asset_key: str) -> dict:
        if asset_key in self.object_multiview_cache:
            return self.object_multiview_cache[asset_key]
        bundle_dir = self.object_multiview_root / asset_key
        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"Missing ForeHOI object multiview bundle: {bundle_dir}")
        with np.load(bundle_dir / "camera_params.npz", allow_pickle=False) as payload:
            intrinsics = payload["K"].astype(np.float32)
            camera_pose = payload["T_oc"].astype(np.float32)
            normalization_center = payload["normalization_center"].astype(np.float32)
            normalization_scale = np.float32(payload["normalization_scale"])
        color_files = sorted(bundle_dir.glob("color_*.jpg"))
        depth_files = sorted(bundle_dir.glob("aligned_depth_to_color_*.png"))
        mask_files = sorted(bundle_dir.glob("mask_*.png"))
        bundle = {
            "bundle_dir": bundle_dir,
            "color_files": color_files,
            "depth_files": depth_files,
            "mask_files": mask_files,
            "camera_intrinsics": intrinsics,
            "camera_pose": camera_pose,
            "normalization_center": normalization_center,
            "normalization_scale": normalization_scale,
        }
        self.object_multiview_cache[asset_key] = bundle
        return bundle

    def _resolve_object_parts_descriptor(self, sequence_meta: dict) -> tuple[str | None, str | None, str]:
        asset_key = str(sequence_meta["objaverse_asset_key"])
        object_parts_root = sequence_meta.get("object_parts_root")
        if object_parts_root:
            object_parts_root = str(object_parts_root)
        else:
            candidate_root = self.data_root.parent / "object_parts"
            object_parts_root = str(candidate_root) if candidate_root.is_dir() else None
        object_parts_key = sequence_meta.get("object_parts_key")
        if object_parts_key:
            object_parts_key = str(object_parts_key)
        else:
            object_parts_key = asset_key
        return object_parts_root, object_parts_key, asset_key

    def _prepare_object_multiview(self, sequence_meta: dict) -> dict:
        object_parts_root, object_parts_key, asset_key = self._resolve_object_parts_descriptor(sequence_meta)
        bundle = self._load_object_multiview_bundle(asset_key)
        imgs = []
        depthmaps = []
        imgs = []
        depthmaps = []
        mask_files = bundle.get("mask_files", [])
        for idx, (color_file, depth_file) in enumerate(zip(bundle["color_files"], bundle["depth_files"])):
            image = self.transform(self._load_rgb_pil(color_file))
            depthmap = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            # Fill background pixels with ImageNet mean so that after
            # normalization the encoder sees a neutral zero signal.
            if idx < len(mask_files):
                mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
                bg_mask = torch.from_numpy(np.asarray(mask == 0, dtype=bool))
            else:
                # Fallback: pyrender background is exact [0,0,0] in RGB.
                bg_mask = (image < 2.0 / 255.0).all(dim=0)
            if bg_mask.any():
                image[:, bg_mask] = torch.from_numpy(_IMAGENET_MEAN_RGB).view(3, 1)
            imgs.append(image)
            depthmaps.append(depthmap)
        return {
            "img": torch.stack(imgs, dim=0),
            "depthmap": np.stack(depthmaps, axis=0).astype(np.float32),
            "camera_intrinsics": bundle["camera_intrinsics"].copy(),
            "camera_pose": bundle["camera_pose"].copy(),
            "normalization_center": bundle["normalization_center"].copy(),
            "normalization_scale": np.float32(bundle["normalization_scale"]),
            "asset": {
                "object_key": asset_key,
                "asset_type": "object_parts_bundle" if object_parts_root else "unknown",
                "asset_path": str(Path(object_parts_root) / object_parts_key) if object_parts_root else "",
            },
        }

    def _load_object_parts_scale_info(self, sequence_meta: dict) -> dict:
        object_parts_root, object_parts_key, object_key = self._resolve_object_parts_descriptor(sequence_meta)
        if object_key in self.object_parts_scale_cache:
            return self.object_parts_scale_cache[object_key]
        manifest_path = None
        if object_parts_root:
            manifest_path = Path(object_parts_root) / object_parts_key / "manifest.json"
        scale_info = {}
        if manifest_path is not None and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                scale_info = manifest.get("scale_info") or {}
        if not scale_info:
            scale_info = sequence_meta.get("object_scale_info") or {}
        self.object_parts_scale_cache[object_key] = scale_info
        return scale_info

    def _build_object_scale_meta(self, sequence_meta: dict, normalization_scale: np.float32) -> dict:
        scale_info = self._load_object_parts_scale_info(sequence_meta)
        alignment_scale = float(scale_info.get("alignment_scale", 1.0))
        canonical_to_metric = float(normalization_scale)
        canonical_to_target_scale = canonical_to_metric * alignment_scale
        return {
            "canonical_to_metric": np.float32(canonical_to_metric),
            "canonical_to_target_scale": np.float32(canonical_to_target_scale),
            "canonical_to_scene_metric": np.float32(canonical_to_target_scale),
            "source_to_target": np.float32(alignment_scale),
        }

    def _finalize_views_sample(self, views):
        object_multiview_payload = getattr(self, "_pending_object_multiview_payload", {})
        return {
            "views": views,
            "object_multiview_payload": object_multiview_payload,
        }

    def _get_views(self, index, resolution, rng):
        sequence_name = self.sequences[index]["sequence_name"]
        bundle = self._load_sequence_bundle(sequence_name)
        sequence_meta = bundle["sequence_meta"]
        frame_count = int(sequence_meta.get("frame_count", len(bundle["sampled_frame_indices"])))
        frame_indices = self._sample_frame_indices(frame_count, rng)
        object_multiview_payload = self._prepare_object_multiview(sequence_meta)
        object_scale_meta = self._build_object_scale_meta(sequence_meta, object_multiview_payload["normalization_scale"])

        views = []
        for view_idx, frame_idx in enumerate(frame_indices):
            rgb_path = bundle["sequence_dir"] / "rgb" / f"frame_{frame_idx:06d}.png"
            depth_path = bundle["sequence_dir"] / "depth" / f"frame_{frame_idx:06d}.npy"
            mask_path = bundle["sequence_dir"] / "mask" / f"frame_{frame_idx:06d}.png"
            rgb_image = self._load_rgb_array(rgb_path)
            depthmap = np.load(depth_path).astype(np.float32)
            with Image.open(mask_path) as image:
                merged_mask = np.asarray(image, dtype=np.uint8)
            hand_mask = merged_mask == 127
            object_mask = merged_mask == 255
            rgb_image, depthmap, intrinsics, masks = self._crop_resize_with_masks(
                rgb_image,
                depthmap,
                bundle["camera_intrinsics"].copy(),
                resolution,
                rng=rng,
                info=str(rgb_path),
                masks={"hand_mask": hand_mask, "object_mask": object_mask},
            )
            hand_mask = masks["hand_mask"]
            object_mask = masks["object_mask"]
            joints_2d, _ = project_points_cam_to_image(bundle["hand_joints_3d_cam"][frame_idx], intrinsics)
            global_orient_rotmat_gt, pose_rotmat_gt = self._decode_hand_pose_gt_rotmats(
                bundle["hand_pose_mano"][frame_idx],
                str(sequence_meta["mano_side"]),
            )
            object_multiview = {
                "normalization_center": object_multiview_payload["normalization_center"],
                "normalization_scale": object_multiview_payload["normalization_scale"],
                "asset": object_multiview_payload["asset"],
            }
            if self.include_object_multiview_payload and view_idx == 0:
                object_multiview.update(object_multiview_payload)
            views.append(
                {
                    "img": rgb_image,
                    "depthmap": depthmap.astype(np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                    "camera_pose": bundle["camera_pose"].astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": sequence_name,
                    "instance": f"{frame_idx:06d}",
                    "sparse_depth": (depthmap > 0).astype(np.uint8),
                    "hand": {
                        "mask": hand_mask.astype(bool),
                        "valid": bool(bundle["hand_valid"][frame_idx]) and bool(hand_mask.any()),
                        "pose_mano": bundle["hand_pose_mano"][frame_idx].astype(np.float32),
                        "pose_repr": "mano_full_aa",
                        "hand_transl": bundle["hand_transl_cam"][frame_idx].astype(np.float32),
                        "joints_3d_cam": bundle["hand_joints_3d_cam"][frame_idx].astype(np.float32),
                        "joints_2d": joints_2d.astype(np.float32),
                        "mano_betas": bundle["hand_mano_betas"].astype(np.float32),
                        "mano_side": str(sequence_meta["mano_side"]),
                        "global_orient_rotmat_gt": global_orient_rotmat_gt.astype(np.float32),
                        "pose_rotmat_gt": pose_rotmat_gt.astype(np.float32),
                    },
                    "object": {
                        "mask": object_mask.astype(bool),
                        "valid": bool(bundle["object_valid"][frame_idx]) and bool(object_mask.any()),
                        "pose_obj2cam": bundle["object_pose_obj2cam"][frame_idx].astype(np.float32),
                        "scale_meta": object_scale_meta,
                        "asset": object_multiview_payload["asset"],
                    },
                    "object_multiview": object_multiview,
                }
            )
        self._pending_object_multiview_payload = object_multiview_payload
        return views
