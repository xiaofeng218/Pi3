from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from .base.base_dataset import BaseDataset
from .base.transforms import *
import pi3.utils.cropping as cropping
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates


_YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


class DexYCBDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        split_style="s0_like_subject01",
        subject="20200709-subject-01",
        local_window_radius=12,
        object_multiview_subdir="canonical_views_224",
        max_tracks=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.data_root = data_root
        self.dataset_label = "DexYCB"
        self.split_style = split_style
        self.subject = subject
        self.local_window_radius = local_window_radius
        self.object_multiview_subdir = object_multiview_subdir
        self.max_tracks = max_tracks

        self.intrinsics_cache = self._load_intrinsics_cache()
        self.extrinsics_cache = {}
        self.mano_cache = {}
        self.sequence_meta_cache = {}
        self.object_multiview_cache = {}
        self.object_model_index = self._build_object_model_index()

        self.tracks = self._build_tracks()
        self.sequences = self.tracks

        print(f"[{self.dataset_label}] Found {len(self.tracks)} camera tracks in {data_root}", flush=True)

    def __len__(self):
        return len(self.tracks)

    def _load_yaml(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _mode_to_split(self):
        return "train" if self.mode == "train" else "valid"

    def _load_intrinsics_cache(self):
        intrinsics_dir = Path(self.data_root) / "calibration" / "intrinsics"
        cache = {}
        for path in sorted(intrinsics_dir.glob("*_640x480.yml")):
            data = self._load_yaml(path)
            color = data["color"]
            k_matrix = np.array(
                [
                    [color["fx"], 0.0, color["ppx"]],
                    [0.0, color["fy"], color["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            serial = path.stem.replace("_640x480", "")
            cache[serial] = k_matrix
        return cache

    def _build_object_model_index(self):
        models_dir = Path(self.data_root) / "models"
        if not models_dir.is_dir():
            return {}

        index = {}
        for object_id, class_name in _YCB_CLASSES.items():
            path = models_dir / class_name
            if path.is_dir():
                index[object_id] = path
        return index

    def _load_extrinsics(self, extrinsics_id):
        if extrinsics_id in self.extrinsics_cache:
            return self.extrinsics_cache[extrinsics_id]

        path = Path(self.data_root) / "calibration" / f"extrinsics_{extrinsics_id}" / "extrinsics.yml"
        data = self._load_yaml(path)
        cache = {}
        for serial, values in data["extrinsics"].items():
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :4] = np.asarray(values, dtype=np.float32).reshape(3, 4)
            cache[serial] = transform

        self.extrinsics_cache[extrinsics_id] = cache
        return cache

    def _load_mano_betas(self, mano_calib):
        if mano_calib in self.mano_cache:
            return self.mano_cache[mano_calib]

        calibration_root = Path(self.data_root) / "calibration"
        candidates = [
            calibration_root / mano_calib / "mano.yml",
            calibration_root / f"mano_{mano_calib}" / "mano.yml",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Cannot resolve MANO calibration for {mano_calib!r} under {calibration_root}")
        data = self._load_yaml(path)
        betas = np.asarray(data["betas"], dtype=np.float32)
        self.mano_cache[mano_calib] = betas
        return betas

    def _list_sequences(self):
        subject_root = Path(self.data_root) / self.subject
        assert subject_root.is_dir(), f"Missing DexYCB subject directory: {subject_root}"
        return sorted(
            path.name
            for path in subject_root.iterdir()
            if path.is_dir() and (path / "meta.yml").exists()
        )

    def _sequence_in_split(self, sequence_idx):
        if self.split_style != "s0_like_subject01":
            raise ValueError(f"Unsupported split_style: {self.split_style}")
        split = self._mode_to_split()
        if split == "train":
            return sequence_idx % 5 != 4
        return sequence_idx % 5 == 4

    def _build_tracks(self):
        tracks = []
        subject_root = Path(self.data_root) / self.subject
        sequences = self._list_sequences()
        for sequence_idx, sequence in enumerate(sequences):
            if not self._sequence_in_split(sequence_idx):
                continue

            meta_path = subject_root / sequence / "meta.yml"
            meta = self._load_yaml(meta_path)
            self.sequence_meta_cache[f"{self.subject}/{sequence}"] = meta

            serials = meta["serials"]
            num_frames = int(meta["num_frames"])
            extrinsics_id = meta["extrinsics"]
            mano_calib = meta["mano_calib"][0]
            mano_side = meta["mano_sides"][0]
            ycb_ids = meta["ycb_ids"]
            ycb_grasp_ind = int(meta["ycb_grasp_ind"])
            grasped_object_id = int(ycb_ids[ycb_grasp_ind])
            grasped_object_local_index = ycb_grasp_ind

            self._load_extrinsics(extrinsics_id)
            self._load_mano_betas(mano_calib)

            for serial in serials:
                tracks.append(
                    {
                        "subject": self.subject,
                        "sequence": sequence,
                        "camera": serial,
                        "split": self._mode_to_split(),
                        "sequence_idx": sequence_idx,
                        "num_frames": num_frames,
                        "extrinsics_id": extrinsics_id,
                        "mano_calib": mano_calib,
                        "mano_side": mano_side,
                        "grasped_object_id": grasped_object_id,
                        "grasped_object_local_index": grasped_object_local_index,
                    }
                )

        if self.max_tracks is not None and len(tracks) > self.max_tracks:
            tracks = tracks[: self.max_tracks]

        return tracks

    def _sample_frame_indices(self, track, rng):
        num_frames = track["num_frames"]
        if num_frames <= self.frame_num:
            should_replace = num_frames < self.frame_num
            return list(rng.choice(np.arange(num_frames), size=self.frame_num, replace=should_replace))

        max_gap = 10
        gap = (num_frames - 1) / (self.frame_num - 1) if self.frame_num > 1 else 0
        if gap <= max_gap:
            indices = np.linspace(0, num_frames - 1, self.frame_num, dtype=int)
            return list(indices)

        window_span = (self.frame_num - 1) * max_gap
        if window_span >= num_frames:
            indices = np.linspace(0, num_frames - 1, self.frame_num, dtype=int)
        else:
            start = int(rng.integers(0, num_frames - window_span))
            indices = np.linspace(start, start + window_span, self.frame_num, dtype=int)
        return list(indices)

    def _to_object_pose_4x4(self, pose_3x4, pose_valid):
        if not pose_valid:
            return np.zeros((4, 4), dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :4] = pose_3x4.astype(np.float32)
        return pose

    def _load_object_multiview_bundle(self, object_id):
        if self.object_multiview_subdir is None:
            return None

        if object_id in self.object_multiview_cache:
            return self.object_multiview_cache[object_id]

        model_dir = self.object_model_index.get(int(object_id))
        if model_dir is None:
            raise KeyError(f"Missing model directory for DexYCB object id {object_id}")

        bundle_dir = model_dir / self.object_multiview_subdir
        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"Missing object multiview bundle: {bundle_dir}")

        camera_params = np.load(bundle_dir / "camera_params.npz", allow_pickle=False)
        intrinsics = camera_params["K"].astype(np.float32)
        camera_pose = camera_params["T_oc"].astype(np.float32)
        normalization_center = camera_params["normalization_center"].astype(np.float32)
        normalization_scale = np.float32(camera_params["normalization_scale"])

        view_count = intrinsics.shape[0]
        color_files = [bundle_dir / f"color_{view_idx:06d}.jpg" for view_idx in range(view_count)]
        depth_files = [bundle_dir / f"aligned_depth_to_color_{view_idx:06d}.png" for view_idx in range(view_count)]
        for color_file, depth_file in zip(color_files, depth_files):
            if not color_file.is_file() or not depth_file.is_file():
                raise FileNotFoundError(
                    f"Incomplete object multiview bundle for object {object_id}: missing {color_file.name} or {depth_file.name}"
                )

        bundle = {
            "object_id": np.int32(object_id),
            "bundle_dir": bundle_dir,
            "color_files": color_files,
            "depth_files": depth_files,
            "camera_intrinsics": intrinsics,
            "camera_pose": camera_pose,
            "normalization_center": normalization_center,
            "normalization_scale": normalization_scale,
        }
        self.object_multiview_cache[object_id] = bundle
        return bundle

    def _prepare_object_multiview(self, object_id):
        bundle = self._load_object_multiview_bundle(object_id)
        if bundle is None:
            return None

        imgs = []
        depthmaps = []
        pts3d_all = []

        for color_file, depth_file, intrinsics, camera_pose in zip(
            bundle["color_files"],
            bundle["depth_files"],
            bundle["camera_intrinsics"],
            bundle["camera_pose"],
        ):
            image = Image.open(color_file).convert("RGB")
            imgs.append(self.transform(image))

            depthmap = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                camera_pose=camera_pose,
                z_far=self.z_far,
            )
            depthmap[~valid_mask] = 0.0
            depthmaps.append(depthmap.astype(np.float32))
            pts3d_all.append(pts3d.astype(np.float32))

        return {
            "img": torch.stack(imgs, dim=0),
            "depthmap": np.stack(depthmaps, axis=0).astype(np.float32),
            "camera_intrinsics": bundle["camera_intrinsics"].copy(),
            "camera_pose": bundle["camera_pose"].copy(),
            "pts3d": np.stack(pts3d_all, axis=0).astype(np.float32),
            "normalization_center": bundle["normalization_center"].copy(),
            "normalization_scale": np.float32(bundle["normalization_scale"]),
        }

    def _build_hand_payload(self, hand_mask, hand_valid, pose_m, joint_3d, joint_2d, mano_betas, mano_side):
        return {
            "mask": hand_mask,
            "valid": hand_valid,
            "pose_mano": pose_m[:48].astype(np.float32),
            "hand_transl": pose_m[48:51].astype(np.float32),
            "joints_3d_cam": joint_3d.astype(np.float32),
            "joints_2d": joint_2d.astype(np.float32),
            "mano_betas": mano_betas.astype(np.float32),
            "mano_side": mano_side,
        }

    def _build_object_multiview_payload(
        self,
        base_object_multiview,
        grasped_object_id,
        grasped_object_mask,
        grasped_object_valid,
        grasped_object_pose_obj2cam,
    ):
        payload = {} if base_object_multiview is None else dict(base_object_multiview)
        payload.update(
            {
                "grasped_object_id": np.int32(grasped_object_id),
                "grasped_object_mask": grasped_object_mask,
                "grasped_object_valid": grasped_object_valid,
                "grasped_object_pose_obj2cam": grasped_object_pose_obj2cam.astype(np.float32),
            }
        )
        return payload

    def _crop_resize_with_masks(self, image, depthmap, intrinsics, resolution, rng=None, info=None, masks=None):
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
            return image, depthmap, intrinsics, mask_arrays

        width, height = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, width - cx)
        min_margin_y = min(cy, height - cy)
        assert min_margin_x > width / 5, f"Bad principal point in view={info}"
        assert min_margin_y > height / 5, f"Bad principal point in view={info}"
        left, top = cx - min_margin_x, cy - min_margin_y
        right, bottom = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (left, top, right, bottom)

        image, depthmap, intrinsics, _, _ = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        mask_arrays = {key: value[top:bottom, left:right] for key, value in mask_arrays.items()}

        input_resolution = np.array(image.size)
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5)
            image, depthmap, intrinsics, _, _ = cropping.center_crop_image_depthmap(
                image, depthmap, intrinsics, crop_scale
            )
            width2, height2 = image.size
            out_resolution = np.floor(input_resolution * crop_scale).astype(int)
            margins = input_resolution - out_resolution
            offx, offy = (margins / 2).astype(int)
            mask_arrays = {
                key: value[offy:offy + height2, offx:offx + width2]
                for key, value in mask_arrays.items()
            }
            input_resolution = np.array(image.size)

        if self.aug_crop > 1:
            target_resolution = target_resolution + rng.integers(0, self.aug_crop)

        scale_final = max(target_resolution / image.size) + 1e-8
        resize_resolution = np.floor(input_resolution * scale_final).astype(int)
        image, depthmap, intrinsics, _, _ = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )
        mask_arrays = {
            key: cv2.resize(value, tuple(resize_resolution), interpolation=cv2.INTER_NEAREST)
            for key, value in mask_arrays.items()
        }

        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        left, top, right, bottom = crop_bbox
        image, depthmap, intrinsics2, _, _ = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        mask_arrays = {key: value[top:bottom, left:right] for key, value in mask_arrays.items()}
        mask_arrays = {key: value.astype(bool) for key, value in mask_arrays.items()}

        return image, depthmap, intrinsics2, mask_arrays

    def _get_views(self, index, resolution, rng):
        track = self.tracks[index]
        frame_indices = self._sample_frame_indices(track, rng)
        self.this_views_info = {
            "track": f'{track["subject"]}/{track["sequence"]}/{track["camera"]}',
            "frame_indices": frame_indices,
        }

        subject_root = Path(self.data_root) / track["subject"] / track["sequence"] / track["camera"]
        base_intrinsics = self.intrinsics_cache[track["camera"]]
        camera_pose = self.extrinsics_cache[track["extrinsics_id"]][track["camera"]]
        mano_betas = self.mano_cache[track["mano_calib"]]
        base_object_multiview = self._prepare_object_multiview(track["grasped_object_id"])

        views = []
        for frame_idx in frame_indices:
            rgb_path = subject_root / f"color_{frame_idx:06d}.jpg"
            depth_path = subject_root / f"aligned_depth_to_color_{frame_idx:06d}.png"
            label_path = subject_root / f"labels_{frame_idx:06d}.npz"

            rgb_image = np.array(Image.open(rgb_path))
            depthmap = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            label = np.load(label_path)
            seg = label["seg"]

            pose_m = label["pose_m"][0].astype(np.float32)
            joint_3d = label["joint_3d"][0].astype(np.float32)
            joint_2d = label["joint_2d"][0].astype(np.float32)

            hand_mask = seg == 255
            local_idx = track["grasped_object_local_index"]
            grasped_object_mask = seg == track["grasped_object_id"]
            pose_y = label["pose_y"]
            object_pose_valid = local_idx < len(pose_y) and not np.allclose(pose_y[local_idx], 0.0)
            if local_idx < len(pose_y):
                object_pose = self._to_object_pose_4x4(pose_y[local_idx], object_pose_valid)
            else:
                object_pose = np.zeros((4, 4), dtype=np.float32)

            rgb_image, depthmap, intrinsics, masks = self._crop_resize_with_masks(
                rgb_image,
                depthmap,
                base_intrinsics.copy(),
                resolution,
                rng=rng,
                info=str(rgb_path),
                masks={
                    "hand_mask": hand_mask,
                    "grasped_object_mask": grasped_object_mask,
                },
            )

            hand_mask = masks["hand_mask"]
            grasped_object_mask = masks["grasped_object_mask"]

            hand_valid = bool(hand_mask.any()) and bool(np.any(pose_m != 0.0)) and bool(np.any(joint_3d != -1.0))
            grasped_object_valid = bool(grasped_object_mask.any()) and bool(object_pose_valid)

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset=self.dataset_label,
                    label=f'{track["subject"]}/{track["sequence"]}/{track["camera"]}',
                    instance=f"{frame_idx:06d}",
                    sparse_depth=(depthmap > 0).astype(np.uint8),
                    hand=self._build_hand_payload(
                        hand_mask=hand_mask,
                        hand_valid=hand_valid,
                        pose_m=pose_m,
                        joint_3d=joint_3d,
                        joint_2d=joint_2d,
                        mano_betas=mano_betas,
                        mano_side=track["mano_side"],
                    ),
                    object_multiview=self._build_object_multiview_payload(
                        base_object_multiview=base_object_multiview,
                        grasped_object_id=track["grasped_object_id"],
                        grasped_object_mask=grasped_object_mask,
                        grasped_object_valid=grasped_object_valid,
                        grasped_object_pose_obj2cam=object_pose,
                    ),
                )
            )

        return views
