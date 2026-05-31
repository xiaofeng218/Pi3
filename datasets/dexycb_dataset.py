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
from pi3.utils.projection import load_obj_vertices, map_points_between_intrinsics
from pi3.models.hamer.config import get_config as get_hamer_config, resolve_mano_path_template
from pi3.models.hamer.mano_layer import build_mano_layer_pair


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

_IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406], dtype=np.float32)


class DexYCBDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        split_style="s0_like_subject01",
        subject=None,
        selected_tracks=None,
        local_window_radius=12,
        object_multiview_subdir="canonical_views_224",
        include_object_multiview_payload=False,
        max_tracks=None,
        vertex_sample_count=2048,
        min_valid_ratio=None,
        max_frame_gap=10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.data_root = data_root
        self.dataset_label = "DexYCB"
        self.split_style = split_style
        self.vertex_sample_count = max(1, int(vertex_sample_count))
        self.max_frame_gap = max_frame_gap
        if subject is None or subject == "all":
            data_path = Path(data_root)
            discovered = []
            for d in sorted(data_path.iterdir()):
                if not d.is_dir() or not d.name.startswith("2020"):
                    continue
                children = list(d.iterdir())
                if any(cd.is_dir() and (cd / "meta.yml").exists() for cd in children):
                    discovered.append(d.name)
            if discovered:
                subject = discovered
            else:
                subject = ["20200709-subject-01"]
        if isinstance(subject, str):
            subject = [subject]
        self.subjects = list(subject)
        self.selected_tracks = selected_tracks
        self.local_window_radius = local_window_radius
        self.object_multiview_subdir = object_multiview_subdir
        self.include_object_multiview_payload = bool(include_object_multiview_payload)
        self.max_tracks = max_tracks

        self.min_valid_ratio = min_valid_ratio

        # 缓存公用数据（内外参、beta、物体）
        self.intrinsics_cache = self._load_intrinsics_cache()
        self.extrinsics_cache = {}
        self.mano_cache = {}
        self.sequence_meta_cache = {}
        self.object_multiview_cache = {}
        self.object_model_index = self._build_object_model_index()
        self.object_vertices_cache = {}
        self.hand_pose_decoder = None

        self.tracks = self._build_tracks()
        self.sequences = self.tracks

        print(f"[{self.dataset_label}] Found {len(self.tracks)} camera tracks in {data_root}", flush=True)

    def __len__(self):
        return len(self.tracks)

    def _load_yaml(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _load_camera_params(self, path):
        with np.load(path, allow_pickle=False) as camera_params:
            intrinsics = camera_params["K"].astype(np.float32)
            camera_pose = camera_params["T_oc"].astype(np.float32)
            normalization_center = camera_params["normalization_center"].astype(np.float32)
            normalization_scale = np.float32(camera_params["normalization_scale"])
        return intrinsics, camera_pose, normalization_center, normalization_scale

    def _load_rgb_array(self, path):
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))

    def _load_rgb_pil(self, path):
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def _load_label_npz(self, path):
        with np.load(path, allow_pickle=False) as label:
            return {key: label[key] for key in label.files}

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

    def _load_object_template_vertices(self, object_id):
        object_id = int(object_id)
        if object_id in self.object_vertices_cache:
            return self.object_vertices_cache[object_id]

        model_dir = self.object_model_index.get(object_id)
        if model_dir is None:
            raise KeyError(f"Missing model directory for DexYCB object id {object_id}")

        obj_path = model_dir / "textured_simple.obj"
        vertices = load_obj_vertices(obj_path)
        self.object_vertices_cache[object_id] = vertices
        return vertices

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
            flat_hand_mean=mano_cfg.get("flat_hand_mean", False),
            ncomps=mano_cfg.get("ncomps", 45),
            use_pca=mano_cfg.get("use_pca", True),
            center_idx=mano_cfg.get("center_idx", None),
            root_rot_mode=mano_cfg.get("root_rot_mode", "axisang"),
            joint_rot_mode=mano_cfg.get("joint_rot_mode", "axisang"),
            robust_rot=mano_cfg.get("robust_rot", False),
        )
        return self.hand_pose_decoder

    def _decode_hand_pose_gt_rotmats(self, pose_mano, mano_side):
        decoder = self._get_hand_pose_decoder()
        layer = decoder["right" if mano_side == "right" else "left"]
        pose_tensor = torch.as_tensor(pose_mano, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            global_orient, hand_pose = layer.decode_pose_coeffs_to_rotmat(pose_tensor)
        return (
            global_orient[0].detach().cpu().numpy().astype(np.float32),
            hand_pose[0].detach().cpu().numpy().astype(np.float32),
        )

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

    def _list_sequences(self, subject):
        subject_root = Path(self.data_root) / subject
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
        for subject in self.subjects:
            subject_root = Path(self.data_root) / subject
            sequences = self._list_sequences(subject)
            for sequence_idx, sequence in enumerate(sequences):
                if not self._sequence_in_split(sequence_idx):
                    continue

                meta_path = subject_root / sequence / "meta.yml"
                meta = self._load_yaml(meta_path)
                self.sequence_meta_cache[f"{subject}/{sequence}"] = meta

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
                            "subject": subject,
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

        if self.selected_tracks is not None:
            tracks = self._select_explicit_tracks(tracks)
        elif self.max_tracks is not None and len(tracks) > self.max_tracks:
            tracks = tracks[: self.max_tracks]

        return tracks

    def _select_explicit_tracks(self, tracks):
        selectors = self.selected_tracks
        if selectors is None:
            return tracks
        if hasattr(selectors, "items"):
            selectors = dict(selectors)
        else:
            raise TypeError("selected_tracks must be a mapping from handedness to track selector")

        selected = []
        for side in ("left", "right"):
            selector = selectors.get(side)
            if selector is None:
                raise ValueError(f"selected_tracks must define both 'left' and 'right'; missing {side!r}")
            if hasattr(selector, "items"):
                selector = dict(selector)
            subject = selector.get("subject")
            sequence = selector.get("sequence")
            camera = selector.get("camera")
            if not all(isinstance(value, str) and value for value in (subject, sequence, camera)):
                raise ValueError(
                    f"selected_tracks[{side!r}] must include non-empty subject/sequence/camera strings"
                )

            matches = [
                track
                for track in tracks
                if track["subject"] == subject
                and track["sequence"] == sequence
                and track["camera"] == camera
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"selected_tracks[{side!r}] expected exactly one track for "
                    f"{subject}/{sequence}/{camera}, found {len(matches)}"
                )

            match = matches[0]
            if match["mano_side"] != side:
                raise ValueError(
                    f"selected_tracks[{side!r}] matched handedness {match['mano_side']!r} "
                    f"for {subject}/{sequence}/{camera}"
                )
            selected.append(match)

        return selected

    def _sample_frame_indices(self, track, rng):
        num_frames = track["num_frames"]
        if num_frames <= self.frame_num:
            should_replace = num_frames < self.frame_num
            return list(rng.choice(np.arange(num_frames), size=self.frame_num, replace=should_replace))

        gap = (num_frames - 1) / (self.frame_num - 1) if self.frame_num > 1 else 0
        if gap <= self.max_frame_gap:
            indices = np.linspace(0, num_frames - 1, self.frame_num, dtype=int)
            return list(indices)

        window_span = (self.frame_num - 1) * self.max_frame_gap
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

        intrinsics, camera_pose, normalization_center, normalization_scale = self._load_camera_params(
            bundle_dir / "camera_params.npz"
        )

        view_count = intrinsics.shape[0]
        color_files = [bundle_dir / f"color_{view_idx:06d}.jpg" for view_idx in range(view_count)]
        depth_files = [bundle_dir / f"aligned_depth_to_color_{view_idx:06d}.png" for view_idx in range(view_count)]
        mask_files = [bundle_dir / f"mask_{view_idx:06d}.png" for view_idx in range(view_count)]
        mask_files = [f for f in mask_files if f.is_file()] or []
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
            "mask_files": mask_files,
            "camera_intrinsics": intrinsics,
            "camera_pose": camera_pose,
            "normalization_center": normalization_center,
            "normalization_scale": normalization_scale,
        }
        self.object_multiview_cache[object_id] = bundle
        return bundle

    @staticmethod
    def _subsample_vertices(vertices: np.ndarray, target_count: int) -> np.ndarray:
        """Uniformly subsample vertices along the first axis to target_count."""
        num = vertices.shape[0]
        if num == target_count:
            return vertices.copy()
        indices = np.linspace(0, num - 1, target_count).round().astype(int)
        return vertices[indices]

    def _prepare_object_multiview(self, object_id, include_pts3d=True):
        bundle = self._load_object_multiview_bundle(object_id)
        if bundle is None:
            return None

        imgs = []
        depthmaps = []
        pts3d_all = [] if include_pts3d else None

        mask_files = bundle.get("mask_files", [])
        for idx, (color_file, depth_file, intrinsics, camera_pose) in enumerate(zip(
            bundle["color_files"],
            bundle["depth_files"],
            bundle["camera_intrinsics"],
            bundle["camera_pose"],
        )):
            image = self._load_rgb_pil(color_file)
            image = self.transform(image)

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
            if include_pts3d:
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                    depthmap=depthmap,
                    camera_intrinsics=intrinsics,
                    camera_pose=camera_pose,
                    z_far=self.z_far,
                )
                depthmap[~valid_mask] = 0.0
                pts3d_all.append(pts3d.astype(np.float32))
            depthmaps.append(depthmap.astype(np.float32))
        template_verts = self._load_object_template_vertices(object_id)
        template_verts = self._subsample_vertices(template_verts, self.vertex_sample_count)
        payload = {
            "img": torch.stack(imgs, dim=0),
            "depthmap": np.stack(depthmaps, axis=0).astype(np.float32),
            "camera_intrinsics": bundle["camera_intrinsics"].copy(),
            "camera_pose": bundle["camera_pose"].copy(),
            "template_vertices": template_verts,
            "normalization_center": bundle["normalization_center"].copy(),
            "normalization_scale": np.float32(bundle["normalization_scale"]),
        }
        if include_pts3d:
            payload["pts3d"] = np.stack(pts3d_all, axis=0).astype(np.float32)
        return payload

    def get_object_multiview_payload(self, object_id, include_pts3d=False):
        payload = self._prepare_object_multiview(object_id, include_pts3d=include_pts3d)
        if payload is None:
            return None
        result = {
            "img": payload["img"],
            "depthmap": payload["depthmap"],
            "camera_intrinsics": payload["camera_intrinsics"],
            "camera_pose": payload["camera_pose"],
        }
        if include_pts3d and "pts3d" in payload:
            result["pts3d"] = payload["pts3d"]
        return result

    def _build_hand_payload(
        self,
        hand_mask,
        hand_valid,
        pose_m,
        joint_3d,
        joint_2d,
        mano_betas,
        mano_side,
        global_orient_rotmat_gt,
        pose_rotmat_gt,
    ):
        return {
            "mask": hand_mask,
            "valid": hand_valid,
            "pose_mano": pose_m[:48].astype(np.float32),
            "pose_repr": "mano_pose_coeffs",
            "hand_transl": pose_m[48:51].astype(np.float32),
            "joints_3d_cam": joint_3d.astype(np.float32),
            "joints_2d": joint_2d.astype(np.float32),
            "mano_betas": mano_betas.astype(np.float32),
            "mano_side": mano_side,
            "global_orient_rotmat_gt": global_orient_rotmat_gt.astype(np.float32),
            "pose_rotmat_gt": pose_rotmat_gt.astype(np.float32),
        }

    def _build_object_payload(
        self,
        object_id,
        object_mask,
        object_valid,
        object_pose,
        scale_meta,
    ):
        return {
            "grasped_object_id": np.int32(object_id),
            "mask": object_mask.astype(bool),
            "valid": bool(object_valid),
            "pose_obj2cam": object_pose.astype(np.float32),
            "scale_meta": scale_meta,
        }

    def _build_scene_inputs(self, views):
        imgs = torch.stack([view["img"] for view in views], dim=0)
        depths = torch.stack([torch.as_tensor(view["depthmap"]).float() for view in views], dim=0)
        intrinsics = torch.stack([torch.as_tensor(view["camera_intrinsics"]).float() for view in views], dim=0)
        poses = torch.stack([torch.as_tensor(view["camera_pose"]).float() for view in views], dim=0)

        hand_masks = torch.stack([torch.as_tensor(view["hand"]["mask"]).float() for view in views], dim=0)
        hand_owner_index = torch.tensor([[0, view_idx, 0] for view_idx in range(len(views))], dtype=torch.long)
        hand_is_right = torch.tensor(
            [view["hand"]["mano_side"] == "right" for view in views],
            dtype=torch.bool,
        )

        object_masks = torch.stack([torch.as_tensor(view["object"]["mask"]).bool() for view in views], dim=0)
        object_valid = torch.tensor([bool(view["object"]["valid"]) for view in views], dtype=torch.bool)

        object_id = int(views[0]["object"]["grasped_object_id"])
        object_multiview_payload = self.get_object_multiview_payload(object_id)
        object_multiview = None
        if object_multiview_payload is not None:
            object_multiview = {
                "img": object_multiview_payload["img"],
                "depthmap": torch.as_tensor(object_multiview_payload["depthmap"]).float(),
                "camera_intrinsics": torch.as_tensor(object_multiview_payload["camera_intrinsics"]).float(),
                "camera_pose": torch.as_tensor(object_multiview_payload["camera_pose"]).float(),
            }

        return {
            "imgs": imgs,
            "depths": depths,
            "intrinsics": intrinsics,
            "poses": poses,
            "hand_masks": hand_masks,
            "hand_owner_index": hand_owner_index,
            "hand_is_right": hand_is_right,
            "object_masks": object_masks,
            "object_valid": object_valid,
            "object_multiview": object_multiview,
        }

    def _build_gt_metric(self, views):
        hand_valid = torch.tensor([bool(view["hand"]["valid"]) for view in views], dtype=torch.bool)
        hand_pose_coeffs = torch.stack([torch.as_tensor(view["hand"]["pose_mano"]).float() for view in views], dim=0)
        hand_global_orient_rotmat_gt = torch.stack(
            [torch.as_tensor(view["hand"]["global_orient_rotmat_gt"]).float() for view in views], dim=0
        )
        hand_pose_rotmat_gt = torch.stack(
            [torch.as_tensor(view["hand"]["pose_rotmat_gt"]).float() for view in views], dim=0
        )
        hand_transl = torch.stack([torch.as_tensor(view["hand"]["hand_transl"]).float() for view in views], dim=0)
        hand_mano_betas = torch.stack([torch.as_tensor(view["hand"]["mano_betas"]).float() for view in views], dim=0)
        hand_joints_3d = torch.stack([torch.as_tensor(view["hand"]["joints_3d_cam"]).float() for view in views], dim=0)
        hand_joints_2d = torch.stack([torch.as_tensor(view["hand"]["joints_2d"]).float() for view in views], dim=0)
        hand_camera_intrinsics = torch.stack([torch.as_tensor(view["camera_intrinsics"]).float() for view in views], dim=0)
        hand_is_right = torch.tensor([view["hand"]["mano_side"] == "right" for view in views], dtype=torch.bool)
        hand_owner_index = torch.tensor([[0, view_idx, 0] for view_idx in range(len(views))], dtype=torch.long)

        object_valid = torch.tensor([bool(view["object"]["valid"]) for view in views], dtype=torch.bool)
        object_pose_obj2cam = torch.stack([torch.as_tensor(view["object"]["pose_obj2cam"]).float() for view in views], dim=0)
        object_camera_intrinsics = torch.stack([torch.as_tensor(view["camera_intrinsics"]).float() for view in views], dim=0)

        object_multiview_shared = views[0]["object_multiview"]
        return {
            "hand_valid": hand_valid,
            "hand_pose_coeffs": hand_pose_coeffs,
            "hand_global_orient_rotmat_gt": hand_global_orient_rotmat_gt,
            "hand_pose_rotmat_gt": hand_pose_rotmat_gt,
            "hand_transl": hand_transl,
            "hand_mano_betas": hand_mano_betas,
            "hand_joints_3d": hand_joints_3d,
            "hand_joints_2d": hand_joints_2d,
            "hand_camera_intrinsics": hand_camera_intrinsics,
            "hand_is_right": hand_is_right,
            "hand_owner_index": hand_owner_index,
            "object_valid": object_valid,
            "object_pose_obj2cam": object_pose_obj2cam,
            "object_camera_intrinsics": object_camera_intrinsics,
            "object_template_vertices": torch.as_tensor(object_multiview_shared["template_vertices"]).float(),
            "object_normalization_center": torch.as_tensor(object_multiview_shared["normalization_center"]).float(),
            "object_normalization_scale": torch.as_tensor(object_multiview_shared["normalization_scale"]).float(),
            "object_scale_canonical_to_target": torch.stack(
                [torch.as_tensor(view["object"]["scale_meta"]["canonical_to_target_scale"]).float() for view in views], dim=0
            ),
        }

    def _build_gt_scale_meta(self, views):
        scene_focus_masks = torch.stack(
            [
                torch.as_tensor(view["hand"]["mask"]).bool() | torch.as_tensor(view["object"]["mask"]).bool()
                for view in views
            ],
            dim=0,
        )
        return {
            "scene_focus_masks": scene_focus_masks,
        }

    def _finalize_views_sample(self, views):
        object_id = int(views[0]["object"]["grasped_object_id"])
        object_multiview_payload = self.get_object_multiview_payload(object_id)
        return {
            "views": views,
            "object_multiview_payload": object_multiview_payload or {},
        }

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
        for view_idx, frame_idx in enumerate(frame_indices):
            rgb_path = subject_root / f"color_{frame_idx:06d}.jpg"
            depth_path = subject_root / f"aligned_depth_to_color_{frame_idx:06d}.png"
            label_path = subject_root / f"labels_{frame_idx:06d}.npz"

            rgb_image = self._load_rgb_array(rgb_path)
            depthmap = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            label = self._load_label_npz(label_path)
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
            joint_2d_processed = map_points_between_intrinsics(joint_2d, base_intrinsics, intrinsics)
            global_orient_rotmat_gt, pose_rotmat_gt = self._decode_hand_pose_gt_rotmats(
                pose_m[:48],
                track["mano_side"],
            )

            hand_valid = bool(hand_mask.any()) and bool(np.any(pose_m != 0.0)) and bool(np.any(joint_3d != -1.0))
            grasped_object_valid = bool(grasped_object_mask.any()) and bool(object_pose_valid)
            # Shared fields: tiny ones (~25KB) are cheap to include in every view;
            # heavy fields (img/depthmap/pts3d, ~56MB total) only in view 0.
            # NB: views are shuffled after _get_views, so consumers must search
            # for the first view that has "img" rather than using a fixed index.
            if base_object_multiview is not None:
                object_multiview = {
                    "template_vertices": base_object_multiview["template_vertices"],
                    "normalization_center": base_object_multiview["normalization_center"],
                    "normalization_scale": base_object_multiview["normalization_scale"],
                }
                if self.include_object_multiview_payload and view_idx == 0:
                    object_multiview.update(
                        {
                            "img": base_object_multiview["img"],
                            "depthmap": base_object_multiview["depthmap"],
                            "camera_intrinsics": base_object_multiview["camera_intrinsics"],
                            "camera_pose": base_object_multiview["camera_pose"],
                            "pts3d": base_object_multiview["pts3d"],
                        }
                    )
            else:
                object_multiview = {}

            object_payload = self._build_object_payload(
                object_id=track["grasped_object_id"],
                object_mask=grasped_object_mask,
                object_valid=grasped_object_valid,
                object_pose=object_pose,
                scale_meta={
                    "canonical_to_metric": np.float32(base_object_multiview["normalization_scale"]),
                    "canonical_to_target_scale": np.float32(base_object_multiview["normalization_scale"]),
                    "canonical_to_scene_metric": np.float32(base_object_multiview["normalization_scale"]),
                },
            )

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
                        joint_2d=joint_2d_processed,
                        mano_betas=mano_betas,
                        mano_side=track["mano_side"],
                        global_orient_rotmat_gt=global_orient_rotmat_gt,
                        pose_rotmat_gt=pose_rotmat_gt,
                    ),
                    object=object_payload,
                    object_multiview=object_multiview,
                )
            )

        if self.min_valid_ratio is not None:
            valid_count = sum(
                1 for v in views
                if v["hand"]["valid"] or v["object"]["valid"]
            )
            if valid_count / len(views) < self.min_valid_ratio:
                raise ValueError(
                    f"Sample has {valid_count}/{len(views)} valid frames "
                    f"(min_valid_ratio={self.min_valid_ratio})"
                )

        return views
