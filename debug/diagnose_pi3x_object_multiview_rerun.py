from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.base.utils import unified_collate_fn
from datasets.dexycb_dataset import DexYCBDataset
from pi3.models.pi3x import Pi3X
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates, homogenize_points


def _as_tensor(value, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def extract_object_multiview_inputs_from_views_batch(batch: dict[str, object]) -> dict[str, torch.Tensor]:
    views = batch.get("views", None)
    if not isinstance(views, list) or not views:
        raise KeyError("batch['views'] must be a non-empty list")
    required = ("img", "depthmap", "camera_intrinsics", "camera_pose")
    payload = None
    for view in views:
        candidate = view.get("object_multiview", None)
        if isinstance(candidate, dict) and all(key in candidate for key in required):
            payload = candidate
            break
    if payload is None:
        raise KeyError("No view contains a complete object_multiview payload")
    return {
        "imgs": _as_tensor(payload["img"], dtype=torch.float32),
        "depths": _as_tensor(payload["depthmap"], dtype=torch.float32),
        "intrinsics": _as_tensor(payload["camera_intrinsics"], dtype=torch.float32),
        "camera_poses": _as_tensor(payload["camera_pose"], dtype=torch.float32),
    }


def _find_object_multiview_payload(batch: dict[str, object]) -> dict[str, object]:
    views = batch["views"]
    required = ("img", "depthmap", "camera_intrinsics", "camera_pose")
    for view in views:
        payload = view.get("object_multiview", None)
        if isinstance(payload, dict) and all(key in payload for key in required):
            return payload
    raise KeyError("No view contains a complete object_multiview payload")


def gt_depth_to_world_points(
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_poses: torch.Tensor,
    imgs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    all_points: list[torch.Tensor] = []
    all_colors: list[torch.Tensor] = []
    batch, views = depths.shape[:2]
    for batch_idx in range(batch):
        for view_idx in range(views):
            depth = depths[batch_idx, view_idx].detach().cpu().numpy().astype(np.float32)
            intrinsic = intrinsics[batch_idx, view_idx].detach().cpu().numpy().astype(np.float32)
            pose = camera_poses[batch_idx, view_idx].detach().cpu().numpy().astype(np.float32)
            world_points, valid_mask = depthmap_to_absolute_camera_coordinates(depth, intrinsic, pose)
            valid_points = torch.from_numpy(world_points[valid_mask]).to(dtype=torch.float32)
            if valid_points.numel() > 0:
                all_points.append(valid_points)
            if imgs is not None:
                rgb = imgs[batch_idx, view_idx].detach().cpu().permute(1, 2, 0)
                colors = rgb[torch.from_numpy(valid_mask)].to(dtype=torch.float32)
                if colors.numel() > 0:
                    all_colors.append(colors)
    flat_points = torch.cat(all_points, dim=0) if all_points else torch.zeros((0, 3), dtype=torch.float32)
    if imgs is None:
        return flat_points, None
    flat_colors = torch.cat(all_colors, dim=0) if all_colors else torch.zeros((0, 3), dtype=torch.float32)
    return flat_points, flat_colors


def align_local_points_with_camera_poses(local_points: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bnij,bnhwj->bnhwi", camera_poses, homogenize_points(local_points))[..., :3]


def flatten_points_with_colors(
    points: torch.Tensor,
    imgs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    flat_points = points.reshape(-1, 3)
    finite_mask = torch.isfinite(flat_points).all(dim=-1)
    flat_points = flat_points[finite_mask]
    if imgs is None:
        return flat_points, None
    flat_colors = imgs.permute(0, 1, 3, 4, 2).reshape(-1, 3)
    flat_colors = flat_colors[finite_mask]
    return flat_points, flat_colors


def _tensor_image_to_uint8(img: torch.Tensor) -> np.ndarray:
    array = img.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).round().astype(np.uint8)


def _tensor_depth_to_uint16(depth: torch.Tensor) -> np.ndarray:
    array = depth.detach().cpu().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(array * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)


def _load_rerun():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("rerun is required to export the diagnostic .rrd output") from exc
    return rr


def _normalize_loader_batch(batch) -> dict[str, object]:
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, list):
        return {"views": batch}
    raise TypeError(f"Unsupported collated batch type: {type(batch).__name__}")


def load_batch(
    *,
    data_root: str,
    subject: str,
    mode: str,
    batch_index: int,
    resolution: tuple[int, int],
    frame_num: int,
) -> dict[str, object]:
    dataset = DexYCBDataset(
        data_root=data_root,
        subject=subject,
        mode=mode,
        resolution=[list(resolution)],
        frame_num=frame_num,
        include_object_multiview_payload=True,
        shuffle=False,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=unified_collate_fn,
    )
    for current_batch_index, batch in enumerate(loader):
        if current_batch_index == batch_index:
            return _normalize_loader_batch(batch)
    raise IndexError(f"batch_index={batch_index} is out of range")


def load_pi3x_image_model(checkpoint: str, device: torch.device) -> Pi3X:
    model = Pi3X(use_multimodal=True).eval()
    if checkpoint:
        state_dict = model._load_checkpoint_state_dict(checkpoint)
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        load_report = model.load_state_dict(compatible_state, strict=False)
        missing = [
            key
            for key in load_report.missing_keys
            if not key.startswith(("hand_", "object_", "ho_", "depth_", "ray_", "pose_inject_"))
        ]
        if missing:
            raise RuntimeError(f"Checkpoint is missing required image-path weights: {missing[:12]}")
    else:  # pragma: no cover
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
    return model.to(device)


def run_pi3x_inference(
    model,
    scene_inputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
    inference_mode: str,
):
    imgs = scene_inputs["imgs"].to(device)
    if inference_mode == "multimodal":
        return model(
            imgs,
            depths=scene_inputs["depths"].to(device),
            intrinsics=scene_inputs["intrinsics"].to(device),
            poses=scene_inputs["camera_poses"].to(device),
        )
    if inference_mode == "image":
        return model(imgs)
    raise ValueError(f"Unsupported inference_mode: {inference_mode}")


def _subsample_points(points: torch.Tensor, colors: torch.Tensor | None, max_points: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    if points.shape[0] <= max_points:
        return points, colors
    indices = torch.linspace(0, points.shape[0] - 1, max_points).round().to(dtype=torch.long)
    points = points[indices]
    if colors is not None:
        colors = colors[indices]
    return points, colors


def _bbox_stats(points: torch.Tensor) -> dict[str, object]:
    if points.numel() == 0:
        return {"count": 0, "min": None, "max": None}
    return {
        "count": int(points.shape[0]),
        "min": points.min(dim=0).values.detach().cpu().tolist(),
        "max": points.max(dim=0).values.detach().cpu().tolist(),
    }


def export_diagnostic_rerun(
    *,
    output_path: Path,
    imgs: torch.Tensor,
    depths: torch.Tensor,
    gt_camera_poses: torch.Tensor,
    pred_camera_poses: torch.Tensor,
    gt_points: torch.Tensor,
    gt_colors: torch.Tensor | None,
    pred_points: torch.Tensor,
    pred_colors: torch.Tensor | None,
    pred_points_gt_pose: torch.Tensor,
    pred_points_gt_pose_colors: torch.Tensor | None,
) -> None:
    rr = _load_rerun()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pi3x_object_multiview_diag", spawn=False)
    rr.save(str(output_path))
    rr.log("world", rr.ViewCoordinates.RDF)

    num_views = imgs.shape[1]
    for view_idx in range(num_views):
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("view", view_idx)
        elif hasattr(rr, "set_time"):
            rr.set_time("view", sequence=view_idx)
        rr.log(f"frames/{view_idx:02d}/rgb", rr.Image(_tensor_image_to_uint8(imgs[0, view_idx])))
        rr.log(f"frames/{view_idx:02d}/depth", rr.Image(_tensor_depth_to_uint16(depths[0, view_idx])))

        gt_pose = gt_camera_poses[0, view_idx].detach().cpu().numpy()
        pred_pose = pred_camera_poses[0, view_idx].detach().cpu().numpy()
        rr.log(
            f"world/cameras/gt/{view_idx:02d}",
            rr.Transform3D(mat3x3=gt_pose[:3, :3], translation=gt_pose[:3, 3]),
        )
        rr.log(
            f"world/cameras/pred/{view_idx:02d}",
            rr.Transform3D(mat3x3=pred_pose[:3, :3], translation=pred_pose[:3, 3]),
        )

    if gt_points.numel() > 0:
        rr.log(
            "world/gt_points",
            rr.Points3D(
                positions=gt_points.detach().cpu().numpy(),
                colors=None if gt_colors is None else gt_colors.detach().cpu().numpy(),
            ),
        )
    if pred_points.numel() > 0:
        rr.log(
            "world/pred_points",
            rr.Points3D(
                positions=pred_points.detach().cpu().numpy(),
                colors=None if pred_colors is None else pred_colors.detach().cpu().numpy(),
            ),
        )
    if pred_points_gt_pose.numel() > 0:
        rr.log(
            "world/pred_points_gt_pose",
            rr.Points3D(
                positions=pred_points_gt_pose.detach().cpu().numpy(),
                colors=None if pred_points_gt_pose_colors is None else pred_points_gt_pose_colors.detach().cpu().numpy(),
            ),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Pi3X object multiview inference and export rerun artifacts.")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "data" / "model" / "pi3x" / "model.safetensors"), help="Checkpoint directory or model file.")
    parser.add_argument("--subject", default="20200709-subject-01")
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "debug_object_multiview_diag"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-mode", choices=("image", "multimodal"), default="multimodal")
    parser.add_argument("--frame-num", type=int, default=8)
    parser.add_argument("--resolution", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--max-points", type=int, default=50000)
    return parser


def _set_default_asset_env() -> None:
    data_root = REPO_ROOT / "data"
    model_root = data_root / "model"
    dataset_root = data_root / "dataset"
    os.environ.setdefault("PI3_REPO_ROOT", str(REPO_ROOT))
    os.environ.setdefault("PI3_DATA_ROOT", str(data_root))
    os.environ.setdefault("PI3_MODEL_ROOT", str(model_root))
    os.environ.setdefault("PI3_DATASET_ROOT", str(dataset_root))
    os.environ.setdefault("DEXYCB_ROOT", str(dataset_root / "dexycb"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.sample_index != 0:
        raise ValueError("sample_index must be 0 for this diagnostic because the loader currently uses batch_size=1")
    _set_default_asset_env()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = load_batch(
        data_root=os.environ["DEXYCB_ROOT"],
        subject=args.subject,
        mode=args.mode,
        batch_index=args.batch_index,
        resolution=tuple(args.resolution),
        frame_num=args.frame_num,
    )
    scene_inputs = extract_object_multiview_inputs_from_views_batch(batch)
    imgs = scene_inputs["imgs"]
    depths = scene_inputs["depths"]
    intrinsics = scene_inputs["intrinsics"]
    gt_camera_poses = scene_inputs["camera_poses"]

    device = torch.device(args.device)
    model = load_pi3x_image_model(args.checkpoint, device=device)
    with torch.no_grad():
        pred = run_pi3x_inference(
            model,
            scene_inputs,
            device=device,
            inference_mode=args.inference_mode,
        )

    pred_local_points = pred["local_points"].detach().cpu()
    pred_camera_poses = pred["camera_poses"].detach().cpu()
    pred_world_points = pred["points"].detach().cpu()

    gt_points, gt_colors = gt_depth_to_world_points(depths, intrinsics, gt_camera_poses, imgs)
    pred_points, pred_colors = flatten_points_with_colors(pred_world_points, imgs)
    pred_points_gt_pose, pred_points_gt_pose_colors = flatten_points_with_colors(
        align_local_points_with_camera_poses(pred_local_points, gt_camera_poses),
        imgs,
    )

    gt_points, gt_colors = _subsample_points(gt_points, gt_colors, args.max_points)
    pred_points, pred_colors = _subsample_points(pred_points, pred_colors, args.max_points)
    pred_points_gt_pose, pred_points_gt_pose_colors = _subsample_points(
        pred_points_gt_pose,
        pred_points_gt_pose_colors,
        args.max_points,
    )

    rrd_path = output_dir / "sample_000.rrd"
    export_diagnostic_rerun(
        output_path=rrd_path,
        imgs=imgs,
        depths=depths,
        gt_camera_poses=gt_camera_poses,
        pred_camera_poses=pred_camera_poses,
        gt_points=gt_points,
        gt_colors=gt_colors,
        pred_points=pred_points,
        pred_colors=pred_colors,
        pred_points_gt_pose=pred_points_gt_pose,
        pred_points_gt_pose_colors=pred_points_gt_pose_colors,
    )

    views = batch["views"]
    payload = _find_object_multiview_payload(batch)
    meta = {
        "checkpoint": args.checkpoint,
        "subject": args.subject,
        "mode": args.mode,
        "inference_mode": args.inference_mode,
        "batch_index": args.batch_index,
        "sample_index": args.sample_index,
        "track_label": views[0]["label"][args.sample_index],
        "instances": [view["instance"][args.sample_index] for view in views],
        "object_multiview_views": int(imgs.shape[1]),
        "object_multiview_shape": list(imgs.shape),
        "object_id": int(views[0]["object"]["grasped_object_id"][args.sample_index]),
        "rrd_path": str(rrd_path.resolve()),
        "payload_keys": sorted(payload.keys()),
    }
    stats = {
        "gt_points": _bbox_stats(gt_points),
        "pred_points": _bbox_stats(pred_points),
        "pred_points_gt_pose": _bbox_stats(pred_points_gt_pose),
        "pred_camera_translation_norms": torch.linalg.norm(pred_camera_poses[..., :3, 3], dim=-1).tolist(),
        "gt_camera_translation_norms": torch.linalg.norm(gt_camera_poses[..., :3, 3], dim=-1).tolist(),
    }
    _write_json(output_dir / "meta.json", meta)
    _write_json(output_dir / "stats.json", stats)
    print(f"Wrote diagnostic rerun export to {rrd_path}")


if __name__ == "__main__":
    main()
