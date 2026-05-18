from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dexycb_dataset import DexYCBDataset
from pi3.models.hand_object_loss import estimate_scene_scale_from_depth
from pi3.models.pi3x import Pi3X
from pi3.utils.projection import compute_2d_point_errors, project_points_cam_to_image
from pi3.visualization.pi3x_rerun_export import _load_object_mesh_template, _transform_vertices


HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--subject", default="all")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--frame-num", type=int, default=4)
    parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _tensor_image_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu().float().clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def _draw_points(base: Image.Image, points_uv: np.ndarray, color: tuple[int, int, int], radius: int = 2) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    for u, v in np.asarray(points_uv, dtype=np.float32):
        if not np.isfinite([u, v]).all():
            continue
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color)
    return image


def _draw_hand(base: Image.Image, joints_uv: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    image = _draw_points(base, joints_uv, color=color, radius=3)
    draw = ImageDraw.Draw(image)
    joints_uv = np.asarray(joints_uv, dtype=np.float32)
    for i, j in HAND_BONES:
        if i >= joints_uv.shape[0] or j >= joints_uv.shape[0]:
            continue
        if not np.isfinite(joints_uv[[i, j]]).all():
            continue
        draw.line((float(joints_uv[i, 0]), float(joints_uv[i, 1]), float(joints_uv[j, 0]), float(joints_uv[j, 1])), fill=color, width=2)
    return image


def _save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _select_frame(views: list[dict], frame_index: int | None) -> int:
    if frame_index is not None:
        if not 0 <= frame_index < len(views):
            raise IndexError(f"frame_index={frame_index} out of range for {len(views)} views")
        return frame_index
    for idx, view in enumerate(views):
        if bool(view["hand"]["valid"]) or bool(view["object"]["valid"]):
            return idx
    return 0


def _build_model_inputs(views: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    imgs = torch.stack([view["img"] for view in views], dim=0).unsqueeze(0).to(device)
    depths = torch.stack([torch.as_tensor(view["depthmap"], dtype=torch.float32) for view in views], dim=0).unsqueeze(0).to(device)
    intrinsics = torch.stack([torch.as_tensor(view["camera_intrinsics"], dtype=torch.float32) for view in views], dim=0).unsqueeze(0).to(device)
    poses = torch.stack([torch.as_tensor(view["camera_pose"], dtype=torch.float32) for view in views], dim=0).unsqueeze(0).to(device)
    return {"imgs": imgs, "depths": depths, "intrinsics": intrinsics, "poses": poses}


def _estimate_pred_scale(views: list[dict], ckpt: str | None, device: torch.device) -> tuple[float, str]:
    if ckpt is None:
        return 1.0, "gt_fallback_no_ckpt"

    model = Pi3X(use_multimodal=True, ckpt=ckpt).to(device).eval()
    model_inputs = _build_model_inputs(views, device)
    with torch.no_grad():
        pred = model(
            imgs=model_inputs["imgs"],
            depths=model_inputs["depths"],
            intrinsics=model_inputs["intrinsics"],
            poses=model_inputs["poses"],
        )

    focus_masks = torch.stack(
        [
            torch.as_tensor(view["hand"]["mask"], dtype=torch.bool) |
            torch.as_tensor(view["object"]["mask"], dtype=torch.bool)
            for view in views
        ],
        dim=0,
    ).unsqueeze(0).to(device)
    scene_scale = estimate_scene_scale_from_depth(
        pred["local_points"][..., 2].detach(),
        model_inputs["depths"],
        valid_mask=model_inputs["depths"] > 0,
        focus_mask=focus_masks,
    )
    return float(scene_scale[0].item()), "gt_fallback_model_has_no_pred_intrinsics"


def generate_projection_consistency_report(
    *,
    data_root: str,
    output_dir: str,
    subject: str = "all",
    mode: str = "train",
    sample_index: int = 0,
    frame_index: int | None = None,
    frame_num: int = 4,
    resolution: tuple[int, int] = (224, 224),
    ckpt: str | None = None,
    device: str = "cpu",
) -> dict:
    dataset = DexYCBDataset(
        data_root=data_root,
        mode=mode,
        subject=subject,
        resolution=[list(resolution)],
        frame_num=frame_num,
        shuffle=False,
        use_crop=True,
    )
    views = dataset[sample_index]
    frame_slot = _select_frame(views, frame_index)
    view = views[frame_slot]
    frame_ids = list(getattr(dataset, "this_views_info", {}).get("frame_indices", list(range(len(views)))))
    actual_frame_idx = int(frame_ids[frame_slot]) if frame_slot < len(frame_ids) else frame_slot

    image = _tensor_image_to_pil(view["img"])
    processed_intrinsics = np.asarray(view["camera_intrinsics"], dtype=np.float32)

    track = dataset.tracks[sample_index]
    joint_2d_processed = np.asarray(view["hand"]["joints_2d"], dtype=np.float32)
    joints_3d_cam = np.asarray(view["hand"]["joints_3d_cam"], dtype=np.float32)
    hand_proj_uv, _ = project_points_cam_to_image(joints_3d_cam, processed_intrinsics)
    hand_reproj_error_per_joint, hand_eval_mask = compute_2d_point_errors(joint_2d_processed, hand_proj_uv)
    hand_reproj_error = hand_reproj_error_per_joint[hand_eval_mask] if hand_eval_mask.any() else np.zeros((0,), dtype=np.float32)

    hand_gt_image = _draw_hand(image, joint_2d_processed, (255, 64, 64))
    hand_reproj_image = _draw_hand(image, hand_proj_uv, (64, 255, 64))

    object_vertices_image = image.copy()
    object_predscale_image = image.copy()
    object_gt_points = np.zeros((0, 2), dtype=np.float32)
    object_predscale_points = np.zeros((0, 2), dtype=np.float32)
    object_valid = bool(view["object"]["valid"])

    if object_valid:
        object_id = int(view["object"]["grasped_object_id"])
        template_vertices, _ = _load_object_mesh_template(str(Path(data_root).resolve()), object_id)
        object_pose = torch.as_tensor(view["object"]["pose_obj2cam"], dtype=torch.float32)
        object_vertices_cam = _transform_vertices(template_vertices.float(), object_pose).detach().cpu().numpy()
        object_gt_points = np.asarray(view["object"]["vertices_2d"], dtype=np.float32)
        object_gt_points_reprojected, _ = project_points_cam_to_image(object_vertices_cam, processed_intrinsics)
        object_vertices_image = _draw_points(image, object_gt_points[:: max(1, len(object_gt_points) // 2000 or 1)], (64, 128, 255), radius=1)
    else:
        object_vertices_cam = np.zeros((0, 3), dtype=np.float32)
        object_gt_points_reprojected = np.zeros((0, 2), dtype=np.float32)

    scene_scale, intrinsics_source = _estimate_pred_scale(views, ckpt, torch.device(device))
    pred_intrinsics = processed_intrinsics.copy()
    pred_hand_joints_cam = joints_3d_cam * scene_scale
    pred_hand_proj, _ = project_points_cam_to_image(pred_hand_joints_cam, pred_intrinsics)
    object_predscale_cam = object_vertices_cam * scene_scale
    if object_predscale_cam.size > 0:
        object_predscale_points, _ = project_points_cam_to_image(object_predscale_cam, pred_intrinsics)
    object_predscale_image = _draw_hand(object_predscale_image, pred_hand_proj, (255, 200, 32))
    if object_predscale_points.size > 0:
        object_predscale_image = _draw_points(
            object_predscale_image,
            object_predscale_points[:: max(1, len(object_predscale_points) // 2000 or 1)],
            (80, 160, 255),
            radius=1,
        )

    object_error_full = np.zeros((0,), dtype=np.float32)
    object_error_sample = []
    object_vertex_count_compared = 0
    if object_gt_points.size > 0 and object_gt_points_reprojected.size > 0:
        object_error_full, object_eval_mask = compute_2d_point_errors(object_gt_points, object_gt_points_reprojected)
        object_vertex_count_compared = int(object_eval_mask.sum())
        valid_errors = object_error_full[object_eval_mask]
        if valid_errors.size > 0:
            sample_step = max(1, valid_errors.size // 128)
            object_error_sample = [float(x) for x in valid_errors[::sample_step].tolist()]
        else:
            valid_errors = np.zeros((0,), dtype=np.float32)
    else:
        valid_errors = np.zeros((0,), dtype=np.float32)

    output_root = Path(output_dir)
    _save_image(output_root / "01_gt_hand_joint2d_overlay.png", hand_gt_image)
    _save_image(output_root / "02_gt_hand_reprojected_overlay.png", hand_reproj_image)
    _save_image(output_root / "03_gt_object_vertices_overlay.png", object_vertices_image)
    _save_image(output_root / "04_predscale_hand_object_overlay.png", object_predscale_image)

    metrics = {
        "sample_index": int(sample_index),
        "frame_slot": int(frame_slot),
        "frame_index": int(actual_frame_idx),
        "track": f'{track["subject"]}/{track["sequence"]}/{track["camera"]}',
        "resolution": list(resolution),
        "hand_valid": bool(view["hand"]["valid"]),
        "object_valid": object_valid,
        "hand_joint_count_total": int(len(joint_2d_processed)),
        "hand_joint_count_compared": int(hand_eval_mask.sum()),
        "hand_reprojection_mean_px": float(hand_reproj_error.mean()) if hand_reproj_error.size > 0 else None,
        "hand_reprojection_median_px": float(np.median(hand_reproj_error)) if hand_reproj_error.size > 0 else None,
        "hand_reprojection_max_px": float(hand_reproj_error.max()) if hand_reproj_error.size > 0 else None,
        "hand_reprojection_per_joint_px": [
            None if not np.isfinite(value) else float(value)
            for value in hand_reproj_error_per_joint.tolist()
        ],
        "object_vertex_count_total": int(len(object_gt_points)),
        "object_vertex_count_compared": int(object_vertex_count_compared),
        "object_reprojection_mean_px": float(valid_errors.mean()) if valid_errors.size > 0 else None,
        "object_reprojection_median_px": float(np.median(valid_errors)) if valid_errors.size > 0 else None,
        "object_reprojection_max_px": float(valid_errors.max()) if valid_errors.size > 0 else None,
        "object_reprojection_sample_px": object_error_sample,
        "scene_scale_pred_over_metric": float(scene_scale),
        "pred_intrinsics_source": intrinsics_source,
        "note": "Pi3X currently does not expose predicted camera intrinsics; script falls back to processed GT intrinsics for image 04.",
        "outputs": {
            "gt_hand_joint2d": "01_gt_hand_joint2d_overlay.png",
            "gt_hand_reprojected": "02_gt_hand_reprojected_overlay.png",
            "gt_object_vertices": "03_gt_object_vertices_overlay.png",
            "predscale_hand_object": "04_predscale_hand_object_overlay.png",
        },
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = generate_projection_consistency_report(
        data_root=args.data_root,
        output_dir=args.output_dir,
        subject=args.subject,
        mode=args.mode,
        sample_index=args.sample_index,
        frame_index=args.frame_index,
        frame_num=args.frame_num,
        resolution=tuple(args.resolution),
        ckpt=args.ckpt,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
