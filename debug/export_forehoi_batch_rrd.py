from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch.utils.data import DataLoader

from datasets.base.utils import unified_collate_fn
from datasets.forehoi_dataset import ForeHOIDataset
from pi3.models.hamer.config import get_config as get_hamer_config, resolve_mano_path_template
from pi3.models.hamer.mano_layer import build_mano_layer_pair
from pi3.visualization import (
    build_scene_gt_metric,
    convert_scene_gt_to_pred_scale,
    export_pi3x_rerun_sample,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _default_hamer_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    config_file = repo_root / "configs" / "hamer" / "model_config.yaml"
    cache_dir = repo_root / "data" / "model" / "hamer" / "_DATA"
    return config_file, cache_dir


def _build_visualization_mano_layers():
    config_file, cache_dir = _default_hamer_paths()
    if not config_file.exists():
        return None
    hamer_cfg = get_hamer_config(str(config_file), merge=True, cache_dir=str(cache_dir), update_cachedir=True)
    mano_cfg = {key.lower(): value for key, value in dict(hamer_cfg.MANO).items()}
    mano_data_dir = mano_cfg.get("data_dir", None)
    mano_cfg["model_path"] = resolve_mano_path_template(mano_cfg.get("model_path"), mano_data_dir)
    mano_root_value = mano_cfg.get("model_path")
    if not mano_root_value:
        return None
    mano_root = Path(mano_root_value)
    if mano_root.is_file():
        mano_root = mano_root.parent
    if not (mano_root / "MANO_RIGHT.pkl").exists() or not (mano_root / "MANO_LEFT.pkl").exists():
        return None
    return build_mano_layer_pair(
        mano_root=str(mano_root),
        flat_hand_mean=mano_cfg.get("flat_hand_mean", False),
        ncomps=mano_cfg.get("ncomps", 45),
        use_pca=mano_cfg.get("use_pca", True),
        center_idx=mano_cfg.get("center_idx", None),
        root_rot_mode=mano_cfg.get("root_rot_mode", "axisang"),
        joint_rot_mode=mano_cfg.get("joint_rot_mode", "axisang"),
        robust_rot=mano_cfg.get("robust_rot", False),
    )


def export_forehoi_batch_rrd(
    data_root,
    object_multiview_root,
    output_path,
    mode="train",
    batch_size=1,
    frame_num=4,
    resolution=(224, 224),
    batch_index=0,
    sample_index=0,
    release="pi3x-forehoi-batch-debug",
    dataset_name="forehoi",
    item_id=None,
):
    dataset = ForeHOIDataset(
        data_root=data_root,
        object_multiview_root=object_multiview_root,
        mode=mode,
        resolution=[list(resolution)],
        frame_num=frame_num,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=unified_collate_fn,
    )

    batch = None
    for current_batch_index, candidate in enumerate(loader):
        if current_batch_index == batch_index:
            batch = candidate
            break
    if batch is None:
        raise IndexError(f"batch_index={batch_index} is out of range")
    if not isinstance(batch, dict) or "views" not in batch:
        raise TypeError("ForeHOI loader is expected to return a batch dict containing `views`")

    views_batch = batch["views"]
    actual_batch_size = len(views_batch[0]["dataset"])
    if sample_index >= actual_batch_size:
        raise IndexError(
            f"sample_index={sample_index} is out of range for batch size {actual_batch_size}"
        )

    scene_gt_metric = build_scene_gt_metric(views_batch)
    scene_scale = torch.ones(actual_batch_size, dtype=scene_gt_metric["local_points"].dtype)
    gt = convert_scene_gt_to_pred_scale(scene_gt_metric, scene_scale)
    gt["object_pose_obj2cam"] = torch.stack(
        [view["object"]["pose_obj2cam"] for view in views_batch],
        dim=1,
    )
    gt["object_valid"] = torch.stack(
        [view["object"]["valid"] for view in views_batch],
        dim=1,
    )
    if "hand" in views_batch[0]:
        gt["hand_valid"] = torch.stack([view["hand"]["valid"] for view in views_batch], dim=1)
        gt["hand_mano_betas"] = torch.stack([view["hand"]["mano_betas"] for view in views_batch], dim=1)
        gt["hand_joints_3d_cam"] = torch.stack([view["hand"]["joints_3d_cam"] for view in views_batch], dim=1)
        gt["hand_transl"] = torch.stack([view["hand"]["hand_transl"] for view in views_batch], dim=1)
        gt["hand_global_orient_rotmat"] = torch.stack(
            [view["hand"]["global_orient_rotmat_gt"] for view in views_batch],
            dim=1,
        )
        gt["hand_pose_rotmat"] = torch.stack(
            [view["hand"]["pose_rotmat_gt"] for view in views_batch],
            dim=1,
        )
        gt["hand_sides"] = views_batch[0]["hand"]["mano_side"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mano_layers = _build_visualization_mano_layers()
    track_label = views_batch[0]["label"][sample_index]
    item_name = item_id or f"{track_label}/batch{batch_index:04d}/sample{sample_index:02d}"

    export_pi3x_rerun_sample(
        output_path=output_path,
        batch=views_batch,
        pred=None,
        gt=gt,
        sample_index=sample_index,
        data_root=data_root,
        mano_layer=mano_layers,
        item_name=item_name,
    )

    instances = [view["instance"][sample_index] for view in views_batch]
    metadata = {
        "data_root": str(Path(data_root).resolve()),
        "object_multiview_root": str(Path(object_multiview_root).resolve()),
        "mode": mode,
        "batch_index": batch_index,
        "sample_index": sample_index,
        "batch_size": batch_size,
        "frame_num": frame_num,
        "resolution": list(resolution),
        "track_label": track_label,
        "instances": instances,
    }
    meta_path = _write_json(output_path.parent / "meta.json", metadata)
    manifest = {
        "schema_version": "1.0",
        "release": release,
        "dataset": dataset_name,
        "item_id": item_name,
        "metadata": metadata,
        "artifacts": [
            {
                "role": "interactive_rrd",
                "local_path": str(output_path.resolve()),
                "remote_name": output_path.name,
                "visibility": "private",
                "content_type": "application/octet-stream",
            },
            {
                "role": "metadata_json",
                "local_path": str(meta_path.resolve()),
                "remote_name": "meta.json",
                "visibility": "private",
                "content_type": "application/json",
            },
        ],
    }
    manifest_path = _write_json(output_path.parent / "manifest.json", manifest)
    return output_path, meta_path, manifest_path


def _build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser("export-forehoi-batch-rrd")
    export_parser.add_argument("--data-root", required=True)
    export_parser.add_argument("--object-multiview-root", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--mode", default="train")
    export_parser.add_argument("--batch-size", type=int, default=1)
    export_parser.add_argument("--frame-num", type=int, default=4)
    export_parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224])
    export_parser.add_argument("--batch-index", type=int, default=0)
    export_parser.add_argument("--sample-index", type=int, default=0)
    export_parser.add_argument("--release", default="pi3x-forehoi-batch-debug")
    export_parser.add_argument("--dataset-name", default="forehoi")
    export_parser.add_argument("--item-id", default=None)
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "export-forehoi-batch-rrd":
        output_path, meta_path, manifest_path = export_forehoi_batch_rrd(
            data_root=args.data_root,
            object_multiview_root=args.object_multiview_root,
            output_path=args.output,
            mode=args.mode,
            batch_size=args.batch_size,
            frame_num=args.frame_num,
            resolution=tuple(args.resolution),
            batch_index=args.batch_index,
            sample_index=args.sample_index,
            release=args.release,
            dataset_name=args.dataset_name,
            item_id=args.item_id,
        )
        print(f"Saved {output_path}")
        print(f"Saved {meta_path}")
        print(f"Saved {manifest_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
