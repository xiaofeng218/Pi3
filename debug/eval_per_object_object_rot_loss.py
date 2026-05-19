from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.dexycb_dataset import _YCB_CLASSES
from pi3.models.hamer.geometry import rot6d_to_rotmat
from pi3.visualization import export_pi3x_rerun_sample
from trainers.pi3x_trainer import Pi3XTrainer
from utils.misc import move_to_device


YCB_SYMMETRIC_OBJECT_IDS = {1, 13, 14, 16, 18, 19, 20, 21}


def _set_default_asset_env() -> None:
    data_root = REPO_ROOT / "data"
    model_root = data_root / "model"
    dataset_root = data_root / "dataset"
    hamer_cache_dir = model_root / "hamer" / "_DATA"

    os.environ.setdefault("PI3_REPO_ROOT", str(REPO_ROOT))
    os.environ.setdefault("PI3_DATA_ROOT", str(data_root))
    os.environ.setdefault("PI3_MODEL_ROOT", str(model_root))
    os.environ.setdefault("PI3_DATASET_ROOT", str(dataset_root))
    os.environ.setdefault("DEXYCB_ROOT", str(dataset_root / "dexycb"))
    os.environ.setdefault("DEX_YCB_DIR", os.environ["DEXYCB_ROOT"])
    os.environ.setdefault("HAMER_CONFIG_FILE", str(REPO_ROOT / "configs" / "hamer" / "model_config.yaml"))
    os.environ.setdefault("HAMER_CACHE_DIR", str(hamer_cache_dir))
    os.environ.setdefault(
        "HAMER_ENCODER_CKPT",
        str(hamer_cache_dir / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"),
    )
    os.environ.setdefault("MANO_ROOT", str(hamer_cache_dir / "data" / "mano"))
    os.environ.setdefault("PI3X_CKPT", str(model_root / "pi3x"))
    os.environ["PYTHONPATH"] = (
        f"{REPO_ROOT}:{REPO_ROOT / 'dex-ycb-toolkit'}"
        + (f":{os.environ['PYTHONPATH']}" if "PYTHONPATH" in os.environ else "")
    )


def _build_trainer(subject: str, output_dir: Path) -> Pi3XTrainer:
    config_dir = str(REPO_ROOT / "configs")
    overrides = [
        "name=eval_per_object_object_rot_loss",
        f"log.output_dir={output_dir}",
        f"log.ckpt_dir={output_dir / 'ckpts'}",
        "train.auto_resume=false",
        "train.resume=null",
        "vis.enabled=false",
        "train.num_workers=0",
        "test.num_workers=0",
        "log.use_tensorboard=false",
        "log.use_wandb=false",
        "train.dynamic_compile=false",
        "train.dynamo_backend=NO",
        "train.batch_size=1",
        "test.batch_size=1",
        "train.model_dtype=fp32",
        f"train_dataset.subject={subject}",
        f"test_dataset.subject={subject}",
    ]
    with initialize_config_dir(config_dir=config_dir, job_name="eval_per_object_object_rot_loss", version_base=None):
        cfg = compose(config_name="pi3x_hand_object", overrides=overrides)

    with open_dict(cfg):
        cfg.extras.print_config = False
        cfg.work_dir = str(REPO_ROOT)
        cfg.working_dir = str(REPO_ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    job_logging_cfg = OmegaConf.load(REPO_ROOT / "configs" / "hydra" / "job_logging" / "custom.yaml")
    with open_dict(job_logging_cfg):
        job_logging_cfg.handlers.file.filename = str(output_dir / "log.log")
    hydra_cfg = type("HydraCfg", (), {"job_logging": job_logging_cfg})()

    with mock.patch("trainers.base_trainer_accelerate.HydraConfig.get", return_value=hydra_cfg), mock.patch(
        "trainers.base_trainer_accelerate.pretty_print_hydra_config",
        lambda *_args, **_kwargs: None,
    ):
        trainer = Pi3XTrainer(cfg)
    return trainer


def _summarize(values: list[float]) -> tuple[float, float, float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0, values[0], values[0]
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = var ** 0.5
    return mean, std, min(values), max(values)


def _normalize_object_multiview_device(batch, device: torch.device):
    if not isinstance(batch, list):
        return batch
    shared_keys = {
        "img",
        "depthmap",
        "camera_intrinsics",
        "camera_pose",
        "template_vertices",
        "normalization_center",
        "normalization_scale",
        "grasped_object_id",
        "grasped_object_mask",
        "grasped_object_valid",
        "grasped_object_pose_obj2cam",
        "grasped_object_vertices_2d",
        "grasped_object_vertices_2d_valid",
    }
    for view in batch:
        if not isinstance(view, dict):
            continue
        object_multiview = view.get("object_multiview", None)
        if not isinstance(object_multiview, dict):
            continue
        for key in shared_keys:
            if key not in object_multiview:
                continue
            value = object_multiview[key]
            if torch.is_tensor(value):
                object_multiview[key] = value.to(device=device)
            elif isinstance(value, (list, tuple)):
                try:
                    object_multiview[key] = torch.as_tensor(value, device=device)
                except Exception:
                    pass
            elif value is not None and hasattr(value, "shape"):
                try:
                    object_multiview[key] = torch.as_tensor(value, device=device)
                except Exception:
                    pass
    return batch


def _maybe_wrap_hydrate_object_payload_to_device(trainer: Pi3XTrainer) -> None:
    original = getattr(trainer, "_hydrate_object_multiview_payload", None)
    if original is None:
        return

    def _wrapped(self, object_multiview, batch_size):
        payload = original(object_multiview, batch_size)
        for key, value in payload.items():
            if torch.is_tensor(value):
                payload[key] = value.to(device=self.accelerator.device)
        return payload

    trainer._hydrate_object_multiview_payload = _wrapped.__get__(trainer, type(trainer))


def _extract_object_ids(batch, device: torch.device) -> torch.Tensor:
    if not isinstance(batch, dict):
        raise TypeError(f"Expected dict batch with `views`, got {type(batch).__name__}")
    views = batch.get("views", None)
    if not isinstance(views, list) or not views:
        raise KeyError("Batch is missing non-empty `views`")
    first_view = views[0]
    if not isinstance(first_view, dict):
        raise TypeError("Batch `views[0]` must be a dict")
    object_payload = first_view.get("object", None)
    if not isinstance(object_payload, dict) or "grasped_object_id" not in object_payload:
        raise KeyError("Batch `views[0].object.grasped_object_id` is required")
    return torch.as_tensor(object_payload["grasped_object_id"], device=device, dtype=torch.long).reshape(-1)


def _extract_views_batch_for_rerun(batch):
    if isinstance(batch, dict):
        views = batch.get("views", None)
        if not isinstance(views, list) or not views:
            raise KeyError("Rerun export requires batch['views'] to be a non-empty list")
        return views
    return batch


def _safe_class_name(object_id: int) -> str:
    return _YCB_CLASSES.get(object_id, f"object_{object_id:02d}").replace("/", "_")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-object object_rot_loss from a saved Pi3X checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs" / "full1.1" / "ckpts" / "checkpoint-epoch-0028"),
        help="Checkpoint directory containing trainable_model.pt/optimizer.pt/scheduler.pt.",
    )
    parser.add_argument(
        "--subject",
        default="20200709-subject-01",
        help="DexYCB subject to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "debug_eval_per_object_rot"),
        help="Temporary output directory for Hydra/trainer logs.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional cap on evaluated batches. Use 0 to evaluate the full test split.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N batches. Use 0 to disable periodic progress logs.",
    )
    parser.add_argument(
        "--export-rerun",
        action="store_true",
        help="Export one representative .rrd sample for each object class encountered.",
    )
    parser.add_argument(
        "--rerun-dir",
        default="",
        help="Directory for per-object rerun exports. Defaults to <output-dir>/rerun_per_object.",
    )
    args = parser.parse_args()

    _set_default_asset_env()

    checkpoint_dir = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    trainer = _build_trainer(subject=args.subject, output_dir=output_dir)
    _maybe_wrap_hydrate_object_payload_to_device(trainer)
    trainer.load_training_state(str(checkpoint_dir))
    trainer.model.eval()
    trainer.before_epoch(0)

    per_object_track_mean: dict[int, list[float]] = defaultdict(list)
    per_object_view_losses: dict[int, list[float]] = defaultdict(list)
    per_object_valid_views: dict[int, int] = defaultdict(int)
    exported_rerun_records: dict[int, dict[str, object]] = {}
    num_batches = 0
    rerun_dir = Path(args.rerun_dir).resolve() if args.rerun_dir else output_dir / "rerun_per_object"

    with torch.no_grad():
        for batch in trainer.test_loader:
            num_batches += 1
            if args.max_batches > 0 and num_batches > args.max_batches:
                break
            batch = move_to_device(batch, device=trainer.accelerator.device)
            batch = _normalize_object_multiview_device(batch, device=trainer.accelerator.device)
            pred, gt = trainer.forward_batch(batch, mode="test")

            gt_rot = gt["object_pose_obj2cam"][..., :3, :3]
            pred_rot = rot6d_to_rotmat(pred["pred_object_rot6d"].reshape(-1, 6)).reshape(gt_rot.shape)
            object_rot_per_view = F.mse_loss(pred_rot, gt_rot, reduction="none").mean(dim=(-1, -2))
            object_valid = gt["object_valid"]
            object_ids = _extract_object_ids(batch, device=object_valid.device)

            batch_size = object_valid.shape[0]
            for batch_idx in range(batch_size):
                object_id = int(object_ids[batch_idx].item())
                valid_losses = object_rot_per_view[batch_idx][object_valid[batch_idx]]
                if valid_losses.numel() == 0:
                    continue
                valid_losses_list = [float(value) for value in valid_losses.detach().cpu().tolist()]
                per_object_track_mean[object_id].append(sum(valid_losses_list) / len(valid_losses_list))
                per_object_view_losses[object_id].extend(valid_losses_list)
                per_object_valid_views[object_id] += len(valid_losses_list)
                if args.export_rerun and object_id not in exported_rerun_records:
                    class_name = _safe_class_name(object_id)
                    export_dir = rerun_dir / f"{object_id:02d}_{class_name}"
                    output_path = export_dir / "sample_000.rrd"
                    export_pi3x_rerun_sample(
                        output_path=output_path,
                        batch=_extract_views_batch_for_rerun(batch),
                        pred=pred,
                        gt=gt,
                        sample_index=batch_idx,
                        data_root=trainer._resolve_data_root(),
                        mano_layer=trainer._base_model.hand_mano_layer,
                        item_name=f"object_{object_id:02d}_{class_name}",
                    )
                    exported_rerun_records[object_id] = {
                        "object_id": object_id,
                        "class_name": _YCB_CLASSES.get(object_id, "?"),
                        "rrd_path": str(output_path.resolve()),
                        "meta_path": str(output_path.with_suffix(".meta.json").resolve()),
                        "manifest_path": str(output_path.with_name("manifest.json").resolve()),
                        "batch_index": batch_idx,
                        "eval_batch_number": num_batches,
                        "valid_views": int(valid_losses.numel()),
                        "track_mean_loss": float(sum(valid_losses_list) / len(valid_losses_list)),
                    }
            if args.progress_every > 0 and num_batches % args.progress_every == 0:
                print(f"progress: evaluated {num_batches} batches", flush=True)

    print(f"checkpoint: {checkpoint_dir}")
    print(f"subject: {args.subject}")
    print(f"num_batches: {num_batches}")
    print()
    print(
        "object_id type class_name                 tracks valid_views "
        "track_mean track_std view_mean view_std"
    )

    symmetric_view_losses: list[float] = []
    asymmetric_view_losses: list[float] = []
    for object_id in sorted(per_object_view_losses):
        track_values = per_object_track_mean[object_id]
        view_values = per_object_view_losses[object_id]
        track_mean, track_std, _, _ = _summarize(track_values)
        view_mean, view_std, _, _ = _summarize(view_values)
        is_symmetric = object_id in YCB_SYMMETRIC_OBJECT_IDS
        if is_symmetric:
            symmetric_view_losses.extend(view_values)
        else:
            asymmetric_view_losses.extend(view_values)
        print(
            f"{object_id:8d} "
            f"{'SYM' if is_symmetric else 'ASYM':4s} "
            f"{_YCB_CLASSES.get(object_id, '?'):24s} "
            f"{len(track_values):6d} "
            f"{per_object_valid_views[object_id]:11d} "
            f"{track_mean:9.4f} "
            f"{track_std:8.4f} "
            f"{view_mean:8.4f} "
            f"{view_std:8.4f}"
        )

    print()
    if symmetric_view_losses:
        sym_mean, sym_std, _, _ = _summarize(symmetric_view_losses)
        print(f"SYM view_mean={sym_mean:.4f} view_std={sym_std:.4f} count={len(symmetric_view_losses)}")
    if asymmetric_view_losses:
        asym_mean, asym_std, _, _ = _summarize(asymmetric_view_losses)
        print(f"ASYM view_mean={asym_mean:.4f} view_std={asym_std:.4f} count={len(asymmetric_view_losses)}")
    if args.export_rerun:
        manifest = {
            "checkpoint": str(checkpoint_dir),
            "subject": args.subject,
            "num_batches": num_batches,
            "rerun_dir": str(rerun_dir.resolve()),
            "exports": [exported_rerun_records[object_id] for object_id in sorted(exported_rerun_records)],
        }
        _write_json(rerun_dir / "manifest.json", manifest)
        print()
        print(f"exported_rerun_objects={len(exported_rerun_records)} rerun_dir={rerun_dir.resolve()}")


if __name__ == "__main__":
    main()
