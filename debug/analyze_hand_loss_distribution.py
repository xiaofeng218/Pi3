from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi3.models.hand_object_loss import _canonicalize_hand_geometry
from trainers.checkpoint_utils import load_trainable_checkpoint
from trainers.pi3x_trainer import Pi3XTrainer
from utils.misc import move_to_device


HAND_JOINT_NAMES = [
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]

HAND_POSE_SLOT_NAMES = [
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
]


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


def _build_trainer(subject: str, output_dir: Path, batch_size: int) -> Pi3XTrainer:
    config_dir = str(REPO_ROOT / "configs")
    overrides = [
        "name=analyze_hand_loss_distribution",
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
        f"test.batch_size={batch_size}",
        "train.model_dtype=fp32",
        f"train_dataset.subject={subject}",
        f"test_dataset.subject={subject}",
    ]
    with initialize_config_dir(config_dir=config_dir, job_name="analyze_hand_loss_distribution", version_base=None):
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


def _init_stats(device: torch.device) -> dict[str, dict[str, torch.Tensor | int]]:
    return {
        "all": {
            "joint_sum": torch.zeros(len(HAND_JOINT_NAMES), dtype=torch.float64, device=device),
            "pose_sum": torch.zeros(len(HAND_POSE_SLOT_NAMES), dtype=torch.float64, device=device),
            "global_orient_sum": torch.zeros((), dtype=torch.float64, device=device),
            "count": 0,
        },
        "right": {
            "joint_sum": torch.zeros(len(HAND_JOINT_NAMES), dtype=torch.float64, device=device),
            "pose_sum": torch.zeros(len(HAND_POSE_SLOT_NAMES), dtype=torch.float64, device=device),
            "global_orient_sum": torch.zeros((), dtype=torch.float64, device=device),
            "count": 0,
        },
        "left": {
            "joint_sum": torch.zeros(len(HAND_JOINT_NAMES), dtype=torch.float64, device=device),
            "pose_sum": torch.zeros(len(HAND_POSE_SLOT_NAMES), dtype=torch.float64, device=device),
            "global_orient_sum": torch.zeros((), dtype=torch.float64, device=device),
            "count": 0,
        },
    }


def _accumulate(
    stats: dict[str, dict[str, torch.Tensor | int]],
    side: str,
    joint_error: torch.Tensor,
    pose_error: torch.Tensor,
    global_orient_error: torch.Tensor,
) -> None:
    bucket = stats[side]
    bucket["joint_sum"] = bucket["joint_sum"] + joint_error.to(torch.float64).sum(dim=0)
    bucket["pose_sum"] = bucket["pose_sum"] + pose_error.to(torch.float64).sum(dim=0)
    bucket["global_orient_sum"] = bucket["global_orient_sum"] + global_orient_error.to(torch.float64).sum()
    bucket["count"] += int(joint_error.shape[0])


def _format_ranked_table(names: list[str], values: list[float]) -> list[str]:
    ranked = sorted(zip(names, values), key=lambda item: item[1], reverse=True)
    return [f"{name:14s} {value:.6f}" for name, value in ranked]


def _build_summary(stats: dict[str, dict[str, torch.Tensor | int]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for side_name, bucket in stats.items():
        count = int(bucket["count"])
        if count == 0:
            summary[side_name] = {
                "count": 0,
                "hand_joints_3d_loss_mean": None,
                "hand_pose_loss_mean": None,
                "hand_global_orient_loss_mean": None,
                "per_joint": {},
                "per_pose_slot": {},
            }
            continue

        joint_mean = (bucket["joint_sum"] / count).detach().cpu().tolist()
        pose_mean = (bucket["pose_sum"] / count).detach().cpu().tolist()
        global_orient_mean = float((bucket["global_orient_sum"] / count).detach().cpu().item())
        summary[side_name] = {
            "count": count,
            "hand_joints_3d_loss_mean": float(sum(joint_mean) / len(joint_mean)),
            "hand_pose_loss_mean": float(sum(pose_mean) / len(pose_mean)),
            "hand_global_orient_loss_mean": global_orient_mean,
            "per_joint": {name: float(value) for name, value in zip(HAND_JOINT_NAMES, joint_mean)},
            "per_pose_slot": {name: float(value) for name, value in zip(HAND_POSE_SLOT_NAMES, pose_mean)},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hand loss distribution from a saved Pi3X checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs" / "full1.2" / "ckpts" / "checkpoint-epoch-0020"),
        help="Checkpoint directory containing trainable_model.pt.",
    )
    parser.add_argument(
        "--subject",
        default="20200709-subject-01",
        help="DexYCB subject to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "debug_hand_loss_distribution"),
        help="Temporary output directory for Hydra/trainer logs.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional JSON path for the aggregated summary.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional cap on evaluated test batches. Use 0 for full test split.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N batches. Use 0 to disable progress logs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Test batch size for evaluation.",
    )
    args = parser.parse_args()

    _set_default_asset_env()

    checkpoint_dir = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    trainer = _build_trainer(subject=args.subject, output_dir=output_dir, batch_size=args.batch_size)
    state = load_trainable_checkpoint(
        checkpoint_dir,
        trainer.model,
        optimizer=None,
        scheduler=None,
        map_location=trainer.accelerator.device,
    )
    trainer.log_info(
        f"Loaded trainable checkpoint from {checkpoint_dir} "
        f"(missing={len(state['missing_keys'])}, unexpected={len(state['unexpected_keys'])})"
    )
    trainer.model.eval()
    trainer.before_epoch(0)

    stats = _init_stats(trainer.accelerator.device)
    num_batches = 0

    with torch.no_grad():
        for batch in trainer.test_loader:
            num_batches += 1
            if args.max_batches > 0 and num_batches > args.max_batches:
                break

            batch = move_to_device(batch, device=trainer.accelerator.device)
            pred, gt = trainer.forward_batch(batch, mode="test")

            hand_valid = gt["hand_valid"].bool()
            if not hand_valid.any():
                continue

            pred_joints = _canonicalize_hand_geometry(
                pred["pred_hand_joints_3d"],
                pred["pred_hand_mano_params"]["global_orient"],
                root_index=trainer.test_loss.root_index,
            )
            gt_joints = _canonicalize_hand_geometry(
                gt["hand_joints_3d"],
                gt["hand_global_orient_rotmat"],
                root_index=trainer.test_loss.root_index,
            )
            joint_error = F.l1_loss(pred_joints, gt_joints, reduction="none").mean(dim=-1) * 10.0
            pose_error = F.mse_loss(
                pred["pred_hand_mano_params"]["hand_pose"],
                gt["hand_pose_rotmat"],
                reduction="none",
            ).mean(dim=(-1, -2))
            global_orient_error = F.mse_loss(
                pred["pred_hand_mano_params"]["global_orient"],
                gt["hand_global_orient_rotmat"],
                reduction="none",
            ).mean(dim=(-1, -2, -3))

            valid_joint_error = joint_error[hand_valid]
            valid_pose_error = pose_error[hand_valid]
            valid_global_orient_error = global_orient_error[hand_valid]
            valid_is_right = gt["hand_is_right"][hand_valid].bool()

            _accumulate(stats, "all", valid_joint_error, valid_pose_error, valid_global_orient_error)
            if valid_is_right.any():
                _accumulate(
                    stats,
                    "right",
                    valid_joint_error[valid_is_right],
                    valid_pose_error[valid_is_right],
                    valid_global_orient_error[valid_is_right],
                )
            if (~valid_is_right).any():
                _accumulate(
                    stats,
                    "left",
                    valid_joint_error[~valid_is_right],
                    valid_pose_error[~valid_is_right],
                    valid_global_orient_error[~valid_is_right],
                )

            if args.progress_every > 0 and num_batches % args.progress_every == 0:
                print(f"progress: evaluated {num_batches} batches", flush=True)

    summary = _build_summary(stats)
    print(f"checkpoint: {checkpoint_dir}")
    print(f"subject: {args.subject}")
    print(f"num_batches: {num_batches}")
    print()

    for side_name in ("all", "right", "left"):
        side_summary = summary[side_name]
        print(f"[{side_name}] count={side_summary['count']}")
        if side_summary["count"] == 0:
            print("  no valid hand samples")
            print()
            continue
        print(f"  hand_joints_3d_loss_mean = {side_summary['hand_joints_3d_loss_mean']:.6f}")
        print(f"  hand_pose_loss_mean      = {side_summary['hand_pose_loss_mean']:.6f}")
        print(f"  hand_global_orient_loss  = {side_summary['hand_global_orient_loss_mean']:.6f}")
        print("  per_joint:")
        for line in _format_ranked_table(
            HAND_JOINT_NAMES,
            [side_summary["per_joint"][name] for name in HAND_JOINT_NAMES],
        ):
            print(f"    {line}")
        print("  per_pose_slot:")
        for line in _format_ranked_table(
            HAND_POSE_SLOT_NAMES,
            [side_summary["per_pose_slot"][name] for name in HAND_POSE_SLOT_NAMES],
        ):
            print(f"    {line}")
        print()

    if args.json_out:
        json_path = Path(args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"json_out: {json_path}")


if __name__ == "__main__":
    main()
