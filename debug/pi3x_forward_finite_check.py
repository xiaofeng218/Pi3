from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.pi3x_trainer import Pi3XTrainer
from utils.misc import move_to_device


def find_first_non_finite(obj: Any, path: str = "root") -> dict[str, Any] | None:
    if torch.is_tensor(obj):
        if not torch.is_floating_point(obj) and not torch.is_complex(obj):
            return None
        finite_mask = torch.isfinite(obj)
        if bool(finite_mask.all()):
            return None
        bad_count = int((~finite_mask).sum().item())
        finite_vals = obj[finite_mask]
        finite_min = float(finite_vals.min().item()) if finite_vals.numel() > 0 else None
        finite_max = float(finite_vals.max().item()) if finite_vals.numel() > 0 else None
        return {
            "path": path,
            "shape": tuple(obj.shape),
            "dtype": str(obj.dtype),
            "bad_count": bad_count,
            "finite_min": finite_min,
            "finite_max": finite_max,
        }
    if isinstance(obj, dict):
        for key, value in obj.items():
            report = find_first_non_finite(value, f"{path}.{key}")
            if report is not None:
                return report
        return None
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            report = find_first_non_finite(value, f"{path}[{idx}]")
            if report is not None:
                return report
        return None
    return None


def summarize_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(batch, list) or not batch:
        return []
    first_view = batch[0]
    labels = first_view.get("label", [])
    if not isinstance(labels, list):
        return []
    batch_size = len(labels)
    summary: list[dict[str, Any]] = []
    for sample_idx in range(batch_size):
        instances = []
        for view in batch:
            instance_list = view.get("instance", [])
            if isinstance(instance_list, list) and sample_idx < len(instance_list):
                instances.append(str(instance_list[sample_idx]))
        summary.append(
            {
                "sample_index": sample_idx,
                "label": str(labels[sample_idx]),
                "instances": instances,
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated Pi3X train-set forward passes and check for NaN/Inf.")
    parser.add_argument("--config-name", default="pi3x_hand_object")
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. May be passed multiple times.",
    )
    return parser.parse_args()


def build_cfg(config_name: str, overrides: list[str]):
    config_dir = str(REPO_ROOT / "configs")
    base_overrides = [
        "train.auto_resume=false",
        "train.resume=null",
        "vis.enabled=false",
        "train.num_workers=0",
        "test.num_workers=0",
        "log.use_tensorboard=false",
        "log.use_wandb=false",
        "train.dynamic_compile=false",
        "train.dynamo_backend=NO",
    ]
    with initialize_config_dir(config_dir=config_dir, job_name="pi3x_forward_finite_check", version_base=None):
        cfg = compose(config_name=config_name, overrides=base_overrides + list(overrides))
    with open_dict(cfg):
        cfg.extras.print_config = False
        cfg.work_dir = str(REPO_ROOT)
        cfg.working_dir = str(REPO_ROOT)
    return cfg


def build_job_logging_cfg(cfg) -> Any:
    job_logging_cfg = OmegaConf.load(REPO_ROOT / "configs" / "hydra" / "job_logging" / "custom.yaml")
    with open_dict(job_logging_cfg):
        job_logging_cfg.handlers.file.filename = str(Path(cfg.log.output_dir) / "log.log")
    return job_logging_cfg


def main() -> int:
    args = parse_args()
    cfg = build_cfg(args.config_name, args.override)
    Path(cfg.log.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log.ckpt_dir).mkdir(parents=True, exist_ok=True)
    job_logging_cfg = build_job_logging_cfg(cfg)
    hydra_cfg = type("HydraCfg", (), {"job_logging": job_logging_cfg})()
    with mock.patch("trainers.base_trainer_accelerate.HydraConfig.get", return_value=hydra_cfg), mock.patch(
        "trainers.base_trainer_accelerate.pretty_print_hydra_config",
        lambda *_args, **_kwargs: None,
    ):
        trainer = Pi3XTrainer(cfg)
    trainer.before_epoch(0)
    trainer.model.eval()

    train_iter = itertools.cycle(trainer.train_loader)
    last_finite_loss = None

    with torch.no_grad():
        for step in range(args.max_iters):
            batch = next(train_iter)
            batch_summary = summarize_batch(batch)
            batch = move_to_device(batch, device=trainer.accelerator.device)
            with trainer.accelerator.autocast():
                forward_output = trainer.forward_batch(batch, mode="train")
                batch_output = trainer.calculate_loss(forward_output, batch, mode="train")

            pred, gt = forward_output
            reports = [
                find_first_non_finite(pred, "pred"),
                find_first_non_finite(gt, "gt"),
                find_first_non_finite(dict(batch_output), "loss_details"),
            ]
            report = next((item for item in reports if item is not None), None)

            loss_value = float(batch_output.loss.detach().item())
            print(f"[iter {step}] loss={loss_value:.6f} batch_samples={len(batch_summary)}")
            if report is not None or not math.isfinite(loss_value):
                print("Non-finite value detected.")
                print(f"batch_summary={batch_summary}")
                if report is not None:
                    print(f"report={report}")
                if "hand_scale_gt" in batch_output:
                    print(f"hand_scale_gt={batch_output['hand_scale_gt']}")
                if "object_scale_gt" in batch_output:
                    print(f"object_scale_gt={batch_output['object_scale_gt']}")
                return 2

            last_finite_loss = loss_value

    print(
        "All checked iterations produced finite forward outputs and finite loss. "
        f"last_loss={last_finite_loss:.6f}"
    )
    print("If training still hits NaN later, the more likely cause is optimizer/update instability rather than pure forward on train data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
