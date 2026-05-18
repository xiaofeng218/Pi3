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

from debug.pi3x_forward_finite_check import find_first_non_finite, summarize_batch
from trainers.pi3x_trainer import Pi3XTrainer
from utils.misc import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real Pi3X train steps and report the first NaN/Inf source.")
    parser.add_argument("--config-name", default="pi3x_hand_object")
    parser.add_argument("--max-iters", type=int, default=12)
    parser.add_argument("--override", action="append", default=[])
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
    with initialize_config_dir(config_dir=config_dir, job_name="pi3x_train_finite_check", version_base=None):
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


def first_non_finite_named_tensor(named_tensors):
    for name, tensor in named_tensors:
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
            continue
        report = find_first_non_finite(tensor, name)
        if report is not None:
            return report
    return None


def find_unmasked_non_finite_gt(gt: dict[str, Any]):
    gt_for_check = dict(gt)
    # These fields intentionally contain NaN for invalid projected points.
    # The corresponding *_valid masks decide which entries participate in loss.
    gt_for_check.pop("object_vertices_2d", None)
    gt_for_check.pop("hand_joints_2d", None)
    return find_first_non_finite(gt_for_check, "gt.before_backward")


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
    trainer.model.train()
    train_iter = itertools.cycle(trainer.train_loader)

    print(f"device={trainer.accelerator.device}, mixed_precision={trainer.accelerator.mixed_precision}")
    print(f"max_iters={args.max_iters}, clip_grad={cfg.train.clip_grad}, lr={cfg.train.optimizer.lr}")

    for step in range(args.max_iters):
        batch = next(train_iter)
        batch_summary = summarize_batch(batch)
        batch = move_to_device(batch, device=trainer.accelerator.device)
        trainer.optimizer.zero_grad(set_to_none=True)

        with trainer.accelerator.autocast():
            forward_output = trainer.forward_batch(batch, mode="train")
            batch_output = trainer.calculate_loss(forward_output, batch, mode="train")
            loss = batch_output.loss

        pred, gt = forward_output
        reports = [
            find_first_non_finite(pred, "pred.before_backward"),
            find_unmasked_non_finite_gt(gt),
            find_first_non_finite(dict(batch_output), "loss_details.before_backward"),
        ]
        report = next((item for item in reports if item is not None), None)
        loss_value = float(loss.detach().item())
        print(f"[iter {step}] loss={loss_value:.6f} samples={batch_summary}")
        if report is not None or not math.isfinite(loss_value):
            print("Non-finite before backward.")
            print(f"report={report}")
            return 2

        trainer.accelerator.backward(loss)
        grad_report = first_non_finite_named_tensor((f"grad.{name}", param.grad) for name, param in trainer.model.named_parameters())
        if grad_report is not None:
            print("Non-finite gradient after backward.")
            print(f"report={grad_report}")
            return 3

        if trainer.accelerator.sync_gradients:
            trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), cfg.train.clip_grad)
        clipped_grad_report = first_non_finite_named_tensor((f"grad_after_clip.{name}", param.grad) for name, param in trainer.model.named_parameters())
        if clipped_grad_report is not None:
            print("Non-finite gradient after clipping.")
            print(f"report={clipped_grad_report}")
            return 4

        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        param_report = first_non_finite_named_tensor(trainer.model.named_parameters())
        if param_report is not None:
            print("Non-finite parameter after optimizer step.")
            print(f"report={param_report}")
            return 5

    print("All checked train steps stayed finite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
