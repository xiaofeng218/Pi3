"""Inspect loss components at iter 0."""
from __future__ import annotations

import itertools, math, sys
from pathlib import Path
from unittest import mock

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.pi3x_trainer import Pi3XTrainer
from utils.misc import move_to_device


def main():
    config_dir = str(REPO_ROOT / "configs")
    overrides = [
        "train.auto_resume=false", "train.resume=null", "vis.enabled=false",
        "train.num_workers=0", "test.num_workers=0",
        "log.use_tensorboard=false", "log.use_wandb=false",
        "train.dynamic_compile=false", "train.dynamo_backend=NO",
    ]
    with initialize_config_dir(config_dir=config_dir, job_name="loss_inspect", version_base=None):
        cfg = compose(config_name="overfit", overrides=overrides)
    with open_dict(cfg):
        cfg.extras.print_config = False
        cfg.work_dir = str(REPO_ROOT)
        cfg.working_dir = str(REPO_ROOT)
    Path(cfg.log.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log.ckpt_dir).mkdir(parents=True, exist_ok=True)
    job_logging_cfg = OmegaConf.load(REPO_ROOT / "configs" / "hydra" / "job_logging" / "custom.yaml")
    with open_dict(job_logging_cfg):
        job_logging_cfg.handlers.file.filename = str(Path(cfg.log.output_dir) / "log.log")
    hydra_cfg = type("HydraCfg", (), {"job_logging": job_logging_cfg})()

    with mock.patch("trainers.base_trainer_accelerate.HydraConfig.get", return_value=hydra_cfg), mock.patch(
        "trainers.base_trainer_accelerate.pretty_print_hydra_config", lambda *_a, **_kw: None,
    ):
        trainer = Pi3XTrainer(cfg)

    trainer.before_epoch(0)
    trainer.model.eval()
    train_iter = itertools.cycle(trainer.train_loader)

    for step in range(3):
        batch = next(train_iter)
        batch = move_to_device(batch, device=trainer.accelerator.device)
        with trainer.accelerator.autocast():
            fwd = trainer.forward_batch(batch, mode="train")
            bo = trainer.calculate_loss(fwd, batch, mode="train")

        pred, gt = fwd
        total = float(bo.loss.item())
        print(f"\n{'='*60}")
        print(f"Iter {step} | total_loss = {total:.4f}")

        # Hand 2D details
        if "hand_2d_loss" in bo:
            K = gt["hand_camera_intrinsics"]
            img_wh = torch.stack([2.0*K[...,0,2].clamp_min(1.0), 2.0*K[...,1,2].clamp_min(1.0)], dim=-1)
            pred_2d, pred_v = project_points_cam_to_image_torch(pred["pred_hand_joints_3d"], K)
            gt_2d = gt["hand_joints_2d"]
            valid = pred_v & torch.isfinite(gt_2d).all(dim=-1)
            hv = gt.get("hand_valid")
            if hv is not None:
                valid = valid & hv.unsqueeze(-1)
            z_ok = (pred["pred_hand_joints_3d"][..., 2] > 1e-6).sum()
            print(f"  HAND_2D: img_wh~[{img_wh[0,0].item():.0f},{img_wh[0,1].item():.0f}]"
                  f" z>0={z_ok.item()}/{pred['pred_hand_joints_3d'].numel()}"
                  f" valid={valid.sum().item()}/{valid.numel()}"
                  f" loss={bo['hand_2d_loss'].item():.6f}")

        # All loss components
        print(f"  {'loss_component':35s}  value")
        print(f"  {'-'*50}")
        for key in sorted(bo.keys()):
            if key == "loss": continue
            v = bo[key].item() if torch.is_tensor(bo[key]) and bo[key].numel()==1 else ""
            print(f"  {key:35s}  {v}")

        if not torch.isfinite(torch.tensor(total)):
            print("  !! NaN/Inf detected!")
            break


if __name__ == "__main__":
    main()
