from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.base_trainer_accelerate import BaseTrainer


class _DummyAccelerator:
    def __init__(self):
        self.logged = []
        self.trackers = []

    def log(self, values, step):
        self.logged.append((step, dict(values)))


class _DummyWriter:
    def __init__(self):
        self.scalars = []
        self.images = []

    def add_scalar(self, tag, value, global_step):
        self.scalars.append((tag, value, global_step))

    def add_image(self, *args, **kwargs):
        self.images.append((args, kwargs))

    def flush(self):
        return None


class TensorboardLogCleanupTests(unittest.TestCase):
    def test_scalar_logs_are_grouped_and_images_are_dropped(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.accelerator = _DummyAccelerator()
        trainer.tb_writer = _DummyWriter()

        trainer.log_all(
            {
                "loss": 1.0,
                "hand_2d_loss": 2.0,
                "lr": 1e-4,
                "min_lr": 5e-5,
                "weight_decay": 0.05,
                "grad_norm": 2.0,
            },
            step=11,
            prefix="train_step",
        )
        trainer.log_all(
            {
                "loss": 4.0,
            },
            step=12,
            prefix="val_epoch",
        )

        tags = [tag for tag, _, _ in trainer.tb_writer.scalars]
        self.assertIn("train_step/loss", tags)
        self.assertIn("train_step/hand_2d_loss", tags)
        self.assertIn("train_step/lr", tags)
        self.assertIn("train_step/min_lr", tags)
        self.assertIn("train_step/weight_decay", tags)
        self.assertIn("train_step/grad_norm", tags)
        self.assertIn("val_epoch/loss", tags)

        logged_tags = [sorted(call[1].keys()) for call in trainer.accelerator.logged]
        self.assertIn(["train_step/grad_norm", "train_step/hand_2d_loss", "train_step/loss", "train_step/lr", "train_step/min_lr", "train_step/weight_decay"], logged_tags)
        self.assertIn(["val_epoch/loss"], logged_tags)

    def test_tensorboard_logging_skips_weighted_epoch_metrics(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.accelerator = _DummyAccelerator()
        trainer.tb_writer = _DummyWriter()
        trainer.cfg = type(
            "_Cfg",
            (),
            {"log": type("_LogCfg", (), {"use_tensorboard": True, "use_wandb": False})()},
        )()

        trainer.log_all(
            {
                "val_loss": 1.0,
                "val_hand_pose_loss": 2.0,
                "val_hand_pose_loss_w": 0.2,
            },
            step=13,
        )

        tensorboard_tags = [tag for tag, _, _ in trainer.tb_writer.scalars]
        self.assertIn("/val_loss", tensorboard_tags)
        self.assertIn("/val_hand_pose_loss", tensorboard_tags)
        self.assertNotIn("/val_hand_pose_loss_w", tensorboard_tags)

        self.assertEqual(
            trainer.accelerator.logged,
            [(13, {"/val_loss": 1.0, "/val_hand_pose_loss": 2.0})],
        )


if __name__ == "__main__":
    unittest.main()
