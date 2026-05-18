from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from contextlib import contextmanager

import torch

from trainers.base_trainer_accelerate import BaseTrainer
from utils.dist import MetricLogger


class _Cfg(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _LossOutput(dict):
    @property
    def loss(self):
        return self["loss"]


class _DummyModel:
    def eval(self):
        return self


class _DummyAccelerator:
    is_main_process = False
    device = torch.device("cpu")
    num_processes = 1
    gradient_accumulation_steps = 1

    def prepare(self, *objects):
        return objects[0] if len(objects) == 1 else objects

    def init_trackers(self, *_args, **_kwargs):
        return None

    def gather(self, tensor):
        return tensor

    def reduce(self, tensor, reduction="mean"):
        assert reduction == "mean"
        return tensor

    def wait_for_everyone(self):
        return None

    def end_training(self):
        return None

    @contextmanager
    def autocast(self):
        yield


class BaseTrainerAccelerateContractTests(unittest.TestCase):
    def test_train_runs_validation_before_first_epoch_by_default(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.accelerator = _DummyAccelerator()
        trainer.cfg = _Cfg(train=_Cfg(num_epoch=1))
        trainer.first_epoch = 0
        trainer.n_learnable_parameters = 1
        trainer.global_step = 0
        trainer.log_info = lambda *_args, **_kwargs: None
        trainer.log_all = lambda *_args, **_kwargs: None
        trainer.before_epoch = lambda epoch: events.append(("before_epoch", epoch))
        trainer._save_ckpt = lambda *_args, **_kwargs: "/tmp/checkpoint"
        trainer._cleanup_checkpoints = lambda *_args, **_kwargs: None
        trainer.tb_writer = None

        events = []
        trainer.train_one_epoch = lambda epoch: events.append(("train_one_epoch", epoch)) or {"loss": 1.0}
        trainer.validate = lambda epoch: events.append(("validate", epoch)) or {"loss": 1.0}

        BaseTrainer.train(trainer)

        self.assertEqual(
            events,
            [
                ("validate", -1),
                ("before_epoch", 0),
                ("train_one_epoch", 0),
                ("validate", 0),
            ],
        )

    def test_reduce_train_loss_value_skips_nested_weighted_loss_details(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.accelerator = _DummyAccelerator()

        reduced = BaseTrainer._reduce_train_loss_value(trainer, {"hand_pose_loss": 0.125})

        self.assertIsNone(reduced)

    def test_reduce_train_loss_value_gathers_scalar_tensor_losses(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.accelerator = _DummyAccelerator()

        reduced = BaseTrainer._reduce_train_loss_value(trainer, torch.tensor(2.5))

        self.assertEqual(reduced, 2.5)

    def test_reduce_train_loss_value_can_use_mean_reduce_mode(self) -> None:
        class _ReduceTrackingAccelerator(_DummyAccelerator):
            def __init__(self):
                self.reduce_called = False

            def reduce(self, tensor, reduction="mean"):
                self.reduce_called = True
                self.seen = tensor
                return tensor

        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.accelerator = _ReduceTrackingAccelerator()
        trainer.cfg = _Cfg(log=_Cfg(train_loss_reduce_mode="mean_reduce"))

        reduced = BaseTrainer._reduce_train_loss_value(trainer, torch.tensor(1.25))

        self.assertEqual(reduced, 1.25)
        self.assertTrue(trainer.accelerator.reduce_called)
        self.assertEqual(trainer.accelerator.seen.dtype, torch.float32)

    def test_reduce_train_loss_value_skips_non_scalar_tensor_losses(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.accelerator = _DummyAccelerator()

        reduced = BaseTrainer._reduce_train_loss_value(trainer, torch.tensor([1.0, 2.0]))

        self.assertIsNone(reduced)

    def test_reduce_train_loss_value_gathers_dense_detached_scalar(self) -> None:
        class _StrictAccelerator(_DummyAccelerator):
            def gather(self, tensor):
                self.seen = tensor
                assert not tensor.requires_grad
                assert tensor.layout == torch.strided
                assert tensor.is_contiguous()
                assert tensor.device == self.device
                return tensor

        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.accelerator = _StrictAccelerator()
        sparse_scalar = torch.sparse_coo_tensor(torch.zeros((0, 1), dtype=torch.long), torch.tensor([3.0]), ())

        reduced = BaseTrainer._reduce_train_loss_value(trainer, sparse_scalar.requires_grad_(True))

        self.assertEqual(reduced, 3.0)
        self.assertIsNotNone(trainer.accelerator.seen)

    def test_format_metric_logger_uses_weighted_global_avg_in_parentheses(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        raw = MetricLogger(delimiter="  ")
        weighted = MetricLogger(delimiter="  ")
        raw.update(hand_pose_loss=0.32, hand_beta_loss=0.18, object_rot_loss=0.0002)
        weighted.update(hand_pose_loss=0.0032, hand_beta_loss=0.0018)

        formatted = BaseTrainer._format_metric_logger_with_weighted(trainer, raw, weighted)

        self.assertIn("hand_pose_loss: 0.3200 (0.0032)", formatted)
        self.assertIn("hand_beta_loss: 0.1800 (0.0018)", formatted)
        self.assertIn("object_rot_loss: 0.0002 (0.0002)", formatted)

    def test_format_step_timing_message_includes_phase_rank_and_elapsed(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)

        message = BaseTrainer._format_step_timing_message(
            trainer,
            epoch=2,
            it=87,
            global_step=123,
            phase="forward",
            state="end",
            rank=1,
            elapsed_s=2.375,
        )

        self.assertIn("STEP_TIMING", message)
        self.assertIn("rank=1", message)
        self.assertIn("epoch=2", message)
        self.assertIn("iter=87", message)
        self.assertIn("global_step=123", message)
        self.assertIn("phase=forward", message)
        self.assertIn("state=end", message)
        self.assertIn("dt=2.375s", message)

    def test_sync_cuda_before_train_loss_reduce_defaults_to_false(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)

        enabled = BaseTrainer._sync_cuda_before_train_loss_reduce(trainer)

        self.assertFalse(enabled)

    def test_snapshot_for_visualization_detaches_nested_tensors(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        source = {
            "tensor": torch.ones(2, requires_grad=True),
            "nested": [{"value": torch.zeros(1, requires_grad=True)}],
            "plain": 3,
        }

        snapshot = BaseTrainer._snapshot_for_visualization(trainer, source)

        self.assertEqual(snapshot["plain"], 3)
        self.assertFalse(snapshot["tensor"].requires_grad)
        self.assertFalse(snapshot["nested"][0]["value"].requires_grad)
        self.assertIsNot(snapshot["tensor"], source["tensor"])
        self.assertIsNot(snapshot["nested"][0]["value"], source["nested"][0]["value"])

    def test_auto_resume_detects_checkpoint_dash_epoch_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            (ckpt_dir / "checkpoint-epoch-0001").mkdir()
            (ckpt_dir / "checkpoint-epoch-0003").mkdir()
            (ckpt_dir / "checkpoint-step-0001000").mkdir()

            trainer = BaseTrainer.__new__(BaseTrainer)
            trainer.cfg = _Cfg(
                train=_Cfg(resume=None),
                log=_Cfg(ckpt_dir=str(ckpt_dir)),
            )
            trainer.log_info = lambda *_args, **_kwargs: None

            start_epoch = BaseTrainer.auto_resume(trainer)

            self.assertEqual(start_epoch, 4)
            self.assertEqual(trainer.resume_path, str(ckpt_dir / "checkpoint-epoch-0003"))

    def test_validate_returns_sample_weighted_loss(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.model = _DummyModel()
        trainer.accelerator = _DummyAccelerator()
        trainer.iters_per_test = 2
        trainer.log_info = lambda *_args, **_kwargs: None
        trainer.maybe_export_validation_sample = lambda **_kwargs: None

        batch_large = [{"img": torch.zeros(3, 1)}]
        batch_small = [{"img": torch.zeros(1, 1)}]
        trainer.test_loader = [batch_large, batch_small]

        losses = iter([1.0, 10.0])
        trainer.forward_batch = lambda batch, mode="test": {"mode": mode, "batch": batch}
        trainer.calculate_loss = lambda _forward, _batch, mode="test": _LossOutput(
            loss=torch.tensor(next(losses), dtype=torch.float32),
            aux_metric=torch.tensor(0.0),
        )

        stats = BaseTrainer.validate(trainer, epoch=0)

        self.assertAlmostEqual(stats["loss"], 3.25, places=6)

    def test_validate_runs_forward_inside_accelerator_autocast(self) -> None:
        class _AutocastTrackingAccelerator(_DummyAccelerator):
            def __init__(self):
                self.autocast_entered = 0
                self.in_autocast = False

            @contextmanager
            def autocast(self):
                self.autocast_entered += 1
                self.in_autocast = True
                try:
                    yield
                finally:
                    self.in_autocast = False

        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.model = _DummyModel()
        trainer.accelerator = _AutocastTrackingAccelerator()
        trainer.iters_per_test = 1
        trainer.log_info = lambda *_args, **_kwargs: None
        trainer.maybe_export_validation_sample = lambda **_kwargs: None
        trainer.test_loader = [[{"img": torch.zeros(1, 1)}]]

        def _forward_batch(_batch, mode="test"):
            self.assertEqual(mode, "test")
            self.assertTrue(trainer.accelerator.in_autocast)
            return {"ok": True}

        trainer.forward_batch = _forward_batch
        trainer.calculate_loss = lambda _forward, _batch, mode="test": _LossOutput(
            loss=torch.tensor(1.0, dtype=torch.float32),
        )

        BaseTrainer.validate(trainer, epoch=0)

        self.assertEqual(trainer.accelerator.autocast_entered, 1)

    def test_prepare_training_loads_auto_resume_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            ckpt_dir = output_dir / "ckpts"
            (ckpt_dir / "checkpoint-epoch-0003").mkdir(parents=True)

            trainer = BaseTrainer.__new__(BaseTrainer)
            trainer.cfg = _Cfg(
                train=_Cfg(
                    batch_size=1,
                    gradient_accumulation_steps=1,
                    optimizer=_Cfg(lr=1e-4, weight_decay=0.0),
                    num_epoch=10,
                    resume=None,
                ),
                log=_Cfg(
                    output_dir=str(output_dir),
                    ckpt_dir=str(ckpt_dir),
                    use_tensorboard=False,
                    use_wandb=False,
                ),
            )
            trainer.accelerator = _DummyAccelerator()
            trainer.optimizer = object()
            trainer.lr_scheduler = object()
            trainer.train_loader = []
            trainer.test_loader = []
            trainer.n_learnable_parameters = 1
            trainer.n_fix_parameters = 1
            trainer.iters_per_epoch = 5
            trainer.log_info = lambda *_args, **_kwargs: None

            loaded = []
            trainer.load_training_state = lambda path: loaded.append(path)

            BaseTrainer.prepare_training(trainer)

            self.assertEqual(loaded, [str(ckpt_dir / "checkpoint-epoch-0003")])
            self.assertEqual(trainer.first_epoch, 4)
            self.assertIsNone(getattr(trainer, "_pending_resume_path", None))

    def test_build_accelerator_rejects_bf16_without_cuda(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.cfg = _Cfg(
            train=_Cfg(
                model_dtype="bf16",
                gradient_accumulation_steps=1,
                find_unused_parameters=False,
            ),
            log=_Cfg(
                output_dir="/tmp/out",
                use_wandb=False,
                use_tensorboard=False,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            BaseTrainer.build_accelerator(trainer)

    def test_infer_batch_sample_count_from_view_batch(self) -> None:
        trainer = BaseTrainer.__new__(BaseTrainer)
        batch = [{"img": torch.zeros(3, 3, 4, 4)}, {"img": torch.zeros(3, 3, 4, 4)}]

        count = BaseTrainer._infer_batch_sample_count(trainer, batch)

        self.assertEqual(count, 3)



if __name__ == "__main__":
    unittest.main()
