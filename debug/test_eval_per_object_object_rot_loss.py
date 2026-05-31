from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug.eval_per_object_object_rot_loss import (  # noqa: E402
    _extract_object_ids,
    _extract_views_batch_for_rerun,
    _resolve_eval_loader,
    _maybe_wrap_hydrate_object_payload_to_device,
    _build_trainer,
)


class _DummyTrainer:
    pass


class EvalPerObjectObjectRotLossTests(unittest.TestCase):
    def test_resolve_eval_loader_uses_train_loader_for_train_split(self) -> None:
        trainer = SimpleNamespace(train_loader="train_loader", test_loader="test_loader")

        loader = _resolve_eval_loader(trainer, split="train")

        self.assertEqual(loader, "train_loader")

    def test_build_trainer_overrides_dataset_modes_for_requested_split(self) -> None:
        fake_cfg = SimpleNamespace(
            extras=SimpleNamespace(print_config=True),
            work_dir="",
            working_dir="",
        )

        with (
            mock.patch("debug.eval_per_object_object_rot_loss.compose", return_value=fake_cfg) as compose_mock,
            mock.patch("debug.eval_per_object_object_rot_loss.initialize_config_dir") as init_mock,
            mock.patch("debug.eval_per_object_object_rot_loss.OmegaConf.load"),
            mock.patch("debug.eval_per_object_object_rot_loss.open_dict") as open_dict_mock,
            mock.patch("debug.eval_per_object_object_rot_loss.Pi3XTrainer"),
        ):
            init_mock.return_value.__enter__.return_value = None
            init_mock.return_value.__exit__.return_value = None
            open_dict_mock.return_value.__enter__.return_value = fake_cfg
            open_dict_mock.return_value.__exit__.return_value = None

            _build_trainer(subject="20200709-subject-01", output_dir=REPO_ROOT / "tmp" / "unit-test", split="train")

        overrides = compose_mock.call_args.kwargs["overrides"]
        self.assertIn("train_dataset.mode=train", overrides)
        self.assertIn("test_dataset.mode=train", overrides)

    def test_wrap_hydrate_payload_is_noop_when_trainer_uses_new_batch_api(self) -> None:
        trainer = _DummyTrainer()

        _maybe_wrap_hydrate_object_payload_to_device(trainer)

        self.assertFalse(hasattr(trainer, "_hydrate_object_multiview_payload"))

    def test_extract_object_ids_reads_current_views_batch_shape(self) -> None:
        batch = {
            "views": [
                {
                    "object": {
                        "grasped_object_id": torch.tensor([11, 14], dtype=torch.int64),
                    }
                },
                {
                    "object": {
                        "grasped_object_id": torch.tensor([11, 14], dtype=torch.int64),
                    }
                },
            ]
        }

        object_ids = _extract_object_ids(batch, device=torch.device("cpu"))

        self.assertTrue(torch.equal(object_ids, torch.tensor([11, 14], dtype=torch.int64)))

    def test_extract_views_batch_for_rerun_returns_views_list_from_dict_batch(self) -> None:
        views = [{"img": torch.zeros(1, 3, 4, 4)}]
        batch = {"views": views}

        rerun_batch = _extract_views_batch_for_rerun(batch)

        self.assertIs(rerun_batch, views)


if __name__ == "__main__":
    unittest.main()
