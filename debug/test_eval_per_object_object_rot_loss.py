from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug.eval_per_object_object_rot_loss import (  # noqa: E402
    _extract_object_ids,
    _extract_views_batch_for_rerun,
    _maybe_wrap_hydrate_object_payload_to_device,
)


class _DummyTrainer:
    pass


class EvalPerObjectObjectRotLossTests(unittest.TestCase):
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
