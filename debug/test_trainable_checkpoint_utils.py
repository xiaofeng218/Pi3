from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from trainers.checkpoint_utils import collect_trainable_state_dict, load_trainable_checkpoint, save_trainable_checkpoint


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(2, 2, bias=False)
        self.trainable = nn.Linear(2, 2, bias=False)
        for param in self.frozen.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.trainable(self.frozen(x))


class TrainableCheckpointUtilsTests(unittest.TestCase):
    def test_collect_trainable_state_dict_only_contains_trainable_params(self):
        module = _TinyModule()
        state = collect_trainable_state_dict(module)
        self.assertIn("trainable.weight", state)
        self.assertNotIn("frozen.weight", state)

    def test_save_and_load_trainable_checkpoint_roundtrip(self):
        module = _TinyModule()
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_trainable_checkpoint(
                tmpdir,
                module,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={"epoch": 3, "global_step": 7},
            )

            module.trainable.weight.data.fill_(0.0)
            module.frozen.weight.data.fill_(0.0)

            result = load_trainable_checkpoint(tmpdir, module, optimizer=optimizer, scheduler=scheduler)
            self.assertEqual(result["meta"]["epoch"], 3)
            self.assertEqual(result["meta"]["global_step"], 7)
            self.assertTrue(torch.allclose(module.trainable.weight, torch.load(tmpdir / "trainable_model.pt")["trainable.weight"]))
            self.assertTrue(torch.all(module.frozen.weight == 0.0))


if __name__ == "__main__":
    unittest.main()
