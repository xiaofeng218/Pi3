from __future__ import annotations

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir


class Pi3XConfigSmokeTests(unittest.TestCase):
    def test_pi3x_hand_object_config_composes(self) -> None:
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=config_dir, job_name="pi3x_smoke", version_base=None):
            cfg = compose(config_name="pi3x_hand_object")
        self.assertEqual(cfg.trainer, "trainers.pi3x_trainer.Pi3XTrainer")
        self.assertEqual(cfg.model._target_, "pi3.models.pi3x.Pi3X")
        self.assertEqual(cfg.loss.train_loss._target_, "pi3.models.hand_object_loss.HandObjectLoss")
        self.assertEqual(cfg.train_dataset._target_, "datasets.dexycb_dataset.DexYCBDataset")
        self.assertEqual(cfg.test_dataset._target_, "datasets.dexycb_dataset.DexYCBDataset")
        self.assertEqual(cfg.hand_encoder._target_, "pi3.models.hamer.load.hamer_encoder")
        self.assertIn("configs/hamer/model_config.yaml", cfg.model.hamer_config_file)
        self.assertIn("configs/hamer/model_config.yaml", cfg.hand_encoder.config_file)
        self.assertIn("data/model/hamer/_DATA", cfg.model.hamer_cache_dir)
        self.assertIn("data/model/hamer/_DATA", cfg.hand_encoder.cache_dir)
        self.assertNotIn("third_party", cfg.model.hamer_config_file)
        self.assertNotIn("third_party", cfg.hand_encoder.config_file)


if __name__ == "__main__":
    unittest.main()
