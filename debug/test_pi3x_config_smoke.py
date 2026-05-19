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
        self.assertEqual(cfg.train.num_workers, 2)
        self.assertEqual(cfg.test.num_workers, 2)
        self.assertEqual(list(cfg.test.image_num_range), [14, 14])
        self.assertFalse(cfg.train_dataloader.persistent_workers)
        self.assertFalse(cfg.test_dataloader.persistent_workers)
        self.assertNotIn("third_party", cfg.model.hamer_config_file)
        self.assertNotIn("third_party", cfg.hand_encoder.config_file)

    def test_overfit_config_composes(self) -> None:
        config_dir = str(Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(config_dir=config_dir, job_name="pi3x_overfit_smoke", version_base=None):
            cfg = compose(config_name="overfit")
        self.assertEqual(cfg.name, "pi3x_overfit")
        self.assertTrue(cfg.test.use_train_loader)
        self.assertEqual(cfg.train_dataset.selected_tracks.left.subject, "20200709-subject-01")
        self.assertEqual(cfg.train_dataset.selected_tracks.left.sequence, "20200709_141931")
        self.assertEqual(cfg.train_dataset.selected_tracks.right.sequence, "20200709_141754")
        self.assertEqual(cfg.train_dataset.selected_tracks.left.camera, "836212060125")
        self.assertEqual(cfg.test_dataset.selected_tracks.right.subject, "20200709-subject-01")
        self.assertEqual(cfg.train.iters_per_epoch, 100)
        self.assertEqual(cfg.test.iters_per_test, 2)
        self.assertEqual(cfg.vis.interval, cfg.train.iters_per_epoch)
        self.assertEqual(cfg.train.num_workers, 0)
        self.assertEqual(cfg.test.num_workers, 0)
        self.assertFalse(cfg.train.auto_resume)


if __name__ == "__main__":
    unittest.main()
