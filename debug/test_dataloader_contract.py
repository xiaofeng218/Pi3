from __future__ import annotations

import unittest
from unittest import mock

import datasets


class _Cfg(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _FakeDataset:
    def __len__(self):
        return 100

    def convert_attributes(self):
        return None


class _FakeSampler:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return 100


class _FakeBatchSampler:
    image_num_range = [6, 12]
    max_img_per_gpu = 24

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return 10


class CreateDataloaderContractTests(unittest.TestCase):
    def test_train_dataloader_passes_multiprocessing_context(self) -> None:
        cfg = _Cfg(
            train=_Cfg(batch_size=1, num_workers=2, image_num_range=[6, 12], base_seed=666, max_img_per_gpu=24, iters_per_epoch=-1),
            test=_Cfg(batch_size=1, num_workers=0, image_num_range=[12, 12]),
            train_dataset=_Cfg(_target_="fake"),
            test_dataset=_Cfg(_target_="fake"),
            train_dataloader=_Cfg(drop_last=True, shuffle=True, persistent_workers=False, multiprocessing_context="spawn"),
            test_dataloader=_Cfg(drop_last=False, shuffle=False, persistent_workers=False),
        )

        captured = {}

        def _fake_dataloader(**kwargs):
            captured.update(kwargs)
            return kwargs

        with mock.patch.object(datasets.hydra.utils, "instantiate", return_value=_FakeDataset()), \
             mock.patch.object(datasets, "DynamicDistributedSampler", _FakeSampler), \
             mock.patch.object(datasets, "DynamicBatchSampler", _FakeBatchSampler), \
             mock.patch.object(datasets, "get_world_size", return_value=1), \
             mock.patch.object(datasets, "get_rank", return_value=0), \
             mock.patch.object(datasets, "DataLoader", side_effect=_fake_dataloader):
            datasets.create_dataloader(cfg, "train")

        self.assertEqual(captured["multiprocessing_context"], "spawn")
        self.assertEqual(captured["num_workers"], 2)
        self.assertEqual(captured["prefetch_factor"], 2)

    def test_train_dataloader_allows_prefetch_override(self) -> None:
        cfg = _Cfg(
            train=_Cfg(batch_size=1, num_workers=2, image_num_range=[6, 12], base_seed=666, max_img_per_gpu=24, iters_per_epoch=-1),
            test=_Cfg(batch_size=1, num_workers=0, image_num_range=[12, 12]),
            train_dataset=_Cfg(_target_="fake"),
            test_dataset=_Cfg(_target_="fake"),
            train_dataloader=_Cfg(drop_last=True, shuffle=True, persistent_workers=False, multiprocessing_context="forkserver", prefetch_factor=1),
            test_dataloader=_Cfg(drop_last=False, shuffle=False, persistent_workers=False),
        )

        captured = {}

        def _fake_dataloader(**kwargs):
            captured.update(kwargs)
            return kwargs

        with mock.patch.object(datasets.hydra.utils, "instantiate", return_value=_FakeDataset()), \
             mock.patch.object(datasets, "DynamicDistributedSampler", _FakeSampler), \
             mock.patch.object(datasets, "DynamicBatchSampler", _FakeBatchSampler), \
             mock.patch.object(datasets, "get_world_size", return_value=1), \
             mock.patch.object(datasets, "get_rank", return_value=0), \
             mock.patch.object(datasets, "DataLoader", side_effect=_fake_dataloader):
            datasets.create_dataloader(cfg, "train")

        self.assertEqual(captured["multiprocessing_context"], "forkserver")
        self.assertEqual(captured["prefetch_factor"], 1)


if __name__ == "__main__":
    unittest.main()
