from __future__ import annotations

import unittest

from datasets.base.batched_sampler import DynamicBatchSampler


class _FakeSampler:
    def __init__(self, length: int) -> None:
        self.length = length
        self.resolution_idx = None
        self.image_num = None

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for idx in range(self.length):
            yield (idx, self.resolution_idx, self.image_num)

    def update_parameters(self, resolution_idx, image_num) -> None:
        self.resolution_idx = resolution_idx
        self.image_num = image_num


class DynamicBatchSamplerTests(unittest.TestCase):
    def test_len_matches_actual_number_of_batches(self) -> None:
        sampler = _FakeSampler(length=17)
        batch_sampler = DynamicBatchSampler(
            sampler=sampler,
            resolution_num=1,
            image_num_range=[6, 12],
            seed=42,
            rank=0,
            max_img_per_gpu=24,
        )

        actual_batches = list(iter(batch_sampler))

        self.assertEqual(len(batch_sampler), len(actual_batches))

    def test_rejects_image_num_range_above_frame_budget(self) -> None:
        sampler = _FakeSampler(length=8)

        with self.assertRaises(ValueError):
            DynamicBatchSampler(
                sampler=sampler,
                resolution_num=1,
                image_num_range=[6, 25],
                seed=42,
                rank=0,
                max_img_per_gpu=24,
            )


if __name__ == "__main__":
    unittest.main()
