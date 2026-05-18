from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from datasets.dexycb_dataset import DexYCBDataset


class _FakeImage:
    def __init__(self):
        self.closed = False

    def convert(self, mode):
        self.mode = mode
        return self

    def __array__(self, dtype=None):
        del dtype
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _FakeImageContext:
    def __init__(self, image):
        self.image = image

    def __enter__(self):
        return self.image

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        self.image.closed = True
        return False


class _FakeNpz(dict):
    @property
    def files(self):
        return list(self.keys())


class _FakeNpzContext:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def __enter__(self):
        return self.payload

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        self.closed = True
        return False


class DexYCBFileHandleTests(unittest.TestCase):
    def test_load_rgb_array_closes_pil_handle(self) -> None:
        dataset = DexYCBDataset.__new__(DexYCBDataset)
        fake_image = _FakeImage()
        fake_ctx = _FakeImageContext(fake_image)

        with mock.patch("datasets.dexycb_dataset.Image.open", return_value=fake_ctx):
            array = DexYCBDataset._load_rgb_array(dataset, "dummy.jpg")

        self.assertEqual(array.shape, (2, 2, 3))
        self.assertTrue(fake_image.closed)

    def test_load_camera_params_closes_npz_handle(self) -> None:
        dataset = DexYCBDataset.__new__(DexYCBDataset)
        payload = _FakeNpz(
            K=np.zeros((1, 3, 3), dtype=np.float32),
            T_oc=np.zeros((1, 4, 4), dtype=np.float32),
            normalization_center=np.zeros(3, dtype=np.float32),
            normalization_scale=np.float32(1.0),
        )
        fake_ctx = _FakeNpzContext(payload)

        with mock.patch("datasets.dexycb_dataset.np.load", return_value=fake_ctx):
            values = DexYCBDataset._load_camera_params(dataset, "camera_params.npz")

        self.assertEqual(values[0].shape, (1, 3, 3))
        self.assertTrue(fake_ctx.closed)

    def test_load_label_npz_closes_handle(self) -> None:
        dataset = DexYCBDataset.__new__(DexYCBDataset)
        payload = _FakeNpz(seg=np.zeros((2, 2), dtype=np.uint8))
        fake_ctx = _FakeNpzContext(payload)

        with mock.patch("datasets.dexycb_dataset.np.load", return_value=fake_ctx):
            loaded = DexYCBDataset._load_label_npz(dataset, "labels.npz")

        self.assertIn("seg", loaded)
        self.assertTrue(fake_ctx.closed)


if __name__ == "__main__":
    unittest.main()
