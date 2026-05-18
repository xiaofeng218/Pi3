from __future__ import annotations

import unittest

import torch

from debug.pi3x_forward_finite_check import find_first_non_finite, summarize_batch


class Pi3XForwardFiniteCheckTests(unittest.TestCase):
    def test_find_first_non_finite_reports_nested_tensor_path(self) -> None:
        payload = {
            "pred": {
                "ok": torch.ones(2, 3),
                "bad": torch.tensor([[1.0, float("nan")]]),
            }
        }

        report = find_first_non_finite(payload)

        self.assertIsNotNone(report)
        self.assertEqual(report["path"], "root.pred.bad")
        self.assertEqual(report["bad_count"], 1)

    def test_summarize_batch_includes_view_labels_and_instances(self) -> None:
        batch = [
            {
                "label": ["seq/a", "seq/b"],
                "instance": ["000001", "000002"],
            },
            {
                "label": ["seq/a", "seq/b"],
                "instance": ["000003", "000004"],
            },
        ]

        summary = summarize_batch(batch)

        self.assertEqual(summary[0]["label"], "seq/a")
        self.assertEqual(summary[0]["instances"], ["000001", "000003"])
        self.assertEqual(summary[1]["label"], "seq/b")
        self.assertEqual(summary[1]["instances"], ["000002", "000004"])


if __name__ == "__main__":
    unittest.main()
