from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from debug.export_forehoi_batch_rrd import export_forehoi_batch_rrd
from debug.test_forehoi_dataset_contract import build_fixture


class ExportForeHOIBatchRRDTests(unittest.TestCase):
    def test_export_uses_dataloader_batch_and_dispatch_exporter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forehoi_export_rrd_") as tmpdir:
            root = Path(tmpdir)
            raw_root, omv_root = build_fixture(root)
            output_path = root / "rrd" / "sample.rrd"

            with mock.patch(
                "debug.export_forehoi_batch_rrd._build_visualization_mano_layers",
                return_value="mano_layers",
            ):
                with mock.patch("debug.export_forehoi_batch_rrd.export_pi3x_rerun_sample") as export_mock:
                    export_mock.return_value = (output_path, output_path.with_suffix(".meta.json"))
                    rrd_path, meta_path, manifest_path = export_forehoi_batch_rrd(
                        data_root=str(raw_root),
                        object_multiview_root=str(omv_root),
                        output_path=str(output_path),
                        frame_num=2,
                        resolution=(64, 64),
                    )

            self.assertEqual(rrd_path, output_path)
            self.assertTrue(meta_path.is_file())
            self.assertTrue(manifest_path.is_file())

            export_kwargs = export_mock.call_args.kwargs
            self.assertIsNone(export_kwargs["pred"])
            self.assertEqual(export_kwargs["mano_layer"], "mano_layers")
            self.assertEqual(export_kwargs["sample_index"], 0)
            self.assertEqual(export_kwargs["batch"][0]["dataset"][0], "ForeHOI")
            self.assertTrue(torch.allclose(export_kwargs["gt"]["scene_scale"], torch.ones(1)))
            self.assertIn("object_pose_obj2cam", export_kwargs["gt"])
            self.assertIn("hand_global_orient_rotmat", export_kwargs["gt"])


if __name__ == "__main__":
    unittest.main()
