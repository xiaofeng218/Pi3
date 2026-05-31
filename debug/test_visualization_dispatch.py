import unittest
from unittest import mock

from pi3.visualization import export_pi3x_rerun_sample


class VisualizationDispatchTests(unittest.TestCase):
    def test_dispatches_forehoi_batch_to_forehoi_exporter(self):
        batch = [{"dataset": ["ForeHOI"]}]
        with mock.patch("pi3.visualization.export_dispatch.export_forehoi_rerun_sample", return_value=("rrd", "meta")) as forehoi_mock:
            with mock.patch("pi3.visualization.export_dispatch.export_dexycb_rerun_sample") as dexycb_mock:
                out = export_pi3x_rerun_sample(
                    output_path="out.rrd",
                    batch=batch,
                    pred={},
                    gt={},
                    sample_index=0,
                    data_root=".",
                )
        self.assertEqual(out, ("rrd", "meta"))
        forehoi_mock.assert_called_once()
        dexycb_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
