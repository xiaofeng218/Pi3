from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug.inspect_pi3x_2d_projection_consistency import generate_projection_consistency_report
from debug.test_dexycb_dataset_contract import build_fixture


def _write_minimal_obj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0.0 0.0 0.5",
                "v 0.05 0.0 0.5",
                "v 0.0 0.05 0.5",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class ProjectionConsistencyScriptTests(unittest.TestCase):
    def test_script_generates_four_images_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="proj_consistency_") as tmpdir:
            root = Path(tmpdir)
            build_fixture(root)
            for model_name in ("011_banana", "019_pitcher_base"):
                _write_minimal_obj(root / "models" / model_name / "textured_simple.obj")

            output_dir = root / "out"
            metrics = generate_projection_consistency_report(
                data_root=str(root),
                output_dir=str(output_dir),
                subject="20200709-subject-01",
                mode="train",
                sample_index=0,
                frame_index=None,
                frame_num=4,
                resolution=(224, 224),
                ckpt=None,
                device="cpu",
            )

            self.assertTrue((output_dir / "01_gt_hand_joint2d_overlay.png").is_file())
            self.assertTrue((output_dir / "02_gt_hand_reprojected_overlay.png").is_file())
            self.assertTrue((output_dir / "03_gt_object_vertices_overlay.png").is_file())
            self.assertTrue((output_dir / "04_predscale_hand_object_overlay.png").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertIn("pred_intrinsics_source", metrics)
            self.assertEqual(metrics["pred_intrinsics_source"], "gt_fallback_no_ckpt")
            self.assertIn("hand_reprojection_mean_px", metrics)
            self.assertIn("hand_reprojection_median_px", metrics)
            self.assertIn("hand_reprojection_max_px", metrics)
            self.assertIn("hand_reprojection_per_joint_px", metrics)
            self.assertEqual(len(metrics["hand_reprojection_per_joint_px"]), 21)
            self.assertIn("object_reprojection_mean_px", metrics)
            self.assertIn("object_reprojection_median_px", metrics)
            self.assertIn("object_reprojection_max_px", metrics)
            self.assertIn("object_reprojection_sample_px", metrics)


if __name__ == "__main__":
    unittest.main()
