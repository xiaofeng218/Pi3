from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pi3.visualization.forehoi_rerun_export import _load_object_meshes


class ForeHOIRerunExportTests(unittest.TestCase):
    def test_object_mesh_loader_flips_uv_v_coordinate_for_rerun(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forehoi_uv_flip_") as tmpdir:
            root = Path(tmpdir)
            part_dir = root / "part_000"
            part_dir.mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(
                '{"parts":[{"key":"part_000","name":"mesh0","texture_file":"albedo.png"}]}',
                encoding="utf-8",
            )
            np.save(part_dir / "vertex_positions_local.npy", np.zeros((3, 3), dtype=np.float32))
            np.save(part_dir / "triangle_indices.npy", np.array([[0, 1, 2]], dtype=np.int32))
            np.save(
                part_dir / "vertex_texcoords.npy",
                np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
            )
            Image.fromarray(np.full((2, 2, 4), 255, dtype=np.uint8), mode="RGBA").save(part_dir / "albedo.png")

            meshes = _load_object_meshes(root)

            self.assertEqual(len(meshes), 1)
            np.testing.assert_allclose(
                meshes[0]["vertex_texcoords"],
                np.array([[0.1, 0.8], [0.3, 0.6], [0.5, 0.4]], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
