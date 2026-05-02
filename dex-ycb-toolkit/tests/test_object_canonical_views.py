import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "dex_ycb_toolkit" / "object_canonical_views.py"
SPEC = importlib.util.spec_from_file_location("object_canonical_views", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FakeVisual:
    def __init__(self):
        self.vertex_colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)


class _FakeMesh:
    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.visual = _FakeVisual()

    def copy(self):
        return _FakeMesh(self.vertices.copy())


class CanonicalObjectViewsTests(unittest.TestCase):
    def test_configure_offscreen_backend_defaults_to_egl_without_display(self):
        with mock.patch.dict(MODULE.os.environ, {}, clear=True):
            platform = MODULE.configure_offscreen_backend()

        self.assertEqual(platform, "egl")

    def test_configure_offscreen_backend_preserves_explicit_platform(self):
        with mock.patch.dict(MODULE.os.environ, {"PYOPENGL_PLATFORM": "osmesa"}, clear=True):
            platform = MODULE.configure_offscreen_backend()

        self.assertEqual(platform, "osmesa")

    def test_configure_offscreen_backend_uses_pyglet_with_display(self):
        with mock.patch.dict(MODULE.os.environ, {"DISPLAY": ":0"}, clear=True):
            platform = MODULE.configure_offscreen_backend()

        self.assertEqual(platform, "pyglet")

    def test_default_light_rig_has_key_fill_and_rim_lights(self):
        lights = MODULE.default_light_rig()

        self.assertEqual(len(lights), 3)
        self.assertEqual([light["name"] for light in lights], ["key", "fill", "rim"])
        self.assertGreater(lights[0]["intensity"], lights[1]["intensity"])
        self.assertGreater(lights[0]["intensity"], lights[2]["intensity"])
        for light in lights:
            self.assertEqual(light["pose"].shape, (4, 4))

    def test_build_camera_intrinsics_uses_tighter_default_fov(self):
        intrinsics = MODULE.build_camera_intrinsics(image_size=224)

        self.assertGreater(float(intrinsics[0, 0]), 150.0)
        self.assertAlmostEqual(float(intrinsics[0, 2]), 111.5)
        self.assertAlmostEqual(float(intrinsics[1, 2]), 111.5)

    def test_generate_canonical_view_specs_uses_fixed_order_and_object_frame(self):
        specs = MODULE.generate_canonical_view_specs(radius=1.5)

        self.assertEqual(
            [spec.name for spec in specs],
            ["nnn", "nnp", "npn", "npp", "pnn", "pnp", "ppn", "ppp"],
        )
        np.testing.assert_allclose(specs[0].target, np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(specs[-1].target, np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(specs[0].eye), 1.5, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(specs[-1].eye), 1.5, atol=1e-6)
        self.assertEqual(specs[0].t_co.shape, (4, 4))
        self.assertEqual(specs[0].t_oc.shape, (4, 4))
        np.testing.assert_allclose(specs[0].t_oc[:3, 3], specs[0].eye, atol=1e-6)
        np.testing.assert_allclose(specs[0].t_co @ specs[0].t_oc, np.eye(4), atol=1e-6)

    def test_normalize_vertices_maps_max_extent_to_unit_bbox(self):
        vertices = np.array(
            [
                [1.0, 2.0, 3.0],
                [5.0, 4.0, 7.0],
            ],
            dtype=np.float32,
        )

        normalized, center, scale = MODULE.normalize_vertices_to_unit_bbox(vertices)

        np.testing.assert_allclose(center, np.array([3.0, 3.0, 5.0], dtype=np.float32))
        self.assertEqual(scale, 4.0)
        np.testing.assert_allclose(
            normalized.min(axis=0),
            np.array([-0.5, -0.25, -0.5], dtype=np.float32),
        )
        np.testing.assert_allclose(
            normalized.max(axis=0),
            np.array([0.5, 0.25, 0.5], dtype=np.float32),
        )

    def test_save_render_bundle_uses_dexycb_names_and_persists_camera_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "canonical_views_224"
            rgb_frames = [np.full((4, 4, 3), 20 + i, dtype=np.uint8) for i in range(8)]
            depth_frames = [np.full((4, 4), i + 1, dtype=np.uint16) for i in range(8)]
            specs = MODULE.generate_canonical_view_specs(radius=1.5)
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 8, axis=0)

            result = MODULE.save_render_bundle(
                output_dir=output_dir,
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                view_specs=specs,
                intrinsics=intrinsics,
                object_id=25,
                object_name="025_mug",
                mesh_file="mesh.obj",
                image_size=224,
                radius=1.5,
                normalization_center=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                normalization_scale=4.0,
            )

            self.assertEqual(result["color_files"][0].name, "color_000000.jpg")
            self.assertEqual(result["depth_files"][-1].name, "aligned_depth_to_color_000007.png")
            self.assertTrue((output_dir / "camera_params.npz").exists())
            self.assertTrue((output_dir / "meta.json").exists())

            camera_params = np.load(output_dir / "camera_params.npz", allow_pickle=False)
            np.testing.assert_allclose(camera_params["T_co"][0], specs[0].t_co)
            np.testing.assert_allclose(camera_params["T_oc"][-1], specs[-1].t_oc)
            self.assertEqual(camera_params["view_names"].tolist(), ["nnn", "nnp", "npn", "npp", "pnn", "pnp", "ppn", "ppp"])

            color = cv2.imread(str(output_dir / "color_000000.jpg"), cv2.IMREAD_COLOR)
            depth = cv2.imread(
                str(output_dir / "aligned_depth_to_color_000007.png"),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(tuple(color.shape[:2]), (4, 4))
            self.assertEqual(depth.dtype, np.uint16)
            self.assertEqual(int(depth[0, 0]), 8)

            meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["object_name"], "025_mug")
            self.assertEqual(meta["image_size"], 224)
            self.assertEqual(meta["view_names"], ["nnn", "nnp", "npn", "npp", "pnn", "pnp", "ppn", "ppp"])

    def test_generate_object_views_wires_loader_renderer_and_bundle_writer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            object_dir = dataset_root / "models" / "025_mug"
            object_dir.mkdir(parents=True)
            mesh = _FakeMesh(
                [
                    [1.0, 2.0, 3.0],
                    [5.0, 2.0, 3.0],
                    [1.0, 4.0, 7.0],
                ]
            )

            captured = {}

            def fake_renderer(mesh, view_specs, intrinsics, image_size, pyopengl_platform=None):
                captured["mesh_vertices"] = mesh.vertices.copy()
                captured["view_names"] = [spec.name for spec in view_specs]
                captured["intrinsics"] = intrinsics.copy()
                captured["image_size"] = image_size
                captured["pyopengl_platform"] = pyopengl_platform
                rgb = [np.zeros((image_size, image_size, 3), dtype=np.uint8) for _ in view_specs]
                depth = [np.ones((image_size, image_size), dtype=np.uint16) for _ in view_specs]
                return rgb, depth

            def fake_bundle_writer(**kwargs):
                captured["bundle_kwargs"] = kwargs
                return {"output_dir": kwargs["output_dir"]}

            result = MODULE.generate_object_views(
                dataset_root=dataset_root,
                object_name="025_mug",
                image_size=224,
                radius=1.5,
                load_mesh=lambda _: mesh,
                renderer=fake_renderer,
                bundle_writer=fake_bundle_writer,
            )

            self.assertEqual(result["output_dir"], object_dir / "canonical_views_224")
            np.testing.assert_allclose(captured["mesh_vertices"].min(axis=0), np.array([-0.5, -0.25, -0.5], dtype=np.float32))
            np.testing.assert_allclose(captured["mesh_vertices"].max(axis=0), np.array([0.5, 0.25, 0.5], dtype=np.float32))
            self.assertEqual(captured["view_names"], ["nnn", "nnp", "npn", "npp", "pnn", "pnp", "ppn", "ppp"])
            self.assertEqual(captured["image_size"], 224)
            self.assertIsNone(captured["pyopengl_platform"])
            self.assertEqual(captured["bundle_kwargs"]["object_name"], "025_mug")
            np.testing.assert_allclose(
                captured["bundle_kwargs"]["normalization_center"],
                np.array([3.0, 3.0, 5.0], dtype=np.float32),
            )
            self.assertEqual(captured["bundle_kwargs"]["normalization_scale"], 4.0)

    def test_main_forwards_single_object_args(self):
        captured = {}

        def fake_generate_dataset_object_views(**kwargs):
            captured.update(kwargs)
            return []

        with mock.patch.object(MODULE, "generate_dataset_object_views", side_effect=fake_generate_dataset_object_views):
            MODULE.main(
                [
                    "--dataset-root",
                    "/data/hanxiaofeng/dataset/dexycb",
                    "--object-name",
                    "025_mug",
                    "--image-size",
                    "224",
                    "--radius",
                    "1.5",
                    "--output-subdir",
                    "canonical_views_224",
                    "--pyopengl-platform",
                    "egl",
                    "--overwrite",
                ]
            )

        self.assertEqual(captured["dataset_root"], "/data/hanxiaofeng/dataset/dexycb")
        self.assertEqual(captured["object_names"], ["025_mug"])
        self.assertEqual(captured["image_size"], 224)
        self.assertEqual(captured["radius"], 1.5)
        self.assertEqual(captured["output_subdir"], "canonical_views_224")
        self.assertEqual(captured["pyopengl_platform"], "egl")
        self.assertTrue(captured["overwrite"])


if __name__ == "__main__":
    unittest.main()
