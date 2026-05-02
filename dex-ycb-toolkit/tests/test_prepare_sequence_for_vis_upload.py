import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "examples" / "prepare_sequence_for_vis_upload.py"
SPEC = importlib.util.spec_from_file_location("prepare_sequence_for_vis_upload", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _FakeHandLayer:
    def __init__(self):
        self.f = np.array([[0, 1, 2]], dtype=np.int32)


class _FakeManoGroupLayer:
    def __init__(self):
        self._layers = [_FakeHandLayer()]


class _FakeYcbGroupLayer:
    def __init__(self, obj_files):
        self.obj_file = obj_files


class _FakeSequenceLoader:
    def __init__(self):
        self.serials = ["cam0"]
        self.num_frames = 2
        self.ycb_ids = [1, 2]
        self.mano_group_layer = _FakeManoGroupLayer()
        self.ycb_group_layer = _FakeYcbGroupLayer(["obj1.obj", "obj2.obj"])
        self._frame = -1
        self._identity = np.eye(4, dtype=np.float32)
        translated = np.eye(4, dtype=np.float32)
        translated[:3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._ycb_frames = [
            [np.stack([self._identity, translated], axis=0)],
            [np.stack([self._identity, np.zeros((4, 4), dtype=np.float32)], axis=0)],
        ]
        self._hand_frames = [
            [[np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float32)]],
            [[np.zeros((3, 3), dtype=np.float32)]],
        ]

    def step(self):
        self._frame = (self._frame + 1) % self.num_frames

    @property
    def ycb_pose(self):
        return self._ycb_frames[self._frame]

    @property
    def mano_vert(self):
        return self._hand_frames[self._frame]


class _FakeMesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces


class _FakeRerun:
    class ViewCoordinates:
        RDF = "RDF"

    class Mesh3D:
        def __init__(self, vertex_positions, triangle_indices=None, albedo_factor=None):
            self.vertex_positions = np.asarray(vertex_positions)
            self.triangle_indices = None if triangle_indices is None else np.asarray(triangle_indices)
            self.albedo_factor = albedo_factor

    class Clear:
        def __init__(self, recursive):
            self.recursive = recursive

    def __init__(self):
        self.calls = []

    def init(self, *args, **kwargs):
        self.calls.append(("init", args, kwargs))

    def save(self, path):
        self.calls.append(("save", path))
        Path(path).write_bytes(b"rrd")

    def set_time_sequence(self, timeline, value):
        self.calls.append(("time", timeline, value))

    def log(self, path, obj, **kwargs):
        self.calls.append(("log", path, obj, kwargs))


class PrepareSequenceForVisUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.camera_dir = self.root / "20200709-subject-01" / "20200709_141754" / "932122061900"
        self.camera_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepare_sequence_bundle_writes_videos_rrd_and_manifest(self):
        output_root = self.root / "prepared"

        def fake_video_exporter(camera_dir, output_dir, prefix, fps):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            files = {}
            for name in ["rgb", "seg", "joints", "overlay", "foreground_overlay"]:
                path = output_dir / f"tmp_{name}.mp4"
                path.write_bytes(b"video")
                files[name] = path
            return files

        def fake_rrd_exporter(sequence_name, camera_serial, output_path, device):
            Path(output_path).write_bytes(b"rrd")

        manifest_path = MODULE.prepare_sequence_bundle_for_vis_upload(
            camera_dir=self.camera_dir,
            output_root=output_root,
            item_id="capture-set/clip-001/view-left",
            release="sequence-vis",
            dataset="dexycb",
            fps=15,
            num_frames=72,
            device="cpu",
            video_exporter=fake_video_exporter,
            rrd_exporter=fake_rrd_exporter,
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_dir = manifest_path.parent

        self.assertTrue((bundle_dir / "rgb.mp4").exists())
        self.assertTrue((bundle_dir / "sequence.rrd").exists())
        self.assertEqual(manifest["item_id"], "capture-set/clip-001/view-left")
        self.assertEqual(len(manifest["artifacts"]), 7)
        self.assertEqual(
            [artifact["role"] for artifact in manifest["artifacts"]],
            [
                "preview_rgb",
                "preview_seg",
                "preview_joints",
                "preview_overlay",
                "preview_foreground_overlay",
                "interactive_rrd",
                "metadata_json",
            ],
        )

    def test_export_rerun_sequence_rrd_logs_meshes_and_clears_missing_entities(self):
        output_path = self.root / "sequence.rrd"
        rr = _FakeRerun()

        MODULE.export_rerun_sequence_rrd(
            sequence_name="20200709-subject-01/20200709_141754",
            camera_serial="cam0",
            output_path=output_path,
            device="cpu",
            rr_module=rr,
            loader_factory=lambda *args, **kwargs: _FakeSequenceLoader(),
            object_mesh_loader=lambda loader: {
                1: _FakeMesh(
                    vertices=np.array(
                        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]],
                        dtype=np.float32,
                    ),
                    faces=np.array([[0, 1, 2]], dtype=np.int32),
                ),
                2: _FakeMesh(
                    vertices=np.array(
                        [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.03, 0.0]],
                        dtype=np.float32,
                    ),
                    faces=np.array([[0, 1, 2]], dtype=np.int32),
                ),
            },
        )

        self.assertTrue(output_path.exists())
        mesh_calls = [call for call in rr.calls if call[0] == "log" and isinstance(call[2], _FakeRerun.Mesh3D)]
        clear_calls = [call for call in rr.calls if call[0] == "log" and isinstance(call[2], _FakeRerun.Clear)]
        view_calls = [call for call in rr.calls if call[0] == "log" and call[2] == _FakeRerun.ViewCoordinates.RDF]
        self.assertTrue(any(call[1] == "world/objects/01_0" for call in mesh_calls))
        self.assertTrue(any(call[1] == "world/hands/0" for call in mesh_calls))
        self.assertTrue(any(call[1] == "world/objects/02_1" for call in clear_calls))
        self.assertTrue(any(call[1] == "world/hands/0" for call in clear_calls))
        self.assertTrue(any(call[1] == "world/camera" and call[3].get("static") for call in view_calls))


if __name__ == "__main__":
    unittest.main()
