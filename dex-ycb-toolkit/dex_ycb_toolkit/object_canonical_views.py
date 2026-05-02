"""Generate fixed canonical RGB/depth views for DexYCB object models."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from dex_ycb_toolkit.dex_ycb import DexYCBDataset


VIEW_SIGNS = [
    (-1, -1, -1, "nnn"),
    (-1, -1, +1, "nnp"),
    (-1, +1, -1, "npn"),
    (-1, +1, +1, "npp"),
    (+1, -1, -1, "pnn"),
    (+1, -1, +1, "pnp"),
    (+1, +1, -1, "ppn"),
    (+1, +1, +1, "ppp"),
]

_CV_TO_GL = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _make_pose(translation: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array(translation, dtype=np.float32)
    return pose


@dataclass(frozen=True)
class CanonicalViewSpec:
    """One canonical view referenced to the canonical object frame."""

    index: int
    name: str
    sign: np.ndarray
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    t_co: np.ndarray
    t_oc: np.ndarray


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def _build_t_oc_from_look_at(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Return camera-to-object pose for an OpenCV-style camera frame."""
    z_axis = _normalize_vector(target - eye)
    x_axis = _normalize_vector(np.cross(z_axis, world_up))
    y_axis = _normalize_vector(np.cross(z_axis, x_axis))

    t_oc = np.eye(4, dtype=np.float32)
    t_oc[:3, 0] = x_axis
    t_oc[:3, 1] = y_axis
    t_oc[:3, 2] = z_axis
    t_oc[:3, 3] = eye.astype(np.float32)
    return t_oc


def normalize_vertices_to_unit_bbox(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize vertices so the max bbox extent becomes one."""
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape [N, 3], got {vertices.shape}")

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = ((bbox_min + bbox_max) / 2.0).astype(np.float32)
    scale = float(np.max(bbox_max - bbox_min))
    if scale <= 0.0:
        raise ValueError("Cannot normalize a degenerate mesh with zero extent")

    normalized = ((vertices - center[None, :]) / scale).astype(np.float32)
    return normalized, center, scale


def normalize_mesh_to_unit_bbox(mesh):
    """Return a mesh copy in the canonical object frame plus normalization parameters."""
    normalized_vertices, center, scale = normalize_vertices_to_unit_bbox(np.asarray(mesh.vertices))
    normalized_mesh = mesh.copy()
    normalized_mesh.vertices = normalized_vertices
    return normalized_mesh, center, scale


def build_camera_intrinsics(image_size: int, fov_y_degrees: float = 70.0) -> np.ndarray:
    """Create a synthetic pinhole camera intrinsics matrix."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    fov_y_radians = np.deg2rad(float(fov_y_degrees))
    focal = (image_size / 2.0) / np.tan(fov_y_radians / 2.0)
    principal = (image_size - 1) / 2.0
    return np.array(
        [
            [focal, 0.0, principal],
            [0.0, focal, principal],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def generate_canonical_view_specs(radius: float = 1.5) -> list[CanonicalViewSpec]:
    """Return the 8 fixed cube-corner views in a stable order."""
    target = np.zeros(3, dtype=np.float32)
    default_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    specs: list[CanonicalViewSpec] = []
    for index, (sx, sy, sz, name) in enumerate(VIEW_SIGNS):
        sign = np.array([sx, sy, sz], dtype=np.float32)
        eye = radius * _normalize_vector(sign)
        view_dir = _normalize_vector(target - eye)
        up = default_up if abs(float(np.dot(view_dir, default_up))) < 0.99 else fallback_up
        t_oc = _build_t_oc_from_look_at(eye, target, up)
        t_co = np.linalg.inv(t_oc).astype(np.float32)
        specs.append(
            CanonicalViewSpec(
                index=index,
                name=name,
                sign=sign,
                eye=eye.astype(np.float32),
                target=target.copy(),
                up=up.copy(),
                t_co=t_co,
                t_oc=t_oc,
            )
        )
    return specs


def load_object_mesh(object_dir: str | Path):
    """Load DexYCB textured_simple.obj as a trimesh mesh."""
    import trimesh

    object_dir = Path(object_dir)
    mesh_file = object_dir / "textured_simple.obj"
    mesh = trimesh.load(mesh_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def configure_offscreen_backend(explicit_platform: str | None = None) -> str:
    """Select an offscreen OpenGL backend suitable for the current environment."""
    if explicit_platform:
        os.environ["PYOPENGL_PLATFORM"] = explicit_platform
        return explicit_platform

    configured = os.environ.get("PYOPENGL_PLATFORM")
    if configured:
        return configured

    if os.environ.get("DISPLAY"):
        return "pyglet"

    os.environ["PYOPENGL_PLATFORM"] = "egl"
    return "egl"


def default_light_rig() -> list[dict[str, object]]:
    """Return a simple three-point light rig in camera-local coordinates."""
    return [
        {
            "name": "key",
            "intensity": 3.8,
            "pose": _make_pose((0.45, 0.35, 0.25)),
        },
        {
            "name": "fill",
            "intensity": 1.6,
            "pose": _make_pose((-0.6, 0.15, 0.35)),
        },
        {
            "name": "rim",
            "intensity": 2.2,
            "pose": _make_pose((-0.25, -0.45, -0.55)),
        },
    ]


def render_canonical_views(
    mesh,
    view_specs: Iterable[CanonicalViewSpec],
    intrinsics: np.ndarray,
    image_size: int,
    pyopengl_platform: str | None = None,
    light_rig: list[dict[str, object]] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render RGB and canonical depth for the provided views."""
    configure_offscreen_backend(explicit_platform=pyopengl_platform)
    import pyrender

    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    renderer = pyrender.OffscreenRenderer(viewport_width=image_size, viewport_height=image_size)
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    lights = light_rig if light_rig is not None else default_light_rig()

    try:
        for index, spec in enumerate(view_specs):
            scene = pyrender.Scene(
                bg_color=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                ambient_light=np.array([0.28, 0.28, 0.28], dtype=np.float32),
            )
            scene.add(mesh_node, pose=np.eye(4, dtype=np.float32))

            K = np.asarray(intrinsics[index], dtype=np.float32)
            camera = pyrender.IntrinsicsCamera(
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
            )
            camera_pose = np.asarray(spec.t_oc, dtype=np.float32) @ _CV_TO_GL
            scene.add(camera, pose=camera_pose)
            for light in lights:
                scene.add(
                    pyrender.DirectionalLight(
                        color=np.ones(3, dtype=np.float32),
                        intensity=float(light["intensity"]),
                    ),
                    pose=camera_pose @ np.asarray(light["pose"], dtype=np.float32),
                )

            rgb, depth = renderer.render(scene)
            rgb_frames.append(np.asarray(rgb, dtype=np.uint8))
            depth_frames.append(_encode_depth_to_uint16(depth))
    finally:
        renderer.delete()

    return rgb_frames, depth_frames


def _encode_depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0.0)
    encoded[valid] = np.clip(np.round(depth[valid] * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return encoded


def _write_rgb_image(path: str | Path, image: np.ndarray) -> Path:
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(target), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise IOError(f"Failed to write RGB image: {target}")
    return target


def _write_depth_image(path: str | Path, depth: np.ndarray) -> Path:
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), np.asarray(depth, dtype=np.uint16)):
        raise IOError(f"Failed to write depth image: {target}")
    return target


def save_render_bundle(
    output_dir: str | Path,
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    view_specs: list[CanonicalViewSpec],
    intrinsics: np.ndarray,
    object_id: int | None,
    object_name: str,
    mesh_file: str | Path,
    image_size: int,
    radius: float,
    normalization_center: np.ndarray,
    normalization_scale: float,
) -> dict[str, object]:
    """Persist one object's canonical render bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(rgb_frames) != len(view_specs) or len(depth_frames) != len(view_specs):
        raise ValueError("Expected one RGB/depth frame per view spec")

    color_files = []
    depth_files = []
    for index, (rgb, depth, _) in enumerate(zip(rgb_frames, depth_frames, view_specs)):
        color_files.append(_write_rgb_image(output_dir / f"color_{index:06d}.jpg", rgb))
        depth_files.append(_write_depth_image(output_dir / f"aligned_depth_to_color_{index:06d}.png", depth))

    np.savez(
        output_dir / "camera_params.npz",
        K=np.asarray(intrinsics, dtype=np.float32),
        T_co=np.stack([spec.t_co for spec in view_specs], axis=0).astype(np.float32),
        T_oc=np.stack([spec.t_oc for spec in view_specs], axis=0).astype(np.float32),
        eye=np.stack([spec.eye for spec in view_specs], axis=0).astype(np.float32),
        target=np.stack([spec.target for spec in view_specs], axis=0).astype(np.float32),
        up=np.stack([spec.up for spec in view_specs], axis=0).astype(np.float32),
        view_dirs=np.stack([spec.sign for spec in view_specs], axis=0).astype(np.int8),
        view_names=np.array([spec.name for spec in view_specs]),
        normalization_center=np.asarray(normalization_center, dtype=np.float32),
        normalization_scale=np.asarray(normalization_scale, dtype=np.float32),
    )

    meta = {
        "object_id": object_id,
        "object_name": object_name,
        "mesh_file": str(mesh_file),
        "image_size": int(image_size),
        "camera_radius": float(radius),
        "view_names": [spec.name for spec in view_specs],
        "depth_encoding": "canonical_depth_times_1000_uint16_png",
        "coordinate_frame": "canonical_object_frame",
        "normalization": {
            "type": "axis_aligned_bbox_max_extent",
            "center": np.asarray(normalization_center, dtype=np.float32).tolist(),
            "scale": float(normalization_scale),
        },
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": output_dir,
        "color_files": color_files,
        "depth_files": depth_files,
        "camera_params_file": output_dir / "camera_params.npz",
        "meta_file": output_dir / "meta.json",
    }


def infer_object_id(object_name: str) -> int | None:
    reverse = {name: object_id for object_id, name in DexYCBDataset.ycb_classes.items()}
    return reverse.get(object_name)


def _intrinsics_batch(image_size: int, count: int, fov_y_degrees: float) -> np.ndarray:
    intrinsics = build_camera_intrinsics(image_size=image_size, fov_y_degrees=fov_y_degrees)
    return np.repeat(intrinsics[None, :, :], count, axis=0).astype(np.float32)


def generate_object_views(
    dataset_root: str | Path,
    object_name: str,
    image_size: int = 224,
    radius: float = 1.5,
    output_subdir: str = "canonical_views_224",
    overwrite: bool = False,
    fov_y_degrees: float = 70.0,
    pyopengl_platform: str | None = None,
    load_mesh: Callable[[str | Path], object] = load_object_mesh,
    renderer: Callable[..., tuple[list[np.ndarray], list[np.ndarray]]] = render_canonical_views,
    bundle_writer: Callable[..., dict[str, object]] = save_render_bundle,
) -> dict[str, object]:
    """Generate a canonical render bundle for a single DexYCB object."""
    dataset_root = Path(dataset_root)
    object_dir = dataset_root / "models" / object_name
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Object directory not found: {object_dir}")

    output_dir = object_dir / output_subdir
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    mesh = load_mesh(object_dir)
    normalized_mesh, center, scale = normalize_mesh_to_unit_bbox(mesh)
    view_specs = generate_canonical_view_specs(radius=radius)
    intrinsics = _intrinsics_batch(image_size=image_size, count=len(view_specs), fov_y_degrees=fov_y_degrees)
    rgb_frames, depth_frames = renderer(
        normalized_mesh,
        view_specs,
        intrinsics,
        image_size,
        pyopengl_platform=pyopengl_platform,
    )

    return bundle_writer(
        output_dir=output_dir,
        rgb_frames=rgb_frames,
        depth_frames=depth_frames,
        view_specs=view_specs,
        intrinsics=intrinsics,
        object_id=infer_object_id(object_name),
        object_name=object_name,
        mesh_file=object_dir / "textured_simple.obj",
        image_size=image_size,
        radius=radius,
        normalization_center=center,
        normalization_scale=scale,
    )


def iter_object_names(dataset_root: str | Path) -> list[str]:
    models_dir = Path(dataset_root) / "models"
    return sorted(path.name for path in models_dir.iterdir() if path.is_dir())


def generate_dataset_object_views(
    dataset_root: str | Path,
    object_names: Iterable[str] | None = None,
    **kwargs,
) -> list[dict[str, object]]:
    """Generate canonical render bundles for all or selected objects."""
    names = list(object_names) if object_names is not None else iter_object_names(dataset_root)
    return [generate_object_views(dataset_root=dataset_root, object_name=name, **kwargs) for name in names]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical DexYCB object renderings.")
    parser.add_argument("--dataset-root", required=True, help="Path to the DexYCB dataset root.")
    parser.add_argument("--object-name", default=None, help="Optional single object name, e.g. 025_mug.")
    parser.add_argument("--image-size", default=224, type=int, help="Square render size.")
    parser.add_argument("--radius", default=1.5, type=float, help="Canonical camera radius.")
    parser.add_argument(
        "--output-subdir",
        default="canonical_views_224",
        help="Output directory name created under each object directory.",
    )
    parser.add_argument("--fov-y-degrees", default=45.0, type=float, help="Synthetic vertical field of view.")
    parser.add_argument(
        "--pyopengl-platform",
        default=None,
        choices=("egl", "osmesa", "pyglet"),
        help="Override the OpenGL backend. Defaults to egl on headless servers.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    object_names = [args.object_name] if args.object_name else None
    generate_dataset_object_views(
        dataset_root=args.dataset_root,
        object_names=object_names,
        image_size=args.image_size,
        radius=args.radius,
        output_subdir=args.output_subdir,
        overwrite=args.overwrite,
        fov_y_degrees=args.fov_y_degrees,
        pyopengl_platform=args.pyopengl_platform,
    )


if __name__ == "__main__":
    main()
