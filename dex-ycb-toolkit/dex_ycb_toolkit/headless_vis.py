"""Headless visualization helpers for DexYCB samples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

from dex_ycb_toolkit.dex_ycb import DexYCBDataset

BACKGROUND_SEG_COLOR = (0, 0, 0)
HAND_SEG_COLOR = (255, 255, 255)
JOINT_COLOR = (255, 64, 64)
BONE_COLOR = (64, 255, 64)

_OBJECT_SEG_COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (128, 0, 0),
    8: (0, 128, 0),
    9: (0, 0, 128),
    10: (128, 128, 0),
    11: (128, 0, 128),
    12: (0, 128, 128),
    13: (64, 0, 0),
    14: (0, 64, 0),
    15: (0, 0, 64),
    16: (64, 64, 0),
    17: (64, 0, 64),
    18: (0, 64, 64),
    19: (192, 0, 0),
    20: (0, 192, 0),
    21: (0, 0, 192),
}


def colorize_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Convert a DexYCB label map into an RGB visualization."""
    seg = segmentation.astype(np.uint8, copy=False)
    colored = np.zeros(seg.shape + (3,), dtype=np.uint8)
    colored[:] = np.array(BACKGROUND_SEG_COLOR, dtype=np.uint8)

    for obj_id, color in _OBJECT_SEG_COLORS.items():
        colored[seg == obj_id] = color

    colored[seg == 255] = HAND_SEG_COLOR
    return colored


def draw_hand_joints(image: np.ndarray, joints_2d: np.ndarray) -> np.ndarray:
    """Draw MANO joints and bones on an RGB image."""
    canvas = image.copy()
    joints = np.asarray(joints_2d, dtype=np.float32)
    if joints.ndim == 3:
        joints = joints[0]

    valid = np.all(joints >= 0, axis=1)
    for start, end in DexYCBDataset.mano_joint_connect:
        if valid[start] and valid[end]:
            p0 = tuple(np.round(joints[start]).astype(int))
            p1 = tuple(np.round(joints[end]).astype(int))
            cv2.line(canvas, p0, p1, BONE_COLOR, 2, lineType=cv2.LINE_AA)

    for point, is_valid in zip(joints, valid):
        if not is_valid:
            continue
        center = tuple(np.round(point).astype(int))
        cv2.circle(canvas, center, 3, JOINT_COLOR, -1, lineType=cv2.LINE_AA)

    return canvas


def compose_overlay(
    color_image: np.ndarray,
    segmentation_rgb: np.ndarray,
    joints_2d: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend RGB image with segmentation and draw joints on top."""
    blended = cv2.addWeighted(color_image, 1.0 - alpha, segmentation_rgb, alpha, 0.0)
    return draw_hand_joints(blended, joints_2d)


def compose_foreground_overlay(
    color_image: np.ndarray,
    segmentation_rgb: np.ndarray,
    segmentation: np.ndarray,
    joints_2d: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend RGB with segmentation but keep only hand/object foreground."""
    overlay = compose_overlay(color_image, segmentation_rgb, joints_2d, alpha=alpha)
    foreground = np.zeros_like(overlay)
    mask = segmentation > 0
    foreground[mask] = overlay[mask]
    return foreground


def load_label_data(label_file: str | Path) -> Dict[str, np.ndarray]:
    """Load raw arrays from a DexYCB label file."""
    label = np.load(str(label_file))
    return {
        "segmentation": label["seg"],
        "joints_2d": label["joint_2d"],
    }


def load_sample_visuals(color_file: str | Path, label_file: str | Path) -> Dict[str, np.ndarray]:
    """Load RGB image and label file and return visualization images."""
    color_path = Path(color_file)

    image_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read color image: {color_path}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    label = load_label_data(label_file)
    segmentation = label["segmentation"]
    joints_2d = label["joints_2d"]

    seg_rgb = colorize_segmentation(segmentation)
    joints_rgb = draw_hand_joints(rgb, joints_2d)
    overlay_rgb = compose_overlay(rgb, seg_rgb, joints_2d)
    foreground_overlay_rgb = compose_foreground_overlay(rgb, seg_rgb, segmentation, joints_2d)

    return {
        "rgb": rgb,
        "seg": seg_rgb,
        "joints": joints_rgb,
        "overlay": overlay_rgb,
        "foreground_overlay": foreground_overlay_rgb,
    }


def save_visualizations(images: Dict[str, np.ndarray], output_dir: str | Path, prefix: str) -> Dict[str, Path]:
    """Save RGB visualization images to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}
    for name, image in images.items():
        target = output_path / f"{prefix}_{name}.png"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(target), image_bgr):
            raise IOError(f"Failed to write visualization: {target}")
        saved_paths[name] = target

    return saved_paths


def get_frame_pairs(camera_dir: str | Path) -> List[tuple[Path, Path]]:
    """Return sorted matching color/label frame pairs for one camera directory."""
    base = Path(camera_dir)
    color_files = sorted(base.glob("color_*.jpg"))
    pairs: List[tuple[Path, Path]] = []
    for color_file in color_files:
        frame_id = color_file.stem.split("_")[-1]
        label_file = base / f"labels_{frame_id}.npz"
        if label_file.is_file():
            pairs.append((color_file, label_file))
    if not pairs:
        raise FileNotFoundError(f"No matching color_/labels_ frame pairs found in {base}")
    return pairs


def save_video(frames: Iterable[np.ndarray], path: str | Path, fps: int = 15) -> Path:
    """Save RGB frames to an mp4 video."""
    frames = list(frames)
    if not frames:
        raise ValueError("No frames provided for video export")

    first = frames[0]
    height, width = first.shape[:2]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise IOError(f"Failed to open video writer: {target}")

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All frames must share the same spatial resolution")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return target


def save_sequence_videos(
    camera_dir: str | Path,
    output_dir: str | Path,
    prefix: str | None = None,
    fps: int = 15,
) -> Dict[str, Path]:
    """Generate and save visualization videos for one camera directory."""
    pairs = get_frame_pairs(camera_dir)
    stem = prefix or Path(camera_dir).name

    frame_store: Dict[str, List[np.ndarray]] = {
        "rgb": [],
        "seg": [],
        "joints": [],
        "overlay": [],
        "foreground_overlay": [],
    }
    for color_file, label_file in pairs:
        visuals = load_sample_visuals(color_file, label_file)
        for name, image in visuals.items():
            frame_store[name].append(image)

    saved: Dict[str, Path] = {}
    for name, frames in frame_store.items():
        saved[name] = save_video(frames, Path(output_dir) / f"{stem}_{name}.mp4", fps=fps)
    return saved
