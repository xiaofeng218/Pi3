from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def project_points_cam_to_image(points_cam: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_cam = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & np.isfinite(z) & (z > 1e-8)
    uv = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    if valid.any():
        xyz = points_cam[valid]
        uv_valid = xyz[:, :2] / xyz[:, 2:3]
        uv_valid[:, 0] = uv_valid[:, 0] * intrinsics[0, 0] + intrinsics[0, 2]
        uv_valid[:, 1] = uv_valid[:, 1] * intrinsics[1, 1] + intrinsics[1, 2]
        uv[valid] = uv_valid
    return uv, valid


def map_points_between_intrinsics(
    points_uv: np.ndarray,
    intrinsics_src: np.ndarray,
    intrinsics_dst: np.ndarray,
) -> np.ndarray:
    points_uv = np.asarray(points_uv, dtype=np.float32).reshape(-1, 2)
    src = np.asarray(intrinsics_src, dtype=np.float32).reshape(3, 3)
    dst = np.asarray(intrinsics_dst, dtype=np.float32).reshape(3, 3)
    out = np.full_like(points_uv, np.nan, dtype=np.float32)
    valid = np.isfinite(points_uv).all(axis=1) & (points_uv[:, 0] >= 0) & (points_uv[:, 1] >= 0)
    if valid.any():
        pts = points_uv[valid]
        x = (pts[:, 0] - src[0, 2]) / src[0, 0]
        y = (pts[:, 1] - src[1, 2]) / src[1, 1]
        out_valid = np.stack(
            [
                x * dst[0, 0] + dst[0, 2],
                y * dst[1, 1] + dst[1, 2],
            ],
            axis=1,
        )
        out[valid] = out_valid
    return out


def compute_2d_point_errors(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    points_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
    if points_a.shape != points_b.shape:
        raise ValueError(f"point shapes must match, got {points_a.shape} vs {points_b.shape}")
    valid = np.isfinite(points_a).all(axis=1) & np.isfinite(points_b).all(axis=1)
    errors = np.full((points_a.shape[0],), np.nan, dtype=np.float32)
    if valid.any():
        errors[valid] = np.linalg.norm(points_a[valid] - points_b[valid], axis=1)
    return errors, valid


def load_obj_vertices(obj_path: str | Path) -> np.ndarray:
    obj_path = Path(obj_path)
    vertices = []
    with obj_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertices found in OBJ file: {obj_path}")
    return np.asarray(vertices, dtype=np.float32)


def project_points_cam_to_image_torch(
    points_cam: torch.Tensor,
    intrinsics: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points_cam.shape[-1] != 3:
        raise ValueError(f"points_cam last dim must be 3, got {points_cam.shape}")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics last dims must be 3x3, got {intrinsics.shape}")

    z = points_cam[..., 2]
    valid = torch.isfinite(points_cam).all(dim=-1) & torch.isfinite(z) & (z > eps)
    uv = torch.full(points_cam.shape[:-1] + (2,), float("nan"), dtype=points_cam.dtype, device=points_cam.device)

    x = points_cam[..., 0] / points_cam[..., 2].clamp_min(eps)
    y = points_cam[..., 1] / points_cam[..., 2].clamp_min(eps)
    fx = intrinsics[..., 0, 0].unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1)
    u = x * fx + cx
    v = y * fy + cy
    uv_valid = torch.stack([u, v], dim=-1)
    uv = torch.where(valid.unsqueeze(-1), uv_valid, uv)
    return uv, valid
