from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi3.models.hamer.geometry import aa_to_rotmat, rot6d_to_rotmat
from pi3.models.hand_object_loss import HandObjectLoss


def _set_default_asset_env() -> None:
    data_root = REPO_ROOT / "data"
    model_root = data_root / "model"
    hamer_cache_dir = model_root / "hamer" / "_DATA"

    os.environ.setdefault("PI3_DATA_ROOT", str(data_root))
    os.environ.setdefault("PI3_MODEL_ROOT", str(model_root))
    os.environ.setdefault("HAMER_CACHE_DIR", str(hamer_cache_dir))


def _mean_pose_to_local_rotmat(mean_pose: np.ndarray) -> tuple[torch.Tensor, str]:
    pose = torch.from_numpy(mean_pose.astype(np.float32)).view(1, -1)
    dims = pose.shape[1]

    if dims == 96:
        pose_rotmat = rot6d_to_rotmat(pose.view(-1, 6)).view(1, 16, 3, 3)
        return pose_rotmat[:, 1:], "6d_16j_with_global"
    if dims == 48:
        pose_rotmat = aa_to_rotmat(pose.view(-1, 3)).view(1, 16, 3, 3)
        return pose_rotmat[:, 1:], "aa_16j_with_global"
    if dims == 45:
        pose_rotmat = aa_to_rotmat(pose.view(-1, 3)).view(1, 15, 3, 3)
        return pose_rotmat, "aa_15j_local_only"

    raise ValueError(f"Unsupported mean pose dimension: {dims}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect discriminator output for MANO mean pose/shape.")
    parser.add_argument(
        "--mean-params",
        default=str(REPO_ROOT / "data" / "model" / "hamer" / "_DATA" / "data" / "mano_mean_params.npz"),
        help="Path to mano_mean_params.npz",
    )
    parser.add_argument(
        "--discriminator-ckpt",
        default=str(REPO_ROOT / "data" / "model" / "hamer" / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"),
        help="Path to hamer.ckpt containing discriminator.* weights",
    )
    parser.add_argument(
        "--use-zero-betas",
        action="store_true",
        help="Use zeros(10) as shape input instead of mean_params['shape']",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to dump results as JSON.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _set_default_asset_env()

    mean_params_path = Path(args.mean_params).resolve()
    disc_ckpt_path = Path(args.discriminator_ckpt).resolve()

    mean_params = np.load(mean_params_path)
    mean_pose = mean_params["pose"]
    shape_key = "shape" if "shape" in mean_params else None
    if args.use_zero_betas or shape_key is None:
        betas = torch.zeros((1, 10), dtype=torch.float32)
        beta_source = "zeros"
    else:
        betas = torch.from_numpy(mean_params[shape_key].astype(np.float32)).view(1, 10)
        beta_source = shape_key

    hand_pose_rotmat, pose_format = _mean_pose_to_local_rotmat(mean_pose)

    loss_module = HandObjectLoss(
        hand_adversarial_weight=0.1,
        hand_discriminator_ckpt=str(disc_ckpt_path),
    )
    discriminator = loss_module.hand_discriminator
    if discriminator is None:
        raise RuntimeError("Failed to build hand discriminator")
    discriminator.eval()

    with torch.no_grad():
        disc_out = discriminator(hand_pose_rotmat, betas)
        prior_loss = ((disc_out - 1.0) ** 2).sum(dim=1)

    result = {
        "mean_params_path": str(mean_params_path),
        "discriminator_ckpt": str(disc_ckpt_path),
        "mean_pose_dim": int(mean_pose.shape[0]),
        "pose_format": pose_format,
        "betas_source": beta_source,
        "disc_out_shape": list(disc_out.shape),
        "disc_out": disc_out[0].detach().cpu().tolist(),
        "disc_out_mean": float(disc_out.mean().item()),
        "disc_out_min": float(disc_out.min().item()),
        "disc_out_max": float(disc_out.max().item()),
        "prior_loss_sum": float(prior_loss[0].item()),
    }

    print(json.dumps(result, indent=2))

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"json_out: {out_path}")


if __name__ == "__main__":
    main()
