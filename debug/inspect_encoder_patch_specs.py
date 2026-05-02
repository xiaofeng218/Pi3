from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi3.models.hamer.load import load_hamer
from pi3.models.pi3x import Pi3X


def normalize_patch_size(patch_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if len(patch_size) != 2:
        raise ValueError(f"Expected patch_size with 2 dims, got {patch_size}")
    return (int(patch_size[0]), int(patch_size[1]))


def summarize_pi3_encoder_output(
    output: dict[str, torch.Tensor],
    image_hw: tuple[int, int],
    patch_hw: tuple[int, int],
) -> dict[str, Any]:
    patch_tokens = output["x_norm_patchtokens"]
    image_h, image_w = image_hw
    patch_h, patch_w = patch_hw
    return {
        "token_shape": list(patch_tokens.shape),
        "token_grid_hw": [image_h // patch_h, image_w // patch_w],
        "token_count": int(patch_tokens.shape[1]),
        "embed_dim": int(patch_tokens.shape[2]),
        "register_token_count": int(output.get("x_norm_regtokens", torch.empty(0)).shape[1])
        if "x_norm_regtokens" in output
        else 0,
    }


def summarize_hamer_backbone_output(
    output: torch.Tensor,
    patch_hw: tuple[int, int],
) -> dict[str, Any]:
    batch, channels, grid_h, grid_w = output.shape
    patch_h, patch_w = patch_hw
    return {
        "feature_shape": [batch, channels, grid_h, grid_w],
        "token_grid_hw": [grid_h, grid_w],
        "token_count": int(grid_h * grid_w),
        "embed_dim": int(channels),
        "effective_input_hw": [grid_h * patch_h, grid_w * patch_w],
    }


def default_hamer_checkpoint() -> Path:
    return Path("/data/hanxiaofeng/models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt")


def resolve_device(device_str: str, cuda_available: bool | None = None) -> torch.device:
    if device_str.startswith("cuda"):
        if cuda_available is None:
            cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise RuntimeError("No CUDA GPUs are available in the current environment.")
    return torch.device(device_str)


def load_pi3x_model(device: torch.device, ckpt: str | None) -> tuple[Pi3X, str]:
    if ckpt is not None:
        ckpt_path = Path(ckpt).expanduser().resolve()
        model = Pi3X(use_multimodal=True).eval()
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(ckpt_path))
        else:
            state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        return model, str(ckpt_path)

    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to(device)
    return model, "hf://yyfz233/Pi3X"


def inspect_pi3x(model: Pi3X, device: torch.device, image_size: int) -> dict[str, Any]:
    dummy = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32, device=device)
    patch_hw = normalize_patch_size(model.encoder.patch_size)
    with torch.no_grad():
        output = model.encoder(dummy, is_training=True)
    return {
        "declared_patch_size_hw": list(normalize_patch_size(model.patch_size)),
        "encoder_patch_size_hw": list(patch_hw),
        "input_hw": [image_size, image_size],
        **summarize_pi3_encoder_output(output, image_hw=(image_size, image_size), patch_hw=patch_hw),
    }


def inspect_hamer(model, image_size: int, device: torch.device) -> dict[str, Any]:
    patch_hw = normalize_patch_size(model.backbone.patch_embed.patch_size)
    dummy = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32, device=device)
    cropped = dummy[:, :, :, 32:-32]
    with torch.no_grad():
        output = model.backbone(cropped)
    return {
        "declared_patch_size_hw": list(patch_hw),
        "full_input_hw": [image_size, image_size],
        "cropped_input_hw": list(cropped.shape[-2:]),
        **summarize_hamer_backbone_output(output, patch_hw=patch_hw),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load Pi3X and HaMeR with real weights and inspect their patch/token specs."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pi3-ckpt", type=str, default=None, help="Optional local Pi3X checkpoint path.")
    parser.add_argument(
        "--hamer-ckpt",
        type=str,
        default=str(default_hamer_checkpoint()),
        help="Local HaMeR checkpoint path.",
    )
    parser.add_argument("--pi3-image-size", type=int, default=224)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)

    pi3_model, pi3_source = load_pi3x_model(device=device, ckpt=args.pi3_ckpt)
    hamer_model, hamer_cfg, incompatible = load_hamer(
        args.hamer_ckpt,
        cache_dir=str(Path(args.hamer_ckpt).resolve().parent.parent.parent),
        map_location=device,
        strict=False,
    )
    hamer_model = hamer_model.eval().to(device)

    summary = {
        "pi3x": {
            "weight_source": pi3_source,
            **inspect_pi3x(pi3_model, device=device, image_size=args.pi3_image_size),
        },
        "hamer": {
            "weight_source": str(Path(args.hamer_ckpt).resolve()),
            "image_size": int(hamer_cfg.MODEL.IMAGE_SIZE),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            **inspect_hamer(hamer_model, image_size=int(hamer_cfg.MODEL.IMAGE_SIZE), device=device),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
