from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pi3.models.hamer import hamer, hamer_encoder
from pi3.models.hamer.geometry import aa_to_rotmat, rot6d_to_rotmat

THIRD_PARTY_HAMER_ROOT = REPO_ROOT / "third_party" / "hamer"
if str(THIRD_PARTY_HAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_HAMER_ROOT))

from demo_mask_bbox_dexycb import default_checkpoint_path, ensure_default_example_data


def load_checkpoint_state_dict(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_obj = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        return checkpoint_obj["state_dict"]
    return checkpoint_obj


def remap_hamer_state_dict_for_encoder(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            mapped[f"backbone.vit.{key[len('backbone.') :]}"] = value
        elif key.startswith("mano_head.transformer."):
            mapped[f"backbone.transformer.{key[len('mano_head.transformer.') :]}"] = value
        elif key.startswith("mano_head.init_"):
            mapped[f"backbone.{key[len('mano_head.') :]}"] = value
    return mapped


def resolve_encoder_config_file(checkpoint: str | Path, config_file: str | Path | None = None) -> Path:
    if config_file is not None:
        return Path(config_file)
    checkpoint_path = Path(checkpoint)
    return checkpoint_path.parent.parent / "model_config.yaml"


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_bbox_from_mask(mask: np.ndarray) -> list[float]:
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Hand mask is empty; cannot compute bounding box")
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def build_encoder_inputs(
    image_path: str | Path,
    mask_path: str | Path,
    hand_side: str,
    device: torch.device,
) -> dict[str, torch.Tensor | list[float]]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)

    image_np = np.array(image, dtype=np.float32) / 255.0
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = np.any(mask_np > 0, axis=2).astype(np.uint8) * 255

    imgs = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    hand_masks = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0).to(device=device, dtype=torch.float32)
    owner_index = torch.tensor([[0, 0, 0]], dtype=torch.long, device=device)
    hand_is_right = torch.tensor([hand_side == "right"], dtype=torch.bool, device=device)

    return {
        "imgs": imgs,
        "hand_masks": hand_masks,
        "owner_index": owner_index,
        "hand_is_right": hand_is_right,
        "bbox_xyxy": compute_bbox_from_mask(mask_np),
    }


def load_encoder_for_demo(
    checkpoint: str | Path | None = None,
    config_file: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint) if checkpoint is not None else default_checkpoint_path()
    config_path = resolve_encoder_config_file(checkpoint=checkpoint_path, config_file=config_file)
    runtime_device = torch.device(device) if not isinstance(device, torch.device) else device

    model = hamer_encoder(
        config_file=str(config_path),
        pretrained=False,
        cache_dir=str(THIRD_PARTY_HAMER_ROOT / "_DATA"),
        eval_mode=True,
    )
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    remapped = remap_hamer_state_dict_for_encoder(state_dict)
    model.load_state_dict(remapped, strict=False)
    model = model.to(runtime_device)
    model.eval()
    return model


def load_prediction_model_for_demo(
    checkpoint: str | Path | None = None,
    config_file: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint) if checkpoint is not None else default_checkpoint_path()
    config_path = resolve_encoder_config_file(checkpoint=checkpoint_path, config_file=config_file)
    runtime_device = torch.device(device) if not isinstance(device, torch.device) else device
    model = hamer(
        checkpoint=str(checkpoint_path),
        config_file=str(config_path),
        pretrained=True,
        cache_dir=str(THIRD_PARTY_HAMER_ROOT / "_DATA"),
        eval_mode=True,
    )
    model = model.to(runtime_device)
    model.eval()
    return model


def prepare_encoder_crops(
    encoder: torch.nn.Module,
    imgs: torch.Tensor,
    hand_masks: torch.Tensor,
    owner_index: torch.Tensor,
    hand_is_right: torch.Tensor,
) -> dict[str, torch.Tensor]:
    hand_masks = encoder._normalize_mask_shape(hand_masks)
    if hand_masks.shape[0] != owner_index.shape[0] or hand_masks.shape[0] != hand_is_right.shape[0]:
        raise ValueError("hand_masks, owner_index, and hand_is_right must align row-wise")

    _, _, _, image_h, image_w = imgs.shape
    valid_boxes = []
    valid_owner = []
    valid_right = []
    normalized_crops = []
    visual_crops = []

    for idx in range(hand_masks.shape[0]):
        mask = hand_masks[idx]
        if int((mask > 0).sum().item()) < encoder.min_mask_area:
            continue

        b, n, _ = owner_index[idx].tolist()
        if not (0 <= b < imgs.shape[0] and 0 <= n < imgs.shape[1]):
            raise ValueError("owner_index points outside imgs")

        box = encoder._expand_box(encoder._compute_bbox_from_mask(mask), image_h, image_w)
        crop = encoder._crop_and_resize(imgs[b, n], box)
        crop = encoder._canonicalize_handedness(crop, hand_is_right[idx])
        visual_crops.append(crop)
        normalized_crops.append(encoder._normalize_input(crop))
        valid_boxes.append(box)
        valid_owner.append(owner_index[idx])
        valid_right.append(hand_is_right[idx])

    if not normalized_crops:
        empty_boxes = imgs.new_zeros((0, 4))
        return {
            "normalized_crops": imgs.new_zeros((0, imgs.shape[2], 256, 192)),
            "visual_crops": imgs.new_zeros((0, imgs.shape[2], 256, 192)),
            "owner_index": owner_index.new_zeros((0, 3)),
            "hand_is_right": hand_is_right.new_zeros((0,), dtype=hand_is_right.dtype),
            "crop_boxes": empty_boxes,
        }

    return {
        "normalized_crops": torch.stack(normalized_crops, dim=0),
        "visual_crops": torch.stack(visual_crops, dim=0),
        "owner_index": torch.stack(valid_owner),
        "hand_is_right": torch.stack(valid_right),
        "crop_boxes": torch.stack(valid_boxes),
    }


def save_encoder_crops(visual_crops: torch.Tensor, out_dir: str | Path, stem: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = []
    for idx in range(visual_crops.shape[0]):
        crop = visual_crops[idx].detach().cpu().clamp(0.0, 1.0)
        crop_image = (crop.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        crop_path = out_dir / f"{stem}_encoder_crop_{idx:02d}.png"
        Image.fromarray(crop_image).save(crop_path)
        crop_paths.append(crop_path)
    return crop_paths


def predict_mano_from_queries(model: torch.nn.Module, hand_queries: torch.Tensor) -> dict[str, torch.Tensor]:
    mano_head = model.mano_head
    i_iters = int(model.cfg.MODEL.MANO_HEAD.get("IEF_ITERS", 1))
    if i_iters != 1:
        raise NotImplementedError(f"Only IEF_ITERS=1 is supported in this debug script, got {i_iters}")
    if bool(mano_head.input_is_mean_shape):
        raise NotImplementedError("Only TRANSFORMER_INPUT=zero is supported in this debug script")

    batch_size = hand_queries.shape[0]
    pred_hand_pose = mano_head.init_hand_pose.expand(batch_size, -1)
    pred_betas = mano_head.init_betas.expand(batch_size, -1)
    pred_cam = mano_head.init_cam.expand(batch_size, -1)

    pred_hand_pose = mano_head.decpose(hand_queries) + pred_hand_pose
    pred_betas = mano_head.decshape(hand_queries) + pred_betas
    pred_cam = mano_head.deccam(hand_queries) + pred_cam

    joint_conversion_fn = {
        "6d": rot6d_to_rotmat,
        "aa": lambda x: aa_to_rotmat(x.view(-1, 3).contiguous()),
    }[mano_head.joint_rep_type]
    pred_hand_pose_rotmat = joint_conversion_fn(pred_hand_pose).view(batch_size, model.cfg.MANO.NUM_HAND_JOINTS + 1, 3, 3)

    pred_mano_params = {
        "global_orient": pred_hand_pose_rotmat[:, [0]],
        "hand_pose": pred_hand_pose_rotmat[:, 1:],
        "betas": pred_betas,
    }

    output = {
        "pred_cam": pred_cam,
        "pred_mano_params": {key: value.clone() for key, value in pred_mano_params.items()},
    }

    device = pred_hand_pose.device
    dtype = pred_hand_pose.dtype
    focal_length = model.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
    pred_cam_t = torch.stack(
        [
            pred_cam[:, 1],
            pred_cam[:, 2],
            2 * focal_length[:, 0] / (model.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
        ],
        dim=-1,
    )
    output["pred_cam_t"] = pred_cam_t
    output["focal_length"] = focal_length

    mano_output = model.mano.forward_rotmat(
        global_orient=pred_mano_params["global_orient"].float(),
        hand_pose=pred_mano_params["hand_pose"].float(),
        betas=pred_mano_params["betas"].float(),
        th_trans=pred_cam_t.reshape(-1, 3).float(),
    )
    output["pred_keypoints_3d"] = mano_output.joints.reshape(batch_size, -1, 3)
    output["pred_vertices"] = mano_output.vertices.reshape(batch_size, -1, 3)
    return output


def render_mesh_visualizations(
    model: torch.nn.Module,
    predictions: dict[str, torch.Tensor],
    normalized_crops: torch.Tensor,
    crop_boxes: torch.Tensor,
    hand_is_right: torch.Tensor,
    image_path: str | Path,
    out_dir: str | Path,
    stem: str,
) -> dict[str, Path]:
    from hamer.utils.renderer import Renderer, cam_crop_to_full

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(model.cfg, faces=model.mano.faces)

    multiplier = (2 * hand_is_right.float() - 1).to(device=predictions["pred_cam"].device, dtype=predictions["pred_cam"].dtype)
    pred_cam = predictions["pred_cam"].clone()
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]

    mesh_crop_paths = []
    mesh_full_paths = []
    image = Image.open(image_path).convert("RGB")
    image_size = torch.tensor([[image.width, image.height]], dtype=pred_cam.dtype, device=pred_cam.device).expand(pred_cam.shape[0], -1)
    box_center = torch.stack([(crop_boxes[:, 0] + crop_boxes[:, 2]) * 0.5, (crop_boxes[:, 1] + crop_boxes[:, 3]) * 0.5], dim=1)
    box_size = torch.maximum(crop_boxes[:, 2] - crop_boxes[:, 0], crop_boxes[:, 3] - crop_boxes[:, 1])
    scaled_focal_length = model.cfg.EXTRA.FOCAL_LENGTH / model.cfg.MODEL.IMAGE_SIZE * image_size.max(dim=1).values
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, image_size, scaled_focal_length)

    for idx in range(predictions["pred_vertices"].shape[0]):
        crop_render = renderer(
            predictions["pred_vertices"][idx].detach().cpu().numpy(),
            predictions["pred_cam_t"][idx].detach().cpu().numpy(),
            normalized_crops[idx],
            mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
            scene_bg_color=(1, 1, 1),
        )
        crop_render_path = out_dir / f"{stem}_mesh_render_crop_{idx:02d}.png"
        Image.fromarray((np.clip(crop_render, 0.0, 1.0) * 255.0).astype(np.uint8)).save(crop_render_path)
        mesh_crop_paths.append(crop_render_path)

        full_render = renderer(
            predictions["pred_vertices"][idx].detach().cpu().numpy(),
            pred_cam_t_full[idx].detach().cpu().numpy(),
            normalized_crops[idx],
            full_frame=True,
            imgname=str(image_path),
            mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
            scene_bg_color=(1, 1, 1),
        )
        full_render_path = out_dir / f"{stem}_mesh_render_full_{idx:02d}.png"
        Image.fromarray((np.clip(full_render, 0.0, 1.0) * 255.0).astype(np.uint8)).save(full_render_path)
        mesh_full_paths.append(full_render_path)

    return {
        "mesh_crop_path": mesh_crop_paths[0] if len(mesh_crop_paths) == 1 else mesh_crop_paths,
        "mesh_full_path": mesh_full_paths[0] if len(mesh_full_paths) == 1 else mesh_full_paths,
    }


def save_visualization(
    image_path: str | Path,
    mask_path: str | Path,
    bbox_xyxy: list[float],
    out_path: str | Path,
) -> Path:
    image = Image.open(image_path).convert("RGBA")
    mask_np = np.array(Image.open(mask_path))
    if mask_np.ndim == 3:
        mask_np = np.any(mask_np > 0, axis=2).astype(np.uint8) * 255
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_np = np.array(overlay)
    overlay_np[mask_np > 0] = np.array([0, 255, 0, 80], dtype=np.uint8)
    overlay = Image.fromarray(overlay_np, mode="RGBA")
    vis = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(vis)
    draw.rectangle(bbox_xyxy, outline=(255, 0, 0, 255), width=3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.convert("RGB").save(out_path)
    return out_path


def run_demo(args) -> dict[str, Path]:
    if args.prepare_example_data:
        exported = ensure_default_example_data(
            data_root=args.data_root,
            example_dir=args.example_dir,
        )
        if args.image_path is None:
            args.image_path = str(exported["image_path"])
        if args.mask_path is None:
            args.mask_path = str(exported["mask_path"])

    if args.image_path is None or args.mask_path is None:
        raise ValueError("image_path and mask_path must be provided, or enable --prepare-example-data")

    device = select_device(args.device)
    batch = build_encoder_inputs(
        image_path=args.image_path,
        mask_path=args.mask_path,
        hand_side=args.hand_side,
        device=device,
    )

    model = load_encoder_for_demo(
        checkpoint=args.checkpoint,
        config_file=args.config_file,
        device=device,
    )
    prepared = prepare_encoder_crops(
        model,
        batch["imgs"],
        batch["hand_masks"],
        batch["owner_index"],
        batch["hand_is_right"],
    )
    with torch.no_grad():
        outputs = model(
            batch["imgs"],
            batch["hand_masks"],
            batch["owner_index"],
            batch["hand_is_right"],
        )

    out_dir = Path(args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image_path).stem

    query_path = out_dir / f"{stem}_encoder_queries.pt"
    meta_path = out_dir / f"{stem}_encoder_meta.json"
    vis_path = out_dir / f"{stem}_encoder_vis.png"
    crop_paths = save_encoder_crops(prepared["visual_crops"], out_dir, stem)

    prediction_model = load_prediction_model_for_demo(
        checkpoint=args.checkpoint,
        config_file=args.config_file,
        device=device,
    )
    with torch.no_grad():
        predictions = predict_mano_from_queries(prediction_model, outputs["hand_queries"])
    render_paths = render_mesh_visualizations(
        prediction_model,
        predictions,
        prepared["normalized_crops"],
        outputs["crop_boxes"],
        outputs["hand_is_right"],
        args.image_path,
        out_dir,
        stem,
    )

    serializable_outputs = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            serializable_outputs[key] = value.detach().cpu()
        else:
            serializable_outputs[key] = value
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            serializable_outputs[key] = value.detach().cpu()
        elif isinstance(value, dict):
            serializable_outputs[key] = {
                sub_key: sub_value.detach().cpu() if isinstance(sub_value, torch.Tensor) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            serializable_outputs[key] = value
    torch.save(serializable_outputs, query_path)

    metadata = {
        "image_path": str(Path(args.image_path).resolve()),
        "mask_path": str(Path(args.mask_path).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint is not None else str(default_checkpoint_path()),
        "hand_side": args.hand_side,
        "bbox_xyxy": batch["bbox_xyxy"],
        "num_hands": int(serializable_outputs["hand_queries"].shape[0]),
        "query_shape": list(serializable_outputs["hand_queries"].shape),
        "query_l2_norms": torch.linalg.vector_norm(serializable_outputs["hand_queries"], dim=1).tolist(),
        "owner_index": serializable_outputs["owner_index"].tolist(),
        "hand_is_right": serializable_outputs["hand_is_right"].tolist(),
        "crop_boxes": serializable_outputs["crop_boxes"].tolist(),
        "encoder_crop_paths": [str(path) for path in crop_paths],
        "mesh_crop_path": str(render_paths["mesh_crop_path"]) if isinstance(render_paths["mesh_crop_path"], Path) else [str(path) for path in render_paths["mesh_crop_path"]],
        "mesh_full_path": str(render_paths["mesh_full_path"]) if isinstance(render_paths["mesh_full_path"], Path) else [str(path) for path in render_paths["mesh_full_path"]],
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    save_visualization(args.image_path, args.mask_path, batch["bbox_xyxy"], vis_path)

    return {
        "query_path": query_path,
        "meta_path": meta_path,
        "vis_path": vis_path,
        "crop_path": crop_paths[0] if len(crop_paths) == 1 else crop_paths,
        "mesh_crop_path": render_paths["mesh_crop_path"],
        "mesh_full_path": render_paths["mesh_full_path"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HaMeREncoder on one RGB image and one hand mask.")
    parser.add_argument("--image-path", type=str, default=None, help="RGB image path")
    parser.add_argument("--mask-path", type=str, default=None, help="Binary hand mask path")
    parser.add_argument("--hand-side", choices=["left", "right"], default="right")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint_path()))
    parser.add_argument("--config-file", type=str, default=None, help="Optional HaMeR model_config.yaml")
    parser.add_argument("--out-folder", type=str, default=str(REPO_ROOT / "debug" / "out_hamer_encoder"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prepare-example-data", action="store_true")
    parser.add_argument("--example-dir", type=str, default=str(THIRD_PARTY_HAMER_ROOT / "example_data"))
    parser.add_argument("--data-root", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    outputs = run_demo(parse_args())
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, ensure_ascii=True))
