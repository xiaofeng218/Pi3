from __future__ import annotations

import torch.nn as nn

from pi3.models.layers.lora import LoRALinear


def _apply_lora_to_linear(module: nn.Module, attr_name: str, rank: int, alpha: float) -> None:
    """Replace an nn.Linear attribute with LoRALinear in-place."""
    linear = getattr(module, attr_name, None)
    if linear is not None and isinstance(linear, nn.Linear):
        setattr(module, attr_name, LoRALinear.from_linear(linear, rank=rank, alpha=alpha))


def _apply_lora_to_ho_decoder(model, rank: int = 4, alpha: float = 8.0) -> None:
    """Apply LoRA to self-attention and FFN Linear layers in ho_decoder.

    Self-attention: qkv (nn.Linear), proj (nn.Linear) — LoRA.
    Cross-attention: q_proj, k_proj, v_proj, proj — full fine-tune (no LoRA).
    FFN: fc1, fc2 (nn.Linear) — LoRA.
    Norms and layer scales: full fine-tune (no LoRA).
    """
    for ho_blk in model.ho_decoder:
        # Self-attention
        attn = ho_blk.attn
        _apply_lora_to_linear(attn, "qkv", rank=rank, alpha=alpha)
        _apply_lora_to_linear(attn, "proj", rank=rank, alpha=alpha)

        # FFN
        mlp = ho_blk.mlp
        _apply_lora_to_linear(mlp, "fc1", rank=rank, alpha=alpha)
        _apply_lora_to_linear(mlp, "fc2", rank=rank, alpha=alpha)


def apply_pi3x_training_policy(
    model,
    ho_decoder_lora_rank: int = 4,
    ho_decoder_lora_alpha: float = 8.0,
) -> None:
    # 1. Freeze everything
    for _, param in model.named_parameters():
        param.requires_grad = False

    # 2. Unfreeze new head and adapter modules (full fine-tune)
    new_head_prefixes = (
        "hand_token_adapter",
        "object_query_adapter",
        "hand_global_decoder",
        "hand_pose_decoder",
        "object_pose_decoder",
        "hand_global_head",
        "hand_pose_head",
        "hand_mano_head",
        "object_pose_head",
        "hand_token_fuse",
        "omv_point_decoder",
        "omv_point_head",
        "omv_camera_decoder",
        "omv_camera_head",
    )
    for name, param in model.named_parameters():
        if name.startswith(new_head_prefixes):
            param.requires_grad = True

    # 3. Unfreeze learnable cross-attention alpha scalars
    for name, param in model.named_parameters():
        if name.startswith(("ho_hand_scene_cross_alpha", "ho_object_scene_cross_alpha")):
            param.requires_grad = True

    # 4. ho_decoder: unfreeze all, then apply LoRA to self-attn + FFN.
    #    Cross-attn, norms, and layer scales remain full fine-tune.
    for name, param in model.named_parameters():
        if name.startswith("ho_decoder"):
            param.requires_grad = True

    _apply_lora_to_ho_decoder(model, rank=ho_decoder_lora_rank, alpha=ho_decoder_lora_alpha)

    # 5. Keep MANO layer frozen (it sits inside hand_mano_head)
    for name, param in model.named_parameters():
        if name.startswith("hand_mano_head.mano"):
            param.requires_grad = False
