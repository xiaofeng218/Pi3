from __future__ import annotations


def apply_pi3x_training_policy(model) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_prefixes = (
        "hand_token_adapter",
        "object_query_adapter",
        "hand_mano_head",
        "object_pose_head",
    )

    for name, param in model.named_parameters():
        if name in {"register_token", "metric_token"}:
            param.requires_grad = True
        elif name.startswith(trainable_prefixes):
            param.requires_grad = True
        elif name.startswith("ho_decoder"):
            if "lora_" in name or "norm" in name or "ls" in name or name.endswith("bias"):
                param.requires_grad = True
