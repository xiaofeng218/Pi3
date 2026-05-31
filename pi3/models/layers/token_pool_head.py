from __future__ import annotations

import torch
import torch.nn as nn


class TokenPoolHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 1024):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens with shape (B, N, T, C), got {tuple(tokens.shape)}")

        logits = self.score(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        pooled = (weights * tokens).sum(dim=2)

        if valid_mask is not None:
            if valid_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"valid_mask must match token batch/view shape {tuple(tokens.shape[:2])}, got {tuple(valid_mask.shape)}"
                )
            pooled = pooled * valid_mask.unsqueeze(-1).to(dtype=pooled.dtype)

        return pooled
