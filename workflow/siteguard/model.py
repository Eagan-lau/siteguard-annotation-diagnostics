"""Frozen SiteGuard neural architecture and score blending."""

from __future__ import annotations

import numpy as np


def network_class():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency gate
        raise RuntimeError("Install siteguard-enzyme[inference] to load the deep model") from exc

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dropout: float) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(width * 2, width), nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(width)

        def forward(self, values):
            return self.norm(values + self.block(values))

    class SiteGuardGlobalNet(nn.Module):
        def __init__(self, input_features: int, width: int = 256, dropout: float = 0.15) -> None:
            super().__init__()
            self.input = nn.Sequential(nn.LayerNorm(input_features), nn.Linear(input_features, width), nn.GELU())
            self.trunk = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(3)])
            self.neck = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(128))
            self.head = nn.Linear(128, 3)

        def forward(self, values):
            return self.head(self.neck(self.trunk(self.input(values))))

    return SiteGuardGlobalNet


def blend_scores(deep: np.ndarray, tree: np.ndarray, weights: dict[str, float], levels: tuple[str, ...]) -> np.ndarray:
    if deep.shape != tree.shape or deep.shape[1] != len(levels):
        raise ValueError(f"Prediction shape mismatch: deep={deep.shape}, tree={tree.shape}, levels={levels}")
    output = np.empty_like(deep, dtype=np.float32)
    for index, level in enumerate(levels):
        alpha = float(weights[level])
        output[:, index] = alpha * deep[:, index] + (1.0 - alpha) * tree[:, index]
    return output
