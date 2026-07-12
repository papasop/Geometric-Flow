"""Small neural networks for first-week geometric optimizer proofs."""

from __future__ import annotations

import torch
from torch import nn

from .layers import GeometricRotation


class GeoMLP(nn.Module):
    """Minimal MLP with an optional parameter-free geometric rotation."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        use_rotation: bool = True,
        learnable_rotation: bool = True,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            GeometricRotation(hidden_dim, learnable=learnable_rotation) if use_rotation else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def geometric_parameters(self):
        """Return trainable phase parameters from geometric rotation layers."""

        return [
            module.angle
            for module in self.modules()
            if isinstance(module, GeometricRotation) and isinstance(module.angle, nn.Parameter)
        ]
