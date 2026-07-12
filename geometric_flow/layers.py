"""Geometry-aware layers."""

from __future__ import annotations

import torch
from torch import nn


class GeometricRotation(nn.Module):
    """Pairwise phase rotation that reshapes flow without changing feature dim."""

    def __init__(self, dim: int, angle: float = 0.125, learnable: bool = True) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.dim = dim
        initial = torch.tensor(float(angle))
        if learnable:
            self.angle = nn.Parameter(initial)
        else:
            self.register_buffer("angle", initial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        if even.shape[-1] != odd.shape[-1]:
            odd = torch.nn.functional.pad(odd, (0, 1))
        angle = self.angle.to(dtype=x.dtype, device=x.device)
        c = torch.cos(angle)
        s = torch.sin(angle)
        rot_even = c * even - s * odd
        rot_odd = s * even + c * odd
        stacked = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        return stacked[..., : self.dim]
