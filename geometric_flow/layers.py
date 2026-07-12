"""Parameter-free geometry-aware layers."""

from __future__ import annotations

import torch
from torch import nn


class GeometricRotation(nn.Module):
    """Fixed pairwise rotation that reshapes flow without adding parameters."""

    def __init__(self, dim: int, angle: float = 0.125) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.dim = dim
        self.angle = angle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        if even.shape[-1] != odd.shape[-1]:
            odd = torch.nn.functional.pad(odd, (0, 1))
        c = torch.cos(torch.as_tensor(self.angle, dtype=x.dtype, device=x.device))
        s = torch.sin(torch.as_tensor(self.angle, dtype=x.dtype, device=x.device))
        rot_even = c * even - s * odd
        rot_odd = s * even + c * odd
        stacked = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        return stacked[..., : self.dim]
