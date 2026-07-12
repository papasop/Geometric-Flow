"""Geometry-aware CNN layers and CIFAR-style baseline models."""

from __future__ import annotations

import torch
from torch import nn

from ..layers import GeometricRotation


class GeoConv2D(nn.Module):
    """Conv2d followed by a channel-wise geometric rotation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        learnable_rotation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.rotation = GeometricRotation(out_channels, learnable=learnable_rotation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.rotation(x)
        return x.permute(0, 3, 1, 2).contiguous()


class GeoCNN(nn.Module):
    """Compact CIFAR-style CNN with geometry-aware convolutional blocks."""

    def __init__(
        self,
        channels: int = 32,
        num_classes: int = 10,
        learnable_rotation: bool = True,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            GeoConv2D(3, channels, learnable_rotation=learnable_rotation),
            nn.GELU(),
            nn.MaxPool2d(2),
            GeoConv2D(channels, channels * 2, learnable_rotation=learnable_rotation),
            nn.GELU(),
            nn.MaxPool2d(2),
            GeoConv2D(channels * 2, channels * 2, learnable_rotation=learnable_rotation),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))

    def geometric_parameters(self):
        """Return trainable phase parameters from geometric rotation layers."""

        return [
            module.angle
            for module in self.modules()
            if isinstance(module, GeometricRotation) and isinstance(module.angle, nn.Parameter)
        ]
