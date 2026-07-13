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


class ChannelGeometricRotation(nn.Module):
    """Apply GeometricRotation to the channel dimension of NCHW tensors."""

    def __init__(self, channels: int, learnable_rotation: bool = True) -> None:
        super().__init__()
        self.rotation = GeometricRotation(channels, learnable=learnable_rotation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        block_rotation: bool = True,
        conv_layers: int = 3,
    ) -> None:
        super().__init__()
        if conv_layers < 1:
            raise ValueError("conv_layers must be >= 1")
        self.conv_layers = conv_layers
        channel_schedule = self._channel_schedule(channels, conv_layers)
        layers = []
        in_channels = 3
        for index, out_channels in enumerate(channel_schedule):
            layers.extend(
                [
                    GeoConv2D(in_channels, out_channels, learnable_rotation=learnable_rotation),
                    nn.GELU(),
                    ChannelGeometricRotation(out_channels, learnable_rotation=learnable_rotation)
                    if block_rotation
                    else nn.Identity(),
                ]
            )
            if index in self._pool_after_indices(conv_layers):
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channel_schedule[-1], num_classes)

    @staticmethod
    def _channel_schedule(channels: int, conv_layers: int):
        if conv_layers == 3:
            return [channels, channels * 2, channels * 2]
        return [channels * min(2 ** (index // 2), 4) for index in range(conv_layers)]

    @staticmethod
    def _pool_after_indices(conv_layers: int):
        if conv_layers == 1:
            return set()
        if conv_layers == 3:
            return {0, 1}
        return {index for index in range(1, conv_layers - 1, 2)}

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
