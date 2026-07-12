"""User-facing toolbox aliases: measure, navigate, plot_boundary, embed."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn

from .curvature import CurvatureKind, CurvatureOperator, compute_curvature
from .layers import GeometricRotation
from .navigation import CGResult, geometric_step
from .phase import PhasePoint, phase_diagram_scanner, write_phase_diagram


def measure(
    model: nn.Module,
    loss: torch.Tensor,
    data=None,
    damping: float = 1e-3,
    kind: CurvatureKind = "hessian",
) -> CurvatureOperator:
    return compute_curvature(model, loss, data=data, damping=damping, kind=kind)


def navigate(
    loss: torch.Tensor,
    params,
    curvature_op: CurvatureOperator,
    max_iter: int = 20,
    tolerance: float = 1e-6,
) -> CGResult:
    return geometric_step(loss, params, curvature_op, max_iter=max_iter, tolerance=tolerance)


def plot_boundary(
    model: nn.Module,
    loss_factory,
    param_range: Iterable[float],
    probe_scale: float = 1e-2,
    probes: int = 8,
    output_path: Optional[str | Path] = None,
) -> list[PhasePoint]:
    points = phase_diagram_scanner(
        model,
        loss_factory,
        param_range=param_range,
        probe_scale=probe_scale,
        probes=probes,
    )
    if output_path is not None:
        write_phase_diagram(points, output_path)
    return points


def embed(module: nn.Module, angle: float = 0.125) -> nn.Module:
    """Insert parameter-free geometric rotations after Linear layers in Sequentials."""

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            layers = []
            for layer in child:
                layers.append(layer)
                if isinstance(layer, nn.Linear) and layer.out_features >= 2:
                    layers.append(GeometricRotation(layer.out_features, angle=angle))
            setattr(module, name, nn.Sequential(*layers))
        else:
            embed(child, angle=angle)
    return module
