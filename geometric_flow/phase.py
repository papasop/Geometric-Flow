"""Boundary engine for curvature phase/topography scans."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List

import torch

from ._tensor import get_flat_params, set_flat_params, trainable_params
from .curvature import compute_curvature, hutchinson_trace


@dataclass
class PhasePoint:
    coordinate: float
    trace: float
    min_rayleigh: float
    max_rayleigh: float
    condition_proxy: float
    regime: str


def phase_diagram_scanner(
    model: torch.nn.Module,
    loss_factory: Callable[[], torch.Tensor],
    param_range: Iterable[float],
    probe_scale: float = 1e-2,
    probes: int = 8,
    damping: float = 1e-3,
    cliff_condition: float = 1e6,
) -> List[PhasePoint]:
    """Scan a one-dimensional parameter ray and classify curvature regimes.

    ``loss_factory`` must recompute the scalar loss for the model's current
    parameters. The scanner restores the original parameters before returning.
    """

    params = trainable_params(model.parameters())
    base = get_flat_params(params)
    if base.numel() == 0:
        return []
    direction = torch.randn_like(base)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)
    results: List[PhasePoint] = []

    try:
        for coordinate in param_range:
            set_flat_params(params, base + float(coordinate) * probe_scale * direction)
            loss = loss_factory()
            curvature = compute_curvature(model, loss, damping=damping)
            rays = []
            for _ in range(probes):
                probe = torch.randn_like(base)
                probe = probe / torch.linalg.vector_norm(probe).clamp_min(1e-30)
                rays.append(curvature.rayleigh(probe))
            min_rayleigh = min(rays)
            max_rayleigh = max(rays)
            condition = abs(max_rayleigh) / max(abs(min_rayleigh), 1e-30)
            if min_rayleigh <= 0:
                regime = "saddle"
            elif condition >= cliff_condition:
                regime = "cliff"
            else:
                regime = "plain"
            results.append(
                PhasePoint(
                    coordinate=float(coordinate),
                    trace=hutchinson_trace(curvature, samples=max(1, min(probes, 4))),
                    min_rayleigh=float(min_rayleigh),
                    max_rayleigh=float(max_rayleigh),
                    condition_proxy=float(condition),
                    regime=regime,
                )
            )
    finally:
        set_flat_params(params, base)

    return results


def write_phase_diagram(points: List[PhasePoint], path: str | Path) -> None:
    payload = [asdict(point) for point in points]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
