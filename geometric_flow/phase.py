"""Boundary engine for curvature phase/topography scans."""

from __future__ import annotations

import json
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import torch

from ._tensor import get_flat_params, set_flat_params, trainable_params
from .curvature import compute_curvature, hutchinson_trace
from .optimizer import GeometricOptimizer


@dataclass
class PhasePoint:
    coordinate: float
    trace: float
    min_rayleigh: float
    max_rayleigh: float
    condition_proxy: float
    regime: str


@dataclass
class PhaseGridPoint:
    param1: float
    param2: float
    final_loss: float
    avg_trace: float
    geodesic_distance: float
    final_mode: str


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


def phase_diagram_scanner_2d(
    model_factory: Callable[[], torch.nn.Module],
    loss_factory: Callable[[torch.nn.Module], torch.Tensor],
    param1_range: Iterable[float],
    param2_range: Iterable[float],
    param1_name: str = "lr",
    param2_name: str = "damping",
    steps: int = 5,
    optimizer_kwargs: Optional[dict] = None,
    optimizer_kwargs_factory: Optional[Callable[[float, float], dict]] = None,
) -> List[PhaseGridPoint]:
    """Run a 2D hyperparameter phase scan with ``GeometricOptimizer``.

    By default ``param1`` maps to optimizer ``lr`` and ``param2`` maps to
    ``damping``. For other axes, pass ``optimizer_kwargs_factory`` and return
    the exact keyword arguments for each grid point.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")

    results: List[PhaseGridPoint] = []
    base_kwargs = dict(optimizer_kwargs or {})
    for param1 in param1_range:
        for param2 in param2_range:
            p1 = float(param1)
            p2 = float(param2)
            model = model_factory()
            if optimizer_kwargs_factory is None:
                kwargs = dict(base_kwargs)
                kwargs[param1_name] = p1
                kwargs[param2_name] = p2
            else:
                kwargs = dict(base_kwargs)
                kwargs.update(optimizer_kwargs_factory(p1, p2))

            optimizer = GeometricOptimizer(model.parameters(), **kwargs)
            last_loss = None
            for _ in range(steps):
                last_loss = optimizer.step(lambda: loss_factory(model))

            traces = [
                entry["trace_estimate"]
                for entry in optimizer.topography_log
                if entry["trace_estimate"] is not None
            ]
            avg_trace = float(sum(traces) / len(traces)) if traces else 0.0
            final_mode = optimizer.topography_log[-1]["mode"] if optimizer.topography_log else "none"
            final_loss = float(last_loss.detach()) if last_loss is not None else float("nan")
            results.append(
                PhaseGridPoint(
                    param1=p1,
                    param2=p2,
                    final_loss=final_loss,
                    avg_trace=avg_trace,
                    geodesic_distance=optimizer.geodesic_distance,
                    final_mode=final_mode,
                )
            )

    return results


def write_phase_diagram(points: List[PhasePoint], path: str | Path) -> None:
    payload = [asdict(point) for point in points]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_phase_diagram_csv(points: List[PhaseGridPoint], path: str | Path) -> None:
    fieldnames = ["param1", "param2", "final_loss", "avg_trace", "geodesic_distance", "final_mode"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))
