"""Navigation engine for geometry-preconditioned updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .curvature import CurvatureOperator


@dataclass
class CGResult:
    solution: torch.Tensor
    converged: bool
    iterations: int
    residual_norm: float


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    max_iter: int = 20,
    tolerance: float = 1e-6,
    initial_guess: torch.Tensor | None = None,
) -> CGResult:
    """Solve A x = rhs for symmetric positive-definite A."""

    x = torch.zeros_like(rhs) if initial_guess is None else initial_guess.detach().clone().to(rhs)
    residual = rhs - matvec(x)
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    initial = torch.sqrt(residual_sq).clamp_min(1e-30)

    for iteration in range(1, max_iter + 1):
        ad = matvec(direction)
        denom = torch.dot(direction, ad)
        if not torch.isfinite(denom) or denom <= 1e-30:
            return CGResult(x, False, iteration, float(torch.sqrt(residual_sq)))

        alpha = residual_sq / denom
        x = x + alpha * direction
        residual = residual - alpha * ad
        next_residual_sq = torch.dot(residual, residual)
        residual_norm = torch.sqrt(next_residual_sq)
        if float(residual_norm / initial) <= tolerance:
            return CGResult(x, True, iteration, float(residual_norm))

        beta = next_residual_sq / residual_sq.clamp_min(1e-30)
        direction = residual + beta * direction
        residual_sq = next_residual_sq

    return CGResult(x, False, max_iter, float(torch.sqrt(residual_sq)))


def geometric_step(
    loss: torch.Tensor,
    params,
    curvature_op: CurvatureOperator,
    max_iter: int = 20,
    tolerance: float = 1e-6,
) -> CGResult:
    """Return the geodesic update direction solving A * step = -grad."""

    del loss, params
    return conjugate_gradient(
        curvature_op.matvec,
        -curvature_op.gradient,
        max_iter=max_iter,
        tolerance=tolerance,
    )
