"""Held-out trust calibration for product-state update proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .product_state import ProductState


@dataclass
class TrustRegionResult:
    """Result of held-out scale selection."""

    selected_scale: float
    accepted: bool
    hit_max_scale: bool
    baseline_loss: float
    selected_loss: float


class HeldOutTrustRegion:
    """Held-out scale selector that restores state after every candidate."""

    def __init__(
        self,
        scale_grid: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0),
        armijo_relative_decrease: float = 1e-5,
    ) -> None:
        self.scale_grid = tuple(float(scale) for scale in scale_grid)
        if not self.scale_grid:
            raise ValueError("scale_grid must not be empty")
        if 0.0 not in self.scale_grid:
            raise ValueError("scale_grid must include 0 as an exact reject option")
        if any(scale < 0 for scale in self.scale_grid):
            raise ValueError("scale_grid values must be non-negative")
        if armijo_relative_decrease < 0:
            raise ValueError("armijo_relative_decrease must be non-negative")
        self.armijo_relative_decrease = float(armijo_relative_decrease)

    def select(
        self,
        product_state: ProductState,
        base_steps: dict[str, torch.Tensor],
        calibration_closure: Callable[[], torch.Tensor],
    ) -> TrustRegionResult:
        """Select a scale without leaving candidate states applied."""

        snapshot = product_state.snapshot()
        try:
            with torch.no_grad():
                baseline_loss = float(calibration_closure().detach().cpu())
            best_scale = 0.0
            best_loss = baseline_loss
            threshold = baseline_loss - self.armijo_relative_decrease * max(abs(baseline_loss), 1.0)
            for scale in self.scale_grid:
                product_state.restore_(snapshot)
                if scale != 0.0:
                    product_state.project_and_retract_(base_steps, scale=scale)
                with torch.no_grad():
                    candidate_loss = float(calibration_closure().detach().cpu())
                if scale != 0.0 and candidate_loss < best_loss:
                    best_scale = scale
                    best_loss = candidate_loss
            accepted = bool(best_scale != 0.0 and best_loss <= threshold)
            if not accepted:
                best_scale = 0.0
                best_loss = baseline_loss
            return TrustRegionResult(
                selected_scale=float(best_scale),
                accepted=accepted,
                hit_max_scale=bool(accepted and best_scale == max(self.scale_grid)),
                baseline_loss=float(baseline_loss),
                selected_loss=float(best_loss),
            )
        finally:
            product_state.restore_(snapshot)
