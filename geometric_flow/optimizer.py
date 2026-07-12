"""PyTorch optimizer that navigates with local curvature when reliable."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from ._tensor import assign_flat_update, trainable_params
from .curvature import CurvatureKind, compute_curvature, hutchinson_trace
from .navigation import conjugate_gradient


class GeometricOptimizer(Optimizer):
    """Geometry-first optimizer with SGD fallback.

    The closure must recompute and return a scalar loss. The optimizer computes
    gradients itself so HVPs can reuse the autograd graph.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        damping: float = 1e-3,
        curvature_interval: int = 1,
        cg_max_iter: int = 20,
        cg_tolerance: float = 1e-6,
        trace_samples: int = 4,
        condition_threshold: float = 1e8,
        path_smoothing: float = 0.0,
        max_update_norm: Optional[float] = None,
        fallback_lr: Optional[float] = None,
        curvature_kind: CurvatureKind = "hessian",
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if damping < 0:
            raise ValueError("damping must be non-negative")
        if curvature_interval < 1:
            raise ValueError("curvature_interval must be >= 1")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.damping = damping
        self.curvature_interval = curvature_interval
        self.cg_max_iter = cg_max_iter
        self.cg_tolerance = cg_tolerance
        self.trace_samples = trace_samples
        self.condition_threshold = condition_threshold
        self.path_smoothing = path_smoothing
        self.max_update_norm = max_update_norm
        self.fallback_lr = lr if fallback_lr is None else fallback_lr
        self.curvature_kind = curvature_kind
        self.topography_log = []
        self.geodesic_distance = 0.0
        self._step_index = 0
        self._previous_direction = None

    @property
    def _params(self):
        params = []
        for group in self.param_groups:
            params.extend(group["params"])
        return trainable_params(params)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        if closure is None:
            raise RuntimeError("GeometricOptimizer requires a closure returning the loss")

        self._step_index += 1
        use_curvature = self._step_index % self.curvature_interval == 0
        params = self._params

        with torch.enable_grad():
            self.zero_grad(set_to_none=True)
            loss = closure()
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise RuntimeError("closure must return a scalar loss tensor")

            curvature = compute_curvature(
                _ParameterView(params),
                loss,
                damping=self.damping,
                kind=self.curvature_kind,
            )

        grad = curvature.gradient
        grad_norm = float(torch.linalg.vector_norm(grad))
        direction = -grad
        mode = "sgd"
        cg_iters = 0
        residual_norm = grad_norm
        rayleigh = None
        trace = None

        if use_curvature and grad.numel() > 0 and grad_norm > 0:
            rayleigh = curvature.rayleigh(grad)
            trace = hutchinson_trace(curvature, self.trace_samples) if self.trace_samples else None
            condition_proxy = abs(trace / rayleigh) if trace is not None and abs(rayleigh) > 1e-30 else 1.0
            if rayleigh > 0 and condition_proxy <= self.condition_threshold:
                cg = conjugate_gradient(
                    curvature.matvec,
                    -grad,
                    max_iter=self.cg_max_iter,
                    tolerance=self.cg_tolerance,
                )
                if cg.converged or torch.isfinite(cg.solution).all():
                    direction = cg.solution
                    mode = "geometric"
                    cg_iters = cg.iterations
                    residual_norm = cg.residual_norm

        if self._previous_direction is not None and self.path_smoothing > 0:
            if self._previous_direction.numel() == direction.numel():
                direction = (1.0 - self.path_smoothing) * direction + self.path_smoothing * self._previous_direction

        update_norm = torch.linalg.vector_norm(direction)
        if self.max_update_norm is not None and float(update_norm) > self.max_update_norm:
            direction = direction * (self.max_update_norm / update_norm.clamp_min(1e-30))
            update_norm = torch.linalg.vector_norm(direction)

        lr = self.param_groups[0]["lr"] if mode == "geometric" else self.fallback_lr
        actual_update_norm = float(torch.linalg.vector_norm(direction * lr))
        self.geodesic_distance += actual_update_norm
        assign_flat_update(params, direction, scale=lr)
        self._previous_direction = direction.detach()

        self.topography_log.append(
            {
                "step": self._step_index,
                "mode": mode,
                "loss": float(loss.detach()),
                "grad_norm": grad_norm,
                "direction_norm": float(update_norm),
                "update_norm": actual_update_norm,
                "geodesic_distance": self.geodesic_distance,
                "rayleigh_grad": rayleigh,
                "trace_estimate": trace,
                "cg_iterations": cg_iters,
                "residual_norm": residual_norm,
            }
        )
        return loss


class _ParameterView(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = torch.nn.ParameterList(params)
