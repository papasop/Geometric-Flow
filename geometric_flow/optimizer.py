"""PyTorch optimizer that navigates with local curvature when reliable."""

from __future__ import annotations

import inspect
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
        max_grad_norm: float = 1.0,
        adaptive_damping: bool = True,
        damping_growth: float = 1.05,
        damping_decay: float = 0.95,
        min_damping: float = 1e-3,
        max_damping: float = 1.0,
        regularization: float = 1e-3,
        warmup_steps: int = 10,
        warmup_lr_scale: float = 0.5,
        curvature_reuse: int = 5,
        lr_scale: float = 3.0,
        adaptive_curvature_reuse: bool = True,
        min_curvature_reuse: int = 1,
        max_curvature_reuse: int = 20,
        reuse_growth: int = 1,
        reuse_decay: int = 1,
        reuse_flat_threshold: float = 0.05,
        reuse_steep_threshold: float = 0.25,
        grad_smoothing: float = 0.9,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if damping < 0:
            raise ValueError("damping must be non-negative")
        if curvature_interval < 1:
            raise ValueError("curvature_interval must be >= 1")
        if max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")
        if min_damping <= 0 or max_damping < min_damping:
            raise ValueError("damping bounds must satisfy 0 < min_damping <= max_damping")
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if warmup_lr_scale <= 0:
            raise ValueError("warmup_lr_scale must be positive")
        if curvature_reuse < 1:
            raise ValueError("curvature_reuse must be >= 1")
        if lr_scale <= 0:
            raise ValueError("lr_scale must be positive")
        if min_curvature_reuse < 1 or max_curvature_reuse < min_curvature_reuse:
            raise ValueError("reuse bounds must satisfy 1 <= min_curvature_reuse <= max_curvature_reuse")
        if reuse_growth < 0 or reuse_decay < 0:
            raise ValueError("reuse growth/decay must be non-negative")
        if not 0 <= grad_smoothing < 1:
            raise ValueError("grad_smoothing must be in [0, 1)")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.damping = damping
        self._damping = damping
        self.curvature_interval = curvature_interval
        self.cg_max_iter = cg_max_iter
        self.cg_tolerance = cg_tolerance
        self.trace_samples = trace_samples
        self.condition_threshold = condition_threshold
        self.path_smoothing = path_smoothing
        self.max_update_norm = max_update_norm
        self.fallback_lr = lr if fallback_lr is None else fallback_lr
        self.curvature_kind = curvature_kind
        self.max_grad_norm = max_grad_norm
        self.adaptive_damping = adaptive_damping
        self.damping_growth = damping_growth
        self.damping_decay = damping_decay
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.regularization = regularization
        self.warmup_steps = warmup_steps
        self.warmup_lr_scale = warmup_lr_scale
        self.curvature_reuse = curvature_reuse
        self.lr_scale = lr_scale
        self.adaptive_curvature_reuse = adaptive_curvature_reuse
        self.min_curvature_reuse = min_curvature_reuse
        self.max_curvature_reuse = max_curvature_reuse
        self.reuse_growth = reuse_growth
        self.reuse_decay = reuse_decay
        self.reuse_flat_threshold = reuse_flat_threshold
        self.reuse_steep_threshold = reuse_steep_threshold
        self.grad_smoothing = grad_smoothing
        self.topography_log = []
        self.geodesic_distance = 0.0
        self._step_index = 0
        self._previous_direction = None
        self._last_preconditioner_gain = 1.0
        self._has_preconditioner = False
        self._last_preconditioned_grad_norm = None
        self._previous_preconditioned_grad_norm = None
        self._ema_grad = None

    @property
    def _params(self):
        params = []
        for group in self.param_groups:
            params.extend(group["params"])
        return trainable_params(params)

    def step(self, closure: Optional[Callable[..., torch.Tensor]] = None):
        if closure is None:
            raise RuntimeError("GeometricOptimizer requires a closure returning the loss")

        self._step_index += 1
        if self._step_index <= self.warmup_steps:
            return self._warmup_step(closure)

        geometric_step = self._step_index - self.warmup_steps
        refresh_curvature = (
            geometric_step == 1
            or geometric_step % self.curvature_reuse == 0
            or not self._has_preconditioner
        )
        use_curvature = refresh_curvature and self._step_index % self.curvature_interval == 0
        params = self._params

        with torch.enable_grad():
            self.zero_grad(set_to_none=True)
            loss = self._call_loss_closure(closure)
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise RuntimeError("closure must return a scalar loss tensor")

            curvature = None
            if use_curvature:
                curvature = compute_curvature(
                    _ParameterView(params),
                    loss,
                    damping=self._damping,
                    kind=self.curvature_kind,
                    regularization=self.regularization,
                )
            loss.backward(retain_graph=use_curvature and self.curvature_kind == "hessian")

        raw_grad = self._flat_current_grad(params)
        raw_grad_norm = float(torch.linalg.vector_norm(raw_grad))
        clipped_grad_norm = self._clip_gradients(params)
        self._adapt_damping(clipped_grad_norm)
        if curvature is not None:
            curvature.damping = self._damping
        grad = self._flat_current_grad(params, fallback=raw_grad)
        grad_norm = float(torch.linalg.vector_norm(grad))
        direction = -grad
        mode = "sgd"
        cg_iters = 0
        residual_norm = grad_norm
        rayleigh = None
        trace = None

        if use_curvature and curvature is not None and grad.numel() > 0 and grad_norm > 0:
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
                    self._cache_preconditioner(direction, grad_norm)
        elif grad.numel() > 0 and grad_norm > 0 and self._has_preconditioner:
            direction = -grad * self._last_preconditioner_gain
            mode = "geometric_reuse"

        direction = self._smooth_direction(direction)
        preconditioned_grad_norm = float(torch.linalg.vector_norm(direction))
        reuse_change_rate = self._update_curvature_reuse(preconditioned_grad_norm)

        if self._previous_direction is not None and self.path_smoothing > 0:
            if self._previous_direction.numel() == direction.numel():
                direction = (1.0 - self.path_smoothing) * direction + self.path_smoothing * self._previous_direction

        update_norm = torch.linalg.vector_norm(direction)
        if self.max_update_norm is not None and float(update_norm) > self.max_update_norm:
            direction = direction * (self.max_update_norm / update_norm.clamp_min(1e-30))
            update_norm = torch.linalg.vector_norm(direction)

        if mode in {"geometric", "geometric_reuse"}:
            lr = self.param_groups[0]["lr"] * self.lr_scale
        else:
            lr = self.fallback_lr
        actual_update_norm = float(torch.linalg.vector_norm(direction * lr))
        self.geodesic_distance += actual_update_norm
        assign_flat_update(params, direction, scale=lr)
        self._previous_direction = direction.detach()

        self.topography_log.append(
            {
                "step": self._step_index,
                "mode": mode,
                "loss": float(loss.detach()),
                "raw_grad_norm": raw_grad_norm,
                "grad_norm": grad_norm,
                "clipped_grad_norm": clipped_grad_norm,
                "current_damping": self._damping,
                "curvature_refreshed": use_curvature,
                "curvature_reuse": self.curvature_reuse,
                "reuse_change_rate": reuse_change_rate,
                "preconditioned_grad_norm": preconditioned_grad_norm,
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

    def _warmup_step(self, closure: Callable[..., torch.Tensor]) -> torch.Tensor:
        params = self._params
        lr = self.param_groups[0]["lr"] * self.warmup_lr_scale
        with torch.enable_grad():
            self.zero_grad(set_to_none=True)
            loss = self._call_loss_closure(closure)
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise RuntimeError("closure must return a scalar loss tensor")
            loss.backward()

        raw_grad = self._flat_current_grad(params)
        raw_grad_norm = float(torch.linalg.vector_norm(raw_grad))
        clipped_grad_norm = self._clip_gradients(params)
        self._adapt_damping(clipped_grad_norm)
        grad = self._flat_current_grad(params, fallback=raw_grad)
        direction = -grad
        actual_update_norm = float(torch.linalg.vector_norm(direction * lr))
        self.geodesic_distance += actual_update_norm
        assign_flat_update(params, direction, scale=lr)
        self._previous_direction = direction.detach()
        self.topography_log.append(
            {
                "step": self._step_index,
                "mode": "warmup",
                "loss": float(loss.detach()),
                "raw_grad_norm": raw_grad_norm,
                "grad_norm": float(torch.linalg.vector_norm(grad)),
                "clipped_grad_norm": clipped_grad_norm,
                "current_damping": self._damping,
                "curvature_refreshed": False,
                "curvature_reuse": self.curvature_reuse,
                "reuse_change_rate": None,
                "preconditioned_grad_norm": float(torch.linalg.vector_norm(direction)),
                "direction_norm": float(torch.linalg.vector_norm(direction)),
                "update_norm": actual_update_norm,
                "geodesic_distance": self.geodesic_distance,
                "rayleigh_grad": None,
                "trace_estimate": None,
                "cg_iterations": 0,
                "residual_norm": float(torch.linalg.vector_norm(grad)),
            }
        )
        return loss

    def _cache_preconditioner(self, direction: torch.Tensor, grad_norm: float) -> None:
        direction_norm = float(torch.linalg.vector_norm(direction))
        self._last_preconditioner_gain = direction_norm / max(float(grad_norm), 1e-30)
        self._has_preconditioner = True

    def _smooth_direction(self, direction: torch.Tensor) -> torch.Tensor:
        if direction.numel() == 0 or self.grad_smoothing == 0:
            self._ema_grad = direction.detach()
            return direction
        if self._ema_grad is None or self._ema_grad.numel() != direction.numel():
            self._ema_grad = direction.detach().clone()
            return direction
        self._ema_grad = self._ema_grad.to(device=direction.device, dtype=direction.dtype)
        self._ema_grad = self.grad_smoothing * self._ema_grad + (1.0 - self.grad_smoothing) * direction.detach()
        return self._ema_grad.clone()

    def _update_curvature_reuse(self, preconditioned_grad_norm: float) -> Optional[float]:
        if self._last_preconditioned_grad_norm is not None:
            self._previous_preconditioned_grad_norm = self._last_preconditioned_grad_norm
        self._last_preconditioned_grad_norm = preconditioned_grad_norm
        if self._previous_preconditioned_grad_norm is None:
            return None

        denom = max(abs(self._previous_preconditioned_grad_norm), 1e-30)
        change_rate = abs(preconditioned_grad_norm - self._previous_preconditioned_grad_norm) / denom
        if self.adaptive_curvature_reuse:
            if change_rate < self.reuse_flat_threshold and self.reuse_growth:
                self.curvature_reuse = min(self.max_curvature_reuse, self.curvature_reuse + self.reuse_growth)
            elif change_rate > self.reuse_steep_threshold and self.reuse_decay:
                self.curvature_reuse = max(self.min_curvature_reuse, self.curvature_reuse - self.reuse_decay)
        return float(change_rate)

    def _clip_gradients(self, params) -> float:
        if not params:
            return 0.0
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        grad = self._flat_current_grad(params)
        return float(torch.linalg.vector_norm(grad))

    def _adapt_damping(self, grad_norm: float) -> None:
        if not self.adaptive_damping:
            return
        if grad_norm > 1.0:
            self._damping *= self.damping_growth
        elif grad_norm < 0.01:
            self._damping *= self.damping_decay
        self._damping = max(self.min_damping, min(self.max_damping, self._damping))

    def _flat_current_grad(self, params, fallback: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not params:
            if fallback is not None:
                return torch.zeros_like(fallback)
            return torch.empty(0)
        chunks = []
        for param in params:
            if param.grad is None:
                chunks.append(torch.zeros_like(param).reshape(-1))
            else:
                chunks.append(param.grad.detach().reshape(-1))
        return torch.cat(chunks)

    def _call_loss_closure(self, closure: Callable[..., torch.Tensor]) -> torch.Tensor:
        """Call closure in loss-only mode when it supports ``backward=False``."""

        try:
            signature = inspect.signature(closure)
        except (TypeError, ValueError):
            return closure()

        parameters = signature.parameters.values()
        accepts_backward = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == "backward"
            for parameter in parameters
        )
        if accepts_backward:
            return closure(backward=False)
        return closure()


class _ParameterView(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = torch.nn.ParameterList(params)
