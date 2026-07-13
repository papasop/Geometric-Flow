"""PyTorch optimizer that navigates with local curvature when reliable."""

from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from ._tensor import assign_flat_update, get_flat_params, trainable_params
from .curvature import CurvatureKind, compute_curvature, hutchinson_trace
from .functional_geometry import projected_functional_geoflow_direction
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
        preconditioner_scale: float = 0.5,
        curvature_scale: float = 1.0,
        preconditioner: str = "cg",
        mode: str = "geometric",
        adam_warmup_steps: int = 0,
        descent_gate: bool = True,
        functional_model: Optional[torch.nn.Module] = None,
        functional_probe: Optional[torch.Tensor] = None,
        functional_representation: str = "logits",
        response_kind: str = "gauss_newton",
        diagonal_beta1: float = 0.9,
        diagonal_beta2: float = 0.999,
        diagonal_eps: float = 1e-8,
        verbose: bool = False,
        diagnostic_log_interval: int = 10,
        diagnostic_log_path: str | Path = "geometric_optimizer_diagnostics.csv",
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
        if preconditioner_scale <= 0:
            raise ValueError("preconditioner_scale must be positive")
        if curvature_scale <= 0:
            raise ValueError("curvature_scale must be positive")
        if preconditioner not in {"cg", "diagonal", "diagonal_grad_square"}:
            raise ValueError("preconditioner must be 'cg' or 'diagonal'")
        if mode not in {
            "geometric",
            "adam",
            "hybrid",
            "adam_continue",
            "hybrid_geometric",
            "functional_geoflow",
        }:
            raise ValueError(
                "mode must be 'geometric', 'adam', 'hybrid', 'adam_continue', 'hybrid_geometric', "
                "or 'functional_geoflow'"
            )
        if adam_warmup_steps < 0:
            raise ValueError("adam_warmup_steps must be non-negative")
        if not 0 <= diagonal_beta1 < 1 or not 0 <= diagonal_beta2 < 1:
            raise ValueError("diagonal betas must be in [0, 1)")
        if diagonal_eps <= 0:
            raise ValueError("diagonal_eps must be positive")
        if diagnostic_log_interval < 1:
            raise ValueError("diagnostic_log_interval must be >= 1")
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
        self.preconditioner_scale = preconditioner_scale
        self.curvature_scale = curvature_scale
        self.preconditioner = "diagonal" if preconditioner == "diagonal_grad_square" else preconditioner
        if preconditioner == "diagonal_grad_square":
            self.curvature_kind = "grad_square"
        self.mode = self._normalize_mode(mode)
        self.adam_warmup_steps = adam_warmup_steps
        self.descent_gate = descent_gate
        self.functional_model = functional_model
        self.functional_probe = functional_probe
        self.functional_representation = functional_representation
        self.response_kind = response_kind
        self.diagonal_beta1 = diagonal_beta1
        self.diagonal_beta2 = diagonal_beta2
        self.diagonal_eps = diagonal_eps
        self.verbose = verbose
        self.diagnostic_log_interval = diagnostic_log_interval
        self.diagnostic_log_path = Path(diagnostic_log_path)
        self.topography_log = []
        self.geodesic_distance = 0.0
        self._step_index = 0
        self._previous_direction = None
        self._last_preconditioner_gain = 1.0
        self._has_preconditioner = False
        self._last_preconditioned_grad_norm = None
        self._previous_preconditioned_grad_norm = None
        self._ema_grad = None
        self._diag_m = None
        self._diag_v = None
        self._diag_step = 0
        self._adam_optimizer = None

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if mode == "adam_continue":
            return "adam"
        if mode == "hybrid_geometric":
            return "hybrid"
        return mode

    @property
    def _params(self):
        params = []
        for group in self.param_groups:
            params.extend(group["params"])
        return trainable_params(params)

    def step(self, closure: Optional[Callable[..., torch.Tensor]] = None, verbose: Optional[bool] = None):
        if closure is None:
            raise RuntimeError("GeometricOptimizer requires a closure returning the loss")

        self._step_index += 1
        if self.mode == "adam":
            return self._adam_step(closure, mode="adam", verbose=verbose)
        if self.mode == "hybrid" and self._step_index <= self.adam_warmup_steps:
            return self._adam_step(closure, mode="adam_warmup", verbose=verbose)
        if self.mode == "functional_geoflow":
            return self._functional_step(closure, verbose=verbose)
        if self.mode == "geometric" and self._step_index <= self.warmup_steps:
            return self._warmup_step(closure, verbose=verbose)

        geometric_step = self._step_index
        if self.mode == "geometric":
            geometric_step -= self.warmup_steps
        elif self.mode == "hybrid":
            geometric_step -= self.adam_warmup_steps
        refresh_curvature = (
            geometric_step == 1
            or geometric_step % self.curvature_reuse == 0
            or not self._has_preconditioner
        )
        use_curvature = (
            refresh_curvature
            and self._step_index % self.curvature_interval == 0
            and self.preconditioner == "cg"
        )
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
                    scale=self.curvature_scale,
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

        if self.preconditioner == "diagonal" and grad.numel() > 0 and grad_norm > 0:
            direction = self._diagonal_precondition(grad)
            mode = "diagonal"
            self._cache_preconditioner(direction, grad_norm)
        elif use_curvature and curvature is not None and grad.numel() > 0 and grad_norm > 0:
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
                    direction = self._scale_preconditioned_direction(cg.solution, grad_norm)
                    mode = "geometric"
                    cg_iters = cg.iterations
                    residual_norm = cg.residual_norm
                    self._cache_preconditioner(direction, grad_norm)
        elif grad.numel() > 0 and grad_norm > 0 and self._has_preconditioner:
            direction = -grad * self._last_preconditioner_gain
            mode = "geometric_reuse"

        direction = self._smooth_direction(direction)

        if self._previous_direction is not None and self.path_smoothing > 0:
            if self._previous_direction.numel() == direction.numel():
                direction = (1.0 - self.path_smoothing) * direction + self.path_smoothing * self._previous_direction

        grad_direction_dot = float(torch.dot(grad, direction)) if grad.numel() and direction.numel() else 0.0
        descent_gate_passed = grad_direction_dot < 0.0
        if self.descent_gate and mode in {"geometric", "geometric_reuse", "diagonal"} and not descent_gate_passed:
            direction = -grad
            mode = "descent_gate_fallback"
            self._has_preconditioner = False
            grad_direction_dot = float(torch.dot(grad, direction)) if grad.numel() else 0.0
            descent_gate_passed = grad_direction_dot < 0.0
        preconditioned_grad_norm = float(torch.linalg.vector_norm(direction))
        reuse_change_rate = self._update_curvature_reuse(preconditioned_grad_norm)

        update_norm = torch.linalg.vector_norm(direction)
        if self.max_update_norm is not None and float(update_norm) > self.max_update_norm:
            direction = direction * (self.max_update_norm / update_norm.clamp_min(1e-30))
            update_norm = torch.linalg.vector_norm(direction)

        if mode in {"geometric", "geometric_reuse", "diagonal"}:
            lr = self.param_groups[0]["lr"] * self.lr_scale
        else:
            lr = self.fallback_lr
        actual_update_norm = float(torch.linalg.vector_norm(direction * lr))
        self.geodesic_distance += actual_update_norm
        assign_flat_update(params, direction, scale=lr)
        self._previous_direction = direction.detach()

        entry = {
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
            "preconditioned_to_raw_ratio": preconditioned_grad_norm / max(raw_grad_norm, 1e-30),
            "direction_norm": float(update_norm),
            "update_norm": actual_update_norm,
            "geodesic_distance": self.geodesic_distance,
            "rayleigh_grad": rayleigh,
            "trace_estimate": trace,
            "cg_iterations": cg_iters,
            "residual_norm": residual_norm,
            "grad_direction_dot": grad_direction_dot,
            "descent_gate_passed": descent_gate_passed,
        }
        self.topography_log.append(entry)
        self._maybe_emit_diagnostics(entry, verbose=verbose)
        return loss

    def _functional_step(self, closure: Callable[..., torch.Tensor], verbose: Optional[bool] = None) -> torch.Tensor:
        if self.functional_model is None or self.functional_probe is None:
            raise RuntimeError("functional_geoflow requires functional_model and functional_probe")
        params = self._params
        with torch.enable_grad():
            self.zero_grad(set_to_none=True)
            loss = self._call_loss_closure(closure)
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise RuntimeError("closure must return a scalar loss tensor")
            result = projected_functional_geoflow_direction(
                self.functional_model,
                loss,
                self.functional_probe.to(device=loss.device),
                params=params,
                representation=self.functional_representation,
                response_kind=self.response_kind,
                damping=self._damping + self.regularization,
                max_update_norm=self.max_update_norm,
                descent_gate=self.descent_gate,
            )
        self._assign_flat_grad(params, result.gradient)
        raw_grad_norm = float(torch.linalg.vector_norm(result.gradient))
        clipped_grad_norm = raw_grad_norm
        self._adapt_damping(clipped_grad_norm)
        direction = result.direction
        direction_norm = float(torch.linalg.vector_norm(direction))
        actual_update_norm = float(torch.linalg.vector_norm(direction * self.param_groups[0]["lr"] * self.lr_scale))
        self.geodesic_distance += actual_update_norm
        assign_flat_update(params, direction, scale=self.param_groups[0]["lr"] * self.lr_scale)
        self._previous_direction = direction.detach()
        self._has_preconditioner = True

        eigvals = result.response_eigenvalues
        entry = {
            "step": self._step_index,
            "mode": "functional_geoflow_fallback" if result.fallback else "functional_geoflow",
            "loss": float(loss.detach()),
            "raw_grad_norm": raw_grad_norm,
            "grad_norm": raw_grad_norm,
            "clipped_grad_norm": clipped_grad_norm,
            "current_damping": self._damping,
            "curvature_refreshed": True,
            "curvature_reuse": 1,
            "reuse_change_rate": None,
            "preconditioned_grad_norm": direction_norm,
            "preconditioned_to_raw_ratio": direction_norm / max(raw_grad_norm, 1e-30),
            "direction_norm": direction_norm,
            "update_norm": actual_update_norm,
            "geodesic_distance": self.geodesic_distance,
            "rayleigh_grad": None,
            "trace_estimate": float(torch.trace(result.projected_response)),
            "cg_iterations": 0,
            "residual_norm": float(torch.linalg.vector_norm(result.projected_gradient)),
            "grad_direction_dot": result.g_dot_d,
            "descent_gate_passed": result.descent_gate_passed,
            "functional_rank": result.projectors.rank,
            "tangent_rank": result.projectors.tangent_rank,
            "normal_rank": result.projectors.normal_rank,
            "functional_tangent_norm": result.tangent_norm,
            "projected_gradient_tangent_norm": result.projected_gradient_tangent_norm,
            "response_min_eigenvalue": float(eigvals.min()) if eigvals.numel() else 0.0,
            "response_max_eigenvalue": float(eigvals.max()) if eigvals.numel() else 0.0,
        }
        self.topography_log.append(entry)
        self._maybe_emit_diagnostics(entry, verbose=verbose)
        return loss

    def _adam_step(
        self,
        closure: Callable[..., torch.Tensor],
        mode: str,
        verbose: Optional[bool] = None,
    ) -> torch.Tensor:
        params = self._params
        before = get_flat_params(params)
        if self._adam_optimizer is None:
            self._adam_optimizer = torch.optim.Adam(params, lr=self.param_groups[0]["lr"])

        with torch.enable_grad():
            self.zero_grad(set_to_none=True)
            self._adam_optimizer.zero_grad(set_to_none=True)
            loss = self._call_loss_closure(closure)
            if not torch.is_tensor(loss) or loss.ndim != 0:
                raise RuntimeError("closure must return a scalar loss tensor")
            loss.backward()

        raw_grad = self._flat_current_grad(params)
        raw_grad_norm = float(torch.linalg.vector_norm(raw_grad))
        clipped_grad_norm = self._clip_gradients(params)
        grad = self._flat_current_grad(params, fallback=raw_grad)
        grad_norm = float(torch.linalg.vector_norm(grad))
        self._adam_optimizer.step()
        after = get_flat_params(params)
        update_norm = float(torch.linalg.vector_norm(after - before)) if before.numel() else 0.0
        self.geodesic_distance += update_norm
        self._adapt_damping(clipped_grad_norm)

        entry = {
            "step": self._step_index,
            "mode": mode,
            "loss": float(loss.detach()),
            "raw_grad_norm": raw_grad_norm,
            "grad_norm": grad_norm,
            "clipped_grad_norm": clipped_grad_norm,
            "current_damping": self._damping,
            "curvature_refreshed": False,
            "curvature_reuse": self.curvature_reuse,
            "reuse_change_rate": None,
            "preconditioned_grad_norm": grad_norm,
            "preconditioned_to_raw_ratio": grad_norm / max(raw_grad_norm, 1e-30),
            "direction_norm": grad_norm,
            "update_norm": update_norm,
            "geodesic_distance": self.geodesic_distance,
            "rayleigh_grad": None,
            "trace_estimate": None,
            "cg_iterations": 0,
            "residual_norm": grad_norm,
            "grad_direction_dot": -grad_norm * grad_norm,
            "descent_gate_passed": True,
        }
        self.topography_log.append(entry)
        self._maybe_emit_diagnostics(entry, verbose=verbose)
        return loss

    def _warmup_step(self, closure: Callable[..., torch.Tensor], verbose: Optional[bool] = None) -> torch.Tensor:
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
        preconditioned_grad_norm = float(torch.linalg.vector_norm(direction))
        entry = {
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
            "preconditioned_grad_norm": preconditioned_grad_norm,
            "preconditioned_to_raw_ratio": preconditioned_grad_norm / max(raw_grad_norm, 1e-30),
            "direction_norm": preconditioned_grad_norm,
            "update_norm": actual_update_norm,
            "geodesic_distance": self.geodesic_distance,
            "rayleigh_grad": None,
            "trace_estimate": None,
            "cg_iterations": 0,
            "residual_norm": float(torch.linalg.vector_norm(grad)),
            "grad_direction_dot": -float(torch.dot(grad, grad)) if grad.numel() else 0.0,
            "descent_gate_passed": True,
        }
        self.topography_log.append(entry)
        self._maybe_emit_diagnostics(entry, verbose=verbose)
        return loss

    def _maybe_emit_diagnostics(self, entry: dict, verbose: Optional[bool] = None) -> None:
        active = self.verbose if verbose is None else verbose
        if not active:
            return

        ratio = entry["preconditioned_to_raw_ratio"]
        print(
            "GeometricOptimizer "
            f"step={entry['step']} mode={entry['mode']} loss={entry['loss']:.6g} "
            f"grad_norm={entry['grad_norm']:.6g} precond/raw={ratio:.6g} "
            f"curvature_reuse={entry['curvature_reuse']}"
        )
        if entry["step"] % self.diagnostic_log_interval != 0:
            return

        fields = [
            "step",
            "loss",
            "grad_norm",
            "raw_grad_norm",
            "preconditioned_grad_norm",
            "preconditioned_to_raw_ratio",
            "mode",
            "curvature_reuse",
        ]
        write_header = not self.diagnostic_log_path.exists()
        self.diagnostic_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.diagnostic_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({field: entry[field] for field in fields})

    def _cache_preconditioner(self, direction: torch.Tensor, grad_norm: float) -> None:
        direction_norm = float(torch.linalg.vector_norm(direction))
        self._last_preconditioner_gain = direction_norm / max(float(grad_norm), 1e-30)
        self._has_preconditioner = True

    def _diagonal_precondition(self, grad: torch.Tensor) -> torch.Tensor:
        if self._diag_m is None or self._diag_m.numel() != grad.numel():
            self._diag_m = torch.zeros_like(grad)
            self._diag_v = torch.zeros_like(grad)
            self._diag_step = 0
        self._diag_step += 1
        self._diag_m = self.diagonal_beta1 * self._diag_m + (1.0 - self.diagonal_beta1) * grad.detach()
        self._diag_v = self.diagonal_beta2 * self._diag_v + (1.0 - self.diagonal_beta2) * grad.detach().pow(2)
        m_hat = self._diag_m / (1.0 - self.diagonal_beta1 ** self._diag_step)
        v_hat = self._diag_v / (1.0 - self.diagonal_beta2 ** self._diag_step)
        diag = torch.sqrt(v_hat + self._damping + self.regularization).add(self.diagonal_eps)
        return self._scale_preconditioned_direction(-m_hat / diag, float(torch.linalg.vector_norm(grad)))

    def _scale_preconditioned_direction(self, direction: torch.Tensor, grad_norm: float) -> torch.Tensor:
        direction_norm = torch.linalg.vector_norm(direction)
        if float(direction_norm) <= 1e-30:
            return direction
        target_norm = max(float(grad_norm), 1e-30) * self.preconditioner_scale
        return direction * (target_norm / direction_norm.clamp_min(1e-30))

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

    def _assign_flat_grad(self, params, flat_grad: torch.Tensor) -> None:
        offset = 0
        for param in params:
            n = param.numel()
            grad = flat_grad[offset : offset + n].view_as(param).detach().clone()
            param.grad = grad
            offset += n

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
