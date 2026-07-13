"""Functional stable-neutral geometry for small neural-network toys.

This module intentionally uses dense Jacobians and matrices. It is meant for
small probe batches and toy networks where the stable/neutral decomposition can
be inspected directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional

import torch
import torch.nn.functional as F

try:  # PyTorch 2.x
    from torch.func import functional_call
except ImportError:  # pragma: no cover - older PyTorch compatibility
    from torch.nn.utils.stateless import functional_call

from ._tensor import flatten_grads, flatten_tensors, trainable_params
from .navigation import conjugate_gradient

FunctionalRepresentation = Literal["logits", "probabilities", "hidden"]
ResponseKind = Literal["gauss_newton", "empirical_response"]
NullThresholdMode = Literal["absolute", "relative", "spectral_gap", "energy_fraction"]
ResponseSolver = Literal["dense", "low_rank", "implicit_cg"]


@dataclass
class FunctionalGeometry:
    """Configuration for dense functional stable-neutral geometry."""

    null_threshold_mode: NullThresholdMode = "relative"
    null_tol: float = 1e-6
    max_tangent_fraction: float = 0.9
    energy_fraction: float = 0.999


@dataclass
class FunctionalJacobian:
    """Dense Jacobian of Phi(theta; X_probe)."""

    jacobian: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    nullity: int
    theta: torch.Tensor
    phi: torch.Tensor


@dataclass
class FunctionalProjectors:
    """Tangent and normal projectors induced by ker(J_phi)."""

    tangent: torch.Tensor
    normal: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    tangent_rank: int
    normal_rank: int
    residuals: dict[str, float]
    selected_threshold: float = 0.0
    spectral_gap_index: int = -1
    condition_number_normal: float = 0.0
    retained_energy_fraction: float = 0.0


@dataclass
class FunctionalGeoFlowResult:
    """Projected functional GeoFlow direction and diagnostics."""

    direction: torch.Tensor
    gradient: torch.Tensor
    projected_gradient: torch.Tensor
    response: torch.Tensor
    projected_response: torch.Tensor
    projectors: FunctionalProjectors
    g_dot_d: float
    descent_gate_passed: bool
    fallback: bool
    response_eigenvalues: torch.Tensor
    tangent_norm: float
    projected_gradient_tangent_norm: float
    response_solver: str = "dense"
    retained_rank: int = 0
    retained_spectral_energy: float = 1.0
    solver_residual: float = 0.0
    memory_estimate_bytes: int = 0
    jvp_count: int = 0
    vjp_count: int = 0
    null_leakage: float = 0.0


class FunctionalMap:
    """Configurable functional map Phi(theta; X_probe).

    The default map is the flattened logits on a fixed probe batch. For hidden
    representations, pass a callable that accepts ``model`` and ``x_probe`` and
    returns the selected representation tensor.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        x_probe: torch.Tensor,
        representation: FunctionalRepresentation = "logits",
        hidden_getter: Optional[Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if representation == "hidden" and hidden_getter is None:
            raise ValueError("hidden representation requires hidden_getter")
        self.model = model
        self.x_probe = x_probe.detach()
        self.representation = representation
        self.hidden_getter = hidden_getter
        self.named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        self.param_names = [name for name, _ in self.named_params]
        self.param_shapes = [tuple(param.shape) for _, param in self.named_params]
        self.param_sizes = [int(param.numel()) for _, param in self.named_params]

    @property
    def params(self) -> list[torch.nn.Parameter]:
        return [param for _, param in self.named_params]

    def flatten_params(self) -> torch.Tensor:
        return flatten_tensors([param.detach().clone() for _, param in self.named_params])

    def unflatten(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        values = {}
        offset = 0
        for name, shape, size in zip(self.param_names, self.param_shapes, self.param_sizes):
            values[name] = theta[offset : offset + size].reshape(shape)
            offset += size
        return values

    def evaluate(self, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if theta is None:
            output = self._evaluate_model(self.model, self.x_probe)
        else:
            params = self.unflatten(theta)
            output = self._evaluate_functional(params)
        return output.reshape(-1)

    def jacobian(self, theta: Optional[torch.Tensor] = None, tolerance: Optional[float] = None) -> FunctionalJacobian:
        if theta is None:
            theta = self.flatten_params()
        theta = theta.detach().clone().requires_grad_(True)

        def phi(flat_theta: torch.Tensor) -> torch.Tensor:
            return self.evaluate(flat_theta)

        jac = torch.autograd.functional.jacobian(phi, theta, vectorize=False)
        jac = jac.reshape(-1, theta.numel()).detach()
        singular_values = torch.linalg.svdvals(jac)
        rank = svd_rank(singular_values, tolerance=tolerance)
        return FunctionalJacobian(
            jacobian=jac,
            singular_values=singular_values,
            rank=rank,
            nullity=int(theta.numel() - rank),
            theta=theta.detach(),
            phi=phi(theta).detach(),
        )

    def _evaluate_functional(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.representation == "hidden":
            shadow = _FunctionalModelView(self.model, params)
            return self.hidden_getter(shadow, self.x_probe)
        output = functional_call(self.model, params, (self.x_probe,))
        if self.representation == "probabilities":
            return F.softmax(output, dim=-1)
        return output

    def _evaluate_model(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.representation == "hidden":
            return self.hidden_getter(model, x)
        output = model(x)
        if self.representation == "probabilities":
            return F.softmax(output, dim=-1)
        return output


class _FunctionalModelView:
    """Tiny callable wrapper for hidden_getter compatibility."""

    def __init__(self, model: torch.nn.Module, params: dict[str, torch.Tensor]) -> None:
        self.model = model
        self.params = params

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, self.params, (x,))


def svd_rank(singular_values: torch.Tensor, tolerance: Optional[float] = None) -> int:
    if singular_values.numel() == 0:
        return 0
    if tolerance is None:
        tolerance = float(torch.finfo(singular_values.dtype).eps * max(singular_values.shape) * singular_values.max())
    return int((singular_values > tolerance).sum().item())


def select_svd_rank(
    singular_values: torch.Tensor,
    n_params: int,
    mode: NullThresholdMode = "relative",
    null_tol: float = 1e-6,
    max_tangent_fraction: float = 0.9,
    energy_fraction: float = 0.999,
    tolerance: Optional[float] = None,
) -> tuple[int, dict[str, float]]:
    """Select functional rank and expose threshold diagnostics."""

    if singular_values.numel() == 0:
        return 0, {
            "selected_threshold": 0.0,
            "spectral_gap_index": -1,
            "condition_number_normal": 0.0,
            "retained_energy_fraction": 0.0,
        }
    max_tangent_rank = int(max(0, min(n_params - 1, round(max_tangent_fraction * n_params))))
    min_rank = max(0, n_params - max_tangent_rank)
    s0 = float(singular_values.max())
    if tolerance is not None:
        threshold = float(tolerance)
        rank = int((singular_values > threshold).sum().item())
        gap_index = -1
    elif mode == "absolute":
        threshold = float(null_tol)
        rank = int((singular_values > threshold).sum().item())
        gap_index = -1
    elif mode == "relative":
        threshold = float(null_tol * max(s0, 1e-30))
        rank = int((singular_values > threshold).sum().item())
        gap_index = -1
    elif mode == "spectral_gap":
        if singular_values.numel() == 1:
            gap_index = 0
            threshold = float(null_tol * max(s0, 1e-30))
            rank = int((singular_values > threshold).sum().item())
        else:
            denom = singular_values[1:].clamp_min(torch.finfo(singular_values.dtype).tiny)
            gaps = singular_values[:-1] / denom
            gap_index = int(torch.argmax(gaps).item())
            candidate_rank = gap_index + 1
            relative_floor = singular_values > (null_tol * max(s0, 1e-30))
            rank = max(candidate_rank, int(relative_floor.sum().item()), min_rank)
            threshold = float(0.5 * (singular_values[rank - 1] + singular_values[rank])) if rank < singular_values.numel() and rank > 0 else float(null_tol * max(s0, 1e-30))
    elif mode == "energy_fraction":
        energy = singular_values.pow(2)
        total = float(energy.sum())
        if total <= 1e-30:
            rank = min_rank
        else:
            cumulative = torch.cumsum(energy, dim=0) / total
            rank = int((cumulative < energy_fraction).sum().item() + 1)
        threshold = float(singular_values[rank - 1]) if rank > 0 and rank <= singular_values.numel() else 0.0
        gap_index = -1
    else:
        raise ValueError(f"unknown null_threshold_mode: {mode}")
    rank = max(min_rank, min(int(rank), min(n_params, singular_values.numel())))
    if rank > 0:
        normal_values = singular_values[:rank].clamp_min(torch.finfo(singular_values.dtype).tiny)
        condition = float(normal_values.max() / normal_values.min())
    else:
        condition = 0.0
    total_energy = float(singular_values.pow(2).sum())
    retained = float(singular_values[:rank].pow(2).sum() / max(total_energy, 1e-30))
    return rank, {
        "selected_threshold": threshold,
        "spectral_gap_index": float(gap_index),
        "condition_number_normal": condition,
        "retained_energy_fraction": retained,
    }


def functional_projectors(
    jacobian: torch.Tensor,
    tolerance: Optional[float] = None,
    null_threshold_mode: NullThresholdMode = "relative",
    null_tol: float = 1e-6,
    max_tangent_fraction: float = 0.9,
    energy_fraction: float = 0.999,
) -> FunctionalProjectors:
    """Construct P_T onto ker(J_phi) and P_N = I - P_T."""

    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    n_params = jacobian.shape[1]
    identity = torch.eye(n_params, dtype=jacobian.dtype, device=jacobian.device)
    if n_params == 0:
        return FunctionalProjectors(identity, identity, torch.empty(0), 0, 0, 0, {})

    _, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=True)
    rank, diagnostics = select_svd_rank(
        singular_values,
        n_params,
        mode=null_threshold_mode,
        null_tol=null_tol,
        max_tangent_fraction=max_tangent_fraction,
        energy_fraction=energy_fraction,
        tolerance=tolerance,
    )
    tangent_basis = vh[rank:].T
    if tangent_basis.numel() == 0:
        p_tangent = torch.zeros_like(identity)
    else:
        p_tangent = tangent_basis @ tangent_basis.T
    p_normal = identity - p_tangent
    residuals = projector_residuals(jacobian, p_tangent, p_normal)
    return FunctionalProjectors(
        tangent=p_tangent,
        normal=p_normal,
        singular_values=singular_values,
        rank=rank,
        tangent_rank=int(torch.linalg.matrix_rank(p_tangent)),
        normal_rank=int(torch.linalg.matrix_rank(p_normal)),
        residuals=residuals,
        selected_threshold=diagnostics["selected_threshold"],
        spectral_gap_index=int(diagnostics["spectral_gap_index"]),
        condition_number_normal=diagnostics["condition_number_normal"],
        retained_energy_fraction=diagnostics["retained_energy_fraction"],
    )


def projector_residuals(jacobian: torch.Tensor, p_tangent: torch.Tensor, p_normal: torch.Tensor) -> dict[str, float]:
    identity = torch.eye(p_tangent.shape[0], dtype=p_tangent.dtype, device=p_tangent.device)
    return {
        "pt_idempotent": float(torch.linalg.vector_norm(p_tangent @ p_tangent - p_tangent)),
        "pn_idempotent": float(torch.linalg.vector_norm(p_normal @ p_normal - p_normal)),
        "orthogonal": float(torch.linalg.vector_norm(p_tangent @ p_normal)),
        "j_pt": float(torch.linalg.vector_norm(jacobian @ p_tangent)),
        "rank_sum_error": float(abs(int(torch.linalg.matrix_rank(p_tangent)) + int(torch.linalg.matrix_rank(p_normal)) - identity.shape[0])),
    }


def functional_response_operator(
    jacobian: torch.Tensor,
    response_kind: ResponseKind = "gauss_newton",
    weight: Optional[torch.Tensor] = None,
    ridge: float = 0.0,
) -> torch.Tensor:
    """Return A_resp for the functional map.

    ``empirical_response`` is reserved for finite-difference response fitting.
    The first theory-aligned implementation is the dense Gauss-Newton response
    J_phi^T W J_phi.
    """

    if response_kind == "empirical_response":
        raise NotImplementedError("empirical_response API is reserved; use gauss_newton for the dense minimal closure")
    if response_kind != "gauss_newton":
        raise ValueError(f"unknown response_kind: {response_kind}")
    if weight is None:
        response = jacobian.T @ jacobian
    else:
        response = jacobian.T @ weight @ jacobian
    if ridge:
        response = response + ridge * torch.eye(response.shape[0], dtype=response.dtype, device=response.device)
    return 0.5 * (response + response.T)


def low_rank_response_direction(
    jacobian: torch.Tensor,
    gradient: torch.Tensor,
    p_normal: torch.Tensor,
    damping: float,
    functional_rank: Optional[int] = None,
    functional_energy_fraction: float = 0.99,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Solve in top singular vector subspace without building A_resp."""

    _, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
    total_energy = float(singular_values.pow(2).sum())
    if singular_values.numel() == 0 or total_energy <= 1e-30:
        return torch.zeros_like(gradient), {
            "retained_rank": 0.0,
            "retained_spectral_energy": 0.0,
            "solver_residual": 0.0,
            "memory_estimate_bytes": 0.0,
            "jvp_count": 0.0,
            "vjp_count": 0.0,
            "null_leakage": 0.0,
        }
    if functional_rank is None:
        cumulative = torch.cumsum(singular_values.pow(2), dim=0) / total_energy
        rank = int((cumulative < functional_energy_fraction).sum().item() + 1)
    else:
        rank = int(functional_rank)
    rank = max(1, min(rank, singular_values.numel()))
    v = vh[:rank].T
    values = singular_values[:rank].pow(2)
    g_normal = p_normal @ gradient
    coeff = v.T @ g_normal
    direction = -(v @ (coeff / (values + damping)))
    direction = p_normal @ direction
    retained = float(values.sum() / max(total_energy, 1e-30))
    residual = float(torch.linalg.vector_norm((jacobian.T @ (jacobian @ direction)) + damping * direction + g_normal))
    bytes_per = jacobian.element_size()
    memory = int((v.numel() + values.numel()) * bytes_per)
    return direction, {
        "retained_rank": float(rank),
        "retained_spectral_energy": retained,
        "solver_residual": residual,
        "memory_estimate_bytes": float(memory),
        "jvp_count": 0.0,
        "vjp_count": 0.0,
        "null_leakage": 0.0,
    }


class FunctionalJTJOperator:
    """Dense-projector test helper for v -> J^T(Jv) + damping P_N v.

    This helper is retained for dense comparison tests. The production
    ``implicit_cg`` path below uses ``MatrixFreeFunctionalJTJOperator`` plus a
    VJP range finder and does not require a dense ``P_N``.
    """

    def __init__(
        self,
        functional_map: FunctionalMap,
        theta: torch.Tensor,
        p_normal: torch.Tensor,
        damping: float,
    ) -> None:
        self.functional_map = functional_map
        self.theta = theta.detach().clone().requires_grad_(True)
        self.p_normal = p_normal
        self.damping = damping

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(self.theta)
        p_vector = self.p_normal @ vector

        def phi(flat_theta: torch.Tensor) -> torch.Tensor:
            return self.functional_map.evaluate(flat_theta)

        _, jv = torch.autograd.functional.jvp(phi, (self.theta,), (p_vector,), create_graph=False)
        _, vjp = torch.autograd.functional.vjp(phi, self.theta, v=jv, create_graph=False)
        return self.p_normal @ vjp + self.damping * p_vector


class MatrixFreeFunctionalJTJOperator:
    """Matrix-free operator for v -> J^T(Jv) using JVP/VJP only."""

    def __init__(self, functional_map: FunctionalMap) -> None:
        self.functional_map = functional_map
        self.theta = functional_map.flatten_params().detach().clone().requires_grad_(True)
        self.jvp_count = 0
        self.vjp_count = 0

    def phi(self, flat_theta: torch.Tensor) -> torch.Tensor:
        return self.functional_map.evaluate(flat_theta)

    def jtj(self, vector: torch.Tensor) -> torch.Tensor:
        vector = vector.to(self.theta)
        _, jv = torch.autograd.functional.jvp(self.phi, (self.theta,), (vector,), create_graph=False)
        self.jvp_count += 1
        _, vjp = torch.autograd.functional.vjp(self.phi, self.theta, v=jv, create_graph=False)
        self.vjp_count += 1
        return vjp

    def vjp(self, output_vector: torch.Tensor) -> torch.Tensor:
        _, vjp = torch.autograd.functional.vjp(self.phi, self.theta, v=output_vector.to(self.theta), create_graph=False)
        self.vjp_count += 1
        return vjp


def randomized_normal_basis(
    functional_map: FunctionalMap,
    rank: Optional[int] = None,
    energy_fraction: float = 0.99,
    oversample: int = 4,
    seed: int = 1234,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Estimate range(J^T) with VJP probes without building J or P_N."""

    operator = MatrixFreeFunctionalJTJOperator(functional_map)
    phi = functional_map.evaluate()
    output_dim = int(phi.numel())
    n_params = int(operator.theta.numel())
    if output_dim == 0 or n_params == 0:
        return torch.empty(n_params, 0), {
            "retained_rank": 0.0,
            "retained_spectral_energy": 0.0,
            "jvp_count": float(operator.jvp_count),
            "vjp_count": float(operator.vjp_count),
            "memory_estimate_bytes": 0.0,
        }
    if rank is None:
        probe_count = output_dim
        probes = torch.eye(output_dim, dtype=phi.dtype, device=phi.device)
    else:
        probe_count = min(output_dim, max(1, int(rank) + int(oversample)))
        generator = torch.Generator(device=phi.device).manual_seed(seed) if phi.device.type != "cpu" else torch.Generator().manual_seed(seed)
        probes = torch.randn(probe_count, output_dim, dtype=phi.dtype, device=phi.device, generator=generator)
    columns = [operator.vjp(probe).detach() for probe in probes]
    sample = torch.stack(columns, dim=1)
    q, r = torch.linalg.qr(sample, mode="reduced")
    diag = torch.abs(torch.diag(r))
    if diag.numel() == 0:
        kept = 0
    elif rank is not None:
        kept = min(int(rank), q.shape[1])
    else:
        energy = diag.pow(2)
        total = float(energy.sum())
        kept = int((torch.cumsum(energy, dim=0) / max(total, 1e-30) < energy_fraction).sum().item() + 1)
        kept = min(max(1, kept), q.shape[1])
    q = q[:, :kept].contiguous()
    retained = float(diag[:kept].pow(2).sum() / max(float(diag.pow(2).sum()), 1e-30)) if diag.numel() else 0.0
    return q, {
        "retained_rank": float(kept),
        "retained_spectral_energy": retained,
        "jvp_count": float(operator.jvp_count),
        "vjp_count": float(operator.vjp_count),
        "memory_estimate_bytes": float(q.numel() * q.element_size()),
    }


def implicit_cg_response_direction(
    functional_map: FunctionalMap,
    gradient: torch.Tensor,
    damping: float,
    functional_rank: Optional[int] = None,
    functional_energy_fraction: float = 0.99,
    cg_max_iter: int = 64,
    cg_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    q, basis_info = randomized_normal_basis(
        functional_map,
        rank=functional_rank,
        energy_fraction=functional_energy_fraction,
    )
    operator = MatrixFreeFunctionalJTJOperator(functional_map)
    if q.numel() == 0:
        return torch.zeros_like(gradient), {
            "retained_rank": 0.0,
            "retained_spectral_energy": 0.0,
            "solver_residual": 0.0,
            "memory_estimate_bytes": 0.0,
            "jvp_count": basis_info["jvp_count"],
            "vjp_count": basis_info["vjp_count"],
            "null_leakage": 0.0,
        }
    q = q.to(device=gradient.device, dtype=gradient.dtype)
    q_grad = q.T @ gradient

    def coeff_matvec(coeff: torch.Tensor) -> torch.Tensor:
        vector = q @ coeff
        return q.T @ operator.jtj(vector) + damping * coeff

    result = conjugate_gradient(coeff_matvec, -q_grad, max_iter=cg_max_iter, tolerance=cg_tolerance)
    direction = q @ result.solution
    projected = q @ (q.T @ direction)
    null_leakage = float(torch.linalg.vector_norm(direction - projected))
    residual = float(torch.linalg.vector_norm(coeff_matvec(result.solution) + q_grad))
    return direction, {
        "retained_rank": basis_info["retained_rank"],
        "retained_spectral_energy": basis_info["retained_spectral_energy"],
        "solver_residual": residual,
        "memory_estimate_bytes": basis_info["memory_estimate_bytes"] + float(q.numel() * gradient.element_size()),
        "jvp_count": basis_info["jvp_count"] + float(operator.jvp_count),
        "vjp_count": basis_info["vjp_count"] + float(operator.vjp_count),
        "null_leakage": null_leakage,
    }


def projected_functional_geoflow_direction(
    model: torch.nn.Module,
    loss: torch.Tensor,
    x_probe: torch.Tensor,
    params: Optional[Iterable[torch.nn.Parameter]] = None,
    representation: FunctionalRepresentation = "logits",
    response_kind: ResponseKind = "gauss_newton",
    damping: float = 1e-3,
    max_update_norm: Optional[float] = None,
    descent_gate: bool = True,
    null_threshold_mode: NullThresholdMode = "relative",
    null_tol: float = 1e-6,
    max_tangent_fraction: float = 0.9,
    energy_fraction: float = 0.999,
    response_solver: ResponseSolver = "dense",
    functional_rank: Optional[int] = None,
    functional_energy_fraction: float = 0.99,
    cg_max_iter: int = 64,
    cg_tolerance: float = 1e-6,
) -> FunctionalGeoFlowResult:
    """Compute d = -pinv(P_N A_resp P_N + damping P_N) P_N g."""

    if params is None:
        params = trainable_params(model.parameters())
    else:
        params = trainable_params(params)
    fmap = FunctionalMap(model, x_probe, representation=representation)
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    gradient = flatten_grads(grads, params).detach()
    if response_solver == "implicit_cg":
        direction, solver_info = implicit_cg_response_direction(
            fmap,
            gradient,
            damping,
            functional_rank=functional_rank,
            functional_energy_fraction=functional_energy_fraction,
            cg_max_iter=cg_max_iter,
            cg_tolerance=cg_tolerance,
        )
        if max_update_norm is not None:
            norm = torch.linalg.vector_norm(direction)
            if float(norm) > max_update_norm:
                direction = direction * (max_update_norm / norm.clamp_min(1e-30))
        g_dot_d = float(torch.dot(gradient, direction)) if gradient.numel() else 0.0
        fallback = False
        if descent_gate and g_dot_d >= 0.0:
            direction = -direction
            fallback = True
            g_dot_d = float(torch.dot(gradient, direction)) if gradient.numel() else 0.0
        empty = torch.empty(0, device=gradient.device, dtype=gradient.dtype)
        projectors = FunctionalProjectors(
            tangent=torch.empty(0, 0, device=gradient.device, dtype=gradient.dtype),
            normal=torch.empty(0, 0, device=gradient.device, dtype=gradient.dtype),
            singular_values=empty,
            rank=int(solver_info["retained_rank"]),
            tangent_rank=max(0, int(gradient.numel() - solver_info["retained_rank"])),
            normal_rank=int(solver_info["retained_rank"]),
            residuals={},
            retained_energy_fraction=float(solver_info["retained_spectral_energy"]),
        )
        return FunctionalGeoFlowResult(
            direction=direction.detach(),
            gradient=gradient.detach(),
            projected_gradient=gradient.detach(),
            response=empty,
            projected_response=empty,
            projectors=projectors,
            g_dot_d=g_dot_d,
            descent_gate_passed=g_dot_d < 0.0,
            fallback=fallback,
            response_eigenvalues=empty,
            tangent_norm=float(solver_info["null_leakage"]),
            projected_gradient_tangent_norm=0.0,
            response_solver=response_solver,
            retained_rank=int(solver_info["retained_rank"]),
            retained_spectral_energy=float(solver_info["retained_spectral_energy"]),
            solver_residual=float(solver_info["solver_residual"]),
            memory_estimate_bytes=int(solver_info["memory_estimate_bytes"]),
            jvp_count=int(solver_info["jvp_count"]),
            vjp_count=int(solver_info["vjp_count"]),
            null_leakage=float(solver_info["null_leakage"]),
        )

    fjac = fmap.jacobian()
    projectors = functional_projectors(
        fjac.jacobian,
        null_threshold_mode=null_threshold_mode,
        null_tol=null_tol,
        max_tangent_fraction=max_tangent_fraction,
        energy_fraction=energy_fraction,
    )
    p_normal = projectors.normal.to(device=gradient.device, dtype=gradient.dtype)
    p_tangent = projectors.tangent.to(device=gradient.device, dtype=gradient.dtype)
    g_normal = p_normal @ gradient
    if response_solver == "dense":
        response = functional_response_operator(fjac.jacobian.to(dtype=gradient.dtype), response_kind=response_kind)
        response = response.to(device=gradient.device, dtype=gradient.dtype)
        projected_response = p_normal @ response @ p_normal + damping * p_normal
        rhs = -g_normal
        direction = torch.linalg.pinv(projected_response) @ rhs
        direction = p_normal @ direction
        eigvals = torch.linalg.eigvalsh(0.5 * (projected_response + projected_response.T))
        solver_info = {
            "retained_rank": float(projectors.normal_rank),
            "retained_spectral_energy": 1.0,
            "solver_residual": float(torch.linalg.vector_norm(projected_response @ direction + g_normal)),
            "memory_estimate_bytes": float((response.numel() + projected_response.numel()) * gradient.element_size()),
            "jvp_count": 0.0,
            "vjp_count": 0.0,
            "null_leakage": 0.0,
        }
    elif response_solver == "low_rank":
        direction, solver_info = low_rank_response_direction(
            fjac.jacobian.to(device=gradient.device, dtype=gradient.dtype),
            gradient,
            p_normal,
            damping,
            functional_rank=functional_rank,
            functional_energy_fraction=functional_energy_fraction,
        )
        response = torch.empty(0, device=gradient.device, dtype=gradient.dtype)
        projected_response = torch.empty(0, device=gradient.device, dtype=gradient.dtype)
        eigvals = fjac.singular_values.to(device=gradient.device, dtype=gradient.dtype).pow(2) + damping
    elif response_solver == "implicit_cg":
        raise AssertionError("implicit_cg is handled before dense Jacobian construction")
    else:
        raise ValueError(f"unknown response_solver: {response_solver}")
    if max_update_norm is not None:
        norm = torch.linalg.vector_norm(direction)
        if float(norm) > max_update_norm:
            direction = direction * (max_update_norm / norm.clamp_min(1e-30))
    g_dot_d = float(torch.dot(gradient, direction)) if gradient.numel() else 0.0
    fallback = False
    if descent_gate and g_dot_d >= 0.0:
        direction = -g_normal
        fallback = True
        if max_update_norm is not None:
            norm = torch.linalg.vector_norm(direction)
            if float(norm) > max_update_norm:
                direction = direction * (max_update_norm / norm.clamp_min(1e-30))
        g_dot_d = float(torch.dot(gradient, direction)) if gradient.numel() else 0.0
    return FunctionalGeoFlowResult(
        direction=direction.detach(),
        gradient=gradient.detach(),
        projected_gradient=g_normal.detach(),
        response=response.detach(),
        projected_response=projected_response.detach(),
        projectors=projectors,
        g_dot_d=g_dot_d,
        descent_gate_passed=g_dot_d < 0.0,
        fallback=fallback,
        response_eigenvalues=eigvals.detach(),
        tangent_norm=float(torch.linalg.vector_norm(p_tangent @ direction.detach())),
        projected_gradient_tangent_norm=float(torch.linalg.vector_norm(p_tangent @ g_normal.detach())),
        response_solver=response_solver,
        retained_rank=int(solver_info["retained_rank"]),
        retained_spectral_energy=float(solver_info["retained_spectral_energy"]),
        solver_residual=float(solver_info["solver_residual"]),
        memory_estimate_bytes=int(solver_info["memory_estimate_bytes"]),
        jvp_count=int(solver_info.get("jvp_count", 0.0)),
        vjp_count=int(solver_info.get("vjp_count", 0.0)),
        null_leakage=float(solver_info.get("null_leakage", 0.0)),
    )
