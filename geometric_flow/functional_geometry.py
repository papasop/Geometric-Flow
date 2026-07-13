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

FunctionalRepresentation = Literal["logits", "probabilities", "hidden"]
ResponseKind = Literal["gauss_newton", "empirical_response"]
NullThresholdMode = Literal["absolute", "relative", "spectral_gap", "energy_fraction"]


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
) -> FunctionalGeoFlowResult:
    """Compute d = -pinv(P_N A_resp P_N + damping P_N) P_N g."""

    if params is None:
        params = trainable_params(model.parameters())
    else:
        params = trainable_params(params)
    fmap = FunctionalMap(model, x_probe, representation=representation)
    fjac = fmap.jacobian()
    projectors = functional_projectors(
        fjac.jacobian,
        null_threshold_mode=null_threshold_mode,
        null_tol=null_tol,
        max_tangent_fraction=max_tangent_fraction,
        energy_fraction=energy_fraction,
    )
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    gradient = flatten_grads(grads, params).detach()
    p_normal = projectors.normal.to(device=gradient.device, dtype=gradient.dtype)
    p_tangent = projectors.tangent.to(device=gradient.device, dtype=gradient.dtype)
    g_normal = p_normal @ gradient
    response = functional_response_operator(fjac.jacobian.to(dtype=gradient.dtype), response_kind=response_kind)
    response = response.to(device=gradient.device, dtype=gradient.dtype)
    projected_response = p_normal @ response @ p_normal + damping * p_normal
    rhs = -g_normal
    direction = torch.linalg.pinv(projected_response) @ rhs
    direction = p_normal @ direction
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
    eigvals = torch.linalg.eigvalsh(0.5 * (projected_response + projected_response.T))
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
    )
