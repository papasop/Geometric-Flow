"""Curvature measurement through Hessian-vector products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from ._tensor import flatten_grads, trainable_params

CurvatureKind = Literal["hessian", "fisher"]


@dataclass
class CurvatureOperator:
    """Implicit local curvature operator A(v).

    The default implementation is Hessian-vector product based. The ``fisher``
    kind currently uses the empirical gradient outer-product diagonal as a
    stable approximation, which is useful as a positive fallback for classifiers.
    """

    loss: torch.Tensor
    params: list[torch.nn.Parameter]
    damping: float = 1e-3
    kind: CurvatureKind = "hessian"
    regularization: float = 0.1

    def __post_init__(self) -> None:
        if self.kind not in {"hessian", "fisher"}:
            raise ValueError(f"unknown curvature kind: {self.kind}")
        self.params = trainable_params(self.params)
        self._grads = torch.autograd.grad(
            self.loss,
            self.params,
            create_graph=self.kind == "hessian",
            retain_graph=True,
            allow_unused=True,
        )
        self.gradient = flatten_grads(self._grads, self.params).detach()
        self.size = int(self.gradient.numel())
        if self.kind == "fisher":
            self._fisher_diag = self.gradient.pow(2).detach().clamp_min(self.damping)
        else:
            self._fisher_diag = None
        self._regularization = float(self.regularization)

    def regularize(self, method: Literal["tikhonov", "identity"] = "tikhonov", alpha: float = 0.1):
        """Add an implicit identity regularizer to every curvature matvec."""

        if method not in {"tikhonov", "identity"}:
            raise ValueError(f"unknown regularization method: {method}")
        if alpha < 0:
            raise ValueError("regularization alpha must be non-negative")
        self._regularization = float(alpha)
        return self

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply the damped curvature operator to a flat vector."""

        vector = vector.to(device=self.gradient.device, dtype=self.gradient.dtype)
        if self.kind == "fisher":
            return self._fisher_diag * vector + self._regularization * vector

        grad_dot_vec = torch.dot(flatten_grads(self._grads, self.params), vector)
        hvp = torch.autograd.grad(
            grad_dot_vec,
            self.params,
            retain_graph=True,
            allow_unused=True,
        )
        flat_hvp = flatten_grads(hvp, self.params).detach()
        return flat_hvp + (self.damping + self._regularization) * vector

    def rayleigh(self, vector: torch.Tensor) -> float:
        denom = torch.dot(vector, vector).clamp_min(1e-30)
        return float(torch.dot(vector, self.matvec(vector)) / denom)


def compute_curvature(
    model: torch.nn.Module,
    loss: torch.Tensor,
    data=None,
    damping: float = 1e-3,
    kind: CurvatureKind = "hessian",
    regularization: float = 0.1,
) -> CurvatureOperator:
    """Return an implicit local curvature operator for ``model``.

    ``data`` is accepted for API symmetry with higher-level callers; the loss
    should already be computed from the relevant batch.
    """

    del data
    return CurvatureOperator(
        loss=loss,
        params=list(model.parameters()),
        damping=damping,
        kind=kind,
        regularization=regularization,
    )


def hutchinson_trace(
    curvature_op: CurvatureOperator,
    samples: int = 8,
    distribution: Literal["rademacher", "normal"] = "rademacher",
) -> float:
    """Estimate trace(A) with random HVP probes."""

    if curvature_op.size == 0:
        return 0.0

    estimates = []
    for _ in range(samples):
        if distribution == "normal":
            probe = torch.randn_like(curvature_op.gradient)
        else:
            probe = torch.empty_like(curvature_op.gradient).bernoulli_(0.5).mul_(2).sub_(1)
        estimates.append(torch.dot(probe, curvature_op.matvec(probe)))
    return float(torch.stack(estimates).mean())
