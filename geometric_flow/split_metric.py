"""Core split-metric geometry for low-rank products.

The public convention is ``M = B A`` with ``A`` shaped ``(rank, in)`` and
``B`` shaped ``(out, rank)``.  The functions in this module are intentionally
stateless so the variational direction can be tested and reused independently
of any finite-step optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InverseGramDiagnostics:
    """Numerical diagnostics for the inverse-Gram direction."""

    condition_a: float
    condition_b: float
    used_pinv_a: bool
    used_pinv_b: bool

    @property
    def condition_max(self) -> float:
        return max(self.condition_a, self.condition_b)

    @property
    def fallback_count(self) -> int:
        return int(self.used_pinv_a) + int(self.used_pinv_b)


@dataclass(frozen=True)
class InverseGramDirection:
    """Split-metric quotient direction for ``M = B A``."""

    velocity_A: torch.Tensor
    velocity_B: torch.Tensor
    diagnostics: InverseGramDiagnostics


def product_velocity(
    A: torch.Tensor,
    B: torch.Tensor,
    velocity_A: torch.Tensor,
    velocity_B: torch.Tensor,
) -> torch.Tensor:
    """Return the first-order product velocity ``V_B A + B V_A``."""

    _validate_factor_shapes(A, B, velocity_A, velocity_B)
    return velocity_B @ A + B @ velocity_A


def split_metric_norm(
    A: torch.Tensor,
    B: torch.Tensor,
    velocity_A: torch.Tensor,
    velocity_B: torch.Tensor,
) -> torch.Tensor:
    """Return ``sqrt(||B V_A||_F^2 + ||V_B A||_F^2)``.

    This is the split executed-information norm used by the local variational
    theorem. It is distinct from the net product-capacity
    ``||V_B A + B V_A||_F`` used by the current public capacity controller.
    """

    _validate_factor_shapes(A, B, velocity_A, velocity_B)
    return torch.sqrt((B @ velocity_A).pow(2).sum() + (velocity_B @ A).pow(2).sum())


def product_capacity(
    A: torch.Tensor,
    B: torch.Tensor,
    velocity_A: torch.Tensor,
    velocity_B: torch.Tensor,
) -> torch.Tensor:
    """Return the net product-motion capacity ``||V_B A + B V_A||_F``."""

    return product_velocity(A, B, velocity_A, velocity_B).norm()


def inverse_gram_direction(
    A: torch.Tensor,
    B: torch.Tensor,
    grad_A: torch.Tensor,
    grad_B: torch.Tensor,
    *,
    scale: float | torch.Tensor = 1.0,
    condition_limit: float = 1e10,
) -> InverseGramDirection:
    """Return the split-metric steepest direction for ``M = B A``.

    The full-rank ordinary-inverse branch computes

    ``V_A = -scale * (B.T @ B)^-1 @ grad_A``

    and

    ``V_B = -scale * grad_B @ (A @ A.T)^-1``.

    If either Gram matrix is ill-conditioned, a Moore-Penrose pseudoinverse is
    used as a numerical safeguard. That fallback is useful in practice but does
    not generally preserve exact covariance under arbitrary non-orthogonal
    gauge transformations.
    """

    if condition_limit <= 1:
        raise ValueError("condition_limit must be > 1")
    _validate_factor_shapes(A, B, grad_A, grad_B)
    inv_b, cond_b, pinv_b = _safe_inverse(B.transpose(-2, -1) @ B, condition_limit)
    inv_a, cond_a, pinv_a = _safe_inverse(A @ A.transpose(-2, -1), condition_limit)
    velocity_a = -scale * (inv_b @ grad_A)
    velocity_b = -scale * (grad_B @ inv_a)
    return InverseGramDirection(
        velocity_A=velocity_a,
        velocity_B=velocity_b,
        diagnostics=InverseGramDiagnostics(
            condition_a=cond_a,
            condition_b=cond_b,
            used_pinv_a=pinv_a,
            used_pinv_b=pinv_b,
        ),
    )


def _safe_inverse(gram: torch.Tensor, condition_limit: float) -> tuple[torch.Tensor, float, bool]:
    condition = torch.linalg.cond(gram.detach())
    condition_value = float(condition.cpu())
    if condition_value < condition_limit:
        return torch.linalg.inv(gram), condition_value, False
    return torch.linalg.pinv(gram, rtol=1.0 / condition_limit), condition_value, True


def _validate_factor_shapes(
    A: torch.Tensor,
    B: torch.Tensor,
    first_like_A: torch.Tensor,
    second_like_B: torch.Tensor,
) -> None:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be matrices")
    if B.shape[1] != A.shape[0]:
        raise ValueError("factor shapes must satisfy A=(rank,in), B=(out,rank)")
    if first_like_A.shape != A.shape:
        raise ValueError("A-like tensor must have the same shape as A")
    if second_like_B.shape != B.shape:
        raise ValueError("B-like tensor must have the same shape as B")
