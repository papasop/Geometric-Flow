"""Fixed-rank matrix manifold utilities.

This module contains the small D7 kernel: tangent projection on the rank-r
matrix manifold and rank-preserving SVD retraction. It is intentionally
independent of optimizers and LoRA adapters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FixedRankDiagnostics:
    """Diagnostics for a fixed-rank tangent/retraction step."""

    tangent_residual: float
    retraction_relative_error: float
    numerical_rank: int
    rank_violation: bool
    smallest_retained_singular_value: float


class FixedRankManifold:
    """Geometry of the fixed-rank matrix manifold around a product matrix."""

    def __init__(
        self,
        rank: int,
        svd_floor: float = 1e-10,
        rank_tolerance: float | None = None,
    ) -> None:
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if svd_floor < 0:
            raise ValueError("svd_floor must be non-negative")
        if rank_tolerance is not None and rank_tolerance < 0:
            raise ValueError("rank_tolerance must be non-negative")
        self.rank = int(rank)
        self.svd_floor = float(svd_floor)
        self.rank_tolerance = rank_tolerance

    def factor_basis(
        self,
        matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the retained SVD basis ``U_r, S_r, V_r`` for ``matrix``."""

        self._validate_matrix(matrix)
        if self.rank > min(matrix.shape):
            raise ValueError("rank must be <= min(matrix.shape)")
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        retained = min(self.rank, s.numel())
        return u[:, :retained], s[:retained].clamp_min(self.svd_floor), vh[:retained, :].transpose(-2, -1)

    def project_tangent(
        self,
        matrix: torch.Tensor,
        ambient: torch.Tensor,
    ) -> torch.Tensor:
        """Project an ambient proposal onto the tangent space at ``matrix``."""

        self._validate_pair(matrix, ambient)
        u, _, v = self.factor_basis(matrix)
        uu_t_d = u @ (u.transpose(-2, -1) @ ambient)
        d_vv_t = (ambient @ v) @ v.transpose(-2, -1)
        uu_t_d_vv_t = u @ (u.transpose(-2, -1) @ ambient @ v) @ v.transpose(-2, -1)
        return uu_t_d + d_vv_t - uu_t_d_vv_t

    def tangent_residual(
        self,
        matrix: torch.Tensor,
        candidate: torch.Tensor,
    ) -> float:
        """Return ``||D - P_T(D)|| / max(||D||, tiny)``."""

        self._validate_pair(matrix, candidate)
        projected = self.project_tangent(matrix, candidate)
        denom = candidate.norm().clamp_min(torch.finfo(candidate.dtype).tiny)
        return float(((candidate - projected).norm() / denom).detach().cpu())

    def retract(
        self,
        matrix: torch.Tensor,
        tangent_step: torch.Tensor,
    ) -> tuple[torch.Tensor, FixedRankDiagnostics]:
        """Retract ``matrix + tangent_step`` back to rank ``self.rank``."""

        self._validate_pair(matrix, tangent_step)
        before = matrix
        candidate = matrix + tangent_step
        u, s, vh = torch.linalg.svd(candidate, full_matrices=False)
        retained = min(self.rank, s.numel())
        new_matrix = (u[:, :retained] * s[:retained]) @ vh[:retained, :]
        numerical_rank = self.numerical_rank(new_matrix)
        step_norm = tangent_step.norm()
        if float(step_norm.detach().cpu()) == 0.0:
            retraction_error = torch.zeros((), dtype=tangent_step.dtype, device=tangent_step.device)
        else:
            denom = step_norm.clamp_min(torch.finfo(tangent_step.dtype).tiny)
            retraction_error = (((new_matrix - before) - tangent_step).norm() / denom).detach()
        smallest = float(s[:retained].min().detach().cpu()) if retained > 0 else 0.0
        diagnostics = FixedRankDiagnostics(
            tangent_residual=self.tangent_residual(matrix, tangent_step),
            retraction_relative_error=float(retraction_error.cpu()),
            numerical_rank=numerical_rank,
            rank_violation=bool(numerical_rank > self.rank),
            smallest_retained_singular_value=smallest,
        )
        return new_matrix, diagnostics

    def numerical_rank(
        self,
        matrix: torch.Tensor,
    ) -> int:
        """Return numerical rank using a stable SVD threshold."""

        self._validate_matrix(matrix)
        s = torch.linalg.svdvals(matrix)
        if s.numel() == 0:
            return 0
        tol = self._rank_threshold(matrix, s)
        return int((s > tol).sum().item())

    def _rank_threshold(self, matrix: torch.Tensor, singular_values: torch.Tensor) -> torch.Tensor:
        if self.rank_tolerance is not None:
            return torch.as_tensor(self.rank_tolerance, dtype=singular_values.dtype, device=singular_values.device)
        eps = torch.finfo(matrix.dtype).eps
        return torch.as_tensor(max(matrix.shape) * eps, dtype=singular_values.dtype, device=singular_values.device) * singular_values.max()

    def _validate_matrix(self, matrix: torch.Tensor) -> None:
        if matrix.ndim != 2:
            raise ValueError("fixed-rank geometry expects a 2-D matrix")
        if not matrix.is_floating_point():
            raise ValueError("fixed-rank geometry expects a floating-point matrix")
        if self.rank > min(matrix.shape):
            raise ValueError("rank must be <= min(matrix.shape)")

    def _validate_pair(self, matrix: torch.Tensor, ambient: torch.Tensor) -> None:
        self._validate_matrix(matrix)
        self._validate_matrix(ambient)
        if matrix.shape != ambient.shape:
            raise ValueError("matrix and proposal must have the same shape")
        if matrix.device != ambient.device:
            raise ValueError("matrix and proposal must be on the same device")
        if matrix.dtype != ambient.dtype:
            raise ValueError("matrix and proposal must have the same dtype")
