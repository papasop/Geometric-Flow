"""Fixed-rank product-coordinate Adam optimizer."""

from __future__ import annotations

from typing import Callable

import torch
from torch.optim import Optimizer

from .fixed_rank import FixedRankManifold
from .product_state import ProductState
from .trust_region import HeldOutTrustRegion, TrustRegionResult


class FixedRankFunctionalAdam(Optimizer):
    """Adam in invariant product coordinates with fixed-rank retraction.

    This optimizer treats product tensors ``M`` as the state variables. It is
    not factor-space Adam, and it does not call the flat update path used by
    :class:`GeometricOptimizer`.
    """

    def __init__(
        self,
        product_state: ProductState,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        max_update_norm: float | None = None,
        trust_region: HeldOutTrustRegion | None = None,
        svd_floor: float = 1e-10,
        rank_tolerance: float | None = None,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if max_update_norm is not None and max_update_norm <= 0:
            raise ValueError("max_update_norm must be positive when set")
        self.product_state = product_state
        self.trust_region = trust_region
        self.svd_floor = svd_floor
        self.rank_tolerance = rank_tolerance
        self.last_diagnostics: dict[str, object] = {}
        super().__init__(
            product_state.parameters(),
            dict(lr=lr, betas=betas, eps=eps, max_update_norm=max_update_norm),
        )

    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        *,
        calibration_closure: Callable[[], torch.Tensor] | None = None,
    ):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        max_update_norm = group["max_update_norm"]
        base_steps: dict[str, torch.Tensor] = {}
        pre_diagnostics: dict[str, dict[str, float | bool]] = {}

        for product in self.product_state.products:
            param = product.tensor
            if param.grad is None:
                continue
            grad = param.grad
            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            step_index = int(state["step"])
            bias_correction1 = 1 - beta1**step_index
            bias_correction2 = 1 - beta2**step_index
            denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
            ambient = exp_avg.div(bias_correction1).div(denom).mul(-lr)

            manifold = self._manifold(product.rank)
            tangent = manifold.project_tangent(param, ambient)
            tangent_residual = manifold.tangent_residual(param, tangent)
            if max_update_norm is not None:
                norm = tangent.norm()
                if float(norm.detach().cpu()) > max_update_norm:
                    tangent = tangent * (max_update_norm / norm.clamp_min(torch.finfo(tangent.dtype).tiny))
            base_steps[product.name] = tangent.detach().clone()
            pre_diagnostics[product.name] = {
                "ambient_proposal_norm": float(ambient.norm().detach().cpu()),
                "tangent_proposal_norm": float(tangent.norm().detach().cpu()),
                "tangent_residual": float(tangent_residual),
            }

        selected_scale = 1.0
        accepted = True
        hit_max_scale = False
        if self.trust_region is not None:
            if calibration_closure is None:
                raise RuntimeError("calibration_closure is required when trust_region is enabled")
            trust_result = self.trust_region.select(self.product_state, base_steps, calibration_closure)
            selected_scale = trust_result.selected_scale
            accepted = trust_result.accepted
            hit_max_scale = trust_result.hit_max_scale
        else:
            trust_result = TrustRegionResult(1.0, True, False, float("nan"), float("nan"))

        diagnostics: dict[str, dict[str, float | bool]] = {}
        if base_steps and selected_scale != 0.0:
            with torch.no_grad():
                for product in self.product_state.products:
                    if product.name not in base_steps:
                        continue
                    param = product.tensor
                    before = param.detach().clone()
                    manifold = self._manifold(product.rank)
                    tangent = manifold.project_tangent(param, base_steps[product.name] * selected_scale)
                    new_param, diag = manifold.retract(param, tangent)
                    param.copy_(new_param)
                    realized = param - before
                    entry = dict(pre_diagnostics[product.name])
                    entry.update(
                        {
                            "realized_update_norm": float(realized.norm().detach().cpu()),
                            "retraction_relative_error": diag.retraction_relative_error,
                            "numerical_rank": diag.numerical_rank,
                            "rank_violation": diag.rank_violation,
                            "selected_scale": float(selected_scale),
                            "accepted": bool(accepted),
                            "hit_max_scale": bool(hit_max_scale),
                        }
                    )
                    diagnostics[product.name] = entry
        else:
            for name, entry in pre_diagnostics.items():
                copied = dict(entry)
                copied.update(
                    {
                        "realized_update_norm": 0.0,
                        "retraction_relative_error": 0.0,
                        "numerical_rank": self._rank_for_name(name),
                        "rank_violation": False,
                        "selected_scale": float(selected_scale),
                        "accepted": bool(accepted),
                        "hit_max_scale": bool(hit_max_scale),
                    }
                )
                diagnostics[name] = copied

        self.last_diagnostics = {
            "products": diagnostics,
            "aggregate": self._aggregate(diagnostics),
            "trust_region": trust_result,
        }
        return loss

    def _manifold(self, rank: int) -> FixedRankManifold:
        return FixedRankManifold(rank, svd_floor=self.svd_floor, rank_tolerance=self.rank_tolerance)

    def _rank_for_name(self, name: str) -> int:
        for product in self.product_state.products:
            if product.name == name:
                return self._manifold(product.rank).numerical_rank(product.tensor)
        return 0

    def _aggregate(self, diagnostics: dict[str, dict[str, float | bool]]) -> dict[str, float]:
        if not diagnostics:
            return {}
        numeric_keys = [
            "ambient_proposal_norm",
            "tangent_proposal_norm",
            "tangent_residual",
            "realized_update_norm",
            "retraction_relative_error",
            "numerical_rank",
            "rank_violation",
            "selected_scale",
            "accepted",
            "hit_max_scale",
        ]
        aggregate = {}
        for key in numeric_keys:
            values = [float(entry[key]) for entry in diagnostics.values() if key in entry]
            if values:
                aggregate[f"mean_{key}"] = sum(values) / len(values)
        return aggregate
