"""Experimental fixed-rank and quotient-flow optimizers."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .fixed_rank import FixedRankManifold
from .product_state import ProductState
from .trust_region import HeldOutTrustRegion, TrustRegionResult


class SubsteppedQuotientFlow(Optimizer):
    """Experimental quotient-flow integrator for LoRA factor modules.

    The optimizer has no Adam-style persistent moment tensors. Each ``step``
    consumes the currently available factor gradients and applies one quotient
    substep. ``macro_step`` calls a user closure once per substep so gradients
    can be recomputed after every factor update.
    """

    def __init__(
        self,
        params=None,
        *,
        factor_modules=None,
        macro_lr: float,
        substeps: int = 2,
        clip_norm: Optional[float] = None,
        balance_after_substep: bool = True,
        gram_condition_limit: float = 1e10,
    ) -> None:
        modules = factor_modules if factor_modules is not None else params
        self.factor_modules = self._collect_factor_modules(modules)
        if macro_lr <= 0:
            raise ValueError("macro_lr must be positive")
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be an integer >= 1")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError("clip_norm must be None or positive")
        if gram_condition_limit <= 1:
            raise ValueError("gram_condition_limit must be > 1")
        self.macro_lr = float(macro_lr)
        self.substeps = int(substeps)
        self.local_lr = self.macro_lr / self.substeps
        self.clip_norm = clip_norm
        self.balance_after_substep = bool(balance_after_substep)
        self.gram_condition_limit = float(gram_condition_limit)
        self.condition_max = 0.0
        self.fallback_count = 0
        self.balance_residual_max = 0.0
        self.last_update_norm = 0.0
        self.last_clip_scale = 1.0
        self.last_diagnostics = self._diagnostics()
        optimizer_params = []
        for module in self.factor_modules:
            optimizer_params.extend([module.A, module.B])
        super().__init__(
            optimizer_params,
            dict(
                macro_lr=self.macro_lr,
                local_lr=self.local_lr,
                substeps=self.substeps,
                clip_norm=self.clip_norm,
            ),
        )

    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        """Execute one quotient substep using current gradients."""

        loss = closure() if closure is not None else None
        updates = []
        squared_norm = None
        for module in self.factor_modules:
            if module.A.grad is None or module.B.grad is None:
                raise RuntimeError("SubsteppedQuotientFlow.step requires gradients for every A and B factor")
            d_a, d_b = self._quotient_direction(module)
            updates.append((module, d_a, d_b))
            term = d_a.pow(2).sum() + d_b.pow(2).sum()
            squared_norm = term if squared_norm is None else squared_norm + term
        update_norm = torch.sqrt(squared_norm) if squared_norm is not None else torch.tensor(0.0)
        clip_scale = 1.0
        if self.clip_norm is not None and float(update_norm.detach().cpu()) > self.clip_norm:
            clip_scale = float(self.clip_norm / update_norm.clamp_min(torch.finfo(update_norm.dtype).tiny).detach().cpu())
        with torch.no_grad():
            for module, d_a, d_b in updates:
                module.A.add_(d_a, alpha=clip_scale)
                module.B.add_(d_b, alpha=clip_scale)
                if self.balance_after_substep:
                    self._balance_(module)
        self.last_update_norm = float((update_norm * clip_scale).detach().cpu())
        self.last_clip_scale = float(clip_scale)
        self.last_diagnostics = self._diagnostics()
        return loss

    def macro_step(self, closure: Callable[[], torch.Tensor]):
        """Run one macro step with fresh gradients at each quotient substep."""

        if closure is None:
            raise RuntimeError("macro_step requires a closure")
        loss = None
        for _ in range(self.substeps):
            loss = closure()
            self.step()
        return loss

    def _quotient_direction(self, module) -> tuple[torch.Tensor, torch.Tensor]:
        a = module.A
        b = module.B
        inv_b = self._stable_inverse(b.transpose(-2, -1) @ b)
        inv_a = self._stable_inverse(a @ a.transpose(-2, -1))
        d_a = -self.local_lr * (inv_b @ a.grad)
        d_b = -self.local_lr * (b.grad @ inv_a)
        return d_a, d_b

    def _stable_inverse(self, gram: torch.Tensor) -> torch.Tensor:
        condition = torch.linalg.cond(gram.detach())
        condition_value = float(condition.cpu())
        self.condition_max = max(self.condition_max, condition_value)
        if condition_value < self.gram_condition_limit:
            return torch.linalg.inv(gram)
        self.fallback_count += 1
        return torch.linalg.pinv(gram, rtol=1.0 / self.gram_condition_limit)

    @torch.no_grad()
    def _balance_(self, module) -> None:
        before = module.B @ module.A
        q_b, r_b = torch.linalg.qr(module.B, mode="reduced")
        a_mid = r_b @ module.A
        q_a, r_a = torch.linalg.qr(a_mid.transpose(-2, -1), mode="reduced")
        module.B.copy_(q_b @ r_a.transpose(-2, -1))
        module.A.copy_(q_a.transpose(-2, -1))
        after = module.B @ module.A
        residual = (after - before).norm() / before.norm().clamp_min(torch.finfo(before.dtype).tiny)
        self.balance_residual_max = max(self.balance_residual_max, float(residual.cpu()))

    def _diagnostics(self) -> dict[str, float | int | bool | None]:
        return {
            "condition_max": self.condition_max,
            "fallback_count": self.fallback_count,
            "balance_residual_max": self.balance_residual_max,
            "last_update_norm": self.last_update_norm,
            "last_clip_scale": self.last_clip_scale,
            "substeps": self.substeps,
            "macro_lr": self.macro_lr,
            "local_lr": self.local_lr,
            "clip_norm": self.clip_norm,
            "balance_after_substep": self.balance_after_substep,
        }

    @staticmethod
    def _collect_factor_modules(modules) -> list:
        if modules is None:
            raise ValueError("SubsteppedQuotientFlow requires params or factor_modules")
        if isinstance(modules, torch.nn.Module):
            modules = [modules]
        collected = list(modules)
        if not collected:
            raise ValueError("SubsteppedQuotientFlow requires at least one factor module")
        for module in collected:
            if not hasattr(module, "A") or not hasattr(module, "B"):
                raise ValueError("factor modules must expose A and B parameters")
            if not isinstance(module.A, torch.nn.Parameter) or not isinstance(module.B, torch.nn.Parameter):
                raise ValueError("factor module A and B attributes must be torch.nn.Parameter instances")
            if module.A.ndim != 2 or module.B.ndim != 2 or module.B.shape[1] != module.A.shape[0]:
                raise ValueError("factor module shapes must satisfy A=(rank,in), B=(out,rank)")
        return collected


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
            base_steps[product.name] = tangent.detach().clone()
            pre_diagnostics[product.name] = {
                "ambient_proposal_norm": float(ambient.norm().detach().cpu()),
                "tangent_proposal_norm": float(tangent.norm().detach().cpu()),
                "tangent_residual": float(tangent_residual),
            }

        if not base_steps:
            self.last_diagnostics = {
                "products": {},
                "aggregate": {},
                "trust_region": TrustRegionResult(0.0, False, False, float("nan"), float("nan")),
            }
            return loss

        selected_scale = 1.0
        accepted = True
        hit_max_scale = False
        if self.trust_region is not None:
            if calibration_closure is None:
                raise RuntimeError("calibration_closure is required when trust_region is enabled")
            trust_result = self.trust_region.select(
                self.product_state,
                base_steps,
                calibration_closure,
                candidate_transform=self._finalize_candidate_steps,
            )
            selected_scale = trust_result.selected_scale
            accepted = trust_result.accepted
            hit_max_scale = trust_result.hit_max_scale
        else:
            trust_result = TrustRegionResult(1.0, True, False, float("nan"), float("nan"))

        diagnostics: dict[str, dict[str, float | bool]] = {}
        if base_steps and selected_scale != 0.0:
            final_steps = self._finalize_candidate_steps(self.product_state, base_steps, selected_scale)
            with torch.no_grad():
                for product in self.product_state.products:
                    if product.name not in final_steps:
                        continue
                    param = product.tensor
                    before = param.detach().clone()
                    manifold = self._manifold(product.rank)
                    final_step = final_steps[product.name]
                    new_param, diag = manifold.retract(param, final_step)
                    param.copy_(new_param)
                    realized = param - before
                    entry = dict(pre_diagnostics[product.name])
                    entry.update(
                        {
                            "final_candidate_norm": float(final_step.norm().detach().cpu()),
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

    def _finalize_candidate_steps(
        self,
        product_state: ProductState,
        base_steps: dict[str, torch.Tensor],
        scale: float,
    ) -> dict[str, torch.Tensor]:
        """Apply the shared final-candidate rule: scale, project, then clip."""

        max_update_norm = self.param_groups[0]["max_update_norm"]
        finalized: dict[str, torch.Tensor] = {}
        for product in product_state.products:
            if product.name not in base_steps:
                continue
            manifold = self._manifold(product.rank)
            scaled = base_steps[product.name] * scale
            tangent = manifold.project_tangent(product.tensor, scaled)
            if max_update_norm is not None:
                norm = tangent.norm()
                if float(norm.detach().cpu()) > max_update_norm:
                    tangent = tangent * (max_update_norm / norm.clamp_min(torch.finfo(tangent.dtype).tiny))
            finalized[product.name] = tangent.detach().clone()
        return finalized

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
            "final_candidate_norm",
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
