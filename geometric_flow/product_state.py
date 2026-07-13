"""Adapters for explicit invariant product-state optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch

from .fixed_rank import FixedRankDiagnostics, FixedRankManifold


@dataclass
class ProductParameter:
    """A named invariant product tensor with an intended fixed rank."""

    name: str
    tensor: torch.nn.Parameter
    rank: int


class ProductState:
    """Container for explicit product-coordinate optimizer state.

    The tensors in this state are the optimizer variables. The class does not
    reconstruct arbitrary third-party LoRA factor modules; callers that need a
    forward pass through product coordinates should provide their own adapter.
    """

    def __init__(
        self,
        products: list[ProductParameter],
    ) -> None:
        if not products:
            raise ValueError("ProductState requires at least one product")
        seen = set()
        for product in products:
            if product.name in seen:
                raise ValueError(f"duplicate product name: {product.name}")
            seen.add(product.name)
            if not isinstance(product.tensor, torch.nn.Parameter):
                raise TypeError("ProductParameter.tensor must be a torch.nn.Parameter")
            if product.tensor.ndim != 2:
                raise ValueError("product tensors must be 2-D matrices")
            if product.rank < 1 or product.rank > min(product.tensor.shape):
                raise ValueError("product rank must satisfy 1 <= rank <= min(shape)")
        self.products = list(products)

    @classmethod
    def from_lora_modules(
        cls,
        model: torch.nn.Module,
        module_filter: Callable[[str, torch.nn.Module], bool] | None = None,
    ) -> "ProductState":
        """Discover simple modules exposing trainable LoRA ``A`` and ``B``.

        This helper intentionally supports only clear conventions. It creates
        detached product parameters initialized from ``B @ A``; it does not wire
        them into the source module's forward pass.
        """

        products: list[ProductParameter] = []
        for module_name, module in model.named_modules():
            if module_filter is not None and not module_filter(module_name, module):
                continue
            pair = _find_lora_pair(module)
            if pair is None:
                continue
            a, b = pair
            if not (a.requires_grad and b.requires_grad):
                continue
            if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                raise ValueError(f"invalid LoRA factor shapes in module {module_name!r}")
            product = torch.nn.Parameter((b.detach() @ a.detach()).clone())
            rank = int(a.shape[0])
            name = module_name or module.__class__.__name__
            products.append(ProductParameter(name=name, tensor=product, rank=rank))
        if not products:
            raise ValueError("no compatible LoRA A/B modules found")
        return cls(products)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [product.tensor for product in self.products]

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        for product in self.products:
            yield product.name, product.tensor

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return an exact detached clone of every product tensor."""

        return {name: tensor.detach().clone() for name, tensor in self.named_parameters()}

    def restore_(self, snapshot: dict[str, torch.Tensor]) -> None:
        """Restore product tensors from ``snapshot`` in place."""

        with torch.no_grad():
            for product in self.products:
                if product.name not in snapshot:
                    raise ValueError(f"snapshot missing product {product.name!r}")
                value = snapshot[product.name]
                if value.shape != product.tensor.shape:
                    raise ValueError(f"snapshot shape mismatch for product {product.name!r}")
                product.tensor.copy_(value.to(device=product.tensor.device, dtype=product.tensor.dtype))

    def project_and_retract_(
        self,
        steps: dict[str, torch.Tensor] | Iterable[torch.Tensor],
        *,
        scale: float = 1.0,
        manifolds: dict[str, FixedRankManifold] | None = None,
    ) -> dict[str, FixedRankDiagnostics]:
        """Apply scaled steps through tangent projection and fixed-rank retraction."""

        if isinstance(steps, dict):
            step_by_name = steps
        else:
            step_by_name = {product.name: step for product, step in zip(self.products, steps)}
        diagnostics: dict[str, FixedRankDiagnostics] = {}
        with torch.no_grad():
            for product in self.products:
                if product.name not in step_by_name:
                    raise ValueError(f"missing step for product {product.name!r}")
                manifold = manifolds[product.name] if manifolds and product.name in manifolds else FixedRankManifold(product.rank)
                tangent = manifold.project_tangent(product.tensor, step_by_name[product.name].to(product.tensor) * scale)
                new_tensor, diag = manifold.retract(product.tensor, tangent)
                product.tensor.copy_(new_tensor)
                diagnostics[product.name] = diag
        return diagnostics


def _find_lora_pair(module: torch.nn.Module) -> tuple[torch.nn.Parameter, torch.nn.Parameter] | None:
    candidates = [
        ("a", "b"),
        ("A", "B"),
        ("lora_a", "lora_b"),
        ("lora_A", "lora_B"),
    ]
    for a_name, b_name in candidates:
        a = getattr(module, a_name, None)
        b = getattr(module, b_name, None)
        if isinstance(a, torch.nn.Parameter) and isinstance(b, torch.nn.Parameter):
            return a, b
    return None
