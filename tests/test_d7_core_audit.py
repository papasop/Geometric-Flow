import math

import pytest
import torch

from geometric_flow import FixedRankFunctionalAdam, HeldOutTrustRegion, ProductParameter, ProductState


def make_state(dtype=torch.float64):
    torch.manual_seed(700)
    left = torch.randn(4, 2, dtype=dtype)
    right = torch.randn(2, 3, dtype=dtype)
    matrix = left @ right
    return ProductState([ProductParameter("adapter", torch.nn.Parameter(matrix.clone()), rank=2)])


def test_final_update_norm_is_bounded_after_trust_scale_selection():
    state = make_state()
    product = state.products[0].tensor
    before = product.detach().clone()
    product.grad = -torch.ones_like(product)
    trust = HeldOutTrustRegion(scale_grid=(0.0, 16.0), armijo_relative_decrease=0.0)
    optimizer = FixedRankFunctionalAdam(state, lr=10.0, max_update_norm=0.05, trust_region=trust)
    target = before + torch.ones_like(before)
    losses_seen = []

    def calibration_closure():
        loss = (product - target).pow(2).sum()
        losses_seen.append(float(loss.detach()))
        return loss

    optimizer.step(calibration_closure=calibration_closure)
    realized_norm = float((product.detach() - before).norm())
    assert realized_norm <= 0.0501
    assert optimizer.last_diagnostics["aggregate"]["mean_final_candidate_norm"] <= 0.0501
    selected_loss = optimizer.last_diagnostics["trust_region"].selected_loss
    final_loss = float(((product.detach() - target) ** 2).sum())
    assert selected_loss == pytest.approx(final_loss, abs=1e-10, rel=1e-10)
    assert len(losses_seen) >= 2


def test_product_state_project_and_retract_allows_partial_steps_and_rejects_unknowns():
    torch.manual_seed(701)
    first = torch.nn.Parameter(torch.randn(4, 2) @ torch.randn(2, 3))
    second = torch.nn.Parameter(torch.randn(5, 2) @ torch.randn(2, 3))
    state = ProductState(
        [
            ProductParameter("first", first, rank=2),
            ProductParameter("second", second, rank=2),
        ]
    )
    before_first = first.detach().clone()
    before_second = second.detach().clone()
    state.project_and_retract_({"first": 1e-3 * torch.randn_like(first)})
    assert not torch.allclose(first.detach(), before_first)
    assert torch.equal(second.detach(), before_second)
    with pytest.raises(ValueError, match="unknown product"):
        state.project_and_retract_({"missing": torch.randn_like(first)})
    with pytest.raises(ValueError, match="shape mismatch"):
        state.project_and_retract_({"first": torch.randn(3, 3)})


def test_fixed_rank_optimizer_empty_gradient_is_clean_noop_with_trust_region():
    state = make_state()
    before = state.snapshot()["adapter"]
    trust = HeldOutTrustRegion(scale_grid=(0.0, 1.0))
    optimizer = FixedRankFunctionalAdam(state, trust_region=trust)
    optimizer.step()
    assert torch.equal(state.products[0].tensor.detach(), before)
    assert optimizer.last_diagnostics["products"] == {}
    assert optimizer.last_diagnostics["aggregate"] == {}
    result = optimizer.last_diagnostics["trust_region"]
    assert result.selected_scale == 0.0
    assert result.accepted is False
    assert result.hit_max_scale is False
    assert math.isnan(result.baseline_loss)
    assert math.isnan(result.selected_loss)
