import unittest
import copy

import torch

from geometric_flow import (
    FixedRankFunctionalAdam,
    FixedRankManifold,
    HeldOutTrustRegion,
    ProductParameter,
    ProductState,
)


def make_product(rank=2, dtype=torch.float32, device="cpu"):
    torch.manual_seed(11)
    a = torch.randn(rank, 4, dtype=dtype, device=device)
    b = torch.randn(5, rank, dtype=dtype, device=device)
    return a, b, b @ a


def state_from_matrix(matrix, rank=2, name="lora"):
    return ProductState([ProductParameter(name, torch.nn.Parameter(matrix.clone()), rank)])


class FixedRankFunctionalAdamTests(unittest.TestCase):
    def test_gauge_equivalent_initialization_matches_product_trajectory(self):
        torch.manual_seed(12)
        a, b, matrix = make_product(dtype=torch.float64)
        s = torch.diag(torch.tensor([1.7, 0.4], dtype=torch.float64))
        a_gauge = s @ a
        b_gauge = b @ torch.linalg.inv(s)
        self.assertTrue(torch.allclose(b_gauge @ a_gauge, matrix, atol=1e-12))

        rank = a.shape[0]
        state_a = state_from_matrix(b @ a, rank)
        state_b = state_from_matrix(b_gauge @ a_gauge, rank)
        opt_a = FixedRankFunctionalAdam(state_a, lr=0.05)
        opt_b = FixedRankFunctionalAdam(state_b, lr=0.05)
        grad = torch.randn_like(matrix)
        state_a.products[0].tensor.grad = grad.clone()
        state_b.products[0].tensor.grad = grad.clone()
        opt_a.step()
        opt_b.step()
        final_a = state_a.products[0].tensor.detach()
        final_b = state_b.products[0].tensor.detach()
        x = torch.randn(6, matrix.shape[1], dtype=matrix.dtype)
        self.assertTrue(torch.allclose(final_a, final_b, atol=1e-12, rtol=1e-10))
        self.assertTrue(torch.allclose(x @ final_a.T, x @ final_b.T, atol=1e-12, rtol=1e-10))

    def test_ambient_adam_control_has_projection_residual(self):
        torch.manual_seed(13)
        _, _, matrix = make_product(dtype=torch.float64)
        manifold = FixedRankManifold(rank=2)
        ambient = torch.randn_like(matrix)
        projected = manifold.project_tangent(matrix, ambient)
        self.assertGreater(manifold.tangent_residual(matrix, ambient), 1e-2)
        self.assertLess(manifold.tangent_residual(matrix, projected), 1e-10)

    def test_rank_preservation_over_multiple_steps_cpu_float32(self):
        torch.manual_seed(14)
        _, _, matrix = make_product(dtype=torch.float32)
        state = state_from_matrix(matrix.float(), rank=2)
        opt = FixedRankFunctionalAdam(state, lr=0.03)
        manifold = FixedRankManifold(rank=2)
        for _ in range(30):
            state.products[0].tensor.grad = torch.randn_like(state.products[0].tensor)
            opt.step()
            self.assertLessEqual(manifold.numerical_rank(state.products[0].tensor.detach()), 2)
            self.assertFalse(opt.last_diagnostics["aggregate"]["mean_rank_violation"])

    def test_state_dict_roundtrip_matches_next_step(self):
        torch.manual_seed(15)
        _, _, matrix = make_product(dtype=torch.float64)
        state_a = state_from_matrix(matrix, rank=2)
        opt_a = FixedRankFunctionalAdam(state_a, lr=0.02)
        state_a.products[0].tensor.grad = torch.randn_like(matrix)
        opt_a.step()
        product_snapshot = state_a.snapshot()
        optimizer_snapshot = copy.deepcopy(opt_a.state_dict())

        state_b = state_from_matrix(product_snapshot["lora"], rank=2)
        opt_b = FixedRankFunctionalAdam(state_b, lr=0.02)
        opt_b.load_state_dict(optimizer_snapshot)

        grad = torch.randn_like(matrix)
        state_a.products[0].tensor.grad = grad.clone()
        state_b.products[0].tensor.grad = grad.clone()
        opt_a.step()
        opt_b.step()
        self.assertTrue(
            torch.allclose(state_a.products[0].tensor, state_b.products[0].tensor, atol=1e-12, rtol=1e-10)
        )

    def test_trust_region_snapshot_restore_is_exact(self):
        torch.manual_seed(16)
        _, _, matrix = make_product(dtype=torch.float64)
        state = state_from_matrix(matrix, rank=2)
        target = torch.zeros_like(matrix)
        step = {"lora": -0.1 * matrix}
        snapshot = state.snapshot()
        trust = HeldOutTrustRegion(scale_grid=(0.0, 0.5, 1.0), armijo_relative_decrease=0.0)

        def calibration_closure():
            return (state.products[0].tensor - target).pow(2).sum()

        result = trust.select(state, step, calibration_closure)
        self.assertTrue(torch.equal(state.products[0].tensor, snapshot["lora"]))
        self.assertTrue(result.accepted)
        self.assertGreater(result.selected_scale, 0.0)

    def test_trust_region_optimizer_rejects_when_no_candidate_improves(self):
        torch.manual_seed(17)
        _, _, matrix = make_product(dtype=torch.float64)
        state = state_from_matrix(matrix, rank=2)
        trust = HeldOutTrustRegion(scale_grid=(0.0, 1.0), armijo_relative_decrease=0.0)
        opt = FixedRankFunctionalAdam(state, lr=0.1, trust_region=trust)
        state.products[0].tensor.grad = -state.products[0].tensor.detach().clone()

        def calibration_closure():
            return state.products[0].tensor.pow(2).sum()

        before = state.snapshot()["lora"]
        opt.step(calibration_closure=calibration_closure)
        self.assertTrue(torch.allclose(state.products[0].tensor, before))
        self.assertFalse(opt.last_diagnostics["trust_region"].accepted)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_step_preserves_rank(self):
        _, _, matrix = make_product(dtype=torch.float32, device="cuda")
        state = state_from_matrix(matrix, rank=2)
        opt = FixedRankFunctionalAdam(state, lr=0.03)
        state.products[0].tensor.grad = torch.randn_like(state.products[0].tensor)
        opt.step()
        self.assertLessEqual(FixedRankManifold(rank=2).numerical_rank(state.products[0].tensor.detach()), 2)


if __name__ == "__main__":
    unittest.main()
