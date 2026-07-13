import unittest

import torch

from geometric_flow import FixedRankManifold


def low_rank_matrix(m=5, n=4, rank=2, dtype=torch.float64, device="cpu"):
    left = torch.randn(m, rank, dtype=dtype, device=device)
    right = torch.randn(rank, n, dtype=dtype, device=device)
    return left @ right


class FixedRankManifoldTests(unittest.TestCase):
    def test_tangent_projection_idempotence(self):
        torch.manual_seed(1)
        manifold = FixedRankManifold(rank=2)
        matrix = low_rank_matrix()
        ambient = torch.randn_like(matrix)
        projected = manifold.project_tangent(matrix, ambient)
        projected_twice = manifold.project_tangent(matrix, projected)
        self.assertTrue(torch.allclose(projected_twice, projected, atol=1e-10, rtol=1e-8))

    def test_tangent_residual_for_projected_direction(self):
        for dtype, tolerance in [(torch.float32, 1e-5), (torch.float64, 1e-10)]:
            torch.manual_seed(2)
            manifold = FixedRankManifold(rank=2)
            matrix = low_rank_matrix(dtype=dtype)
            projected = manifold.project_tangent(matrix, torch.randn_like(matrix))
            self.assertLess(manifold.tangent_residual(matrix, projected), tolerance)

    def test_normal_complement_is_zero(self):
        torch.manual_seed(3)
        manifold = FixedRankManifold(rank=2)
        matrix = low_rank_matrix()
        tangent = manifold.project_tangent(matrix, torch.randn_like(matrix))
        u, _, v = manifold.factor_basis(matrix)
        eye_left = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        eye_right = torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
        normal_block = (eye_left - u @ u.T) @ tangent @ (eye_right - v @ v.T)
        self.assertLess(float(normal_block.norm()), 1e-10)

    def test_retraction_rank(self):
        torch.manual_seed(4)
        manifold = FixedRankManifold(rank=2)
        matrix = low_rank_matrix()
        step = manifold.project_tangent(matrix, 0.1 * torch.randn_like(matrix))
        retracted, diagnostics = manifold.retract(matrix, step)
        self.assertLessEqual(manifold.numerical_rank(retracted), 2)
        self.assertFalse(diagnostics.rank_violation)

    def test_small_step_retraction_consistency_improves_when_step_shrinks(self):
        torch.manual_seed(5)
        manifold = FixedRankManifold(rank=2)
        matrix = low_rank_matrix()
        direction = manifold.project_tangent(matrix, torch.randn_like(matrix))
        _, large = manifold.retract(matrix, 1e-3 * direction)
        _, small = manifold.retract(matrix, 1e-5 * direction)
        self.assertLess(small.retraction_relative_error, large.retraction_relative_error)
        self.assertLess(small.retraction_relative_error, 1e-4)

    def test_zero_update_reproduces_matrix(self):
        torch.manual_seed(6)
        manifold = FixedRankManifold(rank=2)
        matrix = low_rank_matrix()
        retracted, diagnostics = manifold.retract(matrix, torch.zeros_like(matrix))
        self.assertTrue(torch.allclose(retracted, matrix, atol=1e-10, rtol=1e-8))
        self.assertLessEqual(diagnostics.retraction_relative_error, 1e-8)

    def test_inputs_are_not_mutated(self):
        torch.manual_seed(7)
        manifold = FixedRankManifold(rank=2)
        matrix = low_rank_matrix()
        ambient = torch.randn_like(matrix)
        matrix_before = matrix.clone()
        ambient_before = ambient.clone()
        _ = manifold.project_tangent(matrix, ambient)
        _ = manifold.retract(matrix, manifold.project_tangent(matrix, ambient))
        self.assertTrue(torch.equal(matrix, matrix_before))
        self.assertTrue(torch.equal(ambient, ambient_before))

    def test_shape_and_rank_validation(self):
        with self.assertRaises(ValueError):
            FixedRankManifold(rank=0)
        manifold = FixedRankManifold(rank=2)
        with self.assertRaises(ValueError):
            manifold.project_tangent(torch.randn(3), torch.randn(3))
        with self.assertRaises(ValueError):
            manifold.project_tangent(torch.randn(3, 2), torch.randn(3, 3))
        with self.assertRaises(ValueError):
            FixedRankManifold(rank=4).factor_basis(torch.randn(3, 2))


if __name__ == "__main__":
    unittest.main()
