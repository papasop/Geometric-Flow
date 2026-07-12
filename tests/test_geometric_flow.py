import unittest

import torch
import torch.nn.functional as F

from geometric_flow import GeoMLP, GeometricOptimizer, compute_curvature, conjugate_gradient, geo, phase_diagram_scanner


class GeometricFlowTests(unittest.TestCase):
    def test_curvature_matvec_matches_quadratic_hessian(self):
        w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        loss = 0.5 * w @ matrix @ w
        op = compute_curvature(torch.nn.ParameterList([w]), loss, damping=0.0)
        vector = torch.tensor([0.25, -0.5])
        self.assertTrue(torch.allclose(op.matvec(vector), matrix @ vector, atol=1e-5))

    def test_conjugate_gradient_solves_spd_system(self):
        matrix = torch.tensor([[5.0, 1.0], [1.0, 2.0]])
        rhs = torch.tensor([1.0, -3.0])
        result = conjugate_gradient(lambda v: matrix @ v, rhs, max_iter=8, tolerance=1e-8)
        self.assertTrue(torch.allclose(matrix @ result.solution, rhs, atol=1e-5))

    def test_optimizer_runs_and_logs_topography(self):
        torch.manual_seed(3)
        model = GeoMLP(input_dim=4, hidden_dim=6, output_dim=2)
        x = torch.randn(12, 4)
        y = (x[:, 0] > 0).long()
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=0.2,
            damping=1e-2,
            cg_max_iter=4,
            trace_samples=1,
            max_update_norm=0.5,
        )

        before = F.cross_entropy(model(x), y)

        def closure():
            return F.cross_entropy(model(x), y)

        optimizer.step(closure)
        after = F.cross_entropy(model(x), y)
        self.assertTrue(torch.isfinite(after))
        self.assertLessEqual(after.item(), before.item() + 1.0)
        self.assertEqual(len(optimizer.topography_log), 1)
        self.assertIn(optimizer.topography_log[-1]["mode"], {"geometric", "sgd"})

    def test_phase_scanner_restores_parameters(self):
        torch.manual_seed(5)
        model = GeoMLP(input_dim=3, hidden_dim=4, output_dim=2)
        x = torch.randn(8, 3)
        y = (x[:, 0] > 0).long()
        before = [p.detach().clone() for p in model.parameters()]
        points = phase_diagram_scanner(
            model,
            lambda: F.cross_entropy(model(x), y),
            param_range=[-1, 0, 1],
            probes=2,
        )
        self.assertEqual(len(points), 3)
        for param, original in zip(model.parameters(), before):
            self.assertTrue(torch.allclose(param, original))

    def test_geo_embed_adds_parameter_free_rotation(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
        before = sum(p.numel() for p in model.parameters())
        wrapped = torch.nn.Module()
        wrapped.net = model
        geo.embed(wrapped)
        after = sum(p.numel() for p in wrapped.parameters())
        self.assertEqual(before, after)
        self.assertEqual(wrapped.net[1].__class__.__name__, "GeometricRotation")


if __name__ == "__main__":
    unittest.main()
