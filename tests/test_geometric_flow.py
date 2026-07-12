import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from geometric_flow import (
    GeoMLP,
    GeometricOptimizer,
    compute_curvature,
    conjugate_gradient,
    geo,
    phase_diagram_scanner,
    phase_diagram_scanner_2d,
    write_phase_diagram_csv,
)


class GeometricFlowTests(unittest.TestCase):
    def test_curvature_matvec_matches_quadratic_hessian(self):
        w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        loss = 0.5 * w @ matrix @ w
        op = compute_curvature(torch.nn.ParameterList([w]), loss, damping=0.0, regularization=0.0)
        vector = torch.tensor([0.25, -0.5])
        self.assertTrue(torch.allclose(op.matvec(vector), matrix @ vector, atol=1e-5))

    def test_curvature_regularization_adds_identity_term(self):
        w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        loss = 0.5 * w @ matrix @ w
        op = compute_curvature(torch.nn.ParameterList([w]), loss, damping=0.0, regularization=0.2)
        vector = torch.tensor([0.25, -0.5])
        self.assertTrue(torch.allclose(op.matvec(vector), matrix @ vector + 0.2 * vector, atol=1e-5))
        op.regularize(alpha=0.4)
        self.assertTrue(torch.allclose(op.matvec(vector), matrix @ vector + 0.4 * vector, atol=1e-5))

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
            warmup_steps=0,
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
        self.assertIn("geodesic_distance", optimizer.topography_log[-1])
        self.assertGreaterEqual(optimizer.geodesic_distance, 0.0)

    def test_optimizer_calls_backward_false_closure_and_keeps_graph(self):
        torch.manual_seed(4)
        model = GeoMLP(input_dim=4, hidden_dim=5, output_dim=2)
        x = torch.randn(10, 4)
        y = (x[:, 0] > 0).long()
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=0.1,
            damping=1e-2,
            cg_max_iter=3,
            trace_samples=1,
            max_update_norm=0.25,
            warmup_steps=0,
        )
        calls = []

        def closure(backward=True):
            calls.append(backward)
            loss = F.cross_entropy(model(x), y)
            if backward:
                loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.assertTrue(torch.isfinite(loss.detach()))
        self.assertEqual(calls, [False])
        self.assertTrue(any(param.grad is not None for param in model.parameters()))

    def test_optimizer_warmup_clips_gradients_and_adapts_damping(self):
        torch.manual_seed(6)
        model = GeoMLP(input_dim=4, hidden_dim=5, output_dim=2)
        x = 100.0 * torch.randn(10, 4)
        y = (x[:, 0] > 0).long()
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=0.1,
            damping=0.05,
            max_grad_norm=0.1,
            warmup_steps=2,
            warmup_lr_scale=0.5,
        )

        loss = optimizer.step(lambda: F.cross_entropy(model(x), y))
        entry = optimizer.topography_log[-1]
        self.assertTrue(torch.isfinite(loss.detach()))
        self.assertEqual(entry["mode"], "warmup")
        self.assertLessEqual(entry["clipped_grad_norm"], 0.1001)
        self.assertGreaterEqual(entry["current_damping"], 0.05)

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

    def test_geo_embed_adds_learnable_rotation_without_changing_feature_dim(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
        wrapped = torch.nn.Module()
        wrapped.net = model
        geo.embed(wrapped)
        self.assertEqual(wrapped.net[1].__class__.__name__, "GeometricRotation")
        self.assertTrue(wrapped.net[1].angle.requires_grad)
        x = torch.randn(2, 3)
        self.assertEqual(wrapped.net(x).shape, (2, 4))

    def test_geomlp_exposes_trainable_phase_parameters(self):
        model = GeoMLP(input_dim=3, hidden_dim=4, output_dim=2)
        phases = model.geometric_parameters()
        self.assertEqual(len(phases), 1)
        self.assertTrue(phases[0].requires_grad)

    def test_phase_diagram_scanner_2d_writes_csv(self):
        torch.manual_seed(9)
        x = torch.randn(8, 3)
        y = (x[:, 0] > 0).long()

        def model_factory():
            return GeoMLP(input_dim=3, hidden_dim=4, output_dim=2)

        def loss_factory(model):
            return F.cross_entropy(model(x), y)

        points = phase_diagram_scanner_2d(
            model_factory,
            loss_factory,
            param1_range=[0.05, 0.1],
            param2_range=[1e-2],
            steps=1,
            optimizer_kwargs={"cg_max_iter": 2, "trace_samples": 1, "max_update_norm": 0.2},
        )
        self.assertEqual(len(points), 2)
        self.assertTrue(all(point.final_loss == point.final_loss for point in points))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "break_even_boundary.csv"
            write_phase_diagram_csv(points, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("param1,param2,final_loss,avg_trace,geodesic_distance,final_mode", text)


if __name__ == "__main__":
    unittest.main()
