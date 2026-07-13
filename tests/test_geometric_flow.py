import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from geometric_flow import (
    GeoCNN,
    GeoMLP,
    GeometricOptimizer,
    FunctionalMap,
    compute_curvature,
    conjugate_gradient,
    functional_projectors,
    functional_response_operator,
    projected_functional_geoflow_direction,
    geo,
    phase_diagram_scanner,
    phase_diagram_scanner_2d,
    write_phase_diagram_csv,
)
from experiments.train_cifar10_geo import experiment_names, parse_ints
from experiments.cifar10_configs import get_config
from experiments.plot_comparison import ratio_time_rows
from experiments.normal_projection_toy import run_toy
from experiments.functional_projection_toy import TwoLayerLinear, known_tangent_vector, run_toy as run_functional_toy


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

    def test_curvature_scale_controls_hessian_strength(self):
        w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
        loss = 0.5 * w @ matrix @ w
        op = compute_curvature(
            torch.nn.ParameterList([w]),
            loss,
            damping=0.0,
            regularization=0.0,
            scale=0.25,
        )
        vector = torch.tensor([0.25, -0.5])
        self.assertTrue(torch.allclose(op.matvec(vector), 0.25 * (matrix @ vector), atol=1e-5))

    def test_grad_square_curvature_replaces_fisher_name(self):
        w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        loss = (w.pow(2).sum())
        op = compute_curvature(torch.nn.ParameterList([w]), loss, damping=0.0, regularization=0.0, kind="grad_square")
        vector = torch.tensor([0.25, -0.5])
        expected = torch.tensor([4.0, 16.0]) * vector
        self.assertTrue(torch.allclose(op.matvec(vector), expected, atol=1e-5))

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

    def test_optimizer_reuses_curvature_between_refreshes(self):
        weight = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = GeometricOptimizer(
            [weight],
            lr=0.1,
            damping=1e-3,
            regularization=1e-3,
            warmup_steps=0,
            curvature_reuse=3,
            grad_smoothing=0.0,
            cg_max_iter=4,
            trace_samples=0,
        )

        def closure():
            return 0.5 * (weight - 1.0).pow(2).sum()

        optimizer.step(closure)
        optimizer.step(closure)
        first, second = optimizer.topography_log[-2:]
        self.assertEqual(first["mode"], "geometric")
        self.assertTrue(first["curvature_refreshed"])
        self.assertEqual(second["mode"], "geometric_reuse")
        self.assertFalse(second["curvature_refreshed"])
        self.assertGreater(second["preconditioned_grad_norm"], 0.0)

    def test_optimizer_diagonal_preconditioner_runs_without_cg(self):
        weight = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = GeometricOptimizer(
            [weight],
            lr=0.1,
            warmup_steps=0,
            preconditioner="diagonal",
            preconditioner_scale=0.5,
            curvature_kind="grad_square",
            trace_samples=0,
        )

        optimizer.step(lambda: 0.5 * (weight - 1.0).pow(2).sum())
        entry = optimizer.topography_log[-1]
        self.assertEqual(entry["mode"], "diagonal")
        self.assertEqual(entry["cg_iterations"], 0)
        self.assertGreater(entry["preconditioned_to_raw_ratio"], 0.0)
        self.assertLess(entry["grad_direction_dot"], 0.0)
        self.assertTrue(entry["descent_gate_passed"])

    def test_legacy_diagonal_grad_square_alias_runs(self):
        weight = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = GeometricOptimizer(
            [weight],
            lr=0.1,
            warmup_steps=0,
            preconditioner="diagonal_grad_square",
            trace_samples=0,
        )
        optimizer.step(lambda: 0.5 * (weight - 1.0).pow(2).sum())
        self.assertEqual(optimizer.topography_log[-1]["mode"], "diagonal")
        self.assertEqual(optimizer.curvature_kind, "grad_square")

    def test_optimizer_adam_mode_logs_adam_steps(self):
        weight = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = GeometricOptimizer([weight], lr=0.05, mode="adam", warmup_steps=0)

        optimizer.step(lambda: 0.5 * (weight - 1.0).pow(2).sum())
        entry = optimizer.topography_log[-1]
        self.assertEqual(entry["mode"], "adam")
        self.assertFalse(entry["curvature_refreshed"])
        self.assertGreater(entry["update_norm"], 0.0)

    def test_optimizer_hybrid_uses_adam_then_geometric(self):
        weight = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = GeometricOptimizer(
            [weight],
            lr=0.05,
            mode="hybrid",
            adam_warmup_steps=2,
            warmup_steps=0,
            preconditioner="diagonal",
            grad_smoothing=0.0,
        )

        def closure():
            return 0.5 * (weight - 1.0).pow(2).sum()

        optimizer.step(closure)
        optimizer.step(closure)
        optimizer.step(closure)
        modes = [row["mode"] for row in optimizer.topography_log]
        self.assertEqual(modes[:2], ["adam_warmup", "adam_warmup"])
        self.assertEqual(modes[2], "diagonal")

    def test_optimizer_preconditioner_scale_damps_geometric_direction(self):
        def run(scale):
            weight = torch.nn.Parameter(torch.tensor([2.0]))
            optimizer = GeometricOptimizer(
                [weight],
                lr=0.1,
                damping=1e-3,
                regularization=1e-3,
                warmup_steps=0,
                preconditioner_scale=scale,
                grad_smoothing=0.0,
                cg_max_iter=4,
                trace_samples=0,
            )
            optimizer.step(lambda: 0.5 * (weight - 1.0).pow(2).sum())
            return optimizer.topography_log[-1]["preconditioned_grad_norm"]

        self.assertLess(run(0.5), run(1.0))

    def test_optimizer_descent_gate_rejects_uphill_reused_direction(self):
        weight = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = GeometricOptimizer(
            [weight],
            lr=0.1,
            warmup_steps=0,
            curvature_reuse=100,
            grad_smoothing=0.0,
            trace_samples=0,
        )

        def closure():
            return 0.5 * (weight - 1.0).pow(2).sum()

        optimizer.step(closure)
        optimizer._last_preconditioner_gain = -1.0
        optimizer._has_preconditioner = True
        optimizer.step(closure)
        entry = optimizer.topography_log[-1]
        self.assertEqual(entry["mode"], "descent_gate_fallback")
        self.assertLess(entry["grad_direction_dot"], 0.0)
        self.assertTrue(entry["descent_gate_passed"])

    def test_optimizer_lr_scale_grad_smoothing_and_adaptive_reuse(self):
        weight = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = GeometricOptimizer(
            [weight],
            lr=0.1,
            damping=1e-3,
            regularization=1e-3,
            warmup_steps=0,
            curvature_reuse=3,
            lr_scale=3.0,
            grad_smoothing=0.5,
            reuse_flat_threshold=1.0,
            reuse_steep_threshold=10.0,
            max_curvature_reuse=4,
            cg_max_iter=4,
            trace_samples=0,
        )

        def closure():
            return 0.5 * (weight - 1.0).pow(2).sum()

        optimizer.step(closure)
        first = optimizer.topography_log[-1]
        self.assertGreater(first["preconditioned_grad_norm"], 0.0)
        self.assertAlmostEqual(
            first["update_norm"],
            first["direction_norm"] * 0.1 * 3.0,
            places=5,
        )
        optimizer.step(closure)
        second = optimizer.topography_log[-1]
        self.assertEqual(second["mode"], "geometric_reuse")
        self.assertIsNotNone(second["reuse_change_rate"])
        self.assertGreaterEqual(second["curvature_reuse"], 3)
        self.assertIsNotNone(optimizer._ema_grad)

    def test_optimizer_verbose_writes_diagnostic_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostics.csv"
            weight = torch.nn.Parameter(torch.tensor([2.0]))
            optimizer = GeometricOptimizer(
                [weight],
                lr=0.1,
                warmup_steps=1,
                diagnostic_log_interval=1,
                diagnostic_log_path=path,
            )

            optimizer.step(lambda: 0.5 * (weight - 1.0).pow(2).sum(), verbose=True)
            text = path.read_text(encoding="utf-8")
        self.assertIn("preconditioned_to_raw_ratio", text)
        self.assertIn("warmup", text)

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

    def test_geocnn_forward_and_geometric_parameters(self):
        model = GeoCNN(channels=4, num_classes=3)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (2, 3))
        self.assertEqual(len(model.geometric_parameters()), 6)

    def test_deep_geocnn_adds_rotation_after_each_conv_layer(self):
        model = GeoCNN(channels=4, num_classes=3, conv_layers=6)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        self.assertEqual(y.shape, (2, 3))
        self.assertEqual(len(model.geometric_parameters()), 12)

    def test_train_script_auto_warmup_expands_hybrid_modes(self):
        args = type("Args", (), {})()
        args.mode = "all"
        args.auto_warmup = True
        args.auto_warmup_steps = parse_ints("30,50,80")
        self.assertEqual(
            experiment_names(args),
            ["adam", "geometric", "hybrid_30", "hybrid_50", "hybrid_80"],
        )

    def test_train_script_switch_compare_expands_matched_modes(self):
        args = type("Args", (), {})()
        args.mode = "switch_compare"
        args.auto_warmup = False
        self.assertEqual(experiment_names(args), ["adam_continue", "hybrid_geometric"])

    def test_cifar10_config_and_ratio_rows_are_available(self):
        config = get_config("hybrid_diagonal_500")
        self.assertEqual(config["conv_layers"], 6)
        rows = [
            {"step": "1", "preconditioned_to_raw_ratio": "0.5"},
            {"optimizer": "adam", "mean_accuracy": "0.1"},
        ]
        self.assertEqual(len(ratio_time_rows(rows)), 1)

    def test_normal_projection_toy_reports_normal_projected_hessian(self):
        row = run_toy(seed=3, input_dim=2, hidden_dim=2, output_dim=1, samples=4)
        self.assertGreater(row["params"], 0)
        self.assertGreaterEqual(row["normal_rank"], 1)
        self.assertIn("normal_projected_trace", row)

    def test_functional_jacobian_shape_and_response_psd(self):
        torch.manual_seed(12)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(3, 2)
        fmap = FunctionalMap(model, x)
        fjac = fmap.jacobian()
        n_params = sum(param.numel() for param in model.parameters())
        self.assertEqual(tuple(fjac.jacobian.shape), (3, n_params))
        response = functional_response_operator(fjac.jacobian)
        self.assertTrue(torch.allclose(response, response.T, atol=1e-6))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(response).min()), -1e-6)

    def test_functional_projectors_capture_known_tangent(self):
        torch.manual_seed(13)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(4, 2)
        fjac = FunctionalMap(model, x).jacobian()
        projectors = functional_projectors(fjac.jacobian)
        tangent = known_tangent_vector(model)
        identity = torch.eye(projectors.tangent.shape[0])
        self.assertLess(float(torch.linalg.vector_norm(fjac.jacobian @ tangent)), 1e-5)
        self.assertLess(projectors.residuals["j_pt"], 1e-5)
        self.assertTrue(torch.allclose(projectors.tangent + projectors.normal, identity, atol=1e-5))
        self.assertEqual(projectors.tangent_rank + projectors.normal_rank, identity.shape[0])

    def test_functional_geoflow_direction_is_normal_descent_and_lowers_loss(self):
        torch.manual_seed(14)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(5, 2)
        y = torch.randn(5, 1)
        loss = F.mse_loss(model(x), y)
        result = projected_functional_geoflow_direction(model, loss, x, damping=1e-3, max_update_norm=0.2)
        self.assertLess(result.tangent_norm, 1e-5)
        self.assertLess(result.g_dot_d, 0.0)
        params = [param for param in model.parameters() if param.requires_grad]
        before = float(loss.detach())
        from geometric_flow._tensor import assign_flat_update

        assign_flat_update(params, result.direction, scale=0.05)
        after = float(F.mse_loss(model(x), y).detach())
        self.assertLessEqual(after, before + 1e-6)

    def test_functional_optimizer_mode_logs_projection(self):
        torch.manual_seed(15)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(5, 2)
        y = torch.randn(5, 1)
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=0.05,
            lr_scale=1.0,
            mode="functional_geoflow",
            functional_model=model,
            functional_probe=x,
            max_update_norm=0.2,
            warmup_steps=0,
        )
        before = float(F.mse_loss(model(x), y).detach())
        optimizer.step(lambda: F.mse_loss(model(x), y))
        entry = optimizer.topography_log[-1]
        after = float(F.mse_loss(model(x), y).detach())
        self.assertEqual(entry["mode"], "functional_geoflow")
        self.assertLess(entry["functional_tangent_norm"], 1e-5)
        self.assertLess(entry["grad_direction_dot"], 0.0)
        self.assertLessEqual(after, before + 1e-6)

    def test_functional_projection_toy_reports_required_fields(self):
        row = run_functional_toy(seed=3, input_dim=2, hidden_dim=2, output_dim=1, samples=4)
        self.assertIn("known_tangent_residual", row)
        self.assertLess(row["functional_geoflow_tangent_norm"], 1e-5)
        self.assertLess(row["g_dot_d"], 0.0)

    def test_matched_warmup_state_can_be_shared_before_branching(self):
        torch.manual_seed(16)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(4, 2)
        y = torch.randn(4, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        left = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        right = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        left.load_state_dict(state)
        right.load_state_dict(state)
        for left_value, right_value in zip(left.state_dict().values(), right.state_dict().values()):
            self.assertTrue(torch.allclose(left_value, right_value))

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
