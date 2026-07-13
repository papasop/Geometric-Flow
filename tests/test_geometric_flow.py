import unittest
import tempfile
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from geometric_flow._tensor import assign_flat_update

from geometric_flow import (
    GeoCNN,
    GeoMLP,
    GeometricOptimizer,
    FunctionalMap,
    FunctionalJTJOperator,
    MatrixFreeFunctionalJTJOperator,
    compute_curvature,
    conjugate_gradient,
    functional_projectors,
    functional_response_operator,
    projected_functional_geoflow_direction,
    randomized_normal_basis,
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
from experiments.reparameterization_stress_test import (
    diagonal_scaling,
    model_representations,
    parse_optimizers as parse_reparam_optimizers,
    reparameterize_model,
    run as run_reparameterization_stress,
    orthogonal_rotation,
)
from experiments.lora_reparameterization_benchmark import SmallLoRAMLP, make_transform, run_lora_benchmark
from experiments.lora_matched_step_benchmark import (
    calibrate_scale,
    compute_gates_from_statistics,
    configure_train_scope,
    cross_seed_mixed_pairwise_distance,
    functional_step_norm_for,
    gate_rows_for_config,
    make_fmap as make_lora_fmap,
    paired_task_differences,
    phase_g_statistics,
    reference_functional_step,
    representation_fn_for,
    run_config as run_lora_matched_config,
    within_seed_pairwise_sensitivity,
)
from experiments.analyze_phase_g_results import analyze_rows, candidate_run_csvs


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

    def test_equivalent_reparameterizations_keep_logits_and_tangents(self):
        torch.manual_seed(17)
        model = TwoLayerLinear(input_dim=3, hidden_dim=3, output_dim=2)
        x = torch.randn(5, 3)
        base_logits = model(x)
        for transform in [diagonal_scaling(3), orthogonal_rotation(3)]:
            rep = reparameterize_model(model, transform)
            self.assertTrue(torch.allclose(rep(x), base_logits, atol=1e-6))
            fjac = FunctionalMap(rep, x).jacobian()
            projectors = functional_projectors(fjac.jacobian)
            if torch.allclose(transform, torch.diag(torch.diag(transform))):
                generator = torch.diag(torch.ones(3))
            else:
                generator = torch.zeros(3, 3)
                generator[0, 1] = 1.0
                generator[1, 0] = -1.0
            tangent = torch.cat([(generator @ rep.w1.weight.detach()).reshape(-1), (-rep.w2.weight.detach() @ generator).reshape(-1)])
            self.assertLess(float(torch.linalg.vector_norm(fjac.jacobian @ tangent)), 1e-5)
            self.assertLess(float(torch.linalg.vector_norm(projectors.normal @ tangent)), 1e-5)

    def test_functional_projectors_adaptive_threshold_diagnostics(self):
        jacobian = torch.diag(torch.tensor([10.0, 1.0, 1e-4, 1e-8]))
        spectral = functional_projectors(jacobian, null_threshold_mode="spectral_gap", null_tol=1e-6)
        absolute = functional_projectors(jacobian, null_threshold_mode="absolute", null_tol=1e-7)
        energy = functional_projectors(jacobian, null_threshold_mode="energy_fraction", null_tol=1e-6)
        self.assertIn("j_pt", spectral.residuals)
        self.assertGreaterEqual(spectral.selected_threshold, 0.0)
        self.assertGreaterEqual(spectral.spectral_gap_index, 0)
        self.assertGreater(spectral.condition_number_normal, 0.0)
        self.assertGreater(spectral.retained_energy_fraction, 0.0)
        self.assertGreaterEqual(absolute.normal_rank, spectral.normal_rank)
        self.assertLessEqual(energy.tangent_rank, jacobian.shape[1])

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

    def test_low_rank_and_implicit_match_dense_functional_direction(self):
        torch.manual_seed(18)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(5, 2)
        y = torch.randn(5, 1)
        loss = F.mse_loss(model(x), y)
        dense = projected_functional_geoflow_direction(model, loss, x, damping=1e-3, response_solver="dense")
        low_rank = projected_functional_geoflow_direction(
            model,
            loss,
            x,
            damping=1e-3,
            response_solver="low_rank",
            functional_energy_fraction=1.0,
        )
        implicit = projected_functional_geoflow_direction(
            model,
            loss,
            x,
            damping=1e-3,
            response_solver="implicit_cg",
            cg_max_iter=64,
            cg_tolerance=1e-8,
        )
        for candidate in [low_rank, implicit]:
            cosine = torch.dot(candidate.direction, dense.direction) / (
                torch.linalg.vector_norm(candidate.direction) * torch.linalg.vector_norm(dense.direction)
            ).clamp_min(1e-30)
            self.assertGreater(float(cosine), 0.99)
            self.assertLess(candidate.solver_residual, 1e-4)
            self.assertLess(candidate.tangent_norm, 1e-5)
            self.assertLess(candidate.g_dot_d, 0.0)

    def test_implicit_operator_matvec_matches_dense_jtj(self):
        torch.manual_seed(19)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(4, 2)
        fmap = FunctionalMap(model, x)
        fjac = fmap.jacobian()
        projectors = functional_projectors(fjac.jacobian)
        damping = 1e-3
        operator = FunctionalJTJOperator(fmap, fmap.flatten_params(), projectors.normal, damping)
        v = torch.randn(fjac.theta.numel())
        dense_matvec = projectors.normal @ (fjac.jacobian.T @ (fjac.jacobian @ (projectors.normal @ v))) + damping * (projectors.normal @ v)
        self.assertTrue(torch.allclose(operator.matvec(v), dense_matvec, atol=1e-5))

    def test_matrix_free_operator_and_range_finder_match_dense_jtj(self):
        torch.manual_seed(20)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(4, 2)
        fmap = FunctionalMap(model, x)
        fjac = fmap.jacobian()
        operator = MatrixFreeFunctionalJTJOperator(fmap)
        v = torch.randn(fjac.theta.numel())
        self.assertTrue(torch.allclose(operator.jtj(v), fjac.jacobian.T @ (fjac.jacobian @ v), atol=1e-5))
        q, info = randomized_normal_basis(fmap, energy_fraction=1.0)
        self.assertGreaterEqual(q.shape[1], fjac.rank)
        leakage = torch.linalg.vector_norm((torch.eye(q.shape[0]) - q @ q.T) @ (fjac.jacobian.T @ torch.randn(fjac.jacobian.shape[0])))
        self.assertLess(float(leakage), 1e-5)
        self.assertGreater(info["vjp_count"], 0)

    def test_matrix_free_implicit_reports_counts_without_dense_projector(self):
        torch.manual_seed(22)
        model = TwoLayerLinear(input_dim=2, hidden_dim=2, output_dim=1)
        x = torch.randn(5, 2)
        y = torch.randn(5, 1)
        loss = F.mse_loss(model(x), y)
        dense = projected_functional_geoflow_direction(model, loss, x, damping=1e-3, response_solver="dense")
        implicit = projected_functional_geoflow_direction(
            model,
            loss,
            x,
            damping=1e-3,
            response_solver="implicit_cg",
            functional_energy_fraction=1.0,
            cg_max_iter=64,
            cg_tolerance=1e-8,
        )
        cosine = torch.dot(implicit.direction, dense.direction) / (
            torch.linalg.vector_norm(implicit.direction) * torch.linalg.vector_norm(dense.direction)
        ).clamp_min(1e-30)
        self.assertGreater(float(cosine), 0.99)
        self.assertLess(implicit.solver_residual, 1e-4)
        self.assertLess(implicit.null_leakage, 1e-5)
        self.assertGreater(implicit.jvp_count, 0)
        self.assertGreater(implicit.vjp_count, 0)
        self.assertEqual(implicit.projectors.normal.numel(), 0)

    def test_production_matrix_free_uses_cached_basis_and_warm_start(self):
        torch.manual_seed(23)
        model = TwoLayerLinear(input_dim=3, hidden_dim=4, output_dim=2)
        x = torch.randn(7, 3)
        y = torch.randn(7, 2)
        dense_loss = F.mse_loss(model(x), y)
        dense = projected_functional_geoflow_direction(model, dense_loss, x, damping=1e-3, response_solver="dense")
        prod = projected_functional_geoflow_direction(
            model,
            dense_loss,
            x,
            damping=1e-3,
            response_solver="implicit_cg",
            production_mode=True,
            max_basis_rank=16,
            max_vjp_probes=20,
            functional_energy_fraction=1.0,
            cg_max_iter=64,
            cg_tolerance=1e-8,
        )
        cosine = torch.dot(prod.direction, dense.direction) / (
            torch.linalg.vector_norm(prod.direction) * torch.linalg.vector_norm(dense.direction)
        ).clamp_min(1e-30)
        self.assertGreater(float(cosine), 0.99)
        self.assertLess(prod.null_leakage, 1e-5)

        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=1e-3,
            lr_scale=1.0,
            mode="functional_geoflow",
            functional_model=model,
            functional_probe=x,
            response_solver="implicit_cg",
            production_mode=True,
            refresh_interval=4,
            max_basis_rank=16,
            max_vjp_probes=20,
            cg_max_iter=16,
            cg_tolerance=1e-5,
            adaptive_damping=False,
            max_update_norm=0.1,
        )
        optimizer.step(lambda: F.mse_loss(model(x), y))
        first = optimizer.topography_log[-1]
        optimizer.step(lambda: F.mse_loss(model(x), y))
        second = optimizer.topography_log[-1]
        self.assertFalse(first["basis_from_cache"])
        self.assertTrue(second["basis_from_cache"])
        self.assertLess(second["vjp_count"], first["vjp_count"])
        self.assertGreaterEqual(second["jvp_count"], 1)

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
        self.assertIn("direction_cosine_vs_dense", row)
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

    def test_reparameterization_benchmark_keeps_per_seed_representation_rows(self):
        args = type("Args", (), {})()
        args.seed = 21
        args.trials = 1
        args.steps = 1
        args.representations = 2
        args.train_samples = 24
        args.eval_samples = 24
        args.batch_size = 8
        args.probe_size = 4
        args.input_dim = 3
        args.hidden_dim = 3
        args.output_dim = 2
        args.lr = 1e-3
        args.max_update_norm = 0.2
        args.null_threshold_mode = "spectral_gap"
        args.null_tol = 1e-6
        args.optimizers = parse_reparam_optimizers("adam,diagonal_grad_square,functional_geoflow")
        rows, aggregates = run_reparameterization_stress(args)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row.seed == 21 for row in rows))
        self.assertIn("final_phi", rows[0].__dataclass_fields__)
        self.assertIn("loss_win_rate_vs_adam", aggregates[0])

    def test_lora_reparameterization_equivalence_and_benchmark_rows(self):
        torch.manual_seed(24)
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        x = torch.randn(6, 4)
        before = model(x).detach()
        model.lora.reparameterize(make_transform(rank=2, representation=2))
        after = model(x).detach()
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

        args = type("Args", (), {})()
        args.seed = 31
        args.trials = 1
        args.steps = 1
        args.representations = 2
        args.samples = 24
        args.probe_size = 4
        args.batch_size = 8
        args.input_dim = 4
        args.hidden_dim = 5
        args.output_dim = 2
        args.lora_rank = 2
        args.lr = 1e-2
        args.damping = 1e-3
        args.max_update_norm = 0.1
        args.refresh_interval = 4
        args.max_basis_rank = 4
        args.max_vjp_probes = 6
        args.vjp_probe_batch_size = 3
        args.cg_max_iter = 8
        args.cg_tol = 1e-5
        args.optimizers = "adamw,diagonal_grad_square,functional_geoflow"
        rows, aggregates = run_lora_benchmark(args)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row.optimizer for row in rows}, {"adamw", "diagonal_grad_square", "functional_geoflow"})
        self.assertIn("reparameterization_sensitivity", aggregates[0])

    def _phase_g_args(self):
        args = type("Args", (), {})()
        args.seed = 41
        args.trials = 1
        args.steps = 2
        args.representations = 2
        args.samples = 28
        args.probe_size = 4
        args.batch_size = 7
        args.input_dim = 4
        args.hidden_dim = 5
        args.output_dim = 2
        args.lora_rank = 2
        args.train_scope = "lora_only"
        args.functional_map = "hidden"
        args.lr = 1e-2
        args.damping = 1e-3
        args.max_update_norm = 0.1
        args.refresh_interval = 4
        args.max_basis_rank = 6
        args.max_vjp_probes = 8
        args.vjp_probe_batch_size = 4
        args.cg_max_iter = 8
        args.cg_tol = 1e-5
        args.calibration_reference = "diagonal_grad_square"
        args.calibration_steps = 2
        args.target_functional_step = None
        args.functional_step_tolerance = 0.10
        args.lr_scale_min = 1e-3
        args.lr_scale_max = 1e3
        args.calibration_max_iters = 8
        args.loss_parity_margin = 0.02
        args.optimizers = ["adamw", "diagonal_grad_square", "functional_geoflow_fixed_lr", "functional_geoflow_matched_step"]
        return args

    def test_lora_only_freezes_head(self):
        torch.manual_seed(32)
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        configure_train_scope(model, "lora_only")
        x = torch.randn(10, 4)
        y = (x[:, 0] > 0).long()
        before = [model.head.weight.detach().clone(), model.head.bias.detach().clone()]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.allclose(model.head.weight, before[0]))
        self.assertTrue(torch.allclose(model.head.bias, before[1]))
        fmap = make_lora_fmap(model, x[:3], "hidden")
        self.assertEqual(fmap.param_names, ["lora.a", "lora.b"])

    def test_lora_and_head_updates_head(self):
        torch.manual_seed(33)
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        configure_train_scope(model, "lora_and_head")
        x = torch.randn(10, 4)
        y = (x[:, 0] > 0).long()
        before = model.head.weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.allclose(model.head.weight, before))

    def test_functional_map_logits_shape(self):
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        x = torch.randn(3, 4)
        self.assertEqual(make_lora_fmap(model, x, "logits").evaluate().numel(), 3 * 2)

    def test_functional_map_hidden_shape(self):
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        x = torch.randn(3, 4)
        self.assertEqual(make_lora_fmap(model, x, "hidden").evaluate().numel(), 3 * 5)

    def test_functional_map_lora_output_shape(self):
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        x = torch.randn(3, 4)
        self.assertEqual(make_lora_fmap(model, x, "lora_output").evaluate().numel(), 3 * 5)

    def test_functional_map_logits_hidden_shape(self):
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        x = torch.randn(3, 4)
        self.assertEqual(make_lora_fmap(model, x, "logits_hidden").evaluate().numel(), 3 * (2 + 5))

    def test_lora_functional_maps_jacobian_and_implicit_solver_run(self):
        torch.manual_seed(34)
        for mode in ["logits", "lora_output", "hidden", "logits_hidden"]:
            model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
            configure_train_scope(model, "lora_only")
            x = torch.randn(6, 4)
            y = (x[:, 0] > 0).long()
            fmap = make_lora_fmap(model, x[:3], mode)
            fjac = fmap.jacobian()
            self.assertEqual(fjac.jacobian.shape[1], model.lora.a.numel() + model.lora.b.numel())
            loss = F.cross_entropy(model(x), y)
            result = projected_functional_geoflow_direction(
                model,
                loss,
                x[:3],
                params=[p for p in model.parameters() if p.requires_grad],
                representation_fn=representation_fn_for(mode),
                response_solver="implicit_cg",
                production_mode=True,
                max_basis_rank=6,
                max_vjp_probes=8,
                cg_max_iter=8,
            )
            self.assertLess(result.g_dot_d, 0.0)
            self.assertTrue(torch.isfinite(torch.tensor(result.null_leakage)))

    def test_functional_step_measurement_matches_direct_evaluation(self):
        torch.manual_seed(35)
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        configure_train_scope(model, "lora_only")
        params = [p for p in model.parameters() if p.requires_grad]
        x = torch.randn(6, 4)
        direction = torch.randn(sum(p.numel() for p in params)) * 0.01
        before = model.functional_representation(x, "hidden").reshape(-1).detach()
        measured = functional_step_norm_for(model, x, "hidden", params, direction, 0.5)
        assign_flat_update(params, direction, scale=0.5)
        direct = float(torch.linalg.vector_norm(model.functional_representation(x, "hidden").reshape(-1).detach() - before))
        self.assertAlmostEqual(measured, direct, places=6)

    def test_matched_step_calibration_reaches_target(self):
        torch.manual_seed(36)
        args = self._phase_g_args()
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        configure_train_scope(model, "lora_only")
        params = [p for p in model.parameters() if p.requires_grad]
        x = torch.randn(8, 4)
        y = (x[:, 0] > 0).long()
        loss = F.cross_entropy(model(x), y)
        result = projected_functional_geoflow_direction(
            model,
            loss,
            x[:4],
            params=params,
            representation_fn=representation_fn_for("hidden"),
            response_solver="implicit_cg",
            production_mode=True,
            max_basis_rank=6,
            max_vjp_probes=8,
            cg_max_iter=8,
        )
        target = 1e-3
        _, achieved, error = calibrate_scale(model, x[:4], "hidden", params, result.direction, args.lr, target, args)
        self.assertLess(error, 0.10)
        self.assertGreater(achieved, 0.0)

    def test_calibration_does_not_mutate_reference_model(self):
        torch.manual_seed(37)
        model = SmallLoRAMLP(input_dim=4, hidden_dim=5, output_dim=2, rank=2)
        configure_train_scope(model, "lora_only")
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        x = torch.randn(8, 4)
        y = (x[:, 0] > 0).long()
        delta = reference_functional_step(model, x, y, x[:4], "hidden", "lora_only", "diagonal_grad_square", 1e-2, 0.1)
        self.assertGreaterEqual(delta, 0.0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.allclose(value, state[key]))

    def test_matched_step_preserves_descent_and_null_leakage_small(self):
        args = self._phase_g_args()
        args.steps = 1
        rows, step_rows, aggregates, gates = run_lora_matched_config(args)
        matched = [row for row in step_rows if row.optimizer == "functional_geoflow_matched_step"]
        self.assertTrue(all(row.descent_gate_passed for row in matched))
        self.assertLess(max(row.null_leakage for row in matched), 1e-4)
        self.assertIn("TASK_GAP_REDUCED_PASS", gates)
        self.assertTrue(any(row["optimizer"] == "functional_geoflow_matched_step" for row in aggregates))
        self.assertEqual(len(rows), 8)

    def test_trajectory_logging_contains_required_fields_and_cache_age(self):
        args = self._phase_g_args()
        args.steps = 2
        _, step_rows, _, _ = run_lora_matched_config(args)
        row = [item for item in step_rows if item.optimizer == "functional_geoflow_matched_step"][0]
        self.assertIn("functional_step_norm", row.__dataclass_fields__)
        self.assertIn("cache_age", row.__dataclass_fields__)
        self.assertIn("basis_rank", row.__dataclass_fields__)
        cached_rows = [item for item in step_rows if item.optimizer.startswith("functional_geoflow") and item.step == 2]
        self.assertTrue(any(item.cache_hit for item in cached_rows))

    def test_old_lora_benchmark_still_runs_after_phase_g(self):
        args = type("Args", (), {})()
        args.seed = 44
        args.trials = 1
        args.steps = 1
        args.representations = 1
        args.samples = 16
        args.probe_size = 4
        args.batch_size = 4
        args.input_dim = 4
        args.hidden_dim = 5
        args.output_dim = 2
        args.lora_rank = 2
        args.lr = 1e-2
        args.damping = 1e-3
        args.max_update_norm = 0.1
        args.refresh_interval = 4
        args.max_basis_rank = 4
        args.max_vjp_probes = 6
        args.vjp_probe_batch_size = 3
        args.cg_max_iter = 8
        args.cg_tol = 1e-5
        args.optimizers = "functional_geoflow"
        rows, _ = run_lora_benchmark(args)
        self.assertEqual(len(rows), 1)

    def _fake_phase_g_rows(self):
        rows = []
        phis = {
            ("diagonal_grad_square", 1): ["0;0", "2;0"],
            ("functional_geoflow_matched_step", 1): ["0;0", "1;0"],
            ("functional_geoflow_fixed_lr", 1): ["0;0", "3;0"],
            ("diagonal_grad_square", 2): ["10;0", "12;0"],
            ("functional_geoflow_matched_step", 2): ["10;0", "11;0"],
            ("functional_geoflow_fixed_lr", 2): ["10;0", "14;0"],
        }
        losses = {
            "diagonal_grad_square": 1.0,
            "functional_geoflow_matched_step": 1.2,
            "functional_geoflow_fixed_lr": 1.4,
        }
        for (optimizer, seed), values in phis.items():
            for rep, final_phi in enumerate(values):
                rows.append(
                    {
                        "seed": seed,
                        "optimizer": optimizer,
                        "representation": rep,
                        "train_scope": "lora_only",
                        "functional_map": "hidden",
                        "lora_rank": 2,
                        "probe_size": 4,
                        "calibration_reference": "diagonal_grad_square",
                        "initial_equivalence_residual": 0.0,
                        "final_loss": losses[optimizer] + 0.01 * rep,
                        "final_accuracy": 0.5,
                        "final_phi": final_phi,
                        "mean_functional_step": 0.1,
                        "median_functional_step": 0.1,
                        "mean_parameter_step": 0.1,
                        "mean_tangent_step": 0.01,
                        "mean_normal_step": 0.1,
                        "mean_calibration_error": 0.001 if optimizer == "functional_geoflow_matched_step" else 0.0,
                        "mean_null_leakage": 1e-8 if optimizer == "functional_geoflow_matched_step" else 0.0,
                        "total_jvp": 1,
                        "total_vjp": 1,
                        "mean_wall_clock": 0.01,
                        "peak_memory_bytes": 0,
                        "tangent_drift": 0.5 if optimizer == "diagonal_grad_square" else 0.25,
                        "near_null_amplification": 0.5,
                        "seconds": 0.02 if optimizer == "functional_geoflow_matched_step" else 0.01,
                    }
                )
        return rows

    def test_sensitivity_never_pairs_different_seeds(self):
        rows = self._fake_phase_g_rows()
        within = within_seed_pairwise_sensitivity(rows, "diagonal_grad_square", 1)
        mixed = cross_seed_mixed_pairwise_distance(rows, "diagonal_grad_square")
        self.assertAlmostEqual(within, 2.0)
        self.assertGreater(mixed, within)

    def test_within_seed_sensitivity_matches_manual_value(self):
        rows = self._fake_phase_g_rows()
        self.assertAlmostEqual(within_seed_pairwise_sensitivity(rows, "functional_geoflow_matched_step", 2), 1.0)

    def test_structural_win_rate_uses_sensitivity_not_loss(self):
        rows = self._fake_phase_g_rows()
        aggregates, gates = phase_g_statistics(rows)
        self.assertTrue(gates["STRUCTURAL_WIN_RATE_PASS"])
        self.assertEqual(gates["structural_win_rate"], 1.0)
        self.assertGreater(gates["matched_step_loss"], gates["diagonal_loss"])
        self.assertIn("mean_within_seed_sensitivity", aggregates[0])

    def test_task_loss_win_rate_is_separate(self):
        rows = self._fake_phase_g_rows()
        _, gates = phase_g_statistics(rows)
        self.assertFalse(gates["TASK_LOSS_WIN_RATE_PASS"])
        self.assertEqual(gates["task_loss_win_rate"], 0.0)

    def test_task_advantage_gate_uses_ci(self):
        rows = self._fake_phase_g_rows()
        _, gates = phase_g_statistics(rows)
        self.assertGreater(gates["matched_minus_diagonal_loss_ci_high"], 0.0)
        self.assertFalse(gates["TASK_ADVANTAGE_PASS"])

    def test_task_gap_reduced_gate_uses_paired_ci(self):
        rows = self._fake_phase_g_rows()
        _, gates = phase_g_statistics(rows)
        self.assertGreater(gates["calibration_improvement_ci_low"], 0.0)
        self.assertTrue(gates["TASK_GAP_REDUCED_PASS"])
        diffs = paired_task_differences(rows)
        self.assertEqual(len(diffs["calibration_improvement"]), 2)

    def test_cross_seed_metric_is_not_primary_sensitivity(self):
        rows = self._fake_phase_g_rows()
        aggregates, _ = phase_g_statistics(rows)
        diag = [row for row in aggregates if row["optimizer"] == "diagonal_grad_square"][0]
        self.assertIn("cross_seed_mixed_pairwise_distance", diag)
        self.assertIn("mean_within_seed_sensitivity", diag)
        self.assertNotIn("reparameterization_sensitivity", diag)

    def test_offline_analyzer_reads_existing_csv(self):
        rows = self._fake_phase_g_rows()
        aggregates, gates = analyze_rows(rows, loss_parity_margin=0.02, functional_step_tolerance=0.10)
        self.assertTrue(gates["STRUCTURAL_WIN_RATE_PASS"])
        self.assertTrue(any(row["optimizer"] == "functional_geoflow_matched_step" for row in aggregates))

    def test_incomplete_stage_b_is_skipped_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            complete = tmp_path / "run.csv"
            incomplete = tmp_path / "stage_b_partial.csv"
            rows = self._fake_phase_g_rows()
            with complete.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            incomplete.write_text("seed,optimizer\n1,diagonal_grad_square\n", encoding="utf-8")
            candidates, warnings = candidate_run_csvs(tmp_path)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(any("skipped" in warning for warning in warnings))

    def test_sweep_writes_per_configuration_gates(self):
        gates = {"STRUCTURAL_WIN_RATE_PASS": True, "TASK_GAP_REDUCED_PASS": False}
        rows = gate_rows_for_config(gates, "hidden", "lora_only", 2, 4, "diagonal_grad_square")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["functional_map"], "hidden")
        self.assertIn("calibration_reference", rows[0])

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
