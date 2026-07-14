import unittest
import copy

import torch

from geometric_flow import (
    CapacityAdaptiveQuotientFlow,
    FixedRankFunctionalAdam,
    FixedRankManifold,
    HeldOutTrustRegion,
    ProductParameter,
    ProductState,
    SubsteppedQuotientFlow,
)


def make_product(rank=2, dtype=torch.float32, device="cpu"):
    torch.manual_seed(11)
    a = torch.randn(rank, 4, dtype=dtype, device=device)
    b = torch.randn(5, rank, dtype=dtype, device=device)
    return a, b, b @ a


def state_from_matrix(matrix, rank=2, name="lora"):
    return ProductState([ProductParameter(name, torch.nn.Parameter(matrix.clone()), rank)])


class FactorModule(torch.nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.A = torch.nn.Parameter(a.clone())
        self.B = torch.nn.Parameter(b.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.B @ self.A).T

    def product(self) -> torch.Tensor:
        return self.B @ self.A


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


class SubsteppedQuotientFlowTests(unittest.TestCase):
    def test_constructor_validation_and_local_lr(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        with self.assertRaises(ValueError):
            SubsteppedQuotientFlow([module], macro_lr=0.0)
        with self.assertRaises(ValueError):
            SubsteppedQuotientFlow([module], macro_lr=1.0, substeps=0)
        with self.assertRaises(ValueError):
            SubsteppedQuotientFlow([module], macro_lr=1.0, clip_norm=0.0)
        with self.assertRaises(ValueError):
            SubsteppedQuotientFlow([module], macro_lr=1.0, gram_condition_limit=1.0)
        with self.assertRaisesRegex(ValueError, "A and B"):
            SubsteppedQuotientFlow([torch.nn.Linear(2, 2)], macro_lr=1.0)
        optimizer = SubsteppedQuotientFlow([module], macro_lr=3.0, substeps=4)
        self.assertEqual(optimizer.local_lr, 0.75)
        self.assertEqual(optimizer.last_diagnostics["local_lr"], 0.75)

    def test_shape_device_dtype_preserved_and_no_moment_tensors(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        optimizer = SubsteppedQuotientFlow([module], macro_lr=0.1, substeps=2, balance_after_substep=False)
        target = torch.randn_like(module.product())
        loss = (module.product() - target).pow(2).sum()
        loss.backward()
        a_shape, b_shape = module.A.shape, module.B.shape
        optimizer.step()
        self.assertEqual(module.A.shape, a_shape)
        self.assertEqual(module.B.shape, b_shape)
        self.assertEqual(module.A.dtype, torch.float64)
        self.assertEqual(module.B.dtype, torch.float64)
        self.assertEqual(len(optimizer.state), 0)

    def test_balance_preserves_product(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        optimizer = SubsteppedQuotientFlow([module], macro_lr=0.1)
        before = module.product().detach().clone()
        with torch.no_grad():
            optimizer._balance_(module)
        after = module.product().detach()
        self.assertTrue(torch.allclose(after, before, atol=1e-12, rtol=1e-10))
        self.assertLess(optimizer.balance_residual_max, 1e-10)

    def test_substeps_one_matches_single_step_quotient_update(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        target = torch.randn_like(module.product())
        loss = (module.product() - target).pow(2).sum()
        loss.backward()
        grad_a = module.A.grad.detach().clone()
        grad_b = module.B.grad.detach().clone()
        inv_b = torch.linalg.inv(module.B.detach().T @ module.B.detach())
        inv_a = torch.linalg.inv(module.A.detach() @ module.A.detach().T)
        expected_a = module.A.detach() - 0.05 * (inv_b @ grad_a)
        expected_b = module.B.detach() - 0.05 * (grad_b @ inv_a)
        optimizer = SubsteppedQuotientFlow([module], macro_lr=0.05, substeps=1, balance_after_substep=False)
        optimizer.step()
        self.assertTrue(torch.allclose(module.A.detach(), expected_a, atol=1e-12, rtol=1e-10))
        self.assertTrue(torch.allclose(module.B.detach(), expected_b, atol=1e-12, rtol=1e-10))

    def test_full_rank_inverse_branch_is_near_machine_gauge_equivariant(self):
        torch.manual_seed(22)
        a, b, _ = make_product(dtype=torch.float64)
        gauge = torch.tensor([[1.4, 0.2], [0.1, 0.8]], dtype=torch.float64)
        a_gauge = gauge @ a
        b_gauge = b @ torch.linalg.inv(gauge)
        target = torch.randn_like(b @ a)
        self.assertTrue(torch.allclose(b @ a, b_gauge @ a_gauge, atol=1e-12, rtol=1e-12))

        def run_once(a0, b0):
            module = FactorModule(a0, b0)
            optimizer = SubsteppedQuotientFlow(
                [module],
                macro_lr=0.01,
                substeps=1,
                balance_after_substep=False,
                gram_condition_limit=1e12,
            )
            loss = (module.product() - target).pow(2).sum()
            loss.backward()
            optimizer.step()
            return module.product().detach(), optimizer

        product_a, opt_a = run_once(a, b)
        product_b, opt_b = run_once(a_gauge, b_gauge)
        relative_product_gap = (product_a - product_b).norm() / product_a.norm().clamp_min(
            torch.finfo(product_a.dtype).tiny
        )
        self.assertLess(float(relative_product_gap), 1e-10)
        self.assertEqual(opt_a.fallback_count, 0)
        self.assertEqual(opt_b.fallback_count, 0)

    def test_macro_step_calls_closure_and_recomputes_gradients(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        target = torch.randn_like(module.product())
        optimizer = SubsteppedQuotientFlow([module], macro_lr=0.03, substeps=3, balance_after_substep=False)
        calls = []
        grad_snapshots = []

        def closure():
            optimizer.zero_grad()
            loss = (module.product() - target).pow(2).sum()
            loss.backward()
            calls.append(loss.item())
            grad_snapshots.append(module.A.grad.detach().clone())
            return loss

        optimizer.macro_step(closure)
        self.assertEqual(len(calls), 3)
        self.assertFalse(torch.allclose(grad_snapshots[0], grad_snapshots[-1]))

    def test_pseudoinverse_fallback_activates_for_ill_conditioned_gram(self):
        a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1e-8, 0.0, 0.0]], dtype=torch.float64)
        b = torch.tensor([[1.0, 0.0], [0.0, 1e-8], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        module = FactorModule(a, b)
        optimizer = SubsteppedQuotientFlow([module], macro_lr=0.01, gram_condition_limit=10.0, balance_after_substep=False)
        loss = module.product().pow(2).sum()
        loss.backward()
        optimizer.step()
        self.assertGreaterEqual(optimizer.fallback_count, 1)
        self.assertGreater(optimizer.condition_max, 10.0)

    def test_clipping_reports_scale_and_bounds_update_norm(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        optimizer = SubsteppedQuotientFlow([module], macro_lr=10.0, clip_norm=0.01, balance_after_substep=False)
        loss = module.product().pow(2).sum()
        loss.backward()
        optimizer.step()
        self.assertLessEqual(optimizer.last_clip_scale, 1.0)
        self.assertLessEqual(optimizer.last_update_norm, 0.01001)

    def test_state_dict_roundtrip_syncs_custom_hyperparameters(self):
        a, b, _ = make_product(dtype=torch.float64)
        source_module = FactorModule(a, b)
        source = SubsteppedQuotientFlow(
            [source_module],
            macro_lr=2.0,
            substeps=5,
            clip_norm=0.7,
            balance_after_substep=False,
            gram_condition_limit=1234.0,
        )
        state = copy.deepcopy(source.state_dict())

        target_module = FactorModule(a, b)
        target = SubsteppedQuotientFlow(
            [target_module],
            macro_lr=1.0,
            substeps=2,
            clip_norm=None,
            balance_after_substep=True,
            gram_condition_limit=1e10,
        )
        target.load_state_dict(state)

        self.assertEqual(target.macro_lr, 2.0)
        self.assertEqual(target.substeps, 5)
        self.assertEqual(target.local_lr, 0.4)
        self.assertEqual(target.clip_norm, 0.7)
        self.assertFalse(target.balance_after_substep)
        self.assertEqual(target.gram_condition_limit, 1234.0)
        self.assertEqual(target.param_groups[0]["local_lr"], 0.4)
        self.assertEqual(target.last_diagnostics["local_lr"], 0.4)

    def test_gauge_equivalent_factors_stay_closer_than_naive_factor_adam(self):
        torch.manual_seed(21)
        a, b, _ = make_product(dtype=torch.float64)
        scale = torch.diag(torch.tensor([3.0, 0.25], dtype=torch.float64))
        target = torch.randn_like(b @ a)

        def quotient_run(a0, b0):
            module = FactorModule(a0, b0)
            optimizer = SubsteppedQuotientFlow([module], macro_lr=0.02, substeps=2)

            def closure():
                optimizer.zero_grad()
                loss = (module.product() - target).pow(2).sum()
                loss.backward()
                return loss

            for _ in range(8):
                optimizer.macro_step(closure)
            return module.product().detach()

        def adam_run(a0, b0):
            module = FactorModule(a0, b0)
            optimizer = torch.optim.Adam([module.A, module.B], lr=0.02)
            for _ in range(16):
                optimizer.zero_grad()
                loss = (module.product() - target).pow(2).sum()
                loss.backward()
                optimizer.step()
            return module.product().detach()

        q_a = quotient_run(a, b)
        q_b = quotient_run(scale @ a, b @ torch.linalg.inv(scale))
        adam_a = adam_run(a, b)
        adam_b = adam_run(scale @ a, b @ torch.linalg.inv(scale))
        quotient_gap = (q_a - q_b).norm()
        adam_gap = (adam_a - adam_b).norm()
        self.assertLess(quotient_gap, 0.5 * adam_gap)


class CapacityAdaptiveQuotientFlowTests(unittest.TestCase):
    def test_constructor_validation(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        with self.assertRaises(ValueError):
            CapacityAdaptiveQuotientFlow([module], macro_flow_time=0.0, local_function_tolerance=0.1)
        with self.assertRaises(ValueError):
            CapacityAdaptiveQuotientFlow([module], macro_flow_time=1.0, local_function_tolerance=0.0)
        with self.assertRaises(ValueError):
            CapacityAdaptiveQuotientFlow([module], macro_flow_time=1.0, local_function_tolerance=0.1, max_auto_substeps=0)
        with self.assertRaises(ValueError):
            CapacityAdaptiveQuotientFlow([module], macro_flow_time=1.0, local_function_tolerance=0.1, max_flow_dt=0.0)
        with self.assertRaises(ValueError):
            CapacityAdaptiveQuotientFlow(
                [module], macro_flow_time=1.0, local_function_tolerance=0.1, gram_condition_limit=1.0
            )
        optimizer = CapacityAdaptiveQuotientFlow([module], macro_flow_time=1.0, local_function_tolerance=0.1)
        self.assertEqual(optimizer.last_diagnostics["macro_flow_time"], 1.0)

    def test_single_step_direction_capacity_and_dt_match_explicit_formula(self):
        torch.manual_seed(31)
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        target = torch.randn_like(module.product())
        loss = (module.product() - target).pow(2).sum()
        loss.backward()
        grad_a = module.A.grad.detach().clone()
        grad_b = module.B.grad.detach().clone()
        inv_b = torch.linalg.inv(module.B.detach().T @ module.B.detach())
        inv_a = torch.linalg.inv(module.A.detach() @ module.A.detach().T)
        v_a = -(inv_b @ grad_a)
        v_b = -(grad_b @ inv_a)
        capacity = torch.linalg.vector_norm(v_b @ module.A.detach() + module.B.detach() @ v_a)
        flow_dt = min(0.07, 1e9 / float(capacity))
        expected_a = module.A.detach() + flow_dt * v_a
        expected_b = module.B.detach() + flow_dt * v_b

        optimizer = CapacityAdaptiveQuotientFlow(
            [module],
            macro_flow_time=0.07,
            local_function_tolerance=1e9,
            max_auto_substeps=1,
            balance_after_substep=False,
        )
        optimizer.macro_step(lambda: loss)
        self.assertTrue(torch.allclose(module.A.detach(), expected_a, atol=1e-12, rtol=1e-10))
        self.assertTrue(torch.allclose(module.B.detach(), expected_b, atol=1e-12, rtol=1e-10))
        self.assertAlmostEqual(optimizer.last_capacity, float(capacity), places=10)
        self.assertAlmostEqual(optimizer.last_flow_dt, flow_dt, places=12)

    def test_local_product_displacement_tolerance_is_respected(self):
        torch.manual_seed(32)
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        target = torch.randn_like(module.product())
        optimizer = CapacityAdaptiveQuotientFlow(
            [module],
            macro_flow_time=0.03,
            local_function_tolerance=0.01,
            max_auto_substeps=128,
            balance_after_substep=False,
        )

        def closure():
            optimizer.zero_grad()
            loss = (module.product() - target).pow(2).sum()
            loss.backward()
            return loss

        optimizer.macro_step(closure)
        self.assertLessEqual(optimizer.max_predicted_local_dphi, 0.01 * (1.0 + 1e-10) + 1e-12)
        self.assertGreater(optimizer.last_auto_substeps, 1)

    def test_dynamic_substep_count_changes_with_capacity(self):
        torch.manual_seed(33)
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        target = torch.zeros_like(module.product())
        optimizer = CapacityAdaptiveQuotientFlow(
            [module],
            macro_flow_time=0.16,
            local_function_tolerance=0.02,
            max_auto_substeps=256,
            balance_after_substep=False,
        )
        counts = []

        def closure():
            optimizer.zero_grad()
            loss = (module.product() - target).pow(2).sum()
            loss.backward()
            return loss

        for _ in range(3):
            optimizer.macro_step(closure)
            counts.append(optimizer.last_auto_substeps)
        self.assertGreater(max(counts), 1)
        self.assertGreater(len(set(counts)), 1)

    def test_fresh_gradient_closure_called_for_each_auto_substep(self):
        torch.manual_seed(34)
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        target = torch.randn_like(module.product())
        optimizer = CapacityAdaptiveQuotientFlow(
            [module],
            macro_flow_time=0.12,
            local_function_tolerance=0.02,
            max_auto_substeps=128,
            balance_after_substep=False,
        )
        grad_snapshots = []

        def closure():
            optimizer.zero_grad()
            loss = (module.product() - target).pow(2).sum()
            loss.backward()
            grad_snapshots.append(module.A.grad.detach().clone())
            return loss

        optimizer.macro_step(closure)
        self.assertEqual(len(grad_snapshots), optimizer.last_auto_substeps)
        self.assertFalse(torch.allclose(grad_snapshots[0], grad_snapshots[-1]))

    def test_zero_capacity_consumes_remaining_time_without_looping(self):
        torch.manual_seed(341)
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        optimizer = CapacityAdaptiveQuotientFlow(
            [module],
            macro_flow_time=1.25,
            local_function_tolerance=0.01,
            max_auto_substeps=1,
            balance_after_substep=False,
        )
        before_a = module.A.detach().clone()
        before_b = module.B.detach().clone()
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad()
            loss = (module.product() * 0.0).sum()
            loss.backward()
            return loss

        optimizer.macro_step(closure)
        self.assertEqual(calls, 1)
        self.assertEqual(optimizer.last_auto_substeps, 1)
        self.assertEqual(optimizer.last_capacity, 0.0)
        self.assertEqual(optimizer.last_predicted_local_dphi, 0.0)
        self.assertAlmostEqual(optimizer.last_flow_dt, 1.25, places=12)
        self.assertTrue(torch.equal(module.A.detach(), before_a))
        self.assertTrue(torch.equal(module.B.detach(), before_b))

    def test_full_rank_inverse_branch_is_near_machine_gauge_equivariant(self):
        torch.manual_seed(35)
        a, b, _ = make_product(dtype=torch.float64)
        gauge = torch.tensor([[1.25, 0.3], [0.15, 0.85]], dtype=torch.float64)
        a_gauge = gauge @ a
        b_gauge = b @ torch.linalg.inv(gauge)
        target = torch.randn_like(b @ a)

        def run_once(a0, b0):
            module = FactorModule(a0, b0)
            optimizer = CapacityAdaptiveQuotientFlow(
                [module],
                macro_flow_time=0.03,
                local_function_tolerance=1.0,
                max_auto_substeps=4,
                balance_after_substep=False,
                gram_condition_limit=1e12,
            )

            def closure():
                optimizer.zero_grad()
                loss = (module.product() - target).pow(2).sum()
                loss.backward()
                return loss

            optimizer.macro_step(closure)
            return module.product().detach(), optimizer

        product_a, opt_a = run_once(a, b)
        product_b, opt_b = run_once(a_gauge, b_gauge)
        relative_gap = (product_a - product_b).norm() / product_a.norm().clamp_min(torch.finfo(product_a.dtype).tiny)
        self.assertLess(float(relative_gap), 1e-10)
        self.assertEqual(opt_a.fallback_count, 0)
        self.assertEqual(opt_b.fallback_count, 0)

    def test_qr_canonicalization_preserves_product(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        optimizer = CapacityAdaptiveQuotientFlow([module], macro_flow_time=0.1, local_function_tolerance=0.1)
        before = module.product().detach().clone()
        with torch.no_grad():
            optimizer._balance_(module)
        after = module.product().detach()
        self.assertTrue(torch.allclose(after, before, atol=1e-12, rtol=1e-10))

    def test_pseudoinverse_fallback_activates_for_ill_conditioned_gram(self):
        a = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1e-8, 0.0, 0.0]], dtype=torch.float64)
        b = torch.tensor([[1.0, 0.0], [0.0, 1e-8], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        module = FactorModule(a, b)
        optimizer = CapacityAdaptiveQuotientFlow(
            [module], macro_flow_time=0.01, local_function_tolerance=0.1, gram_condition_limit=10.0
        )

        def closure():
            optimizer.zero_grad()
            loss = module.product().pow(2).sum()
            loss.backward()
            return loss

        optimizer.macro_step(closure)
        self.assertGreaterEqual(optimizer.fallback_count, 1)
        self.assertGreater(optimizer.condition_max, 10.0)

    def test_no_adam_moment_state(self):
        a, b, _ = make_product(dtype=torch.float64)
        module = FactorModule(a, b)
        optimizer = CapacityAdaptiveQuotientFlow([module], macro_flow_time=0.1, local_function_tolerance=0.1)
        self.assertEqual(len(optimizer.state), 0)

    def test_state_dict_roundtrip_syncs_custom_hyperparameters(self):
        a, b, _ = make_product(dtype=torch.float64)
        source = CapacityAdaptiveQuotientFlow(
            [FactorModule(a, b)],
            macro_flow_time=2.0,
            local_function_tolerance=0.3,
            max_auto_substeps=17,
            max_flow_dt=0.25,
            balance_after_substep=False,
            gram_condition_limit=1234.0,
        )
        state = copy.deepcopy(source.state_dict())
        target = CapacityAdaptiveQuotientFlow(
            [FactorModule(a, b)],
            macro_flow_time=1.0,
            local_function_tolerance=0.1,
            max_auto_substeps=5,
            max_flow_dt=None,
            balance_after_substep=True,
            gram_condition_limit=1e10,
        )
        target.load_state_dict(state)
        self.assertEqual(target.macro_flow_time, 2.0)
        self.assertEqual(target.local_function_tolerance, 0.3)
        self.assertEqual(target.max_auto_substeps, 17)
        self.assertEqual(target.max_flow_dt, 0.25)
        self.assertFalse(target.balance_after_substep)
        self.assertEqual(target.gram_condition_limit, 1234.0)

    def test_public_export(self):
        from geometric_flow import CapacityAdaptiveQuotientFlow as Exported

        self.assertIs(Exported, CapacityAdaptiveQuotientFlow)


if __name__ == "__main__":
    unittest.main()
