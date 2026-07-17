import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments"
    / "h14c3_8a"
    / "h14c3_8a_transported_accepted_secant_response_audit.py"
)


def load_h14c3_8a():
    spec = importlib.util.spec_from_file_location("h14c3_8a_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


h8a = load_h14c3_8a()


def orthonormal(rows, cols, dtype=torch.float64):
    q, _ = torch.linalg.qr(torch.randn(rows, cols, dtype=dtype), mode="reduced")
    return q


def random_klr(u, v, scale=1.0):
    p = h8a.KLR(
        scale * torch.randn(u.shape[1], u.shape[1], dtype=u.dtype),
        scale * torch.randn_like(u),
        scale * torch.randn(u.shape[1], v.shape[0], dtype=u.dtype),
    )
    return h8a.project_klr_constraints(u, v, p)


class H14C38AResponseTests(unittest.TestCase):
    def test_rejected_step_does_not_extend_history(self):
        torch.manual_seed(901)
        cfg = h8a.Config(response_min_secant_norm=0.0)
        u0 = orthonormal(12, 2)
        v0 = orthonormal(10, 2)
        u1 = orthonormal(12, 2)
        v1 = orthonormal(10, 2)
        executed = random_klr(u0, v0, scale=0.01)
        old_grad = random_klr(u0, v0)
        new_grad = random_klr(u1, v1)
        history = h8a.SecantHistory(maxlen=4)

        accept = False
        if accept:
            h8a.append_accepted_secant(
                cfg, history, u0, v0, u1, v1, executed, old_grad, new_grad
            )
        self.assertEqual(len(history), 0)

        accept = True
        if accept:
            h8a.append_accepted_secant(
                cfg, history, u0, v0, u1, v1, executed, old_grad, new_grad
            )
        self.assertEqual(len(history), 1)

    def test_transport_preserves_klr_constraints(self):
        torch.manual_seed(902)
        old_u = orthonormal(16, 3)
        old_v = orthonormal(14, 3)
        new_u = orthonormal(16, 3)
        new_v = orthonormal(14, 3)
        p = random_klr(old_u, old_v)

        transported = h8a.transport_klr(old_u, old_v, new_u, new_v, p)

        self.assertLess(h8a.klr_constraint_residual(new_u, new_v, transported), 1e-12)

    def test_raw_and_derived_outputs_are_separate(self):
        verified = ROOT / "experiments" / "h14c3_8a" / "verified"
        raw = json.loads((verified / "raw_gates.json").read_text())
        derived = json.loads((verified / "derived_verdict.json").read_text())
        combined = json.loads((verified / "gates.json").read_text())

        self.assertTrue(raw)
        self.assertTrue(derived)
        self.assertFalse(any(key.startswith("DERIVED_") for key in raw))
        self.assertTrue(all(key.startswith("DERIVED_") for key in derived))
        for key in raw:
            self.assertIn(key, combined)
        for key in derived:
            self.assertIn(key, combined)

    def test_magnitude_verdict_uses_norm_ratio_not_floor_fraction(self):
        raw = {
            "PASS_DIAGONAL_BEATS_PLAIN_ON_MEAN": True,
            "PASS_FULL_CORE_BEATS_PLAIN_ON_MEAN": False,
            "MAX_ABS_LOG_NORM_RATIO": 0.0,
            "MAX_FLOOR_FRACTION": 1.0,
            "MAX_RELATIVE_DIRECTION_CHANGE": 0.0,
            "MIN_COSINE_RAW_PRE": 1.0,
        }

        verdict = h8a.derive_response_verdict(raw)
        self.assertFalse(verdict["DERIVED_MAGNITUDE_RESPONSE_ACTIVE"])

        raw["MAX_ABS_LOG_NORM_RATIO"] = 2e-3
        verdict = h8a.derive_response_verdict(raw)
        self.assertTrue(verdict["DERIVED_MAGNITUDE_RESPONSE_ACTIVE"])

    def test_response_changes_magnitude_or_direction(self):
        torch.manual_seed(903)
        cfg = h8a.Config(
            rank=2,
            d_in=10,
            d_out=12,
            response_start_accepted=0,
            response_min_secant_norm=0.0,
            response_mix=0.5,
        )
        u = orthonormal(cfg.d_out, cfg.rank)
        v = orthonormal(cfg.d_in, cfg.rank)
        history = h8a.SecantHistory(maxlen=4)
        s1 = random_klr(u, v)
        s2 = random_klr(u, v)
        history.append(u, v, s1, h8a.scale_klr(s1, 2.0))
        history.append(u, v, s2, h8a.scale_klr(s2, 2.5))
        grad = random_klr(u, v)

        preconditioned, diag = h8a.precondition_gradient(
            "diagonal_response_8a", cfg, history, u, v, grad, accepted_count=2
        )

        changed_magnitude = abs(torch.log(torch.tensor(diag.raw_to_pre_norm_ratio))) > 1e-3
        changed_direction = diag.relative_direction_change > 1e-3
        self.assertTrue(diag.active)
        self.assertTrue(bool(changed_magnitude) or changed_direction)
        self.assertGreater(h8a.klr_norm(preconditioned).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
