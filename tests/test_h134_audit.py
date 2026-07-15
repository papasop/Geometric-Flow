import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import torch


def load_h134_module():
    os.environ["GEOFLOW_H134_SKIP_BOOTSTRAP"] = "1"
    path = Path(__file__).resolve().parents[1] / "experiments" / "h134_full_product_audit.py"
    spec = importlib.util.spec_from_file_location("h134_full_product_audit_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_branch(module, *, optimizer="capacity", seed=11, requested_kappa=1.0, scale=1.0):
    product0 = torch.tensor([1.0, 2.0, 3.0])
    rows = []
    current = product0.clone()
    for step in range(1, 3):
        delta = scale * torch.tensor([1e-6, 2e-6, 3e-6])
        current = current + delta
        rows.append(
            {
                "optimizer": optimizer,
                "representation": "balanced" if requested_kappa == 1.0 else "gauge",
                "requested_kappa": requested_kappa,
                "seed": seed,
                "step": step,
                "batch_hash": f"batch-{step}",
                "loss_before": 1.0,
                "same_batch_loss_delta": 0.01,
                "_product": current.clone(),
                "_delta_product": delta.clone(),
            }
        )
    return {
        "optimizer": optimizer,
        "representation": "balanced" if requested_kappa == 1.0 else "gauge",
        "requested_kappa": requested_kappa,
        "seed": seed,
        "initial_product": product0,
        "initial_condition": {"cond_max_mean": 1.0},
        "rows": rows,
    }


def test_h134_cosine_clamps_roundoff_above_one():
    module = load_h134_module()
    x = torch.tensor([1.0, 0.0])
    y = torch.tensor([1.0, 0.0])
    with mock.patch.object(module.torch, "dot", return_value=torch.tensor(1.000001)):
        assert module._h134_cosine(x, y) == 1.0
    with mock.patch.object(module.torch, "dot", return_value=torch.tensor(-1.000001)):
        assert module._h134_cosine(x, -y) == -1.0


def test_h134_pair_metrics_take_kappa_from_gauge_branch():
    module = load_h134_module()
    balanced = make_branch(module, optimizer="capacity", requested_kappa=1.0)
    gauge = make_branch(module, optimizer="capacity", requested_kappa=100.0)

    rows = module._h134_pair_metrics(balanced, gauge)

    assert {row["kappa"] for row in rows} == {100.0}
    assert all(row["initial_product_relative_gap"] == 0.0 for row in rows)


def test_h134_summary_groups_four_kappas_for_both_optimizers():
    module = load_h134_module()
    pair_rows = []
    for optimizer in ("adamw", "capacity"):
        for kappa in (5.0, 10.0, 100.0, 1000.0):
            balanced = make_branch(module, optimizer=optimizer, requested_kappa=1.0, scale=1.0)
            gauge_scale = 1.0 if optimizer == "capacity" else 1.0 + kappa / 1000.0
            gauge = make_branch(module, optimizer=optimizer, requested_kappa=kappa, scale=gauge_scale)
            pair_rows.extend(module._h134_pair_metrics(balanced, gauge))

    summaries = module._h134_summarize(pair_rows, steps=2)

    assert len(summaries) == 8
    assert {row["kappa"] for row in summaries} == {5.0, 10.0, 100.0, 1000.0}
    assert {row["optimizer"] for row in summaries} == {"adamw", "capacity"}


def test_h134_capacity_summary_sets_near_equivariance_pass_gate():
    module = load_h134_module()
    pair_rows = []
    for kappa in (5.0, 10.0, 100.0, 1000.0):
        balanced = make_branch(module, optimizer="capacity", requested_kappa=1.0, scale=1.0)
        gauge = make_branch(module, optimizer="capacity", requested_kappa=kappa, scale=1.0)
        pair_rows.extend(module._h134_pair_metrics(balanced, gauge))

    summaries = module._h134_summarize(pair_rows, steps=2)

    assert all(row["pass_initial_equivalence"] for row in summaries)
    assert all(row["pass_near_gauge_equivariance"] for row in summaries)
