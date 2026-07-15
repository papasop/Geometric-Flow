import importlib.util
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch


def load_h135_module():
    os.environ["GEOFLOW_H135_SKIP_BOOTSTRAP"] = "1"
    path = Path(__file__).resolve().parents[1] / "experiments" / "h135_rebalance_counterfactual.py"
    spec = importlib.util.spec_from_file_location("h135_rebalance_counterfactual_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_import_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "import_h135_results.py"
    spec = importlib.util.spec_from_file_location("import_h135_results_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyAdapter(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        torch.manual_seed(713)
        self.A = torch.nn.Parameter(torch.randn(3, 7, dtype=dtype))
        self.B = torch.nn.Parameter(torch.randn(5, 3, dtype=dtype))


def make_branch(optimizer="capacity", seed=11, requested_kappa=1.0, scale=1.0):
    product0 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    rows = []
    current = product0.clone()
    for step in range(1, 3):
        delta = scale * torch.tensor([1e-6, 2e-6, 3e-6], dtype=torch.float64)
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


def test_h135_optimizer_plan_is_exact():
    module = load_h135_module()
    variants = [
        "adamw",
        "adamw_rebalance_1",
        "adamw_rebalance_10",
        "sgd",
        "sgd_rebalance_1",
        "capacity",
    ]

    assert {name for name in variants} == {
        "adamw",
        "adamw_rebalance_1",
        "adamw_rebalance_10",
        "sgd",
        "sgd_rebalance_1",
        "capacity",
    }
    assert [module._h135_optimizer_spec(name) for name in variants] == [
        ("adamw", 0),
        ("adamw", 1),
        ("adamw", 10),
        ("sgd", 0),
        ("sgd", 1),
        ("capacity", 0),
    ]
    with pytest.raises(ValueError):
        module._h135_optimizer_spec("adamw_rebalance_2")


def test_h135_rebalance_preserves_product_and_balances_grams_float64():
    module = load_h135_module()
    adapter = DummyAdapter(dtype=torch.float64)
    before = adapter.B.detach() @ adapter.A.detach()

    diag = module._h135_rebalance_adapters([adapter])

    after = adapter.B.detach() @ adapter.A.detach()
    relative_gap = torch.linalg.vector_norm(after - before) / torch.linalg.vector_norm(before)
    gram_gap = torch.linalg.vector_norm((adapter.B.T @ adapter.B - adapter.A @ adapter.A.T).detach())
    assert float(relative_gap) < 1e-10
    assert diag["rebalance_product_gap"] < 1e-10
    assert float(gram_gap) < 1e-10
    assert diag["rebalance_gram_gap"] < 1e-10


def test_h135_rebalance_preserves_product_float32():
    module = load_h135_module()
    adapter = DummyAdapter(dtype=torch.float32)
    before = adapter.B.detach() @ adapter.A.detach()

    diag = module._h135_rebalance_adapters([adapter])

    after = adapter.B.detach() @ adapter.A.detach()
    relative_gap = torch.linalg.vector_norm(after - before) / torch.linalg.vector_norm(before)
    assert float(relative_gap) < 1e-5
    assert diag["rebalance_product_gap"] < 1e-5


def test_h135_rebalance_interval_behavior():
    module = load_h135_module()
    assert module._h135_optimizer_spec("adamw_rebalance_1")[1] == 1
    assert module._h135_optimizer_spec("adamw_rebalance_10")[1] == 10
    assert module._h135_optimizer_spec("adamw")[1] == 0

    def applied_steps(interval):
        return [step for step in range(1, 31) if interval > 0 and step % interval == 0]

    assert applied_steps(1) == list(range(1, 31))
    assert applied_steps(10) == [10, 20, 30]
    assert applied_steps(0) == []


def test_h135_cosine_clamps_and_pair_metrics_use_gauge_kappa():
    module = load_h135_module()
    x = torch.tensor([1.0, 0.0])
    with mock.patch.object(module.torch, "dot", return_value=torch.tensor(1.000001)):
        assert module._h135_cosine(x, x) == 1.0
    with mock.patch.object(module.torch, "dot", return_value=torch.tensor(-1.000001)):
        assert module._h135_cosine(x, -x) == -1.0

    balanced = make_branch(optimizer="capacity", requested_kappa=1.0)
    gauge = make_branch(optimizer="capacity", requested_kappa=1000.0)
    rows = module._h135_pair_metrics(balanced, gauge)
    assert {row["kappa"] for row in rows} == {1000.0}


def test_h135_summary_groups_6_methods_by_3_kappas_and_only_capacity_passes():
    module = load_h135_module()
    variants = [
        "adamw",
        "adamw_rebalance_1",
        "adamw_rebalance_10",
        "sgd",
        "sgd_rebalance_1",
        "capacity",
    ]
    pair_rows = []
    for optimizer in variants:
        for kappa in (5.0, 100.0, 1000.0):
            balanced = make_branch(optimizer=optimizer, requested_kappa=1.0, scale=1.0)
            gauge_scale = 1.0 if optimizer == "capacity" else 1.0 + kappa / 1000.0
            gauge = make_branch(optimizer=optimizer, requested_kappa=kappa, scale=gauge_scale)
            pair_rows.extend(module._h135_pair_metrics(balanced, gauge))

    summaries = module._h135_summarize(pair_rows, steps=2)

    assert len(summaries) == 18
    assert {row["optimizer"] for row in summaries} == set(variants)
    assert {row["kappa"] for row in summaries} == {5.0, 100.0, 1000.0}
    assert all(row["pass_near_gauge_equivariance"] for row in summaries if row["optimizer"] == "capacity")
    assert not any(row["pass_near_gauge_equivariance"] for row in summaries if row["optimizer"] != "capacity")


def test_h135_import_tool_refuses_incomplete_results_without_allow_partial(tmp_path):
    module = load_import_tool()
    source = tmp_path / "source"
    repo = tmp_path / "repo"
    source.mkdir()
    (repo / ".git").mkdir(parents=True)
    (source / module.EXPECTED[0]).write_text("partial", encoding="utf-8")

    with mock.patch.object(
        sys,
        "argv",
        ["import_h135_results.py", "--source", str(source), "--repo", str(repo)],
    ):
        with pytest.raises(SystemExit):
            module.main()

    with mock.patch.object(
        sys,
        "argv",
        [
            "import_h135_results.py",
            "--source",
            str(source),
            "--repo",
            str(repo),
            "--allow-partial",
        ],
    ):
        module.main()
    assert (repo / "results" / "h135" / "manifest.json").is_file()
