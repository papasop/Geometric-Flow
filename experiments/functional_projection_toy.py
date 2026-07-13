"""Two-layer linear stable-neutral functional GeoFlow toy benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow._tensor import assign_flat_update, flatten_grads
from geometric_flow.functional_geometry import (
    FunctionalMap,
    functional_projectors,
    projected_functional_geoflow_direction,
)


class TwoLayerLinear(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x))


def known_tangent_vector(model: TwoLayerLinear) -> torch.Tensor:
    hidden_dim = model.w1.weight.shape[0]
    generator = torch.zeros(hidden_dim, hidden_dim, dtype=model.w1.weight.dtype)
    generator[0, 0] = 1.0
    if hidden_dim > 1:
        generator[0, 1] = 0.25
    delta_w1 = generator @ model.w1.weight.detach()
    delta_w2 = -model.w2.weight.detach() @ generator
    return torch.cat([delta_w1.reshape(-1), delta_w2.reshape(-1)])


def evaluate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(model(x), y)


def run_toy(
    seed: int = 7,
    input_dim: int = 3,
    hidden_dim: int = 3,
    output_dim: int = 2,
    samples: int = 8,
    lr: float = 0.05,
    damping: float = 1e-3,
    response_solver: str = "dense",
    functional_rank: int | None = None,
    functional_energy_fraction: float = 1.0,
):
    torch.manual_seed(seed)
    x = torch.randn(samples, input_dim)
    true_matrix = torch.randn(output_dim, input_dim)
    y = x @ true_matrix.T
    model = TwoLayerLinear(input_dim, hidden_dim, output_dim)
    with torch.no_grad():
        model.w1.weight.mul_(0.2)
        model.w2.weight.mul_(0.2)
    initial_state = copy.deepcopy(model.state_dict())

    fmap = FunctionalMap(model, x, representation="logits")
    fjac = fmap.jacobian()
    projectors = functional_projectors(fjac.jacobian)
    tangent = known_tangent_vector(model)
    known_tangent_residual = float(torch.linalg.vector_norm(fjac.jacobian @ tangent))

    loss_before = evaluate_loss(model, x, y)
    grads = torch.autograd.grad(loss_before, list(model.parameters()), retain_graph=True)
    raw_grad = flatten_grads(grads, list(model.parameters())).detach()
    raw_gradient_tangent_norm = float(torch.linalg.vector_norm(projectors.tangent @ raw_grad))

    dense_start = time.perf_counter()
    dense_result = projected_functional_geoflow_direction(
        model,
        loss_before,
        x,
        damping=damping,
        max_update_norm=1.0,
        response_solver="dense",
    )
    dense_seconds = time.perf_counter() - dense_start
    start = time.perf_counter()
    result = projected_functional_geoflow_direction(
        model,
        loss_before,
        x,
        damping=damping,
        max_update_norm=1.0,
        response_solver=response_solver,
        functional_rank=functional_rank,
        functional_energy_fraction=functional_energy_fraction,
    )
    solver_seconds = time.perf_counter() - start
    params = [param for param in model.parameters() if param.requires_grad]
    assign_flat_update(params, result.direction, scale=lr)
    loss_after = evaluate_loss(model, x, y)
    dense_model = TwoLayerLinear(input_dim, hidden_dim, output_dim)
    dense_model.load_state_dict(initial_state)
    dense_params = [param for param in dense_model.parameters() if param.requires_grad]
    assign_flat_update(dense_params, dense_result.direction, scale=lr)
    dense_loss_after = evaluate_loss(dense_model, x, y)
    eigvals = result.response_eigenvalues
    cosine = float(
        torch.dot(result.direction, dense_result.direction)
        / (torch.linalg.vector_norm(result.direction) * torch.linalg.vector_norm(dense_result.direction)).clamp_min(1e-30)
    )

    return {
        "response_solver": response_solver,
        "total_params": int(fjac.theta.numel()),
        "functional_jacobian_rank": fjac.rank,
        "tangent_rank": projectors.tangent_rank,
        "normal_rank": projectors.normal_rank,
        "pt_idempotent": projectors.residuals["pt_idempotent"],
        "pn_idempotent": projectors.residuals["pn_idempotent"],
        "orthogonal": projectors.residuals["orthogonal"],
        "j_pt": projectors.residuals["j_pt"],
        "rank_sum_error": projectors.residuals["rank_sum_error"],
        "known_tangent_residual": known_tangent_residual,
        "raw_gradient_tangent_norm": raw_gradient_tangent_norm,
        "projected_gradient_tangent_norm": result.projected_gradient_tangent_norm,
        "functional_geoflow_tangent_norm": result.tangent_norm,
        "loss_before": float(loss_before.detach()),
        "loss_after": float(loss_after.detach()),
        "dense_loss_after": float(dense_loss_after.detach()),
        "one_step_loss_difference_vs_dense": float(loss_after.detach() - dense_loss_after.detach()),
        "g_dot_d": result.g_dot_d,
        "response_min_eigenvalue": float(eigvals.min()) if eigvals.numel() else 0.0,
        "response_max_eigenvalue": float(eigvals.max()) if eigvals.numel() else 0.0,
        "direction_cosine_vs_dense": cosine,
        "solver_residual": result.solver_residual,
        "retained_rank": result.retained_rank,
        "retained_spectral_energy": result.retained_spectral_energy,
        "memory_estimate_bytes": result.memory_estimate_bytes,
        "jvp_count": result.jvp_count,
        "vjp_count": result.vjp_count,
        "null_leakage": result.null_leakage,
        "dense_seconds": dense_seconds,
        "solver_seconds": solver_seconds,
        "speedup_vs_dense": dense_seconds / max(solver_seconds, 1e-30),
        "descent_gate_passed": result.descent_gate_passed,
        "fallback": result.fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--input-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=3)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--response-solver", choices=["dense", "low_rank", "implicit_cg"], default="dense")
    parser.add_argument("--functional-rank", type=int, default=None)
    parser.add_argument("--functional-energy-fraction", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/functional_projection_toy.csv"))
    args = parser.parse_args()

    row = run_toy(
        seed=args.seed,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        samples=args.samples,
        lr=args.lr,
        damping=args.damping,
        response_solver=args.response_solver,
        functional_rank=args.functional_rank,
        functional_energy_fraction=args.functional_energy_fraction,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    for key, value in row.items():
        print(f"{key}={value}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
