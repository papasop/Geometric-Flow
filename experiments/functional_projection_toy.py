"""Two-layer linear stable-neutral functional GeoFlow toy benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
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
):
    torch.manual_seed(seed)
    x = torch.randn(samples, input_dim)
    true_matrix = torch.randn(output_dim, input_dim)
    y = x @ true_matrix.T
    model = TwoLayerLinear(input_dim, hidden_dim, output_dim)
    with torch.no_grad():
        model.w1.weight.mul_(0.2)
        model.w2.weight.mul_(0.2)

    fmap = FunctionalMap(model, x, representation="logits")
    fjac = fmap.jacobian()
    projectors = functional_projectors(fjac.jacobian)
    tangent = known_tangent_vector(model)
    known_tangent_residual = float(torch.linalg.vector_norm(fjac.jacobian @ tangent))

    loss_before = evaluate_loss(model, x, y)
    grads = torch.autograd.grad(loss_before, list(model.parameters()), retain_graph=True)
    raw_grad = flatten_grads(grads, list(model.parameters())).detach()
    raw_gradient_tangent_norm = float(torch.linalg.vector_norm(projectors.tangent @ raw_grad))

    result = projected_functional_geoflow_direction(
        model,
        loss_before,
        x,
        damping=damping,
        max_update_norm=1.0,
    )
    params = [param for param in model.parameters() if param.requires_grad]
    assign_flat_update(params, result.direction, scale=lr)
    loss_after = evaluate_loss(model, x, y)
    eigvals = result.response_eigenvalues

    return {
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
        "g_dot_d": result.g_dot_d,
        "response_min_eigenvalue": float(eigvals.min()) if eigvals.numel() else 0.0,
        "response_max_eigenvalue": float(eigvals.max()) if eigvals.numel() else 0.0,
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
