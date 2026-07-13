"""Toy normal-projected Hessian benchmark for a two-layer linear network.

The model is f(x) = W2 W1 x. The factorization has a reparameterization gauge:
W1 -> A W1, W2 -> W2 A^{-1}. Infinitesimally, tangent directions have
delta W1 = B W1 and delta W2 = -W2 B. This script constructs that tangent space,
projects onto the normal space, and reports eigenvalues of P_N H P_N.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F


def unpack(theta: torch.Tensor, input_dim: int, hidden_dim: int, output_dim: int):
    split = hidden_dim * input_dim
    w1 = theta[:split].reshape(hidden_dim, input_dim)
    w2 = theta[split:].reshape(output_dim, hidden_dim)
    return w1, w2


def loss_from_theta(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    w1, w2 = unpack(theta, input_dim, hidden_dim, output_dim)
    prediction = x @ w1.T @ w2.T
    return F.mse_loss(prediction, y)


def tangent_basis(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    hidden_dim, input_dim = w1.shape
    output_dim = w2.shape[0]
    basis = []
    for row in range(hidden_dim):
        for col in range(hidden_dim):
            generator = torch.zeros(hidden_dim, hidden_dim, dtype=w1.dtype, device=w1.device)
            generator[row, col] = 1.0
            delta_w1 = generator @ w1
            delta_w2 = -w2 @ generator
            basis.append(torch.cat([delta_w1.reshape(-1), delta_w2.reshape(-1)]))
    if not basis:
        return torch.empty(0, w1.numel() + w2.numel(), dtype=w1.dtype, device=w1.device)
    return torch.stack(basis, dim=1)


def normal_projector(tangent: torch.Tensor, size: int) -> torch.Tensor:
    identity = torch.eye(size, dtype=tangent.dtype, device=tangent.device)
    if tangent.numel() == 0:
        return identity
    tangent_projector = tangent @ torch.linalg.pinv(tangent)
    return identity - tangent_projector


def normal_projected_hessian(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor, hidden_dim: int):
    theta = theta.detach().clone().requires_grad_(True)
    hessian = torch.autograd.functional.hessian(lambda p: loss_from_theta(p, x, y, hidden_dim), theta)
    w1, w2 = unpack(theta.detach(), x.shape[1], hidden_dim, y.shape[1])
    tangent = tangent_basis(w1, w2)
    projector = normal_projector(tangent, theta.numel())
    projected = projector @ hessian.detach() @ projector
    return projected, hessian.detach(), projector


def run_toy(seed: int, input_dim: int, hidden_dim: int, output_dim: int, samples: int):
    torch.manual_seed(seed)
    x = torch.randn(samples, input_dim)
    true_matrix = torch.randn(output_dim, input_dim)
    y = x @ true_matrix.T
    w1 = 0.2 * torch.randn(hidden_dim, input_dim)
    w2 = 0.2 * torch.randn(output_dim, hidden_dim)
    theta = torch.cat([w1.reshape(-1), w2.reshape(-1)])
    projected, hessian, projector = normal_projected_hessian(theta, x, y, hidden_dim)
    eigvals = torch.linalg.eigvalsh(projected)
    tangent_rank = int(torch.linalg.matrix_rank(torch.eye(theta.numel()) - projector))
    return {
        "loss": float(loss_from_theta(theta, x, y, hidden_dim)),
        "params": int(theta.numel()),
        "tangent_rank": tangent_rank,
        "normal_rank": int(torch.linalg.matrix_rank(projector)),
        "hessian_trace": float(torch.trace(hessian)),
        "normal_projected_trace": float(torch.trace(projected)),
        "min_normal_eigenvalue": float(eigvals.min()),
        "max_normal_eigenvalue": float(eigvals.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--input-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=2)
    parser.add_argument("--output-dim", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/normal_projection_toy.csv"))
    args = parser.parse_args()

    row = run_toy(args.seed, args.input_dim, args.hidden_dim, args.output_dim, args.samples)
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
