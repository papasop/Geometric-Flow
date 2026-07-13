"""Noise stress test for redundant functional geometry."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import FunctionalMap, GeometricOptimizer, functional_projectors
from geometric_flow._tensor import assign_flat_update, flatten_tensors
from experiments.reparameterization_stress_test import (
    RedundantLinearNet,
    collect_batches,
    evaluate,
    init_base_model,
    make_classification,
    reparameterize_model,
)


@dataclass
class NoiseRun:
    seed: int
    optimizer: str
    sigma_g: float
    sigma_theta: float
    final_loss: float
    final_accuracy: float
    functional_output_error: float
    recovery_steps: int
    injected_tangent_norm: float
    injected_normal_norm: float
    post_projection_tangent_norm: float
    post_projection_normal_norm: float
    tangent_component_retained: float
    gate_accept_rate: float
    fallback_rate: float
    wall_clock: float


def flat_params(params) -> torch.Tensor:
    return flatten_tensors([param.reshape(-1) for param in params])


def noisy_loss(model, x, y, grad_noise: torch.Tensor | None) -> torch.Tensor:
    loss = F.cross_entropy(model(x), y)
    if grad_noise is None:
        return loss
    theta = flat_params([param for param in model.parameters() if param.requires_grad])
    return loss + torch.dot(theta, grad_noise.to(theta))


def decompose(model, probe_x: torch.Tensor, vector: torch.Tensor, null_threshold_mode: str, null_tol: float):
    projectors = functional_projectors(
        FunctionalMap(model, probe_x).jacobian().jacobian,
        null_threshold_mode=null_threshold_mode,
        null_tol=null_tol,
    )
    tangent = projectors.tangent @ vector
    normal = projectors.normal @ vector
    return float(torch.linalg.vector_norm(tangent)), float(torch.linalg.vector_norm(normal))


def run_branch(args, optimizer_name: str, sigma_g: float, sigma_theta: float, seed: int) -> NoiseRun:
    torch.manual_seed(seed)
    train_set = make_classification(args.train_samples, args.input_dim, args.output_dim, seed)
    eval_set = make_classification(args.eval_samples, args.input_dim, args.output_dim, seed + 3000)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    batches = collect_batches(loader, args.steps)
    probe_x = train_set.tensors[0][: args.probe_size].detach().clone()
    base = init_base_model(args.input_dim, args.hidden_dim, args.output_dim, seed)
    model = reparameterize_model(base, torch.eye(args.hidden_dim))
    initial_phi = FunctionalMap(model, probe_x).evaluate().detach()
    params = [param for param in model.parameters() if param.requires_grad]
    n_params = sum(param.numel() for param in params)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
    elif optimizer_name == "diagonal_grad_square":
        optimizer = GeometricOptimizer(
            params,
            lr=args.lr,
            lr_scale=1.0,
            mode="geometric",
            warmup_steps=0,
            preconditioner="diagonal_grad_square",
            max_update_norm=args.max_update_norm,
            grad_smoothing=0.0,
        )
    elif optimizer_name == "functional_geoflow":
        optimizer = GeometricOptimizer(
            params,
            lr=args.lr,
            lr_scale=1.0,
            mode="functional_geoflow",
            functional_model=model,
            functional_probe=probe_x,
            max_update_norm=args.max_update_norm,
            null_threshold_mode=args.null_threshold_mode,
            null_tol=args.null_tol,
        )
    else:
        raise ValueError(optimizer_name)

    injected_tangent = []
    injected_normal = []
    post_tangent = []
    post_normal = []
    phi_errors = []
    start = time.perf_counter()
    generator = torch.Generator().manual_seed(seed + 9000)
    for step, (x, y) in enumerate(batches, start=1):
        grad_noise = sigma_g * torch.randn(n_params, generator=generator) if sigma_g else None
        if grad_noise is not None:
            t_norm, n_norm = decompose(model, probe_x, grad_noise, args.null_threshold_mode, args.null_tol)
            injected_tangent.append(t_norm)
            injected_normal.append(n_norm)
        if optimizer_name == "adam":
            optimizer.zero_grad(set_to_none=True)
            loss = noisy_loss(model, x, y, grad_noise)
            loss.backward()
            optimizer.step()
        else:
            optimizer.step(lambda x=x, y=y, grad_noise=grad_noise: noisy_loss(model, x, y, grad_noise))
        if sigma_theta and step % args.parameter_noise_interval == 0:
            noise = sigma_theta * torch.randn(n_params, generator=generator)
            t_norm, n_norm = decompose(model, probe_x, noise, args.null_threshold_mode, args.null_tol)
            injected_tangent.append(t_norm)
            injected_normal.append(n_norm)
            assign_flat_update(params, noise, scale=1.0)
            post_t, post_n = decompose(model, probe_x, noise, args.null_threshold_mode, args.null_tol)
            post_tangent.append(post_t)
            post_normal.append(post_n)
        phi_errors.append(float(torch.linalg.vector_norm(FunctionalMap(model, probe_x).evaluate().detach() - initial_phi)))
    seconds = time.perf_counter() - start
    loss, accuracy = evaluate(model, eval_set, args.batch_size)
    logs = getattr(optimizer, "topography_log", [])
    gate = statistics.mean(1.0 if row.get("descent_gate_passed", True) else 0.0 for row in logs) if logs else 1.0
    fallback = statistics.mean(1.0 if "fallback" in row.get("mode", "") else 0.0 for row in logs) if logs else 0.0
    recovery_threshold = args.recovery_threshold
    recovery_steps = next((idx for idx, value in enumerate(phi_errors, start=1) if value < recovery_threshold), args.steps)
    mean_injected_tangent = statistics.mean(injected_tangent) if injected_tangent else 0.0
    mean_post_tangent = statistics.mean(post_tangent) if post_tangent else 0.0
    return NoiseRun(
        seed=seed,
        optimizer=optimizer_name,
        sigma_g=sigma_g,
        sigma_theta=sigma_theta,
        final_loss=loss,
        final_accuracy=accuracy,
        functional_output_error=phi_errors[-1] if phi_errors else 0.0,
        recovery_steps=recovery_steps,
        injected_tangent_norm=mean_injected_tangent,
        injected_normal_norm=statistics.mean(injected_normal) if injected_normal else 0.0,
        post_projection_tangent_norm=mean_post_tangent,
        post_projection_normal_norm=statistics.mean(post_normal) if post_normal else 0.0,
        tangent_component_retained=mean_post_tangent / max(mean_injected_tangent, 1e-30),
        gate_accept_rate=gate,
        fallback_rate=fallback,
        wall_clock=seconds,
    )


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--train-samples", type=int, default=192)
    parser.add_argument("--eval-samples", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.5)
    parser.add_argument("--sigma-g-list", type=parse_floats, default=parse_floats("0,1e-4,1e-3,1e-2"))
    parser.add_argument("--sigma-theta-list", type=parse_floats, default=parse_floats("0,1e-4,1e-3,1e-2"))
    parser.add_argument("--parameter-noise-interval", type=int, default=10)
    parser.add_argument("--recovery-threshold", type=float, default=1e-2)
    parser.add_argument("--null-threshold-mode", default="spectral_gap")
    parser.add_argument("--null-tol", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=Path("artifacts/noisy_redundancy_validation.csv"))
    args = parser.parse_args()

    rows = []
    optimizers = ["adam", "diagonal_grad_square", "functional_geoflow"]
    for trial in range(args.trials):
        seed = args.seed + trial
        for sigma_g in args.sigma_g_list:
            for optimizer_name in optimizers:
                rows.append(run_branch(args, optimizer_name, sigma_g, 0.0, seed))
        for sigma_theta in args.sigma_theta_list:
            for optimizer_name in optimizers:
                rows.append(run_branch(args, optimizer_name, 0.0, sigma_theta, seed))
        print(f"finished seed={seed}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    for optimizer_name in optimizers:
        group = [row for row in rows if row.optimizer == optimizer_name]
        print(
            f"{optimizer_name} mean_loss={statistics.mean(row.final_loss for row in group):.4f} "
            f"mean_error={statistics.mean(row.functional_output_error for row in group):.4f} "
            f"mean_retained_T={statistics.mean(row.tangent_component_retained for row in group):.4f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
