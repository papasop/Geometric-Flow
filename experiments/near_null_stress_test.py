"""Near-null weak-breaking stress test for functional GeoFlow."""

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

from geometric_flow import FunctionalMap, GeometricOptimizer, functional_projectors, functional_response_operator
from geometric_flow._tensor import assign_flat_update, flatten_grads
from experiments.reparameterization_stress_test import (
    RedundantLinearNet,
    collect_batches,
    evaluate,
    init_base_model,
    make_classification,
    reparameterize_model,
)


@dataclass
class NearNullRun:
    seed: int
    epsilon: float
    optimizer: str
    null_threshold_mode: str
    null_tol: float
    smallest_singular_value: float
    spectral_gap: float
    inferred_tangent_rank: int
    normal_rank: int
    condition_number: float
    tangent_amplification: float
    mean_update_norm: float
    final_loss: float
    final_accuracy: float
    output_drift: float
    wall_clock: float


def augmented_jacobian(model: RedundantLinearNet, probe_x: torch.Tensor, epsilon: float) -> torch.Tensor:
    jac = FunctionalMap(model, probe_x).jacobian().jacobian
    if epsilon <= 0:
        return jac
    identity = torch.eye(jac.shape[1], dtype=jac.dtype, device=jac.device)
    return torch.cat([jac, epsilon * identity], dim=0)


def manual_functional_step(
    model: RedundantLinearNet,
    x: torch.Tensor,
    y: torch.Tensor,
    probe_x: torch.Tensor,
    epsilon: float,
    lr: float,
    damping: float,
    max_update_norm: float,
    null_threshold_mode: str,
    null_tol: float,
) -> tuple[float, float, float, float, int, int, float]:
    params = [param for param in model.parameters() if param.requires_grad]
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params)
    grad = flatten_grads(grads, params).detach()
    jac = augmented_jacobian(model, probe_x, epsilon).to(grad)
    projectors = functional_projectors(jac, null_threshold_mode=null_threshold_mode, null_tol=null_tol)
    p_normal = projectors.normal.to(grad)
    p_tangent = projectors.tangent.to(grad)
    response = functional_response_operator(jac)
    projected_response = p_normal @ response @ p_normal + damping * p_normal
    direction = torch.linalg.pinv(projected_response) @ (-(p_normal @ grad))
    direction = p_normal @ direction
    norm = torch.linalg.vector_norm(direction)
    if float(norm) > max_update_norm:
        direction = direction * (max_update_norm / norm.clamp_min(1e-30))
    g_dot_d = float(torch.dot(grad, direction))
    if g_dot_d >= 0:
        direction = -(p_normal @ grad)
        norm = torch.linalg.vector_norm(direction)
        if float(norm) > max_update_norm:
            direction = direction * (max_update_norm / norm.clamp_min(1e-30))
    tangent_update = float(torch.linalg.vector_norm(p_tangent @ direction))
    normal_update = float(torch.linalg.vector_norm(p_normal @ direction))
    assign_flat_update(params, direction, scale=lr)
    singular_values = projectors.singular_values
    if singular_values.numel() > 1:
        gaps = singular_values[:-1] / singular_values[1:].clamp_min(torch.finfo(singular_values.dtype).tiny)
        spectral_gap = float(gaps.max())
    else:
        spectral_gap = 0.0
    return (
        tangent_update,
        normal_update,
        float(torch.linalg.vector_norm(direction * lr)),
        float(singular_values[-1]) if singular_values.numel() else 0.0,
        projectors.tangent_rank,
        projectors.normal_rank,
        projectors.condition_number_normal,
    )


def run_branch(args, epsilon: float, optimizer_name: str, mode: str, null_tol: float) -> NearNullRun:
    torch.manual_seed(args.seed)
    train_set = make_classification(args.train_samples, args.input_dim, args.output_dim, args.seed)
    eval_set = make_classification(args.eval_samples, args.input_dim, args.output_dim, args.seed + 5000)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    batches = collect_batches(loader, args.steps)
    probe_x = train_set.tensors[0][: args.probe_size].detach().clone()
    base = init_base_model(args.input_dim, args.hidden_dim, args.output_dim, args.seed)
    model = reparameterize_model(base, torch.eye(args.hidden_dim))
    initial_phi = FunctionalMap(model, probe_x).evaluate().detach()
    params = [param for param in model.parameters() if param.requires_grad]

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
    else:
        optimizer = None

    tangent_updates = []
    normal_updates = []
    update_norms = []
    smallest_sv = 0.0
    tangent_rank = 0
    normal_rank = 0
    condition = 0.0
    start = time.perf_counter()
    for x, y in batches:
        if optimizer_name == "adam":
            before = torch.cat([param.detach().reshape(-1) for param in params])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            update = torch.cat([param.detach().reshape(-1) for param in params]) - before
            projectors = functional_projectors(augmented_jacobian(model, probe_x, epsilon), null_threshold_mode=mode, null_tol=null_tol)
            tangent_updates.append(float(torch.linalg.vector_norm(projectors.tangent @ update)))
            normal_updates.append(float(torch.linalg.vector_norm(projectors.normal @ update)))
            update_norms.append(float(torch.linalg.vector_norm(update)))
            smallest_sv = float(projectors.singular_values[-1])
            tangent_rank = projectors.tangent_rank
            normal_rank = projectors.normal_rank
            condition = projectors.condition_number_normal
        elif optimizer_name == "diagonal_grad_square":
            before = torch.cat([param.detach().reshape(-1) for param in params])
            optimizer.step(lambda x=x, y=y: F.cross_entropy(model(x), y))
            update = torch.cat([param.detach().reshape(-1) for param in params]) - before
            projectors = functional_projectors(augmented_jacobian(model, probe_x, epsilon), null_threshold_mode=mode, null_tol=null_tol)
            tangent_updates.append(float(torch.linalg.vector_norm(projectors.tangent @ update)))
            normal_updates.append(float(torch.linalg.vector_norm(projectors.normal @ update)))
            update_norms.append(float(torch.linalg.vector_norm(update)))
            smallest_sv = float(projectors.singular_values[-1])
            tangent_rank = projectors.tangent_rank
            normal_rank = projectors.normal_rank
            condition = projectors.condition_number_normal
        else:
            t_update, n_update, update_norm, smallest_sv, tangent_rank, normal_rank, condition = manual_functional_step(
                model,
                x,
                y,
                probe_x,
                epsilon,
                args.lr,
                args.damping,
                args.max_update_norm,
                mode,
                null_tol,
            )
            tangent_updates.append(t_update)
            normal_updates.append(n_update)
            update_norms.append(update_norm)
    seconds = time.perf_counter() - start
    loss, accuracy = evaluate(model, eval_set, args.batch_size)
    final_phi = FunctionalMap(model, probe_x).evaluate().detach()
    mean_tangent = statistics.mean(tangent_updates) if tangent_updates else 0.0
    mean_normal = statistics.mean(normal_updates) if normal_updates else 0.0
    jac = augmented_jacobian(model, probe_x, epsilon)
    singular_values = torch.linalg.svdvals(jac)
    spectral_gap = 0.0
    if singular_values.numel() > 1:
        spectral_gap = float((singular_values[:-1] / singular_values[1:].clamp_min(torch.finfo(singular_values.dtype).tiny)).max())
    return NearNullRun(
        seed=args.seed,
        epsilon=epsilon,
        optimizer=optimizer_name,
        null_threshold_mode=mode,
        null_tol=null_tol,
        smallest_singular_value=float(singular_values[-1]) if singular_values.numel() else smallest_sv,
        spectral_gap=spectral_gap,
        inferred_tangent_rank=tangent_rank,
        normal_rank=normal_rank,
        condition_number=condition,
        tangent_amplification=mean_tangent / max(mean_normal, 1e-30),
        mean_update_norm=statistics.mean(update_norms) if update_norms else 0.0,
        final_loss=loss,
        final_accuracy=accuracy,
        output_drift=float(torch.linalg.vector_norm(final_phi - initial_phi)),
        wall_clock=seconds,
    )


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--train-samples", type=int, default=192)
    parser.add_argument("--eval-samples", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.5)
    parser.add_argument("--epsilons", type=parse_floats, default=parse_floats("0,1e-5,1e-4,1e-3,1e-2,1e-1"))
    parser.add_argument("--threshold-modes", default="spectral_gap")
    parser.add_argument("--null-tols", type=parse_floats, default=parse_floats("1e-6"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/near_null_stress_test.csv"))
    args = parser.parse_args()

    modes = [part.strip() for part in args.threshold_modes.split(",") if part.strip()]
    rows = []
    for trial in range(args.trials):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.seed = args.seed + trial
        for epsilon in args.epsilons:
            for mode in modes:
                for null_tol in args.null_tols:
                    for optimizer_name in ["adam", "diagonal_grad_square", "functional_geoflow"]:
                        rows.append(run_branch(trial_args, epsilon, optimizer_name, mode, null_tol))
            print(f"finished seed={trial_args.seed} epsilon={epsilon}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    for optimizer_name in ["adam", "diagonal_grad_square", "functional_geoflow"]:
        group = [row for row in rows if row.optimizer == optimizer_name]
        print(
            f"{optimizer_name} mean_tangent_amp={statistics.mean(row.tangent_amplification for row in group):.6g} "
            f"mean_loss={statistics.mean(row.final_loss for row in group):.4f} "
            f"mean_condition={statistics.mean(row.condition_number for row in group):.4g}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
