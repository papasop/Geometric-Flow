"""Reparameterization-invariance stress test for functional GeoFlow."""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import FunctionalMap, GeometricOptimizer
from geometric_flow._tensor import get_flat_params


class RedundantLinearNet(nn.Module):
    """Two-layer linear network with hidden-basis reparameterization symmetry."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x))


@dataclass
class StressRun:
    seed: int
    representation: str
    optimizer: str
    initial_functional_residual: float
    parameter_distance: float
    final_loss: float
    final_accuracy: float
    final_phi: str
    functional_output_drift: float
    task_functional_progress: float
    equivalent_branch_divergence: float
    noise_induced_functional_error: float
    tangent_parameter_motion: float
    normal_parameter_motion: float
    tangent_drift: float
    normal_update_norm: float
    gate_accept_rate: float
    fallback_rate: float
    wall_clock: float


def make_classification(samples: int, input_dim: int, output_dim: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(output_dim, input_dim, generator=generator)
    y = torch.arange(samples) % output_dim
    x = prototypes[y] + 0.2 * torch.randn(samples, input_dim, generator=generator)
    return TensorDataset(x, y.long())


def init_base_model(input_dim: int, hidden_dim: int, output_dim: int, seed: int) -> RedundantLinearNet:
    torch.manual_seed(seed)
    model = RedundantLinearNet(input_dim, hidden_dim, output_dim)
    with torch.no_grad():
        model.w1.weight.mul_(0.25)
        model.w2.weight.mul_(0.25)
    return model


def reparameterize_model(base: RedundantLinearNet, transform: torch.Tensor) -> RedundantLinearNet:
    model = RedundantLinearNet(base.w1.weight.shape[1], base.w1.weight.shape[0], base.w2.weight.shape[0])
    inv = torch.linalg.inv(transform)
    with torch.no_grad():
        model.w1.weight.copy_(transform @ base.w1.weight)
        model.w2.weight.copy_(base.w2.weight @ inv)
    return model


def diagonal_scaling(hidden_dim: int, scale: float = 0.7) -> torch.Tensor:
    values = torch.linspace(-scale, scale, hidden_dim)
    return torch.diag(torch.exp(values))


def orthogonal_rotation(hidden_dim: int, epsilon: float = 0.35) -> torch.Tensor:
    generator = torch.zeros(hidden_dim, hidden_dim)
    if hidden_dim > 1:
        for idx in range(hidden_dim - 1):
            generator[idx, idx + 1] = 1.0
            generator[idx + 1, idx] = -1.0
    return torch.matrix_exp(epsilon * generator)


def model_representations(base: RedundantLinearNet, count: int = 3) -> dict[str, RedundantLinearNet]:
    hidden_dim = base.w1.weight.shape[0]
    reps = {
        "identity": reparameterize_model(base, torch.eye(hidden_dim)),
        "diagonal_scaling": reparameterize_model(base, diagonal_scaling(hidden_dim)),
        "orthogonal_rotation": reparameterize_model(base, orthogonal_rotation(hidden_dim)),
    }
    for idx in range(max(0, count - len(reps))):
        scale = 0.25 + 0.1 * idx
        transform = orthogonal_rotation(hidden_dim, epsilon=scale) @ diagonal_scaling(hidden_dim, scale=0.3 + 0.05 * idx)
        reps[f"mixed_{idx + 1}"] = reparameterize_model(base, transform)
    return dict(list(reps.items())[:count])


def collect_batches(loader: DataLoader, steps: int):
    iterator = iter(loader)
    batches = []
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batches.append(batch)
    return batches


def evaluate(model: nn.Module, dataset: TensorDataset, batch_size: int) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            losses.append(F.cross_entropy(logits, y).item())
            correct += int((logits.argmax(dim=1) == y).sum())
            total += int(y.numel())
    model.train()
    return sum(losses) / max(len(losses), 1), correct / max(total, 1)


def train_branch(
    model: RedundantLinearNet,
    optimizer_name: str,
    batches,
    probe_x: torch.Tensor,
    lr: float,
    max_update_norm: float,
    null_threshold_mode: str,
    null_tol: float,
) -> tuple[object, float, list[torch.Tensor]]:
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "diagonal_grad_square":
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=lr,
            lr_scale=1.0,
            mode="geometric",
            warmup_steps=0,
            preconditioner="diagonal_grad_square",
            max_update_norm=max_update_norm,
            grad_smoothing=0.0,
        )
    elif optimizer_name == "functional_geoflow":
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=lr,
            lr_scale=1.0,
            mode="functional_geoflow",
            functional_model=model,
            functional_probe=probe_x,
            max_update_norm=max_update_norm,
            null_threshold_mode=null_threshold_mode,
            null_tol=null_tol,
        )
    else:
        raise ValueError(optimizer_name)

    trajectory = [FunctionalMap(model, probe_x).evaluate().detach()]
    start = time.perf_counter()
    for x, y in batches:
        if optimizer_name == "adam":
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        else:
            optimizer.step(lambda x=x, y=y: F.cross_entropy(model(x), y))
        trajectory.append(FunctionalMap(model, probe_x).evaluate().detach())
    return optimizer, time.perf_counter() - start, trajectory


def pairwise_mean(vectors: list[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 0.0
    distances = [float(torch.linalg.vector_norm(a - b)) for a, b in itertools.combinations(vectors, 2)]
    return statistics.mean(distances)


def run_seed(args, seed: int) -> tuple[list[StressRun], list[dict[str, float | str]]]:
    torch.manual_seed(seed)
    train_set = make_classification(args.train_samples, args.input_dim, args.output_dim, seed)
    eval_set = make_classification(args.eval_samples, args.input_dim, args.output_dim, seed + 2000)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batches = collect_batches(loader, args.steps)
    probe_x = train_set.tensors[0][: args.probe_size].detach().clone()
    probe_y = train_set.tensors[1][: args.probe_size].detach().clone()
    base = init_base_model(args.input_dim, args.hidden_dim, args.output_dim, seed)
    base_phi = FunctionalMap(base, probe_x).evaluate().detach()
    base_theta = get_flat_params(list(base.parameters()))
    base_probe_loss = float(F.cross_entropy(base(probe_x), probe_y).detach())
    rows = []

    for rep_name, rep_model in model_representations(base, count=args.representations).items():
        initial_phi = FunctionalMap(rep_model, probe_x).evaluate().detach()
        initial_residual = float(torch.linalg.vector_norm(initial_phi - base_phi))
        parameter_distance = float(torch.linalg.vector_norm(get_flat_params(list(rep_model.parameters())) - base_theta))
        for optimizer_name in args.optimizers:
            model = reparameterize_model(rep_model, torch.eye(args.hidden_dim))
            optimizer, seconds, trajectory = train_branch(
                model,
                optimizer_name,
                batches,
                probe_x,
                args.lr,
                args.max_update_norm,
                args.null_threshold_mode,
                args.null_tol,
            )
            loss, accuracy = evaluate(model, eval_set, args.batch_size)
            final_phi = trajectory[-1]
            logs = getattr(optimizer, "topography_log", [])
            tangent_drift = statistics.mean(row.get("functional_tangent_norm", 0.0) for row in logs) if logs else 0.0
            normal_update_norm = statistics.mean(row.get("update_norm", 0.0) for row in logs) if logs else 0.0
            gate = statistics.mean(1.0 if row.get("descent_gate_passed", True) else 0.0 for row in logs) if logs else 1.0
            fallback = statistics.mean(1.0 if "fallback" in row.get("mode", "") else 0.0 for row in logs) if logs else 0.0
            final_probe_loss = float(F.cross_entropy(model(probe_x), probe_y).detach())
            rows.append(
                StressRun(
                    seed=seed,
                    representation=rep_name,
                    optimizer=optimizer_name,
                    initial_functional_residual=initial_residual,
                    parameter_distance=parameter_distance,
                    final_loss=loss,
                    final_accuracy=accuracy,
                    final_phi="|".join(f"{float(value):.9g}" for value in final_phi.reshape(-1)),
                    functional_output_drift=float(torch.linalg.vector_norm(final_phi - initial_phi)),
                    task_functional_progress=base_probe_loss - final_probe_loss,
                    equivalent_branch_divergence=0.0,
                    noise_induced_functional_error=0.0,
                    tangent_parameter_motion=tangent_drift,
                    normal_parameter_motion=normal_update_norm,
                    tangent_drift=tangent_drift,
                    normal_update_norm=normal_update_norm,
                    gate_accept_rate=gate,
                    fallback_rate=fallback,
                    wall_clock=seconds,
                )
            )
    return rows, []


def run(args) -> tuple[list[StressRun], list[dict[str, float | str]]]:
    all_rows = []
    for trial in range(args.trials):
        seed_rows, _ = run_seed(args, args.seed + trial)
        all_rows.extend(seed_rows)
        print(f"finished seed={args.seed + trial}")
    aggregates = aggregate_rows(all_rows, args.optimizers)
    return all_rows, aggregates


def aggregate_rows(rows: list[StressRun], optimizers: list[str]) -> list[dict[str, float | str]]:
    aggregates = []
    for optimizer_name in optimizers:
        group = [row for row in rows if row.optimizer == optimizer_name]
        loss_vs_adam = []
        acc_vs_adam = []
        loss_vs_diag = []
        acc_vs_diag = []
        speed_vs_diag = []
        for row in group:
            adam_row = next((candidate for candidate in rows if candidate.seed == row.seed and candidate.representation == row.representation and candidate.optimizer == "adam"), None)
            diag_row = next((candidate for candidate in rows if candidate.seed == row.seed and candidate.representation == row.representation and candidate.optimizer == "diagonal_grad_square"), None)
            if adam_row is not None:
                loss_vs_adam.append(adam_row.final_loss - row.final_loss)
                acc_vs_adam.append(row.final_accuracy - adam_row.final_accuracy)
            if diag_row is not None:
                loss_vs_diag.append(diag_row.final_loss - row.final_loss)
                acc_vs_diag.append(row.final_accuracy - diag_row.final_accuracy)
                speed_vs_diag.append(row.wall_clock / max(diag_row.wall_clock, 1e-30))
        by_seed = {}
        for row in group:
            by_seed.setdefault(row.seed, []).append(row)
        sensitivity = statistics.mean(
            pairwise_mean([torch.tensor([float(value) for value in row.final_phi.split("|")]) for row in seed_rows])
            for seed_rows in by_seed.values()
        )
        aggregates.append(
            {
                "optimizer": optimizer_name,
                "reparameterization_sensitivity": sensitivity,
                "final_loss_dispersion": statistics.pstdev([row.final_loss for row in group]) if group else 0.0,
                "final_accuracy_dispersion": statistics.pstdev([row.final_accuracy for row in group]) if group else 0.0,
                "loss_win_rate_vs_adam": sum(1 for value in loss_vs_adam if value > 0) / max(len(loss_vs_adam), 1),
                "accuracy_win_rate_vs_adam": sum(1 for value in acc_vs_adam if value > 0) / max(len(acc_vs_adam), 1),
                "loss_win_rate_vs_diagonal": sum(1 for value in loss_vs_diag if value > 0) / max(len(loss_vs_diag), 1),
                "accuracy_win_rate_vs_diagonal": sum(1 for value in acc_vs_diag if value > 0) / max(len(acc_vs_diag), 1),
                "mean_wall_clock": statistics.mean(row.wall_clock for row in group) if group else 0.0,
                "speed_ratio_vs_diagonal": statistics.mean(speed_vs_diag) if speed_vs_diag else 0.0,
                "mean_tangent_drift": statistics.mean(row.tangent_drift for row in group) if group else 0.0,
            }
        )
    return aggregates


def parse_optimizers(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--representations", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=192)
    parser.add_argument("--eval-samples", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=6)
    parser.add_argument("--output-dim", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.5)
    parser.add_argument("--null-threshold-mode", default="spectral_gap")
    parser.add_argument("--null-tol", type=float, default=1e-6)
    parser.add_argument("--optimizers", type=parse_optimizers, default=parse_optimizers("adam,diagonal_grad_square,functional_geoflow"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/reparameterization_stress_runs.csv"))
    parser.add_argument("--aggregate-out", type=Path, default=Path("artifacts/reparameterization_stress_aggregate.csv"))
    args = parser.parse_args()

    rows, aggregates = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with args.aggregate_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)
    for row in aggregates:
        print(
            f"{row['optimizer']} sensitivity={row['reparameterization_sensitivity']:.6g} "
            f"loss_dispersion={row['final_loss_dispersion']:.6g} "
            f"acc_dispersion={row['final_accuracy_dispersion']:.6g} time={row['mean_wall_clock']:.3f}"
        )
    print(f"wrote {args.out}")
    print(f"wrote {args.aggregate_out}")


if __name__ == "__main__":
    main()
