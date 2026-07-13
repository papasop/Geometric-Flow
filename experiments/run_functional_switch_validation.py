"""Matched small-MLP validation for functional GeoFlow.

Each seed warms one model with Adam, snapshots the warm-up state, then forks the
same remaining batch sequence into:

* adam_continue
* diagonal_grad_square (legacy heuristic baseline)
* functional_geoflow (projected functional response)
"""

from __future__ import annotations

import argparse
import copy
import csv
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


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RawResult:
    seed: int
    optimizer: str
    final_loss: float
    final_accuracy: float
    loss_gain_vs_adam: float
    accuracy_gain_vs_adam: float
    tangent_drift: float
    functional_output_drift: float
    task_functional_progress: float
    equivalent_branch_divergence: float
    noise_induced_functional_error: float
    tangent_parameter_motion: float
    normal_parameter_motion: float
    gate_accept_rate: float
    fallback_rate: float
    mean_update_norm: float
    wall_clock: float


def make_dataset(samples: int, input_dim: int, classes: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(classes, input_dim, generator=generator)
    y = torch.arange(samples) % classes
    x = prototypes[y] + 0.35 * torch.randn(samples, input_dim, generator=generator)
    return TensorDataset(x, y.long())


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


def train_adam(model: nn.Module, optimizer, batches) -> None:
    for x, y in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()


def train_geometric(model: nn.Module, optimizer: GeometricOptimizer, batches) -> None:
    for x, y in batches:
        optimizer.step(lambda x=x, y=y: F.cross_entropy(model(x), y))


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


def branch_metrics(
    seed: int,
    name: str,
    model: nn.Module,
    optimizer,
    eval_set: TensorDataset,
    probe_x: torch.Tensor,
    probe_y: torch.Tensor,
    warmup_phi: torch.Tensor,
    warmup_probe_loss: float,
    adam_loss: float,
    adam_accuracy: float,
    batch_size: int,
    wall_clock: float,
) -> RawResult:
    loss, accuracy = evaluate(model, eval_set, batch_size)
    fmap = FunctionalMap(model, probe_x)
    final_phi = fmap.evaluate().detach()
    tangent_drift = 0.0
    output_drift = float(torch.linalg.vector_norm(final_phi - warmup_phi))
    probe_loss = float(F.cross_entropy(model(probe_x), probe_y).detach())
    gate_accept_rate = 1.0
    fallback_rate = 0.0
    mean_update_norm = 0.0
    if isinstance(optimizer, GeometricOptimizer) and optimizer.topography_log:
        rows = optimizer.topography_log
        gate_accept_rate = sum(1 for row in rows if row.get("descent_gate_passed")) / len(rows)
        fallback_rate = sum(1 for row in rows if "fallback" in row.get("mode", "")) / len(rows)
        mean_update_norm = statistics.mean(row.get("update_norm", 0.0) for row in rows)
        tangent_rows = [row.get("functional_tangent_norm", 0.0) for row in rows]
        tangent_drift = statistics.mean(tangent_rows) if tangent_rows else 0.0
    tangent_parameter_motion = tangent_drift
    normal_parameter_motion = mean_update_norm
    return RawResult(
        seed=seed,
        optimizer=name,
        final_loss=loss,
        final_accuracy=accuracy,
        loss_gain_vs_adam=adam_loss - loss,
        accuracy_gain_vs_adam=accuracy - adam_accuracy,
        tangent_drift=tangent_drift,
        functional_output_drift=output_drift,
        task_functional_progress=warmup_probe_loss - probe_loss,
        equivalent_branch_divergence=0.0,
        noise_induced_functional_error=0.0,
        tangent_parameter_motion=tangent_parameter_motion,
        normal_parameter_motion=normal_parameter_motion,
        gate_accept_rate=gate_accept_rate,
        fallback_rate=fallback_rate,
        mean_update_norm=mean_update_norm,
        wall_clock=wall_clock,
    )


def run_seed(args, seed: int) -> list[RawResult]:
    torch.manual_seed(seed)
    train_set = make_dataset(args.train_samples, args.input_dim, args.classes, seed)
    eval_set = make_dataset(args.eval_samples, args.input_dim, args.classes, seed + 10_000)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batches = collect_batches(loader, args.steps)
    warmup_steps = min(args.adam_warmup_steps, args.steps)
    remaining = batches[warmup_steps:]
    probe_x = batches[0][0][: args.probe_size].detach().clone()
    probe_y = batches[0][1][: args.probe_size].detach().clone()

    base_model = SmallMLP(args.input_dim, args.hidden_dim, args.classes)
    warmup_optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr)
    start = time.perf_counter()
    train_adam(base_model, warmup_optimizer, batches[:warmup_steps])
    warmup_seconds = time.perf_counter() - start
    model_state = copy.deepcopy(base_model.state_dict())
    adam_state = copy.deepcopy(warmup_optimizer.state_dict())
    warmup_phi = FunctionalMap(base_model, probe_x).evaluate().detach()
    warmup_probe_loss = float(F.cross_entropy(base_model(probe_x), probe_y).detach())

    adam_model = SmallMLP(args.input_dim, args.hidden_dim, args.classes)
    adam_model.load_state_dict(model_state)
    adam_optimizer = torch.optim.Adam(adam_model.parameters(), lr=args.lr)
    adam_optimizer.load_state_dict(adam_state)
    start = time.perf_counter()
    train_adam(adam_model, adam_optimizer, remaining)
    adam_seconds = warmup_seconds + time.perf_counter() - start
    adam_loss, adam_accuracy = evaluate(adam_model, eval_set, args.batch_size)

    diagonal_model = SmallMLP(args.input_dim, args.hidden_dim, args.classes)
    diagonal_model.load_state_dict(model_state)
    diagonal_optimizer = GeometricOptimizer(
        diagonal_model.parameters(),
        lr=args.geo_lr,
        mode="geometric",
        warmup_steps=0,
        preconditioner="diagonal_grad_square",
        grad_smoothing=0.0,
        max_update_norm=args.max_update_norm,
        lr_scale=args.lr_scale,
        damping=args.damping,
        regularization=args.damping,
    )
    start = time.perf_counter()
    train_geometric(diagonal_model, diagonal_optimizer, remaining)
    diagonal_seconds = warmup_seconds + time.perf_counter() - start

    functional_model = SmallMLP(args.input_dim, args.hidden_dim, args.classes)
    functional_model.load_state_dict(model_state)
    functional_optimizer = GeometricOptimizer(
        functional_model.parameters(),
        lr=args.geo_lr,
        mode="functional_geoflow",
        functional_model=functional_model,
        functional_probe=probe_x,
        damping=args.damping,
        regularization=args.damping,
        max_update_norm=args.max_update_norm,
        lr_scale=args.lr_scale,
    )
    start = time.perf_counter()
    train_geometric(functional_model, functional_optimizer, remaining)
    functional_seconds = warmup_seconds + time.perf_counter() - start

    return [
        branch_metrics(
            seed,
            "adam_continue",
            adam_model,
            adam_optimizer,
            eval_set,
            probe_x,
            probe_y,
            warmup_phi,
            warmup_probe_loss,
            adam_loss,
            adam_accuracy,
            args.batch_size,
            adam_seconds,
        ),
        branch_metrics(
            seed,
            "diagonal_grad_square",
            diagonal_model,
            diagonal_optimizer,
            eval_set,
            probe_x,
            probe_y,
            warmup_phi,
            warmup_probe_loss,
            adam_loss,
            adam_accuracy,
            args.batch_size,
            diagonal_seconds,
        ),
        branch_metrics(
            seed,
            "functional_geoflow",
            functional_model,
            functional_optimizer,
            eval_set,
            probe_x,
            probe_y,
            warmup_phi,
            warmup_probe_loss,
            adam_loss,
            adam_accuracy,
            args.batch_size,
            functional_seconds,
        ),
    ]


def bootstrap_ci(values: list[float], samples: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    generator = torch.Generator().manual_seed(12345)
    means = []
    tensor = torch.tensor(values, dtype=torch.float64)
    for _ in range(samples):
        idx = torch.randint(0, len(values), (len(values),), generator=generator)
        means.append(float(tensor[idx].mean()))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def paired_stats(rows: list[RawResult], target: str, baseline: str) -> dict[str, float]:
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row.seed, {})[row.optimizer] = row
    loss_diffs = []
    accuracy_diffs = []
    speed_ratios = []
    for seed_rows in by_seed.values():
        if target not in seed_rows or baseline not in seed_rows:
            continue
        target_row = seed_rows[target]
        baseline_row = seed_rows[baseline]
        loss_diffs.append(baseline_row.final_loss - target_row.final_loss)
        accuracy_diffs.append(target_row.final_accuracy - baseline_row.final_accuracy)
        speed_ratios.append(target_row.wall_clock / max(baseline_row.wall_clock, 1e-30))
    loss_ci = bootstrap_ci(loss_diffs)
    acc_ci = bootstrap_ci(accuracy_diffs)
    return {
        "loss_win_rate": sum(1 for value in loss_diffs if value > 0) / max(len(loss_diffs), 1),
        "accuracy_win_rate": sum(1 for value in accuracy_diffs if value > 0) / max(len(accuracy_diffs), 1),
        "paired_mean_loss_difference": statistics.mean(loss_diffs) if loss_diffs else 0.0,
        "paired_median_loss_difference": statistics.median(loss_diffs) if loss_diffs else 0.0,
        "paired_std_loss_difference": statistics.stdev(loss_diffs) if len(loss_diffs) > 1 else 0.0,
        "loss_sign_test_wins": float(sum(1 for value in loss_diffs if value > 0)),
        "loss_bootstrap_ci_low": loss_ci[0],
        "loss_bootstrap_ci_high": loss_ci[1],
        "paired_mean_accuracy_difference": statistics.mean(accuracy_diffs) if accuracy_diffs else 0.0,
        "accuracy_bootstrap_ci_low": acc_ci[0],
        "accuracy_bootstrap_ci_high": acc_ci[1],
        "speed_ratio": statistics.mean(speed_ratios) if speed_ratios else 0.0,
    }


def print_summary(rows: list[RawResult]) -> None:
    print("\noptimizer  mean_acc  std_acc  mean_loss  mean_progress  mean_time")
    print("---------  --------  -------  ---------  -------------  ---------")
    for optimizer in ["adam_continue", "diagonal_grad_square", "functional_geoflow"]:
        group = [row for row in rows if row.optimizer == optimizer]
        accuracies = [row.final_accuracy for row in group]
        losses = [row.final_loss for row in group]
        times = [row.wall_clock for row in group]
        progress = [row.task_functional_progress for row in group]
        print(
            f"{optimizer:<24} {statistics.mean(accuracies):>8.3f} "
            f"{(statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0):>7.3f} "
            f"{statistics.mean(losses):>9.4f} {statistics.mean(progress):>13.4f} {statistics.mean(times):>9.2f}"
        )
    for baseline in ["adam_continue", "diagonal_grad_square"]:
        stats = paired_stats(rows, "functional_geoflow", baseline)
        label = "adam" if baseline == "adam_continue" else "diagonal"
        print(
            f"functional_vs_{label}: loss_win_rate={stats['loss_win_rate']:.3f} "
            f"accuracy_win_rate={stats['accuracy_win_rate']:.3f} "
            f"mean_loss_diff={stats['paired_mean_loss_difference']:.6f} "
            f"median_loss_diff={stats['paired_median_loss_difference']:.6f} "
            f"std_loss_diff={stats['paired_std_loss_difference']:.6f} "
            f"sign_wins={int(stats['loss_sign_test_wins'])} "
            f"loss_ci=[{stats['loss_bootstrap_ci_low']:.6f},{stats['loss_bootstrap_ci_high']:.6f}] "
            f"speed_ratio={stats['speed_ratio']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["functional_switch_compare"], default="functional_switch_compare")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--adam-warmup-steps", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-size", type=int, default=12)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--geo-lr", type=float, default=1e-3)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", type=Path, default=Path("artifacts/functional_switch_validation.csv"))
    args = parser.parse_args()

    rows = []
    for trial in range(args.trials):
        seed = args.seed + trial
        seed_rows = run_seed(args, seed)
        rows.extend(seed_rows)
        functional = next(row for row in seed_rows if row.optimizer == "functional_geoflow")
        diagonal = next(row for row in seed_rows if row.optimizer == "diagonal_grad_square")
        print(
            f"seed={seed} functional_acc={functional.final_accuracy:.3f} "
            f"functional_gain={functional.accuracy_gain_vs_adam:+.3f} "
            f"diagonal_gain={diagonal.accuracy_gain_vs_adam:+.3f} "
            f"gate={functional.gate_accept_rate:.3f} fallback={functional.fallback_rate:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print_summary(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
