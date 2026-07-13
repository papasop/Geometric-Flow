"""Train GeoCNN on CIFAR-10 or a synthetic CIFAR-shaped task."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import GeoCNN, GeometricOptimizer


@dataclass
class TrainResult:
    optimizer: str
    final_loss: float
    final_accuracy: float
    train_loss: float
    train_accuracy: float
    generalization_loss_gap: float
    generalization_accuracy_gap: float
    train_seconds: float
    steps: int
    avg_preconditioned_to_raw_ratio: float


@dataclass
class SummaryResult:
    optimizer: str
    trials: int
    mean_accuracy: float
    std_accuracy: float
    mean_loss: float
    std_loss: float
    mean_generalization_loss_gap: float
    std_generalization_loss_gap: float
    mean_generalization_accuracy_gap: float
    std_generalization_accuracy_gap: float
    mean_seconds: float
    std_seconds: float
    mean_preconditioned_to_raw_ratio: float
    steps: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_cifar(train_samples: int, eval_samples: int, seed: int) -> Tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(10, 3, 32, 32, generator=generator)

    def build(n: int) -> TensorDataset:
        y = torch.arange(n) % 10
        x = prototypes[y] + 0.35 * torch.randn(n, 3, 32, 32, generator=generator)
        return TensorDataset(x, y.long())

    return build(train_samples), build(eval_samples)


def load_cifar10(data_root: str, train_samples: int, eval_samples: int, download: bool) -> Tuple[TensorDataset, TensorDataset]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for --dataset cifar10") from exc

    transform = transforms.ToTensor()
    train = datasets.CIFAR10(data_root, train=True, download=download, transform=transform)
    test = datasets.CIFAR10(data_root, train=False, download=download, transform=transform)

    def subset(dataset, n: int) -> TensorDataset:
        xs, ys = [], []
        for idx in range(min(n, len(dataset))):
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)
        return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    return subset(train, train_samples), subset(test, eval_samples)


def make_loaders(args) -> Tuple[DataLoader, DataLoader]:
    if args.dataset == "cifar10":
        train_set, eval_set = load_cifar10(args.data_root, args.train_samples, args.eval_samples, args.download)
    else:
        train_set, eval_set = make_synthetic_cifar(args.train_samples, args.eval_samples, args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, eval_loader


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            losses.append(F.cross_entropy(logits, y).item())
            correct += int((logits.argmax(dim=1) == y).sum())
            total += int(y.numel())
    model.train()
    return sum(losses) / max(len(losses), 1), correct / max(total, 1)


def train_one(args, optimizer_name: str, train_loader: DataLoader, eval_loader: DataLoader) -> TrainResult:
    set_seed(args.seed)
    device = torch.device(args.device)
    model = GeoCNN(channels=args.channels, conv_layers=args.conv_layers).to(device)
    optimizer_mode, adam_warmup_steps = resolve_optimizer_config(optimizer_name, args.adam_warmup_steps)
    if optimizer_mode == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=args.geo_lr or args.lr,
            damping=args.damping,
            lr_scale=args.lr_scale,
            curvature_reuse=args.curvature_reuse,
            warmup_steps=args.warmup_steps,
            regularization=args.regularization,
            max_update_norm=args.max_update_norm,
            max_grad_norm=args.max_grad_norm,
            grad_smoothing=args.grad_smoothing,
            preconditioner_scale=args.precond_scale,
            curvature_scale=args.curvature_scale,
            curvature_kind="fisher" if args.use_fisher else "hessian",
            preconditioner=args.preconditioner,
            mode=optimizer_mode,
            adam_warmup_steps=adam_warmup_steps,
            cg_max_iter=args.cg_max_iter,
            trace_samples=args.trace_samples,
            diagnostic_log_path=args.diagnostic_log,
        )

    iterator = iter(train_loader)
    start = time.perf_counter()
    for _ in range(args.steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x = x.to(device)
        y = y.to(device)
        if optimizer_mode == "adam":
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        else:
            optimizer.step(lambda: F.cross_entropy(model(x), y), verbose=args.verbose)

    seconds = time.perf_counter() - start
    train_loss, train_accuracy = evaluate(model, train_loader, device)
    loss, accuracy = evaluate(model, eval_loader, device)
    ratios = []
    if isinstance(optimizer, GeometricOptimizer):
        geometric_modes = {"geometric", "geometric_reuse", "diagonal"}
        ratios = [
            row["preconditioned_to_raw_ratio"]
            for row in optimizer.topography_log
            if row["raw_grad_norm"] > 0 and row["mode"] in geometric_modes
        ]
    return TrainResult(
        optimizer=optimizer_name,
        final_loss=loss,
        final_accuracy=accuracy,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        generalization_loss_gap=loss - train_loss,
        generalization_accuracy_gap=train_accuracy - accuracy,
        train_seconds=seconds,
        steps=args.steps,
        avg_preconditioned_to_raw_ratio=sum(ratios) / max(len(ratios), 1),
    )


def parse_warmup_label(label: str):
    if label.startswith("hybrid_"):
        return int(label.rsplit("_", 1)[1])
    return None


def resolve_optimizer_config(optimizer_name: str, default_adam_warmup_steps: int):
    warmup = parse_warmup_label(optimizer_name)
    if warmup is not None:
        return "hybrid", warmup
    return optimizer_name, default_adam_warmup_steps


def parse_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def optimizer_names(mode: str) -> List[str]:
    if mode == "all":
        return ["adam", "geometric", "hybrid"]
    return [mode]


def experiment_names(args) -> List[str]:
    names = optimizer_names(args.mode)
    if not args.auto_warmup:
        return names
    expanded = []
    for name in names:
        if name == "hybrid":
            expanded.extend(f"hybrid_{steps}" for steps in args.auto_warmup_steps)
        else:
            expanded.append(name)
    return expanded


def summarize(name: str, results: List[TrainResult]) -> SummaryResult:
    accuracies = [row.final_accuracy for row in results]
    losses = [row.final_loss for row in results]
    loss_gaps = [row.generalization_loss_gap for row in results]
    accuracy_gaps = [row.generalization_accuracy_gap for row in results]
    seconds = [row.train_seconds for row in results]
    ratios = [row.avg_preconditioned_to_raw_ratio for row in results]
    return SummaryResult(
        optimizer=name,
        trials=len(results),
        mean_accuracy=mean(accuracies),
        std_accuracy=stdev(accuracies) if len(accuracies) > 1 else 0.0,
        mean_loss=mean(losses),
        std_loss=stdev(losses) if len(losses) > 1 else 0.0,
        mean_generalization_loss_gap=mean(loss_gaps),
        std_generalization_loss_gap=stdev(loss_gaps) if len(loss_gaps) > 1 else 0.0,
        mean_generalization_accuracy_gap=mean(accuracy_gaps),
        std_generalization_accuracy_gap=stdev(accuracy_gaps) if len(accuracy_gaps) > 1 else 0.0,
        mean_seconds=mean(seconds),
        std_seconds=stdev(seconds) if len(seconds) > 1 else 0.0,
        mean_preconditioned_to_raw_ratio=mean(ratios),
        steps=results[0].steps if results else 0,
    )


def run_trials(args) -> List[SummaryResult]:
    summaries = []
    for name in experiment_names(args):
        trial_results = []
        for trial in range(args.trials):
            trial_args = argparse.Namespace(**vars(args))
            trial_args.seed = args.seed + trial
            set_seed(trial_args.seed)
            train_loader, eval_loader = make_loaders(trial_args)
            result = train_one(trial_args, name, train_loader, eval_loader)
            trial_results.append(result)
            print(
                f"{name} trial={trial + 1}/{args.trials}: loss={result.final_loss:.4f} "
                f"acc={result.final_accuracy:.3f} seconds={result.train_seconds:.2f} "
                f"ratio={result.avg_preconditioned_to_raw_ratio:.3f}"
            )
        summaries.append(summarize(name, trial_results))
    return summaries


def print_comparison_table(rows: List[SummaryResult]) -> None:
    print("\noptimizer  mean_acc  std_acc  mean_loss  gen_gap  mean_sec  ratio")
    print("---------  --------  -------  ---------  -------  --------  -----")
    for row in rows:
        print(
            f"{row.optimizer:<9}  {row.mean_accuracy:>8.3f}  {row.std_accuracy:>7.3f}  "
            f"{row.mean_loss:>9.4f}  {row.mean_generalization_loss_gap:>7.4f}  {row.mean_seconds:>8.2f}  "
            f"{row.mean_preconditioned_to_raw_ratio:>5.3f}"
        )


def print_best_auto_warmup(rows: List[SummaryResult]) -> None:
    hybrids = [row for row in rows if row.optimizer.startswith("hybrid_")]
    if not hybrids:
        return
    best = max(hybrids, key=lambda row: row.mean_accuracy)
    print(
        f"\nbest_auto_warmup={best.optimizer} mean_acc={best.mean_accuracy:.3f} "
        f"mean_loss={best.mean_loss:.4f} ratio={best.mean_preconditioned_to_raw_ratio:.3f} "
        f"gen_gap={best.mean_generalization_loss_gap:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "cifar10"], default="synthetic")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--conv-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--geo-lr", type=float, default=None)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--lr-scale", type=float, default=3.0)
    parser.add_argument("--curvature-reuse", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-update-norm", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--grad-smoothing", type=float, default=0.0)
    parser.add_argument("--precond-scale", type=float, default=0.5)
    parser.add_argument("--curvature-scale", type=float, default=1.0)
    parser.add_argument("--use-fisher", action="store_true")
    parser.add_argument("--preconditioner", choices=["cg", "diagonal"], default="cg")
    parser.add_argument("--mode", choices=["adam", "geometric", "hybrid", "all"], default="all")
    parser.add_argument("--adam-warmup-steps", type=int, default=30)
    parser.add_argument("--auto-warmup", action="store_true")
    parser.add_argument("--auto-warmup-steps", type=parse_ints, default=parse_ints("30,50,80"))
    parser.add_argument("--cg-max-iter", type=int, default=8)
    parser.add_argument("--trace-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="artifacts/cifar10_geo_baseline.csv")
    parser.add_argument("--diagnostic-log", default="artifacts/cifar10_geo_diagnostics.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.conv_layers < 1:
        raise ValueError("--conv-layers must be >= 1")
    if args.auto_warmup and not args.auto_warmup_steps:
        raise ValueError("--auto-warmup-steps must not be empty")

    rows = run_trials(args)
    print_comparison_table(rows)
    if args.auto_warmup:
        print_best_auto_warmup(rows)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
