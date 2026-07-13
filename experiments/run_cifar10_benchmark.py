"""Run a full CIFAR-10 benchmark for Adam, geometric, and hybrid modes."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cifar10_configs import config_names, get_config
from train_cifar10_geo import make_loaders, print_comparison_table, summarize, train_one


def parse_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_floats(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def benchmark_configs(warmup_steps: List[int]) -> List[Tuple[str, int]]:
    configs = [("adam", 0), ("geometric", 0)]
    configs.extend((f"hybrid_{steps}", steps) for steps in warmup_steps)
    return configs


def run_config(args, label: str, mode: str, adam_warmup_steps: int):
    trial_results = []
    for trial in range(args.trials):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.mode = mode
        trial_args.seed = args.seed + trial
        trial_args.adam_warmup_steps = adam_warmup_steps
        trial_args.dataset = "cifar10"
        train_loader, eval_loader = make_loaders(trial_args)
        result = train_one(trial_args, mode, train_loader, eval_loader)
        trial_results.append(result)
        print(
            f"{label} trial={trial + 1}/{args.trials}: loss={result.final_loss:.4f} "
            f"acc={result.final_accuracy:.3f} seconds={result.train_seconds:.2f} "
            f"ratio={result.avg_preconditioned_to_raw_ratio:.3f}"
        )
    return summarize(label, trial_results)


def apply_recommended_config(args) -> None:
    if args.config is None:
        return
    config = get_config(args.config)
    mapping = {
        "hybrid_warmup_steps": "hybrid_warmup_steps",
    }
    for key, value in config.items():
        if key == "description":
            continue
        setattr(args, mapping.get(key, key), value)


def sensitivity_values(args):
    precond_scales = args.precond_scales or [args.precond_scale]
    grad_smoothing_values = args.grad_smoothing_values or [args.grad_smoothing]
    values = []
    multi = len(precond_scales) * len(grad_smoothing_values) > 1
    for precond_scale in precond_scales:
        for grad_smoothing in grad_smoothing_values:
            suffix = f"_p{precond_scale:g}_gs{grad_smoothing:g}" if multi else ""
            values.append((suffix, precond_scale, grad_smoothing))
    return values


def print_sensitivity_report(rows) -> None:
    if not rows:
        return
    print("\nparameter_sensitivity")
    print("optimizer                 mean_acc  mean_loss  seconds  ratio")
    print("------------------------  --------  ---------  -------  -----")
    for row in sorted(rows, key=lambda item: item.mean_accuracy, reverse=True):
        print(
            f"{row.optimizer:<24}  {row.mean_accuracy:>8.3f}  {row.mean_loss:>9.4f}  "
            f"{row.mean_seconds:>7.2f}  {row.mean_preconditioned_to_raw_ratio:>5.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=config_names(), default=None)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--hybrid-warmup-steps", type=parse_ints, default=parse_ints("10,30,50,80"))
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--conv-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--eval-samples", type=int, default=10000)
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
    parser.add_argument("--grad-smoothing-values", type=parse_floats, default=None)
    parser.add_argument("--precond-scale", type=float, default=0.5)
    parser.add_argument("--precond-scales", type=parse_floats, default=None)
    parser.add_argument("--curvature-scale", type=float, default=1.0)
    parser.add_argument("--use-grad-square", "--use-fisher", action="store_true", default=True, dest="use_fisher")
    parser.add_argument("--no-grad-square", "--no-fisher", action="store_false", dest="use_fisher")
    parser.add_argument("--preconditioner", choices=["cg", "diagonal"], default="diagonal")
    parser.add_argument("--cg-max-iter", type=int, default=8)
    parser.add_argument("--trace-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diagnostic-log", default="artifacts/cifar10_benchmark_diagnostics.csv")
    parser.add_argument("--out", default="artifacts/cifar10_benchmark.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    apply_recommended_config(args)
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.conv_layers < 1:
        raise ValueError("--conv-layers must be >= 1")

    rows = []
    for suffix, precond_scale, grad_smoothing in sensitivity_values(args):
        args.precond_scale = precond_scale
        args.grad_smoothing = grad_smoothing
        for label, warmup in benchmark_configs(args.hybrid_warmup_steps):
            mode = "hybrid" if label.startswith("hybrid_") else label
            rows.append(run_config(args, f"{label}{suffix}", mode, warmup))

    print_comparison_table(rows)
    if args.precond_scales or args.grad_smoothing_values:
        print_sensitivity_report(rows)
    best = max(rows, key=lambda row: row.mean_accuracy)
    adam = next((row for row in rows if row.optimizer == "adam"), None)
    if adam is not None:
        delta = best.mean_accuracy - adam.mean_accuracy
        print(f"\nbest={best.optimizer} mean_acc={best.mean_accuracy:.3f} delta_vs_adam={delta:+.3f}")

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
