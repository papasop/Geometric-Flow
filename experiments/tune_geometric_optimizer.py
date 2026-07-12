"""Grid-search GeometricOptimizer hyperparameters on an MNIST-style task."""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import GeoMLP, GeometricOptimizer


@dataclass
class TuneResult:
    mode: str
    lr: float
    damping: float
    lr_scale: float
    curvature_reuse: int
    final_loss: float
    final_accuracy: float
    train_seconds: float
    steps: int
    best_loss: float
    avg_preconditioned_to_raw_ratio: float


def parse_floats(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_mnist(train_samples: int, eval_samples: int, seed: int) -> Tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(10, 28 * 28, generator=generator)

    def build(n: int) -> TensorDataset:
        y = torch.arange(n) % 10
        x = prototypes[y] + 0.25 * torch.randn(n, 28 * 28, generator=generator)
        return TensorDataset(x.view(n, 1, 28, 28), y.long())

    return build(train_samples), build(eval_samples)


def load_mnist(data_root: str, train_samples: int, eval_samples: int) -> Tuple[TensorDataset, TensorDataset]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for --dataset mnist") from exc

    transform = transforms.ToTensor()
    train = datasets.MNIST(data_root, train=True, download=False, transform=transform)
    test = datasets.MNIST(data_root, train=False, download=False, transform=transform)

    def subset(dataset, n: int) -> TensorDataset:
        xs, ys = [], []
        for idx in range(min(n, len(dataset))):
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)
        return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    return subset(train, train_samples), subset(test, eval_samples)


def make_loaders(args) -> Tuple[DataLoader, DataLoader]:
    if args.dataset == "mnist":
        train_set, eval_set = load_mnist(args.data_root, args.train_samples, args.eval_samples)
    else:
        train_set, eval_set = make_synthetic_mnist(args.train_samples, args.eval_samples, args.seed)
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


def run_trial(args, train_loader, eval_loader, lr, damping, lr_scale, curvature_reuse) -> TuneResult:
    set_seed(args.seed)
    device = torch.device(args.device)
    model = GeoMLP(hidden_dim=args.hidden_dim, output_dim=10).to(device)
    if args.mode == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=lr,
            damping=damping,
            lr_scale=lr_scale,
            curvature_reuse=curvature_reuse,
            warmup_steps=args.warmup_steps,
            regularization=args.regularization,
            cg_max_iter=args.cg_max_iter,
            trace_samples=args.trace_samples,
            max_update_norm=args.max_update_norm,
            max_grad_norm=args.max_grad_norm,
            grad_smoothing=args.grad_smoothing,
            preconditioner_scale=args.precond_scale,
            curvature_scale=args.curvature_scale,
            curvature_kind="fisher" if args.use_fisher else "hessian",
            preconditioner=args.preconditioner,
            mode=args.mode,
            adam_warmup_steps=args.adam_warmup_steps,
        )
    iterator = iter(train_loader)
    best_loss = float("inf")
    start = time.perf_counter()
    last_loss = None

    for _ in range(args.steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x = x.to(device)
        y = y.to(device)
        if args.mode == "adam":
            optimizer.zero_grad(set_to_none=True)
            last_loss = F.cross_entropy(model(x), y)
            last_loss.backward()
            optimizer.step()
        else:
            last_loss = optimizer.step(lambda: F.cross_entropy(model(x), y), verbose=args.verbose)
        best_loss = min(best_loss, float(last_loss.detach()))

    seconds = time.perf_counter() - start
    final_loss, final_accuracy = evaluate(model, eval_loader, device)
    if isinstance(optimizer, GeometricOptimizer):
        geometric_modes = {"geometric", "geometric_reuse", "diagonal"}
        ratios = [
            row["preconditioned_to_raw_ratio"]
            for row in optimizer.topography_log
            if row["raw_grad_norm"] > 0 and row["mode"] in geometric_modes
        ]
    else:
        ratios = []
    avg_ratio = sum(ratios) / max(len(ratios), 1)
    return TuneResult(
        mode=args.mode,
        lr=lr,
        damping=damping,
        lr_scale=lr_scale,
        curvature_reuse=curvature_reuse,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        train_seconds=seconds,
        steps=args.steps,
        best_loss=best_loss if last_loss is not None else float("nan"),
        avg_preconditioned_to_raw_ratio=avg_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "mnist"], default="synthetic")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--lrs", type=parse_floats, default=parse_floats("0.001,0.003"))
    parser.add_argument("--dampings", type=parse_floats, default=parse_floats("0.001,0.003"))
    parser.add_argument("--lr-scales", type=parse_floats, default=parse_floats("1.0,3.0"))
    parser.add_argument("--curvature-reuses", type=parse_ints, default=parse_ints("3,5"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--cg-max-iter", type=int, default=8)
    parser.add_argument("--trace-samples", type=int, default=0)
    parser.add_argument("--max-update-norm", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-smoothing", type=float, default=0.9)
    parser.add_argument("--precond-scale", type=float, default=0.5)
    parser.add_argument("--curvature-scale", type=float, default=1.0)
    parser.add_argument("--use-fisher", action="store_true")
    parser.add_argument("--preconditioner", choices=["cg", "diagonal"], default="cg")
    parser.add_argument("--mode", choices=["geometric", "adam", "hybrid"], default="geometric")
    parser.add_argument("--adam-warmup-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="artifacts/tune_geometric_optimizer.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train_loader, eval_loader = make_loaders(args)
    rows = []
    for lr, damping, lr_scale, reuse in itertools.product(
        args.lrs,
        args.dampings,
        args.lr_scales,
        args.curvature_reuses,
    ):
        result = run_trial(args, train_loader, eval_loader, lr, damping, lr_scale, reuse)
        rows.append(result)
        print(
            f"mode={args.mode} lr={lr:g} damping={damping:g} lr_scale={lr_scale:g} reuse={reuse} "
            f"loss={result.final_loss:.4f} acc={result.final_accuracy:.3f} seconds={result.train_seconds:.2f}"
        )

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
