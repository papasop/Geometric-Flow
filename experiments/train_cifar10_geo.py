"""Train GeoCNN on CIFAR-10 or a synthetic CIFAR-shaped task."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

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
    train_seconds: float
    steps: int
    avg_preconditioned_to_raw_ratio: float


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
    model = GeoCNN(channels=args.channels).to(device)
    if optimizer_name == "adam":
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
            mode=args.mode,
            adam_warmup_steps=args.adam_warmup_steps,
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
        if optimizer_name == "adam":
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        else:
            optimizer.step(lambda: F.cross_entropy(model(x), y), verbose=args.verbose)

    seconds = time.perf_counter() - start
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
        train_seconds=seconds,
        steps=args.steps,
        avg_preconditioned_to_raw_ratio=sum(ratios) / max(len(ratios), 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "cifar10"], default="synthetic")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--channels", type=int, default=32)
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
    parser.add_argument("--mode", choices=["geometric", "adam", "hybrid"], default="geometric")
    parser.add_argument("--adam-warmup-steps", type=int, default=48)
    parser.add_argument("--cg-max-iter", type=int, default=8)
    parser.add_argument("--trace-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="artifacts/cifar10_geo_baseline.csv")
    parser.add_argument("--diagnostic-log", default="artifacts/cifar10_geo_diagnostics.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train_loader, eval_loader = make_loaders(args)
    if args.mode == "adam":
        rows = [train_one(args, "adam", train_loader, eval_loader)]
    else:
        rows = [
            train_one(args, "adam", train_loader, eval_loader),
            train_one(args, args.mode, train_loader, eval_loader),
        ]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
            print(
                f"{row.optimizer}: loss={row.final_loss:.4f} acc={row.final_accuracy:.3f} "
                f"seconds={row.train_seconds:.2f} ratio={row.avg_preconditioned_to_raw_ratio:.3f}"
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
