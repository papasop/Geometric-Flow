"""Scale-curve experiment for Adam vs GeometricOptimizer.

The experiment measures how many optimization steps and seconds are needed to
reach the same target accuracy as model width grows. It defaults to a synthetic
CIFAR-shaped classification task so it can run without network downloads, and
can use real CIFAR-10 when torchvision and local data are available.

Example quick smoke run:

    python experiments/scale_curve.py --widths 16,32 --max-steps 3 --target-accuracy 0.99

Example full intended run:

    python experiments/scale_curve.py --dataset cifar10 --data-root ./data
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import GeoMLP, GeometricOptimizer


@dataclass
class RunResult:
    width: int
    optimizer: str
    params: int
    reached_target: bool
    target_accuracy: float
    final_accuracy: float
    final_loss: float
    steps_to_target: int
    seconds_to_target: float
    total_steps: int
    total_seconds: float
    avg_trace: float
    geodesic_distance: float


@dataclass
class ScalePoint:
    width: int
    params: int
    adam_steps: int
    geometric_steps: int
    adam_seconds: float
    geometric_seconds: float
    step_speedup: float
    time_speedup: float
    adam_reached: bool
    geometric_reached: bool


def parse_widths(value: str) -> List[int]:
    widths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not widths:
        raise argparse.ArgumentTypeError("at least one width is required")
    if any(width < 2 for width in widths):
        raise argparse.ArgumentTypeError("all widths must be >= 2")
    return widths


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_cifar(
    train_samples: int,
    eval_samples: int,
    classes: int,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(3 * 32 * 32, classes, generator=generator)
    teacher = teacher / teacher.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def build(n: int) -> TensorDataset:
        x = torch.randn(n, 3, 32, 32, generator=generator)
        logits = x.flatten(1) @ teacher + 0.15 * torch.randn(n, classes, generator=generator)
        y = logits.argmax(dim=1)
        return TensorDataset(x, y)

    return build(train_samples), build(eval_samples)


def load_cifar10(
    data_root: str,
    train_samples: int,
    eval_samples: int,
) -> Tuple[TensorDataset, TensorDataset]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for --dataset cifar10") from exc

    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(data_root, train=True, download=False, transform=transform)
    test = datasets.CIFAR10(data_root, train=False, download=False, transform=transform)

    def subset(dataset, n: int) -> TensorDataset:
        n = min(n, len(dataset))
        xs, ys = [], []
        for idx in range(n):
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)
        return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))

    return subset(train, train_samples), subset(test, eval_samples)


def make_loaders(
    dataset: str,
    data_root: str,
    train_samples: int,
    eval_samples: int,
    classes: int,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    if dataset == "cifar10":
        train_set, eval_set = load_cifar10(data_root, train_samples, eval_samples)
    else:
        train_set, eval_set = make_synthetic_cifar(train_samples, eval_samples, classes, seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader


def count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            losses.append(F.cross_entropy(logits, y).item())
            correct += int((logits.argmax(dim=1) == y).sum())
            total += int(y.numel())
    accuracy = correct / max(total, 1)
    loss = sum(losses) / max(len(losses), 1)
    model.train()
    return accuracy, loss


def train_one(
    width: int,
    optimizer_name: str,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    target_accuracy: float,
    max_steps: int,
    eval_interval: int,
    lr: float,
    damping: float,
    cg_max_iter: int,
    trace_samples: int,
    max_update_norm: float,
    max_grad_norm: float,
    regularization: float,
    warmup_steps: int,
    warmup_lr_scale: float,
    seed: int,
) -> RunResult:
    set_seed(seed)
    model = GeoMLP(input_dim=3 * 32 * 32, hidden_dim=width, output_dim=10).to(device)
    params = count_params(model)
    iterator = iter(train_loader)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "geometric":
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=lr,
            damping=damping,
            cg_max_iter=cg_max_iter,
            trace_samples=trace_samples,
            max_update_norm=max_update_norm,
            max_grad_norm=max_grad_norm,
            regularization=regularization,
            warmup_steps=warmup_steps,
            warmup_lr_scale=warmup_lr_scale,
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name}")

    start = time.perf_counter()
    steps_to_target = max_steps
    seconds_to_target = 0.0
    reached = False
    final_accuracy = 0.0
    final_loss = float("inf")

    for step in range(1, max_steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

        if optimizer_name == "adam":
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        else:
            loss = optimizer.step(lambda: F.cross_entropy(model(x), y))

        if step == 1 or step % eval_interval == 0 or step == max_steps:
            final_accuracy, final_loss = evaluate(model, eval_loader, device)
            if not reached and final_accuracy >= target_accuracy:
                reached = True
                steps_to_target = step
                seconds_to_target = time.perf_counter() - start
                break

    total_seconds = time.perf_counter() - start
    if not reached:
        seconds_to_target = total_seconds

    traces = []
    geodesic_distance = 0.0
    if isinstance(optimizer, GeometricOptimizer):
        traces = [
            entry["trace_estimate"]
            for entry in optimizer.topography_log
            if entry["trace_estimate"] is not None
        ]
        geodesic_distance = optimizer.geodesic_distance

    avg_trace = float(sum(traces) / len(traces)) if traces else 0.0
    return RunResult(
        width=width,
        optimizer=optimizer_name,
        params=params,
        reached_target=reached,
        target_accuracy=target_accuracy,
        final_accuracy=final_accuracy,
        final_loss=float(final_loss),
        steps_to_target=steps_to_target,
        seconds_to_target=float(seconds_to_target),
        total_steps=step,
        total_seconds=float(total_seconds),
        avg_trace=avg_trace,
        geodesic_distance=float(geodesic_distance),
    )


def build_scale_points(results: Iterable[RunResult]) -> List[ScalePoint]:
    by_width = {}
    for result in results:
        by_width.setdefault(result.width, {})[result.optimizer] = result

    points = []
    for width in sorted(by_width):
        pair = by_width[width]
        if "adam" not in pair or "geometric" not in pair:
            continue
        adam = pair["adam"]
        geometric = pair["geometric"]
        points.append(
            ScalePoint(
                width=width,
                params=adam.params,
                adam_steps=adam.steps_to_target,
                geometric_steps=geometric.steps_to_target,
                adam_seconds=adam.seconds_to_target,
                geometric_seconds=geometric.seconds_to_target,
                step_speedup=adam.steps_to_target / max(geometric.steps_to_target, 1),
                time_speedup=adam.seconds_to_target / max(geometric.seconds_to_target, 1e-12),
                adam_reached=adam.reached_target,
                geometric_reached=geometric.reached_target,
            )
        )
    return points


def write_csv(rows, path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def scale(values: List[float], low: float, high: float, log: bool = False) -> List[float]:
    if log:
        values = [math.log10(max(value, 1.0)) for value in values]
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [(low + high) / 2 for _ in values]
    return [low + (value - lo) * (high - low) / (hi - lo) for value in values]


def polyline(points: List[Tuple[float, float]], color: str) -> str:
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{coords}" />'


def circle_points(points: List[Tuple[float, float]], color: str) -> str:
    return "\n".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />' for x, y in points)


def write_svg(
    points: List[ScalePoint],
    path: Path,
    y_getters: List[Tuple[str, str, callable]],
    title: str,
    y_label: str,
) -> None:
    width = 860
    height = 520
    left, right, top, bottom = 82, 36, 54, 74
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [point.params for point in points]
    x_scaled = scale([float(x) for x in xs], left, left + plot_w, log=True)
    ys = [float(getter(point)) for _, _, getter in y_getters for point in points]
    y_scaled_all = scale(ys, top + plot_h, top)

    offset = 0
    lines = []
    for label, color, getter in y_getters:
        raw = [float(getter(point)) for point in points]
        scaled = y_scaled_all[offset : offset + len(points)]
        offset += len(points)
        line_points = list(zip(x_scaled, scaled))
        lines.append(polyline(line_points, color))
        lines.append(circle_points(line_points, color))
        lines.append(f'<text x="{left + 14}" y="{top + 22 + 20 * len(lines)}" fill="{color}">{label}</text>')

    x_labels = "\n".join(
        f'<text x="{x:.2f}" y="{height - 35}" text-anchor="middle">{param_count}</text>'
        for x, param_count in zip(x_scaled, xs)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111111" />
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111111" />
<text x="{width / 2}" y="{height - 10}" text-anchor="middle" font-family="Arial">trainable parameters (log scale)</text>
<text x="18" y="{height / 2}" transform="rotate(-90 18,{height / 2})" text-anchor="middle" font-family="Arial">{y_label}</text>
{x_labels}
{''.join(lines)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_plots(points: List[ScalePoint], out_dir: Path) -> None:
    if not points:
        return
    write_svg(
        points,
        out_dir / "scale_curve_steps.svg",
        [
            ("Adam steps", "#2563eb", lambda p: p.adam_steps),
            ("Geometric steps", "#dc2626", lambda p: p.geometric_steps),
        ],
        "Parameter Count vs Steps to Target Accuracy",
        "steps to target",
    )
    write_svg(
        points,
        out_dir / "scale_curve_time.svg",
        [
            ("Adam seconds", "#2563eb", lambda p: p.adam_seconds),
            ("Geometric seconds", "#dc2626", lambda p: p.geometric_seconds),
        ],
        "Parameter Count vs Time to Target Accuracy",
        "seconds to target",
    )
    write_svg(
        points,
        out_dir / "geometric_speedup.svg",
        [
            ("step speedup", "#16a34a", lambda p: p.step_speedup),
            ("time speedup", "#9333ea", lambda p: p.time_speedup),
        ],
        "Geometric Acceleration Ratio vs Model Scale",
        "Adam / Geometric",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "cifar10"], default="synthetic")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--widths", type=parse_widths, default=parse_widths("64,128,256,512"))
    parser.add_argument("--target-accuracy", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--geo-lr", type=float, default=None)
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--cg-max-iter", type=int, default=8)
    parser.add_argument("--trace-samples", type=int, default=1)
    parser.add_argument("--max-update-norm", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--warmup-lr-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="artifacts/scale_curve")
    args = parser.parse_args()

    if not 0 < args.target_accuracy <= 1:
        raise ValueError("--target-accuracy must be in (0, 1]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, eval_loader = make_loaders(
        args.dataset,
        args.data_root,
        args.train_samples,
        args.eval_samples,
        classes=10,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    results: List[RunResult] = []
    for width in args.widths:
        for optimizer_name in ("adam", "geometric"):
            opt_lr = args.lr if optimizer_name == "adam" else (args.geo_lr or args.lr)
            result = train_one(
                width=width,
                optimizer_name=optimizer_name,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                target_accuracy=args.target_accuracy,
                max_steps=args.max_steps,
                eval_interval=args.eval_interval,
                lr=opt_lr,
                damping=args.damping,
                cg_max_iter=args.cg_max_iter,
                trace_samples=args.trace_samples,
                max_update_norm=args.max_update_norm,
                max_grad_norm=args.max_grad_norm,
                regularization=args.regularization,
                warmup_steps=args.warmup_steps,
                warmup_lr_scale=args.warmup_lr_scale,
                seed=args.seed + width,
            )
            results.append(result)
            print(
                f"width={width:4d} opt={optimizer_name:9s} params={result.params:8d} "
                f"steps={result.steps_to_target:4d} acc={result.final_accuracy:.3f} "
                f"seconds={result.seconds_to_target:.2f} reached={result.reached_target}"
            )

    scale_points = build_scale_points(results)
    write_csv(results, out_dir / "scale_curve_runs.csv")
    write_csv(scale_points, out_dir / "scale_curve_summary.csv")
    (out_dir / "scale_curve_runs.json").write_text(
        json.dumps([asdict(row) for row in results], indent=2),
        encoding="utf-8",
    )
    write_plots(scale_points, out_dir)
    print(f"wrote results to {out_dir}")


if __name__ == "__main__":
    main()
