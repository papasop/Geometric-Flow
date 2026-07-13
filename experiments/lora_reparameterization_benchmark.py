"""Small controlled LoRA reparameterization benchmark.

This benchmark keeps the model deliberately small. It checks whether
functional GeoFlow is less sensitive to the LoRA gauge transform
``A -> S A, B -> B S^{-1}`` than AdamW or the legacy diagonal grad-square
baseline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import GeometricOptimizer
from geometric_flow.functional_geometry import FunctionalMap, functional_projectors


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.register_buffer("base_weight", torch.randn(out_features, in_features) * 0.15)
        self.a = nn.Parameter(torch.randn(rank, in_features) * 0.05)
        self.b = nn.Parameter(torch.randn(out_features, rank) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base_weight + self.b @ self.a)

    def reparameterize(self, transform: torch.Tensor) -> None:
        inv = torch.linalg.inv(transform)
        with torch.no_grad():
            self.a.copy_(transform @ self.a)
            self.b.copy_(self.b @ inv)


class SmallLoRAMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.lora = LoRALinear(input_dim, hidden_dim, rank)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.lora(x)))


@dataclass
class LoRARow:
    seed: int
    optimizer: str
    representation: int
    initial_equivalence_residual: float
    final_loss: float
    final_accuracy: float
    final_phi: str
    tangent_drift: float
    near_null_amplification: float
    mean_null_leakage: float
    mean_jvp_count: float
    mean_vjp_count: float
    peak_memory_bytes: int
    seconds: float


def make_data(seed: int, samples: int, input_dim: int, output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, input_dim, generator=generator)
    teacher = torch.randn(input_dim, output_dim, generator=generator)
    y = (x @ teacher).argmax(dim=1)
    return x, y


def make_transform(rank: int, representation: int) -> torch.Tensor:
    if representation == 0:
        return torch.eye(rank)
    if representation % 2 == 1:
        scales = torch.linspace(-0.6, 0.6, rank) * (representation / 2.0)
        return torch.diag(torch.exp(scales))
    generator = torch.zeros(rank, rank)
    if rank >= 2:
        generator[0, 1] = 0.4 * representation
        generator[1, 0] = -0.4 * representation
    return torch.matrix_exp(generator)


def pairwise_phi_sensitivity(rows: list[LoRARow], optimizer: str) -> float:
    selected = [row for row in rows if row.optimizer == optimizer]
    if len(selected) < 2:
        return 0.0
    phis = [torch.tensor([float(value) for value in row.final_phi.split(";")]) for row in selected]
    distances = [torch.linalg.vector_norm(left - right).item() for left, right in itertools.combinations(phis, 2)]
    return float(sum(distances) / max(len(distances), 1))


def train_once(
    model: SmallLoRAMLP,
    optimizer_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    probe: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    args,
) -> tuple[float, float, torch.Tensor, list[dict], float]:
    generator = torch.Generator().manual_seed(args.seed + 1009)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "diagonal_grad_square":
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=lr,
            lr_scale=1.0,
            mode="geometric",
            preconditioner="diagonal_grad_square",
            warmup_steps=0,
            max_update_norm=args.max_update_norm,
            grad_smoothing=0.0,
            adaptive_damping=False,
        )
    elif optimizer_name == "functional_geoflow":
        optimizer = GeometricOptimizer(
            model.parameters(),
            lr=lr,
            lr_scale=1.0,
            mode="functional_geoflow",
            functional_model=model,
            functional_probe=probe,
            response_solver="implicit_cg",
            production_mode=True,
            refresh_interval=args.refresh_interval,
            max_basis_rank=args.max_basis_rank,
            max_vjp_probes=args.max_vjp_probes,
            vjp_probe_batch_size=args.vjp_probe_batch_size,
            cg_max_iter=args.cg_max_iter,
            cg_tolerance=args.cg_tol,
            damping=args.damping,
            regularization=0.0,
            adaptive_damping=False,
            max_update_norm=args.max_update_norm,
            functional_energy_fraction=1.0,
        )
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name}")

    start = time.perf_counter()
    logs: list[dict] = []
    n = x.shape[0]
    for _ in range(steps):
        indices = torch.randint(0, n, (batch_size,), generator=generator)
        xb = x[indices]
        yb = y[indices]
        if optimizer_name == "adamw":
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
        else:
            optimizer.step(lambda xb=xb, yb=yb: F.cross_entropy(model(xb), yb))
            logs.append(dict(optimizer.topography_log[-1]))
    seconds = time.perf_counter() - start
    with torch.no_grad():
        logits = model(x)
        final_loss = float(F.cross_entropy(logits, y))
        final_accuracy = float((logits.argmax(dim=1) == y).float().mean())
        final_phi = model(probe).reshape(-1).detach().clone()
    return final_loss, final_accuracy, final_phi, logs, seconds


def run_lora_benchmark(args) -> tuple[list[LoRARow], list[dict]]:
    rows: list[LoRARow] = []
    for trial in range(args.trials):
        seed = args.seed + trial
        torch.manual_seed(seed)
        x, y = make_data(seed, args.samples, args.input_dim, args.output_dim)
        probe = x[: args.probe_size].clone()
        base = SmallLoRAMLP(args.input_dim, args.hidden_dim, args.output_dim, args.lora_rank)
        base_state = copy.deepcopy(base.state_dict())
        reference_phi = base(probe).detach()
        for optimizer_name in args.optimizers.split(","):
            for representation in range(args.representations):
                model = SmallLoRAMLP(args.input_dim, args.hidden_dim, args.output_dim, args.lora_rank)
                model.load_state_dict(base_state)
                model.lora.reparameterize(make_transform(args.lora_rank, representation))
                initial_residual = float(torch.linalg.vector_norm(model(probe).detach() - reference_phi))
                fmap = FunctionalMap(model, probe)
                fjac = fmap.jacobian()
                projectors = functional_projectors(fjac.jacobian, null_threshold_mode="spectral_gap")
                loss, acc, phi, logs, seconds = train_once(
                    model,
                    optimizer_name.strip(),
                    x,
                    y,
                    probe,
                    args.steps,
                    args.batch_size,
                    args.lr,
                    args,
                )
                flat_after = fmap.flatten_params()
                tangent_drift = float(torch.linalg.vector_norm(projectors.tangent @ (flat_after - fjac.theta)))
                normal_drift = float(torch.linalg.vector_norm(projectors.normal @ (flat_after - fjac.theta)))
                mean_null = float(sum(float(log.get("null_leakage", 0.0)) for log in logs) / max(len(logs), 1))
                mean_jvp = float(sum(float(log.get("jvp_count", 0.0)) for log in logs) / max(len(logs), 1))
                mean_vjp = float(sum(float(log.get("vjp_count", 0.0)) for log in logs) / max(len(logs), 1))
                peak_mem = int(max([int(log.get("peak_memory_bytes", 0)) for log in logs] or [0]))
                rows.append(
                    LoRARow(
                        seed=seed,
                        optimizer=optimizer_name.strip(),
                        representation=representation,
                        initial_equivalence_residual=initial_residual,
                        final_loss=loss,
                        final_accuracy=acc,
                        final_phi=";".join(f"{float(v):.9g}" for v in phi.reshape(-1)),
                        tangent_drift=tangent_drift,
                        near_null_amplification=tangent_drift / max(normal_drift, 1e-30),
                        mean_null_leakage=mean_null,
                        mean_jvp_count=mean_jvp,
                        mean_vjp_count=mean_vjp,
                        peak_memory_bytes=peak_mem,
                        seconds=seconds,
                    )
                )
    aggregates = []
    for optimizer_name in sorted({row.optimizer for row in rows}):
        selected = [row for row in rows if row.optimizer == optimizer_name]
        aggregates.append(
            {
                "optimizer": optimizer_name,
                "mean_loss": sum(row.final_loss for row in selected) / len(selected),
                "mean_accuracy": sum(row.final_accuracy for row in selected) / len(selected),
                "reparameterization_sensitivity": pairwise_phi_sensitivity(rows, optimizer_name),
                "mean_tangent_drift": sum(row.tangent_drift for row in selected) / len(selected),
                "mean_near_null_amplification": sum(row.near_null_amplification for row in selected) / len(selected),
                "mean_seconds": sum(row.seconds for row in selected) / len(selected),
                "peak_memory_bytes": max(row.peak_memory_bytes for row in selected),
            }
        )
    return rows, aggregates


def write_csv(rows: list[LoRARow], aggregates: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    aggregate_path = out.with_name(out.stem + "_aggregate.csv")
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--representations", type=int, default=4)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--probe-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.25)
    parser.add_argument("--refresh-interval", type=int, default=5)
    parser.add_argument("--max-basis-rank", type=int, default=16)
    parser.add_argument("--max-vjp-probes", type=int, default=24)
    parser.add_argument("--vjp-probe-batch-size", type=int, default=8)
    parser.add_argument("--cg-max-iter", type=int, default=24)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--optimizers", default="adamw,diagonal_grad_square,functional_geoflow")
    parser.add_argument("--out", type=Path, default=Path("artifacts/lora_reparameterization.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, aggregates = run_lora_benchmark(args)
    write_csv(rows, aggregates, args.out)
    for item in aggregates:
        print(
            f"{item['optimizer']}: sensitivity={item['reparameterization_sensitivity']:.6g} "
            f"loss={item['mean_loss']:.6g} acc={item['mean_accuracy']:.3f} "
            f"seconds={item['mean_seconds']:.3f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
