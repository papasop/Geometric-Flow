"""H10 progress-budgeted quotient-flow benchmark.

This script is a small scientific regression benchmark for
``SubsteppedQuotientFlow``. It compares factor-space Adam with the quotient
integrator at comparable loss progress, then measures how much final functional
trajectories diverge across gauge-equivalent LoRA representations.

The default ``macro_lr=2.6`` and ``substeps=16`` match the H10.6/H10.7
progress-budgeted configuration documented in the README. This is not a
production benchmark and not a large-language-model claim.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import SubsteppedQuotientFlow


@dataclass
class RunResult:
    seed: int
    representation: int
    optimizer: str
    steps: int
    initial_loss: float
    final_loss: float
    loss_progress: float
    product_displacement: float
    elapsed_seconds: float
    fallback_count: int
    balance_residual_max: float


@dataclass
class SeedSummary:
    seed: int
    adam_mean_loss_progress: float
    quotient_mean_loss_progress: float
    loss_progress_ratio: float
    adam_mean_product_displacement: float
    quotient_mean_product_displacement: float
    product_displacement_ratio: float
    adam_gauge_divergence: float
    quotient_gauge_divergence: float
    gauge_divergence_ratio: float
    gauge_suppression: float
    matched_progress_pass: bool
    fallback_count: int
    no_fallback_pass: bool
    balance_residual_max: float
    balance_pass: bool


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, generator: torch.Generator) -> None:
        super().__init__()
        self.rank = rank
        weight = torch.randn(out_features, in_features, generator=generator) * 0.12
        self.register_buffer("weight", weight)
        self.A = nn.Parameter(torch.randn(rank, in_features, generator=generator) * 0.04)
        self.B = nn.Parameter(torch.randn(out_features, rank, generator=generator) * 0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight + self.B @ self.A)

    @torch.no_grad()
    def apply_gauge(self, transform: torch.Tensor) -> None:
        self.A.copy_(transform @ self.A)
        self.B.copy_(self.B @ torch.linalg.inv(transform))

    def product(self) -> torch.Tensor:
        return self.B @ self.A


class TinyLoRANextToken(nn.Module):
    """A tiny GPT-style next-token model with LoRA factors in two projections."""

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, d_hidden: int, rank: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        embedding = torch.randn(vocab_size, d_model, generator=generator) * 0.18
        position = torch.randn(seq_len, d_model, generator=generator) * 0.03
        self.register_buffer("embedding", embedding)
        self.register_buffer("position", position)
        self.up = LoRALinear(d_model, d_hidden, rank, generator)
        self.down = LoRALinear(d_hidden, d_model, rank, generator)
        head = torch.randn(vocab_size, d_model, generator=generator) * 0.14
        self.register_buffer("head", head)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.shape[1]
        x = F.embedding(tokens, self.embedding) + self.position[:seq_len]
        hidden = torch.tanh(self.up(x))
        x = x + self.down(hidden)
        return F.linear(torch.tanh(x), self.head)

    def lora_modules(self) -> list[LoRALinear]:
        return [self.up, self.down]

    def factor_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self.lora_modules():
            params.extend([module.A, module.B])
        return params

    @torch.no_grad()
    def apply_gauge(self, transform: torch.Tensor) -> None:
        for module in self.lora_modules():
            module.apply_gauge(transform.to(device=module.A.device, dtype=module.A.dtype))

    def product_vector(self) -> torch.Tensor:
        return torch.cat([module.product().detach().reshape(-1).cpu() for module in self.lora_modules()])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_transform(rank: int, representation: int) -> torch.Tensor:
    if representation == 0:
        return torch.eye(rank)
    if representation % 2 == 1:
        scales = torch.linspace(-0.65, 0.65, rank) * representation
        return torch.diag(torch.exp(scales))
    generator = torch.zeros(rank, rank)
    generator[0, 1] = 0.35 * representation
    generator[1, 0] = -0.35 * representation
    return torch.matrix_exp(generator)


@torch.no_grad()
def make_task(args, seed: int, base: TinyLoRANextToken) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 50_000)
    teacher = deepcopy(base)
    for module in teacher.lora_modules():
        module.A.add_(0.18 * torch.randn(module.A.shape, generator=generator))
        module.B.add_(0.18 * torch.randn(module.B.shape, generator=generator))
    train_x = torch.randint(0, args.vocab_size, (args.train_samples, args.seq_len), generator=generator)
    eval_x = torch.randint(0, args.vocab_size, (args.eval_samples, args.seq_len), generator=generator)
    train_y = teacher(train_x).argmax(dim=-1)
    eval_y = teacher(eval_x).argmax(dim=-1)
    return train_x, train_y, eval_x, eval_y


def make_batches(seed: int, samples: int, batch_size: int, steps: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 90_000)
    return [torch.randint(0, samples, (batch_size,), generator=generator) for _ in range(steps)]


@torch.no_grad()
def evaluate_loss(model: TinyLoRANextToken, x: torch.Tensor, y: torch.Tensor, vocab_size: int) -> float:
    logits = model(x)
    return float(F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)).item())


@torch.no_grad()
def logits_vector(model: TinyLoRANextToken, x: torch.Tensor) -> torch.Tensor:
    return model(x).detach().reshape(-1).cpu()


def train_factor_adam(
    model: TinyLoRANextToken,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    batches: list[torch.Tensor],
    args,
    seed: int,
    representation: int,
) -> tuple[RunResult, torch.Tensor, torch.Tensor]:
    initial_loss = evaluate_loss(model, eval_x, eval_y, args.vocab_size)
    initial_product = model.product_vector()
    optimizer = torch.optim.Adam(model.factor_parameters(), lr=args.factor_lr)
    started = time.perf_counter()
    for indices in batches[: args.adam_steps]:
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x[indices])
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), train_y[indices].reshape(-1))
        loss.backward()
        if args.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.factor_parameters(), args.clip_norm)
        optimizer.step()
    elapsed = time.perf_counter() - started
    final_loss = evaluate_loss(model, eval_x, eval_y, args.vocab_size)
    product_displacement = float(torch.linalg.vector_norm(model.product_vector() - initial_product))
    result = RunResult(
        seed=seed,
        representation=representation,
        optimizer="factor_adam",
        steps=args.adam_steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_progress=initial_loss - final_loss,
        product_displacement=product_displacement,
        elapsed_seconds=elapsed,
        fallback_count=0,
        balance_residual_max=0.0,
    )
    return result, logits_vector(model, eval_x), model.product_vector()


def train_quotient_flow(
    model: TinyLoRANextToken,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    batches: list[torch.Tensor],
    target_progress: float,
    args,
    seed: int,
    representation: int,
) -> tuple[RunResult, torch.Tensor, torch.Tensor]:
    initial_loss = evaluate_loss(model, eval_x, eval_y, args.vocab_size)
    initial_product = model.product_vector()
    optimizer = SubsteppedQuotientFlow(
        model.lora_modules(),
        macro_lr=args.macro_lr,
        substeps=args.substeps,
        clip_norm=args.clip_norm,
        balance_after_substep=True,
        gram_condition_limit=args.gram_condition_limit,
    )
    steps = 0
    started = time.perf_counter()
    final_loss = initial_loss
    for indices in batches[: args.max_quotient_steps]:
        def closure(indices=indices):
            optimizer.zero_grad()
            logits = model(train_x[indices])
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), train_y[indices].reshape(-1))
            loss.backward()
            return loss

        optimizer.macro_step(closure)
        steps += 1
        final_loss = evaluate_loss(model, eval_x, eval_y, args.vocab_size)
        if steps >= args.min_quotient_steps and initial_loss - final_loss >= args.progress_fraction * target_progress:
            break
    elapsed = time.perf_counter() - started
    product_displacement = float(torch.linalg.vector_norm(model.product_vector() - initial_product))
    result = RunResult(
        seed=seed,
        representation=representation,
        optimizer="substepped_quotient_flow",
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_progress=initial_loss - final_loss,
        product_displacement=product_displacement,
        elapsed_seconds=elapsed,
        fallback_count=optimizer.fallback_count,
        balance_residual_max=optimizer.balance_residual_max,
    )
    return result, logits_vector(model, eval_x), model.product_vector()


def mean_pairwise_distance(vectors: list[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 0.0
    distances = [torch.linalg.vector_norm(left - right).item() for left, right in itertools.combinations(vectors, 2)]
    return float(sum(distances) / len(distances))


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def geometric_mean(values: list[float]) -> float:
    positive = [max(value, 1e-30) for value in values if math.isfinite(value)]
    if not positive:
        return float("nan")
    return float(math.exp(sum(math.log(value) for value in positive) / len(positive)))


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite or samples <= 0:
        return float("nan"), float("nan")
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [finite[generator.randrange(len(finite))] for _ in finite]
        estimates.append(geometric_mean(draw))
    estimates.sort()
    lo = estimates[min(int(0.025 * samples), samples - 1)]
    hi = estimates[min(int(0.975 * samples), samples - 1)]
    return lo, hi


def run_seed(seed: int, args) -> tuple[list[RunResult], SeedSummary]:
    seed_everything(seed)
    base = TinyLoRANextToken(args.vocab_size, args.seq_len, args.d_model, args.d_hidden, args.rank, seed)
    train_x, train_y, eval_x, eval_y = make_task(args, seed, base)
    batches = make_batches(seed, args.train_samples, args.batch_size, max(args.adam_steps, args.max_quotient_steps))
    rows: list[RunResult] = []
    adam_logits: list[torch.Tensor] = []
    quotient_logits: list[torch.Tensor] = []
    adam_progress: list[float] = []
    quotient_progress: list[float] = []
    adam_displacements: list[float] = []
    quotient_displacements: list[float] = []
    fallback_count = 0
    balance_residual = 0.0

    for representation in range(args.representations):
        transform = make_transform(args.rank, representation)
        adam_model = deepcopy(base)
        adam_model.apply_gauge(transform)
        quotient_model = deepcopy(base)
        quotient_model.apply_gauge(transform)
        adam_result, adam_phi, _ = train_factor_adam(
            adam_model, train_x, train_y, eval_x, eval_y, batches, args, seed, representation
        )
        quotient_result, quotient_phi, _ = train_quotient_flow(
            quotient_model,
            train_x,
            train_y,
            eval_x,
            eval_y,
            batches,
            max(adam_result.loss_progress, 1e-12),
            args,
            seed,
            representation,
        )
        rows.extend([adam_result, quotient_result])
        adam_logits.append(adam_phi)
        quotient_logits.append(quotient_phi)
        adam_progress.append(adam_result.loss_progress)
        quotient_progress.append(quotient_result.loss_progress)
        adam_displacements.append(adam_result.product_displacement)
        quotient_displacements.append(quotient_result.product_displacement)
        fallback_count += quotient_result.fallback_count
        balance_residual = max(balance_residual, quotient_result.balance_residual_max)

    adam_divergence = mean_pairwise_distance(adam_logits)
    quotient_divergence = mean_pairwise_distance(quotient_logits)
    gauge_ratio = quotient_divergence / max(adam_divergence, 1e-30)
    loss_ratio = finite_mean(quotient_progress) / max(finite_mean(adam_progress), 1e-30)
    displacement_ratio = finite_mean(quotient_displacements) / max(finite_mean(adam_displacements), 1e-30)
    summary = SeedSummary(
        seed=seed,
        adam_mean_loss_progress=finite_mean(adam_progress),
        quotient_mean_loss_progress=finite_mean(quotient_progress),
        loss_progress_ratio=loss_ratio,
        adam_mean_product_displacement=finite_mean(adam_displacements),
        quotient_mean_product_displacement=finite_mean(quotient_displacements),
        product_displacement_ratio=displacement_ratio,
        adam_gauge_divergence=adam_divergence,
        quotient_gauge_divergence=quotient_divergence,
        gauge_divergence_ratio=gauge_ratio,
        gauge_suppression=1.0 / max(gauge_ratio, 1e-30),
        matched_progress_pass=loss_ratio >= args.progress_fraction,
        fallback_count=fallback_count,
        no_fallback_pass=fallback_count == 0,
        balance_residual_max=balance_residual,
        balance_pass=balance_residual <= args.balance_tol,
    )
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="101,211,307")
    parser.add_argument("--representations", type=int, default=4)
    parser.add_argument("--adam-steps", type=int, default=20)
    parser.add_argument("--max-quotient-steps", type=int, default=40)
    parser.add_argument("--min-quotient-steps", type=int, default=1)
    parser.add_argument("--progress-fraction", type=float, default=0.95)
    parser.add_argument("--macro-lr", type=float, default=2.6)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--factor-lr", type=float, default=0.03)
    parser.add_argument("--clip-norm", type=float, default=None)
    parser.add_argument("--gram-condition-limit", type=float, default=1e10)
    parser.add_argument("--balance-tol", type=float, default=1e-5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--vocab-size", type=int, default=29)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--d-hidden", type=int, default=48)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/h10_progress_budget"))
    args = parser.parse_args()

    all_rows: list[RunResult] = []
    seed_summaries: list[SeedSummary] = []
    for seed in parse_seeds(args.seeds):
        rows, summary = run_seed(seed, args)
        all_rows.extend(rows)
        seed_summaries.append(summary)

    gauge_ratios = [row.gauge_divergence_ratio for row in seed_summaries]
    loss_ratios = [row.loss_progress_ratio for row in seed_summaries]
    displacement_ratios = [row.product_displacement_ratio for row in seed_summaries]
    fallback_counts = [row.fallback_count for row in seed_summaries]
    balance_residuals = [row.balance_residual_max for row in seed_summaries]
    suppressions = [row.gauge_suppression for row in seed_summaries]
    suppression_ci = bootstrap_ci(suppressions, args.bootstrap_samples, args.bootstrap_seed)
    aggregate = {
        "seeds": parse_seeds(args.seeds),
        "macro_lr": args.macro_lr,
        "substeps": args.substeps,
        "mean_loss_progress_ratio": finite_mean(loss_ratios),
        "mean_product_displacement_ratio": finite_mean(displacement_ratios),
        "geomean_gauge_divergence_ratio": geometric_mean(gauge_ratios),
        "geomean_gauge_suppression": 1.0 / max(geometric_mean(gauge_ratios), 1e-30),
        "per_seed_gauge_suppression_10x_fraction": finite_mean(
            [1.0 if value >= 10.0 else 0.0 for value in suppressions]
        ),
        "bootstrap_gauge_suppression_95ci_low": suppression_ci[0],
        "bootstrap_gauge_suppression_95ci_high": suppression_ci[1],
        "matched_progress_pass_all_seeds": all(row.matched_progress_pass for row in seed_summaries),
        "no_fallback_pass": sum(fallback_counts) == 0,
        "balance_pass": max(balance_residuals) <= args.balance_tol if balance_residuals else True,
    }
    aggregate["H106_ALL_SEEDS_MATCHED_PROGRESS"] = aggregate["matched_progress_pass_all_seeds"]
    aggregate["H106_MEAN_GAUGE_SUPPRESSION_10X_PASS"] = aggregate["geomean_gauge_suppression"] >= 10.0
    aggregate["H106_NO_FALLBACK_PASS"] = aggregate["no_fallback_pass"]
    aggregate["H106_BALANCE_PASS"] = aggregate["balance_pass"]
    aggregate["H107_ALL_SEEDS_MATCHED_PROGRESS"] = aggregate["matched_progress_pass_all_seeds"]
    aggregate["H107_MEAN_GAUGE_SUPPRESSION_10X_PASS"] = aggregate["geomean_gauge_suppression"] >= 10.0
    aggregate["H107_ALL_SEEDS_GAUGE_SUPPRESSION_10X_PASS"] = (
        aggregate["per_seed_gauge_suppression_10x_fraction"] >= 1.0
    )
    aggregate["H107_BOOTSTRAP_CI_EXCLUDES_10X_PASS"] = aggregate["bootstrap_gauge_suppression_95ci_low"] >= 10.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_run.csv", [asdict(row) for row in all_rows])
    write_csv(args.out_dir / "per_seed.csv", [asdict(row) for row in seed_summaries])
    with (args.out_dir / "summary.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)

    print(
        "H10 progress-budget summary: "
        f"loss_ratio={aggregate['mean_loss_progress_ratio']:.4g} "
        f"disp_ratio={aggregate['mean_product_displacement_ratio']:.4g} "
        f"gauge_ratio={aggregate['geomean_gauge_divergence_ratio']:.4g} "
        f"suppression={aggregate['geomean_gauge_suppression']:.4g} "
        f"per_seed_10x={aggregate['per_seed_gauge_suppression_10x_fraction']:.2f} "
        f"matched={aggregate['H106_ALL_SEEDS_MATCHED_PROGRESS']} "
        f"fallback_free={aggregate['H106_NO_FALLBACK_PASS']} "
        f"balance={aggregate['H106_BALANCE_PASS']}"
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
