#!/usr/bin/env python3
"""Prospective neural response-sufficiency / jet-filtration audit v0.12.0.

The audit uses one no-download residual-CNN teacher per seed and compares three
declared response maps from the same initial implementation:

  R0      : logits on 24 probe images;
  R_dense : logits on 96 probe images;
  R_jet   : logits plus two analytic input-directional derivatives on 24 probes.

Every branch follows the same SVD-projected intrinsic descent and SVD-Newton
response retraction.  The primary question is whether an enriched response map
reduces held-out response drift relative to R0 and closes the preregistered
held-out task gate.  This is a floating-point prospective experiment, not an
interval proof and not evidence for large models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, jacrev, jvp

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


VERSION = "0.12.0"
LEVELS = ("R0_logits24", "R_dense_logits96", "R_jet_logits24_directions2")
DEVELOPMENT_SEEDS = (
    20260801,
    20260817,
    20260829,
    20260907,
    20260919,
    20261003,
    20261107,
    20261119,
    20261203,
)
PROSPECTIVE_SEEDS = (20270107, 20270119, 20270203)


@dataclass(frozen=True)
class Gates:
    minimum_teacher_clean_accuracy: float = 0.95
    minimum_teacher_noisy_accuracy: float = 0.85
    maximum_declared_response_relative_drift: float = 1.0e-3
    maximum_heldout_response_relative_drift: float = 1.0e-3
    minimum_secondary_objective_reduction: float = 1.0e-4
    maximum_clean_accuracy_drop: float = 0.01
    minimum_noisy_accuracy_gain: float = 0.0
    minimum_successful_seed_fraction: float = 2.0 / 3.0


class TinyResidualCNN(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.readout = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.stem(x))
        residual = self.conv2(torch.tanh(self.conv1(hidden)))
        hidden = torch.tanh(hidden + 0.25 * residual)
        return self.readout(hidden.mean(dim=(-2, -1))).squeeze(-1)


class FlatModel:
    def __init__(self, model: nn.Module):
        self.model = model
        self.names: List[str] = []
        self.shapes: List[torch.Size] = []
        self.sizes: List[int] = []
        for name, parameter in model.named_parameters():
            self.names.append(name)
            self.shapes.append(parameter.shape)
            self.sizes.append(parameter.numel())

    def flatten(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])

    def unpack(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        chunks = torch.split(theta, self.sizes)
        return {
            name: chunk.reshape(shape)
            for name, shape, chunk in zip(self.names, self.shapes, chunks)
        }

    def __call__(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return functional_call(self.model, self.unpack(theta), (x,))


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_task(seed: int, n: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=generator).to(dtype)
    x = 0.10 * torch.randn((n, 1, 8, 8), generator=generator, dtype=dtype)
    locations = torch.randint(1, 6, (n,), generator=generator)
    intensities = 0.8 + 0.4 * torch.rand(n, generator=generator, dtype=dtype)
    diagonal = torch.arange(8)
    for index in range(n):
        location = int(locations[index])
        if int(y[index]) == 0:
            x[index, 0, :, location : location + 2] += intensities[index]
        else:
            x[index, 0, location : location + 2, :] += intensities[index]
        nuisance = 0.08 * torch.randn((), generator=generator, dtype=dtype)
        x[index, 0, diagonal, diagonal] += nuisance
    order = torch.randperm(n, generator=generator)
    return x[order], y[order]


def train_teacher(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(((logits >= 0) == (y >= 0.5)).to(torch.float64).mean())


def ece(logits: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    probability = torch.sigmoid(logits)
    confidence = torch.maximum(probability, 1.0 - probability)
    correct = (probability >= 0.5) == (y >= 0.5)
    edges = torch.linspace(0.5, 1.0, bins + 1, dtype=logits.dtype)
    value = torch.zeros((), dtype=logits.dtype)
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if bool(mask.any()):
            value += mask.sum() / logits.numel() * torch.abs(
                correct[mask].to(logits.dtype).mean() - confidence[mask].mean()
            )
    return float(value)


def relative_drift(now: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(reference)), 1.0)
    return float(torch.linalg.vector_norm(now - reference)) / denominator


def secondary_objective(
    flat: FlatModel,
    theta: torch.Tensor,
    clean_inputs: torch.Tensor,
    noisy_inputs: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(
        (flat(theta, noisy_inputs) - flat(theta, clean_inputs)) ** 2
    )


def response_jacobian(response_fn, theta: torch.Tensor) -> torch.Tensor:
    return jacrev(response_fn)(theta)


def svd_row_space(
    jacobian: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
    tolerance = (
        max(jacobian.shape)
        * torch.finfo(jacobian.dtype).eps
        * float(singular_values[0].detach())
    )
    rank = int((singular_values.detach() > tolerance).sum())
    return u, singular_values, vh, rank


def svd_tangent_projection(
    jacobian: torch.Tensor,
    vector: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    _, _, vh, rank = svd_row_space(jacobian)
    if rank != jacobian.shape[0]:
        raise ArithmeticError(
            f"response Jacobian is not full row rank: {rank}/{jacobian.shape[0]}"
        )
    return vector - vh.T @ (vh @ vector), rank


def svd_minimum_norm_correction(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    u, singular_values, vh, rank = svd_row_space(jacobian)
    if rank != jacobian.shape[0]:
        raise ArithmeticError(
            f"response Jacobian is not full row rank: {rank}/{jacobian.shape[0]}"
        )
    return -vh.T @ ((u.T @ residual) / singular_values)


def retract_to_response(
    theta: torch.Tensor,
    response_fn,
    target: torch.Tensor,
    iterations: int,
    absolute_tolerance: float,
) -> Tuple[torch.Tensor, int, float]:
    current = theta.detach()
    for used in range(iterations + 1):
        residual = response_fn(current) - target
        residual_norm = float(torch.linalg.vector_norm(residual))
        if residual_norm <= absolute_tolerance or used == iterations:
            return current, used, residual_norm
        jacobian = response_jacobian(response_fn, current)
        current = (
            current + svd_minimum_norm_correction(jacobian, residual)
        ).detach()
    raise RuntimeError("unreachable retraction state")


def bounded_euler_step(
    direction: torch.Tensor,
    learning_rate: float,
    maximum_radius: float,
) -> torch.Tensor:
    step = learning_rate * direction
    norm = torch.linalg.vector_norm(step)
    if not bool(torch.isfinite(norm)) or float(norm) == 0.0:
        return torch.zeros_like(direction)
    if float(norm) > maximum_radius:
        step = step * (maximum_radius / norm)
    return step


def normalised_directions(
    generator: torch.Generator,
    shape: torch.Size,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = []
    for _ in range(2):
        direction = torch.randn(shape, generator=generator, dtype=dtype)
        norms = torch.linalg.vector_norm(
            direction.reshape(direction.shape[0], -1), dim=1
        ).reshape(-1, 1, 1, 1)
        values.append(direction / torch.clamp(norms, min=1.0e-30))
    return values[0], values[1]


def build_response_fn(
    level: str,
    flat: FlatModel,
    probe24: torch.Tensor,
    probe96: torch.Tensor,
    direction1: torch.Tensor,
    direction2: torch.Tensor,
):
    if level == "R0_logits24":
        return lambda theta: flat(theta, probe24)
    if level == "R_dense_logits96":
        return lambda theta: flat(theta, probe96)
    if level == "R_jet_logits24_directions2":
        def response(theta: torch.Tensor) -> torch.Tensor:
            logits = flat(theta, probe24)
            _, derivative1 = jvp(
                lambda inputs: flat(theta, inputs),
                (probe24,),
                (direction1,),
            )
            _, derivative2 = jvp(
                lambda inputs: flat(theta, inputs),
                (probe24,),
                (direction2,),
            )
            return torch.cat((logits, derivative1, derivative2))

        return response
    raise ValueError(f"unknown response level: {level}")


def optimise_level(
    level: str,
    flat: FlatModel,
    theta0: torch.Tensor,
    response_fn,
    heldout_fn,
    x_robust: torch.Tensor,
    x_robust_noisy: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    x_test_noisy: torch.Tensor,
    steps: int,
    initial_step_radius: float,
    maximum_step_multiplier: float,
    retract_iterations: int,
    response_absolute_tolerance: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    target = response_fn(theta0).detach()
    heldout_target = heldout_fn(theta0).detach()
    initial_jacobian = response_jacobian(response_fn, theta0)
    singular_values = torch.linalg.svdvals(initial_jacobian).detach()
    initial_variable = theta0.detach().requires_grad_(True)
    initial_value = secondary_objective(
        flat, initial_variable, x_robust, x_robust_noisy
    )
    initial_gradient = torch.autograd.grad(initial_value, initial_variable)[0].detach()
    projected_initial, response_rank = svd_tangent_projection(
        initial_jacobian, initial_gradient
    )
    projected_norm = float(torch.linalg.vector_norm(projected_initial))
    if projected_norm <= 0.0:
        raise ArithmeticError(f"{level}: zero initial projected gradient")
    learning_rate = initial_step_radius / projected_norm
    maximum_step_radius = maximum_step_multiplier * initial_step_radius

    theta = theta0.detach().clone()
    trajectory: List[Dict[str, object]] = []
    total_retractions = 0

    def record(step: int) -> None:
        with torch.no_grad():
            clean_test_logits = flat(theta, x_test)
            noisy_test_logits = flat(theta, x_test_noisy)
            trajectory.append(
                {
                    "level": level,
                    "step": step,
                    "secondary_objective": float(
                        secondary_objective(
                            flat, theta, x_robust, x_robust_noisy
                        )
                    ),
                    "declared_response_relative_drift": relative_drift(
                        response_fn(theta), target
                    ),
                    "heldout_response_relative_drift": relative_drift(
                        heldout_fn(theta), heldout_target
                    ),
                    "clean_accuracy": accuracy(clean_test_logits, y_test),
                    "noisy_accuracy": accuracy(noisy_test_logits, y_test),
                    "ece": ece(clean_test_logits, y_test),
                }
            )

    record(0)
    for step in range(1, steps + 1):
        variable = theta.detach().requires_grad_(True)
        value = secondary_objective(flat, variable, x_robust, x_robust_noisy)
        gradient = torch.autograd.grad(value, variable)[0]
        jacobian = response_jacobian(response_fn, variable)
        projected, _ = svd_tangent_projection(jacobian, gradient)
        direction = -projected
        proposal = (
            theta
            + bounded_euler_step(
                direction.detach(), learning_rate, maximum_step_radius
            )
        ).detach()
        proposal, used, _ = retract_to_response(
            proposal,
            response_fn,
            target,
            retract_iterations,
            response_absolute_tolerance,
        )
        total_retractions += used
        theta = proposal
        record(step)

    initial = trajectory[0]
    final = trajectory[-1]
    reduction = float(initial["secondary_objective"]) - float(
        final["secondary_objective"]
    )
    finite = all(
        math.isfinite(float(row[key]))
        for row in trajectory
        for key in (
            "secondary_objective",
            "declared_response_relative_drift",
            "heldout_response_relative_drift",
            "clean_accuracy",
            "noisy_accuracy",
            "ece",
        )
    )
    return {
        "level": level,
        "finite": finite,
        "response_dimension": target.numel(),
        "response_jacobian_rank": response_rank,
        "response_jacobian_minimum_singular_value": float(singular_values[-1]),
        "response_jacobian_condition_number": float(
            singular_values[0] / singular_values[-1]
        ),
        "initial_projected_gradient_norm": projected_norm,
        "common_intrinsic_learning_rate": learning_rate,
        "initial_secondary_objective": initial["secondary_objective"],
        "final_secondary_objective": final["secondary_objective"],
        "secondary_objective_reduction": reduction,
        "maximum_declared_response_relative_drift": max(
            float(row["declared_response_relative_drift"]) for row in trajectory
        ),
        "maximum_heldout_response_relative_drift": max(
            float(row["heldout_response_relative_drift"]) for row in trajectory
        ),
        "final_heldout_response_relative_drift": final[
            "heldout_response_relative_drift"
        ],
        "clean_accuracy_change": float(final["clean_accuracy"])
        - float(initial["clean_accuracy"]),
        "noisy_accuracy_change": float(final["noisy_accuracy"])
        - float(initial["noisy_accuracy"]),
        "initial_clean_accuracy": initial["clean_accuracy"],
        "final_clean_accuracy": final["clean_accuracy"],
        "initial_noisy_accuracy": initial["noisy_accuracy"],
        "final_noisy_accuracy": final["noisy_accuracy"],
        "initial_ece": initial["ece"],
        "final_ece": final["ece"],
        "retraction_iterations_total": total_retractions,
    }, trajectory


def level_gates(
    result: Dict[str, object],
    teacher_clean_accuracy: float,
    teacher_noisy_accuracy: float,
    gates: Gates,
) -> Dict[str, bool]:
    return {
        "teacher_clean_task_learned": teacher_clean_accuracy
        >= gates.minimum_teacher_clean_accuracy,
        "teacher_noisy_task_nontrivial": teacher_noisy_accuracy
        >= gates.minimum_teacher_noisy_accuracy,
        "response_full_row_rank": int(result["response_jacobian_rank"])
        == int(result["response_dimension"]),
        "all_finite": bool(result["finite"]),
        "declared_response_preserved": float(
            result["maximum_declared_response_relative_drift"]
        )
        <= gates.maximum_declared_response_relative_drift,
        "heldout_response_preserved": float(
            result["maximum_heldout_response_relative_drift"]
        )
        <= gates.maximum_heldout_response_relative_drift,
        "secondary_objective_decreased": float(
            result["secondary_objective_reduction"]
        )
        >= gates.minimum_secondary_objective_reduction,
        "clean_accuracy_preserved": float(result["clean_accuracy_change"])
        >= -gates.maximum_clean_accuracy_drop,
        "noisy_accuracy_nonworse": float(result["noisy_accuracy_change"])
        >= gates.minimum_noisy_accuracy_gain,
    }


def plot_seed(rows: List[Dict[str, object]], path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for level in LEVELS:
        selected = [row for row in rows if row["level"] == level]
        steps = [int(row["step"]) for row in selected]
        axes[0].plot(
            steps,
            [row["secondary_objective"] for row in selected],
            label=level,
        )
        axes[1].semilogy(
            steps,
            [
                max(float(row["declared_response_relative_drift"]), 1.0e-16)
                for row in selected
            ],
        )
        axes[2].semilogy(
            steps,
            [
                max(float(row["heldout_response_relative_drift"]), 1.0e-16)
                for row in selected
            ],
        )
    axes[0].set_title("Secondary objective")
    axes[1].set_title("Declared response drift")
    axes[2].set_title("Held-out response drift")
    for axis in axes:
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_seed(seed: int, args, gates: Gates, output: Path) -> Dict[str, object]:
    seed_all(seed)
    dtype = torch.float64
    x_train, y_train = make_task(seed + 11, args.train_size, dtype)
    x_test, y_test = make_task(seed + 29, args.test_size, dtype)
    model = TinyResidualCNN(args.width).to(dtype)
    train_teacher(
        model, x_train, y_train, args.teacher_epochs, args.teacher_lr
    )
    flat = FlatModel(model)
    theta0 = flat.flatten().to(dtype)

    probe_generator = torch.Generator().manual_seed(seed + 101)
    robust_generator = torch.Generator().manual_seed(seed + 103)
    test_noise_generator = torch.Generator().manual_seed(seed + 107)
    direction_generator = torch.Generator().manual_seed(seed + 109)
    train_order = torch.randperm(args.train_size, generator=probe_generator)
    probe96 = x_train[train_order[: args.dense_probe_size]]
    probe24 = probe96[: args.base_probe_size]
    heldout_order = torch.randperm(args.test_size, generator=probe_generator)
    x_heldout = x_test[heldout_order[: args.heldout_size]]
    robust_order = torch.randperm(args.train_size, generator=robust_generator)
    x_robust = x_train[robust_order[: args.robust_size]]
    robust_noise = torch.randn(
        x_robust.shape, generator=robust_generator, dtype=dtype
    )
    x_robust_noisy = x_robust + args.noise_sigma * robust_noise
    test_noise = torch.randn(
        x_test.shape, generator=test_noise_generator, dtype=dtype
    )
    x_test_noisy = x_test + args.noise_sigma * test_noise
    direction1, direction2 = normalised_directions(
        direction_generator, probe24.shape, dtype
    )

    heldout_fn = lambda theta: flat(theta, x_heldout)
    teacher_clean_accuracy = accuracy(flat(theta0, x_test), y_test)
    teacher_noisy_accuracy = accuracy(flat(theta0, x_test_noisy), y_test)

    results: Dict[str, Dict[str, object]] = {}
    all_rows: List[Dict[str, object]] = []
    for level in LEVELS:
        print(f"  [{seed}] level={level}", flush=True)
        response_fn = build_response_fn(
            level,
            flat,
            probe24,
            probe96,
            direction1,
            direction2,
        )
        result, rows = optimise_level(
            level,
            flat,
            theta0,
            response_fn,
            heldout_fn,
            x_robust,
            x_robust_noisy,
            x_test,
            y_test,
            x_test_noisy,
            args.steps,
            args.step_radius,
            args.maximum_step_multiplier,
            args.retract_iterations,
            args.response_absolute_tolerance,
        )
        gates_for_level = level_gates(
            result, teacher_clean_accuracy, teacher_noisy_accuracy, gates
        )
        result["gates"] = gates_for_level
        result["all_level_gates_pass"] = all(gates_for_level.values())
        results[level] = result
        all_rows.extend(rows)

    r0_drift = float(
        results["R0_logits24"]["maximum_heldout_response_relative_drift"]
    )
    enriched_levels = ("R_dense_logits96", "R_jet_logits24_directions2")
    best_enriched_level = min(
        enriched_levels,
        key=lambda name: float(
            results[name]["maximum_heldout_response_relative_drift"]
        ),
    )
    best_enriched_drift = float(
        results[best_enriched_level]["maximum_heldout_response_relative_drift"]
    )
    enriched_candidates_passing = [
        name
        for name in enriched_levels
        if bool(results[name]["all_level_gates_pass"])
    ]
    filtration_gates = {
        "at_least_one_enriched_response_closes_all_level_gates": bool(
            enriched_candidates_passing
        ),
        "best_enriched_response_strictly_reduces_heldout_drift": (
            best_enriched_drift < r0_drift
        ),
    }
    filtration_pass = all(filtration_gates.values())

    csv_path = output / f"trajectory_seed_{seed}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    plot_seed(all_rows, output / f"trajectory_seed_{seed}.png")

    return {
        "seed": seed,
        "filtration_pass": filtration_pass,
        "filtration_gates": filtration_gates,
        "parameter_dimension": theta0.numel(),
        "teacher_clean_accuracy": teacher_clean_accuracy,
        "teacher_noisy_accuracy": teacher_noisy_accuracy,
        "R0_maximum_heldout_drift": r0_drift,
        "best_enriched_level": best_enriched_level,
        "best_enriched_maximum_heldout_drift": best_enriched_drift,
        "heldout_drift_improvement_factor": (
            r0_drift / best_enriched_drift
            if best_enriched_drift > 0.0
            else None
        ),
        "enriched_levels_passing_all_gates": enriched_candidates_passing,
        "levels": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output", default="response_fibre_nn_jet_filtration_v0_12_0_results"
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in PROSPECTIVE_SEEDS),
        help="overrides are diagnostic; only frozen defaults support claims",
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--base-probe-size", type=int, default=24)
    parser.add_argument("--dense-probe-size", type=int, default=96)
    parser.add_argument("--heldout-size", type=int, default=96)
    parser.add_argument("--robust-size", type=int, default=128)
    parser.add_argument("--teacher-epochs", type=int, default=300)
    parser.add_argument("--teacher-lr", type=float, default=2.0e-2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--step-radius", type=float, default=2.0e-3)
    parser.add_argument("--maximum-step-multiplier", type=float, default=4.0)
    parser.add_argument("--noise-sigma", type=float, default=0.12)
    parser.add_argument("--retract-iterations", type=int, default=5)
    parser.add_argument("--response-absolute-tolerance", type=float, default=1.0e-11)
    args, unknown = parser.parse_known_args()
    ignored: List[str] = []
    unresolved: List[str] = []
    index = 0
    while index < len(unknown):
        if unknown[index] == "-f" and index + 1 < len(unknown):
            candidate = unknown[index + 1]
            name = Path(candidate).name
            if name.startswith("kernel-") and name.endswith(".json"):
                ignored.extend((unknown[index], candidate))
                index += 2
                continue
        unresolved.append(unknown[index])
        index += 1
    if unresolved:
        parser.error("unrecognized arguments: " + " ".join(unresolved))
    setattr(args, "_ignored_notebook_arguments", ignored)
    return args


def main() -> int:
    args = parse_args()
    if args._ignored_notebook_arguments:
        print(
            f"[notice] ignored notebook arguments: "
            f"{args._ignored_notebook_arguments}"
        )
    if args.quick:
        args.teacher_epochs = min(args.teacher_epochs, 100)
        args.steps = min(args.steps, 5)
        args.train_size = min(args.train_size, 256)
        args.test_size = min(args.test_size, 256)
        args.dense_probe_size = min(args.dense_probe_size, 48)
        args.heldout_size = min(args.heldout_size, 48)
        args.robust_size = min(args.robust_size, 64)
    if args.base_probe_size > args.dense_probe_size:
        raise ValueError("base_probe_size must not exceed dense_probe_size")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    frozen_prospective_cohort = tuple(seeds) == PROSPECTIVE_SEEDS
    if args.quick:
        seeds = seeds[:1]
    if not seeds:
        raise ValueError("at least one seed is required")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    gates = Gates()
    protocol = {
        "title": "RESPONSE-FIBRE NEURAL RESPONSE-SUFFICIENCY / JET-FILTRATION AUDIT",
        "version": VERSION,
        "formal_interval_arithmetic": False,
        "purpose": "test whether denser value responses or analytic input-direction jets repair held-out task drift in a residual CNN",
        "model": "teacher-trained residual CNN on a no-download synthetic 8x8 bar-orientation task",
        "levels": {
            "R0_logits24": "24 probe logits",
            "R_dense_logits96": "96 probe logits",
            "R_jet_logits24_directions2": "24 logits plus two analytic fixed input-direction derivatives per probe",
        },
        "response_projection": "thin-SVD Euclidean tangent projection and SVD pseudoinverse Newton retraction",
        "primary_claim": "on at least two of three frozen prospective seeds, at least one enriched response closes every level gate and has strictly smaller maximum held-out drift than R0",
        "development_seeds_excluded_from_claims": list(DEVELOPMENT_SEEDS),
        "prospective_seeds_frozen_before_run": list(PROSPECTIVE_SEEDS),
        "frozen_prospective_cohort_used": bool(
            frozen_prospective_cohort and not args.quick
        ),
        "gates": asdict(gates),
        "quick_mode": args.quick,
        "seeds": seeds,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key != "output" and not key.startswith("_")
        },
        "uses_cloud_or_qpu": False,
    }
    protocol_hash = sha256_bytes(canonical_json(protocol))
    print("=" * 112)
    print(f"RESPONSE-FIBRE NEURAL RESPONSE-SUFFICIENCY / JET-FILTRATION AUDIT v{VERSION}")
    print("=" * 112)
    print(json.dumps(protocol, indent=2))
    print(f"protocol_sha256 = {protocol_hash}")

    started = time.time()
    seed_results = []
    for index, seed in enumerate(seeds, 1):
        print(f"\n[seed {index:02d}/{len(seeds):02d}] {seed}", flush=True)
        result = run_seed(seed, args, gates, output)
        seed_results.append(result)
        print(
            f"  filtration={result['filtration_pass']} "
            f"R0={result['R0_maximum_heldout_drift']:.3e} "
            f"best={result['best_enriched_level']} "
            f"enriched={result['best_enriched_maximum_heldout_drift']:.3e} "
            f"factor={result['heldout_drift_improvement_factor']}",
            flush=True,
        )

    passing = sum(bool(result["filtration_pass"]) for result in seed_results)
    fraction = passing / len(seed_results)
    cohort_gate = fraction >= gates.minimum_successful_seed_fraction
    claim_eligible = bool(
        not args.quick and frozen_prospective_cohort
    )
    supported = bool(claim_eligible and cohort_gate)
    if args.quick:
        status = "NEURAL_RESPONSE_JET_FILTRATION_QUICK_DIAGNOSTIC_COMPLETE"
    elif not claim_eligible:
        status = "NEURAL_RESPONSE_JET_FILTRATION_CUSTOM_COHORT_DIAGNOSTIC_COMPLETE"
    elif supported:
        status = "NEURAL_RESPONSE_JET_FILTRATION_SUPPORTED"
    else:
        status = "NEURAL_RESPONSE_JET_FILTRATION_NOT_SUPPORTED"

    report = {
        "scientific_status": status,
        "all_gates_pass": supported,
        "formal_interval_arithmetic": False,
        "neural_response_jet_filtration_claimed": supported,
        "protocol_sha256": protocol_hash,
        "claim_eligible_frozen_prospective_cohort": claim_eligible,
        "seeds_declared": len(seeds),
        "seeds_passing": passing,
        "successful_seed_fraction": fraction,
        "cohort_gate_pass": cohort_gate,
        "seed_results": seed_results,
        "elapsed_seconds": time.time() - started,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scope": "small synthetic residual-CNN numerical response-sufficiency audit; not an interval theorem, real-data result, or large-model claim",
    }
    report_hash = sha256_bytes(canonical_json(report))
    report["certificate_sha256_before_self_field"] = report_hash
    (output / "protocol.json").write_bytes(canonical_json(protocol) + b"\n")
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 112)
    print("FINAL RESULT")
    print("=" * 112)
    compact = {key: value for key, value in report.items() if key != "seed_results"}
    print(json.dumps(compact, indent=2))
    if args.quick:
        print("\nQUICK MODE: diagnostic only.")
    elif not claim_eligible:
        print("\nCUSTOM COHORT: diagnostic only; frozen default seeds are required for claims.")
    elif supported:
        print("\nPASS: enriched neural response sufficiency / jet filtration is prospectively supported.")
    else:
        print("\nFAIL-CLOSED: the preregistered enriched-response filtration claim is not supported.")
    return 0


if __name__ == "__main__":
    main()
