#!/usr/bin/env python3
"""Prospective dimension-matched neural task-advantage audit v0.13.0.

This no-download residual-CNN experiment compares two declared response maps
with exactly 72 scalar constraints:

  R_value72 : logits on 72 probe images;
  R_jet72   : logits and two analytic input-direction jets on 24 probes.

For each response map it compares unconstrained descent, a preregistered grid
of soft penalties, ambient descent followed by response retraction, and
intrinsic projected descent followed by response retraction.  Comparisons use
the best response-feasible trajectory checkpoint, not only the final iterate.

The audit separates three claims: task-preserving intrinsic motion,
dimension-matched jet information efficiency, and optimisation advantage over
non-geometric methods.  It is a floating-point prospective experiment, not an
interval theorem, real-data result, or large-model claim.
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


VERSION = "0.13.0"
LEVELS = ("R_value72", "R_jet72")
SOFT_LAMBDAS = (1.0e2, 1.0e3, 1.0e4, 1.0e5)


def soft_method_name(value: float) -> str:
    return f"soft_penalty_lambda_{value:.0e}".replace("+", "")


SOFT_METHODS = tuple(soft_method_name(value) for value in SOFT_LAMBDAS)
METHODS = ("unconstrained", *SOFT_METHODS, "hard_retract", "intrinsic")
NON_GEOMETRIC_METHODS = ("unconstrained", *SOFT_METHODS)
DEVELOPMENT_SEEDS = (
    20260801, 20260817, 20260829,
    20260907, 20260919, 20261003,
    20261107, 20261119, 20261203,
    20270107, 20270119, 20270203,
)
PROSPECTIVE_SEEDS = (20270307, 20270319, 20270403)


@dataclass(frozen=True)
class Gates:
    minimum_teacher_clean_accuracy: float = 0.95
    minimum_teacher_noisy_accuracy: float = 0.85
    maximum_declared_response_relative_drift: float = 1.0e-3
    maximum_heldout_response_relative_drift: float = 1.0e-3
    minimum_secondary_objective_reduction: float = 1.0e-4
    minimum_advantage_ratio_over_best_feasible_baseline: float = 1.25
    minimum_jet_heldout_improvement_factor: float = 1.05
    minimum_jet_reduction_retention_fraction: float = 0.75
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
    return torch.mean((flat(theta, noisy_inputs) - flat(theta, clean_inputs)) ** 2)


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
    probe72: torch.Tensor,
    direction1: torch.Tensor,
    direction2: torch.Tensor,
):
    if level == "R_value72":
        return lambda theta: flat(theta, probe72)
    if level == "R_jet72":
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


def checkpoint_is_task_feasible(
    row: Dict[str, object],
    initial: Dict[str, object],
    gates: Gates,
) -> bool:
    return bool(
        math.isfinite(float(row["secondary_objective"]))
        and float(row["declared_response_relative_drift"])
        <= gates.maximum_declared_response_relative_drift
        and float(row["heldout_response_relative_drift"])
        <= gates.maximum_heldout_response_relative_drift
        and float(row["clean_accuracy"]) - float(initial["clean_accuracy"])
        >= -gates.maximum_clean_accuracy_drop
        and float(row["noisy_accuracy"]) - float(initial["noisy_accuracy"])
        >= gates.minimum_noisy_accuracy_gain
    )


def optimise_method(
    level: str,
    method: str,
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
    learning_rate: float,
    maximum_step_radius: float,
    soft_lambda: float,
    retract_iterations: int,
    response_absolute_tolerance: float,
    gates: Gates,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    target = response_fn(theta0).detach()
    heldout_target = heldout_fn(theta0).detach()
    theta = theta0.detach().clone()
    trajectory: List[Dict[str, object]] = []
    total_retractions = 0

    def record(step: int) -> None:
        with torch.no_grad():
            clean_logits = flat(theta, x_test)
            noisy_logits = flat(theta, x_test_noisy)
            trajectory.append(
                {
                    "level": level,
                    "method": method,
                    "step": step,
                    "secondary_objective": float(
                        secondary_objective(flat, theta, x_robust, x_robust_noisy)
                    ),
                    "declared_response_relative_drift": relative_drift(
                        response_fn(theta), target
                    ),
                    "heldout_response_relative_drift": relative_drift(
                        heldout_fn(theta), heldout_target
                    ),
                    "clean_accuracy": accuracy(clean_logits, y_test),
                    "noisy_accuracy": accuracy(noisy_logits, y_test),
                    "ece": ece(clean_logits, y_test),
                }
            )

    record(0)
    for step in range(1, steps + 1):
        variable = theta.detach().requires_grad_(True)
        objective = secondary_objective(flat, variable, x_robust, x_robust_noisy)
        if method.startswith("soft_penalty_lambda_"):
            residual = response_fn(variable) - target
            objective = objective + soft_lambda * torch.mean(residual ** 2)
        gradient = torch.autograd.grad(objective, variable)[0]
        if method == "intrinsic":
            jacobian = response_jacobian(response_fn, variable)
            projected, _ = svd_tangent_projection(jacobian, gradient)
            direction = -projected
        else:
            direction = -gradient
        proposal = (
            theta
            + bounded_euler_step(
                direction.detach(), learning_rate, maximum_step_radius
            )
        ).detach()
        if method in ("hard_retract", "intrinsic"):
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
    for row in trajectory:
        row["secondary_objective_reduction"] = (
            float(initial["secondary_objective"])
            - float(row["secondary_objective"])
        )
        row["task_feasible"] = checkpoint_is_task_feasible(row, initial, gates)
    feasible_rows = [row for row in trajectory if bool(row["task_feasible"])]
    best = max(
        feasible_rows,
        key=lambda row: float(row["secondary_objective_reduction"]),
    )
    positive_feasible = bool(
        float(best["secondary_objective_reduction"])
        >= gates.minimum_secondary_objective_reduction
    )
    return {
        "level": level,
        "method": method,
        "soft_penalty_lambda": (
            soft_lambda if method.startswith("soft_penalty_lambda_") else None
        ),
        "finite": all(
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
        ),
        "best_feasible_checkpoint_exists": bool(feasible_rows),
        "positive_feasible_checkpoint_exists": positive_feasible,
        "best_feasible_step": int(best["step"]),
        "best_feasible_secondary_objective_reduction": float(
            best["secondary_objective_reduction"]
        ),
        "best_feasible_declared_response_relative_drift": float(
            best["declared_response_relative_drift"]
        ),
        "best_feasible_heldout_response_relative_drift": float(
            best["heldout_response_relative_drift"]
        ),
        "best_feasible_clean_accuracy_change": float(best["clean_accuracy"])
        - float(initial["clean_accuracy"]),
        "best_feasible_noisy_accuracy_change": float(best["noisy_accuracy"])
        - float(initial["noisy_accuracy"]),
        "maximum_declared_response_relative_drift": max(
            float(row["declared_response_relative_drift"]) for row in trajectory
        ),
        "maximum_heldout_response_relative_drift": max(
            float(row["heldout_response_relative_drift"]) for row in trajectory
        ),
        "endpoint_secondary_objective_reduction": float(
            trajectory[-1]["secondary_objective_reduction"]
        ),
        "retraction_iterations_total": total_retractions,
    }, trajectory


def plot_seed(rows: List[Dict[str, object]], path: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for level_index, level in enumerate(LEVELS):
        for method in METHODS:
            selected = [
                row for row in rows
                if row["level"] == level and row["method"] == method
            ]
            step = [int(row["step"]) for row in selected]
            axes[level_index, 0].plot(
                step,
                [row["secondary_objective"] for row in selected],
                label=method,
            )
            axes[level_index, 1].semilogy(
                step,
                [max(float(row["declared_response_relative_drift"]), 1e-16)
                 for row in selected],
            )
            axes[level_index, 2].semilogy(
                step,
                [max(float(row["heldout_response_relative_drift"]), 1e-16)
                 for row in selected],
            )
        axes[level_index, 0].set_ylabel(level)
    for column, title in enumerate(
        ("Secondary objective", "Declared drift", "Held-out drift")
    ):
        axes[0, column].set_title(title)
    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_seed(seed: int, args, gates: Gates, output: Path) -> Dict[str, object]:
    seed_all(seed)
    dtype = torch.float64
    x_train, y_train = make_task(seed + 11, args.train_size, dtype)
    x_test, y_test = make_task(seed + 29, args.test_size, dtype)
    model = TinyResidualCNN(args.width).to(dtype)
    train_teacher(model, x_train, y_train, args.teacher_epochs, args.teacher_lr)
    flat = FlatModel(model)
    theta0 = flat.flatten().to(dtype)

    probe_generator = torch.Generator().manual_seed(seed + 101)
    robust_generator = torch.Generator().manual_seed(seed + 103)
    test_noise_generator = torch.Generator().manual_seed(seed + 107)
    direction_generator = torch.Generator().manual_seed(seed + 109)
    train_order = torch.randperm(args.train_size, generator=probe_generator)
    probe72 = x_train[train_order[: args.value_probe_size]]
    probe24 = probe72[: args.jet_probe_size]
    heldout_order = torch.randperm(args.test_size, generator=probe_generator)
    x_heldout = x_test[heldout_order[: args.heldout_size]]
    robust_order = torch.randperm(args.train_size, generator=robust_generator)
    x_robust = x_train[robust_order[: args.robust_size]]
    x_robust_noisy = x_robust + args.noise_sigma * torch.randn(
        x_robust.shape, generator=robust_generator, dtype=dtype
    )
    x_test_noisy = x_test + args.noise_sigma * torch.randn(
        x_test.shape, generator=test_noise_generator, dtype=dtype
    )
    direction1, direction2 = normalised_directions(
        direction_generator, probe24.shape, dtype
    )
    heldout_fn = lambda theta: flat(theta, x_heldout)
    teacher_clean_accuracy = accuracy(flat(theta0, x_test), y_test)
    teacher_noisy_accuracy = accuracy(flat(theta0, x_test_noisy), y_test)

    level_results: Dict[str, Dict[str, object]] = {}
    all_rows: List[Dict[str, object]] = []
    for level in LEVELS:
        response_fn = build_response_fn(
            level, flat, probe24, probe72, direction1, direction2
        )
        target = response_fn(theta0).detach()
        initial_jacobian = response_jacobian(response_fn, theta0)
        singular_values = torch.linalg.svdvals(initial_jacobian).detach()
        _, response_rank = svd_tangent_projection(
            initial_jacobian, torch.zeros_like(theta0)
        )
        variable = theta0.detach().requires_grad_(True)
        initial_secondary = secondary_objective(
            flat, variable, x_robust, x_robust_noisy
        )
        initial_gradient = torch.autograd.grad(initial_secondary, variable)[0].detach()
        projected_initial, _ = svd_tangent_projection(
            initial_jacobian, initial_gradient
        )
        projected_norm = float(torch.linalg.vector_norm(projected_initial))
        if projected_norm <= 0.0:
            raise ArithmeticError(f"{level}: zero initial projected gradient")
        learning_rate = args.step_radius / projected_norm
        maximum_step_radius = args.maximum_step_multiplier * args.step_radius

        methods: Dict[str, Dict[str, object]] = {}
        for method in METHODS:
            print(f"  [{seed}] level={level} method={method}", flush=True)
            soft_lambda = (
                SOFT_LAMBDAS[SOFT_METHODS.index(method)]
                if method in SOFT_METHODS else 0.0
            )
            summary, rows = optimise_method(
                level,
                method,
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
                learning_rate,
                maximum_step_radius,
                soft_lambda,
                args.retract_iterations,
                args.response_absolute_tolerance,
                gates,
            )
            methods[method] = summary
            all_rows.extend(rows)

        intrinsic = methods["intrinsic"]
        intrinsic_reduction = float(
            intrinsic["best_feasible_secondary_objective_reduction"]
        )
        eligible_baselines = [
            methods[name] for name in NON_GEOMETRIC_METHODS
            if bool(methods[name]["positive_feasible_checkpoint_exists"])
        ]
        best_baseline_reduction = max(
            [float(item["best_feasible_secondary_objective_reduction"])
             for item in eligible_baselines],
            default=0.0,
        )
        advantage_ratio = (
            intrinsic_reduction / best_baseline_reduction
            if best_baseline_reduction > 0.0 else None
        )
        mechanism_gates = {
            "teacher_clean_task_learned": teacher_clean_accuracy
            >= gates.minimum_teacher_clean_accuracy,
            "teacher_noisy_task_nontrivial": teacher_noisy_accuracy
            >= gates.minimum_teacher_noisy_accuracy,
            "response_dimension_is_72": int(target.numel()) == 72,
            "response_full_row_rank": response_rank == int(target.numel()),
            "intrinsic_positive_feasible_checkpoint_exists": bool(
                intrinsic["positive_feasible_checkpoint_exists"]
            ),
        }
        advantage_gates = {
            "positive_feasible_non_geometric_comparator_exists": bool(
                eligible_baselines
            ),
            "intrinsic_beats_best_feasible_non_geometric_checkpoint": bool(
                advantage_ratio is not None
                and advantage_ratio
                >= gates.minimum_advantage_ratio_over_best_feasible_baseline
            ),
        }
        level_results[level] = {
            "response_dimension": int(target.numel()),
            "response_jacobian_rank": response_rank,
            "response_jacobian_minimum_singular_value": float(singular_values[-1]),
            "response_jacobian_condition_number": float(
                singular_values[0] / singular_values[-1]
            ),
            "local_fibre_dimension": int(theta0.numel() - response_rank),
            "initial_projected_gradient_norm": projected_norm,
            "common_learning_rate": learning_rate,
            "mechanism_gates": mechanism_gates,
            "mechanism_pass": all(mechanism_gates.values()),
            "advantage_gates": advantage_gates,
            "task_advantage_pass": bool(
                all(mechanism_gates.values()) and all(advantage_gates.values())
            ),
            "eligible_non_geometric_methods": [
                item["method"] for item in eligible_baselines
            ],
            "best_feasible_non_geometric_reduction": best_baseline_reduction,
            "intrinsic_advantage_ratio": advantage_ratio,
            "methods": methods,
        }

    value_intrinsic = level_results["R_value72"]["methods"]["intrinsic"]
    jet_intrinsic = level_results["R_jet72"]["methods"]["intrinsic"]
    value_drift = float(
        value_intrinsic["best_feasible_heldout_response_relative_drift"]
    )
    jet_drift = float(
        jet_intrinsic["best_feasible_heldout_response_relative_drift"]
    )
    value_reduction = float(
        value_intrinsic["best_feasible_secondary_objective_reduction"]
    )
    jet_reduction = float(
        jet_intrinsic["best_feasible_secondary_objective_reduction"]
    )
    drift_factor = value_drift / jet_drift if jet_drift > 0.0 else None
    reduction_retention = (
        jet_reduction / value_reduction if value_reduction > 0.0 else None
    )
    information_efficiency_gates = {
        "both_dimension_matched_mechanisms_pass": all(
            bool(level_results[level]["mechanism_pass"]) for level in LEVELS
        ),
        "jet_has_lower_heldout_drift_by_preregistered_factor": bool(
            drift_factor is not None
            and drift_factor >= gates.minimum_jet_heldout_improvement_factor
        ),
        "jet_retains_preregistered_fraction_of_value_descent": bool(
            reduction_retention is not None
            and reduction_retention
            >= gates.minimum_jet_reduction_retention_fraction
        ),
    }
    information_efficiency_pass = all(information_efficiency_gates.values())
    any_level_advantage = any(
        bool(level_results[level]["task_advantage_pass"]) for level in LEVELS
    )

    csv_path = output / f"trajectory_seed_{seed}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    plot_seed(all_rows, output / f"trajectory_seed_{seed}.png")
    return {
        "seed": seed,
        "parameter_dimension": int(theta0.numel()),
        "teacher_clean_accuracy": teacher_clean_accuracy,
        "teacher_noisy_accuracy": teacher_noisy_accuracy,
        "dimension_matched_information_efficiency_pass": information_efficiency_pass,
        "information_efficiency_gates": information_efficiency_gates,
        "value_to_jet_heldout_drift_improvement_factor": drift_factor,
        "jet_to_value_reduction_retention_fraction": reduction_retention,
        "at_least_one_response_level_task_advantage_pass": any_level_advantage,
        "levels": level_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        default="response_fibre_nn_dimension_matched_task_advantage_v0_13_0_results",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in PROSPECTIVE_SEEDS),
        help="overrides are diagnostic; only frozen defaults support claims",
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--value-probe-size", type=int, default=72)
    parser.add_argument("--jet-probe-size", type=int, default=24)
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
        print(f"[notice] ignored notebook arguments: {args._ignored_notebook_arguments}")
    if args.value_probe_size != 72 or 3 * args.jet_probe_size != 72:
        raise ValueError(
            "claim-bearing defaults require value_probe_size=72 and "
            "3*jet_probe_size=72"
        )
    if args.quick:
        args.teacher_epochs = min(args.teacher_epochs, 100)
        args.steps = min(args.steps, 5)
        args.train_size = min(args.train_size, 256)
        args.test_size = min(args.test_size, 256)
        args.heldout_size = min(args.heldout_size, 48)
        args.robust_size = min(args.robust_size, 64)
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
        "title": "RESPONSE-FIBRE DIMENSION-MATCHED NEURAL TASK-ADVANTAGE AUDIT",
        "version": VERSION,
        "formal_interval_arithmetic": False,
        "purpose": "test dimension-matched value-versus-jet response efficiency and intrinsic optimisation advantage on a residual CNN",
        "model": "teacher-trained residual CNN on a no-download synthetic 8x8 bar-orientation task",
        "response_levels": {
            "R_value72": "72 probe logits",
            "R_jet72": "24 probe logits plus two analytic fixed input-direction derivatives per probe",
        },
        "methods": list(METHODS),
        "soft_penalty_lambda_grid": list(SOFT_LAMBDAS),
        "response_projection": "thin-SVD Euclidean tangent projection and SVD-pseudoinverse Newton retraction",
        "comparison_unit": "best response-feasible trajectory checkpoint under common gates",
        "hard_retract_policy": "geometric comparator excluded from the non-geometric advantage baseline",
        "primary_claims": {
            "information_efficiency": "at equal response dimension, R_jet72 lowers held-out drift by at least 1.05x while retaining at least 75% of R_value72 descent",
            "task_advantage": "intrinsic descent beats the best positive response-feasible non-geometric checkpoint by at least 1.25x",
        },
        "development_seeds_excluded_from_claims": list(DEVELOPMENT_SEEDS),
        "prospective_seeds_frozen_before_run": list(PROSPECTIVE_SEEDS),
        "frozen_prospective_cohort_used": bool(
            frozen_prospective_cohort and not args.quick
        ),
        "gates": asdict(gates),
        "quick_mode": args.quick,
        "seeds": seeds,
        "arguments": {
            key: value for key, value in vars(args).items()
            if key != "output" and not key.startswith("_")
        },
        "uses_cloud_or_qpu": False,
    }
    protocol_hash = sha256_bytes(canonical_json(protocol))
    print("=" * 112)
    print(f"RESPONSE-FIBRE DIMENSION-MATCHED NEURAL TASK-ADVANTAGE AUDIT v{VERSION}")
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
            f"  information_efficiency="
            f"{result['dimension_matched_information_efficiency_pass']} "
            f"task_advantage="
            f"{result['at_least_one_response_level_task_advantage_pass']} "
            f"drift_factor="
            f"{result['value_to_jet_heldout_drift_improvement_factor']} "
            f"reduction_retention="
            f"{result['jet_to_value_reduction_retention_fraction']}",
            flush=True,
        )

    information_passing = sum(
        bool(item["dimension_matched_information_efficiency_pass"])
        for item in seed_results
    )
    advantage_passing = sum(
        bool(item["at_least_one_response_level_task_advantage_pass"])
        for item in seed_results
    )
    information_fraction = information_passing / len(seed_results)
    advantage_fraction = advantage_passing / len(seed_results)
    information_cohort_pass = (
        information_fraction >= gates.minimum_successful_seed_fraction
    )
    advantage_cohort_pass = (
        advantage_fraction >= gates.minimum_successful_seed_fraction
    )
    claim_eligible = bool(not args.quick and frozen_prospective_cohort)
    information_supported = bool(claim_eligible and information_cohort_pass)
    advantage_supported = bool(claim_eligible and advantage_cohort_pass)
    all_supported = information_supported and advantage_supported
    if args.quick:
        status = "DIMENSION_MATCHED_NEURAL_TASK_ADVANTAGE_QUICK_DIAGNOSTIC_COMPLETE"
    elif not claim_eligible:
        status = "DIMENSION_MATCHED_NEURAL_TASK_ADVANTAGE_CUSTOM_COHORT_DIAGNOSTIC_COMPLETE"
    elif all_supported:
        status = "DIMENSION_MATCHED_NEURAL_RESPONSE_EFFICIENCY_AND_TASK_ADVANTAGE_SUPPORTED"
    elif information_supported:
        status = "DIMENSION_MATCHED_NEURAL_RESPONSE_EFFICIENCY_SUPPORTED_TASK_ADVANTAGE_UNRESOLVED"
    elif advantage_supported:
        status = "NEURAL_TASK_ADVANTAGE_SUPPORTED_DIMENSION_MATCHED_RESPONSE_EFFICIENCY_UNRESOLVED"
    else:
        status = "DIMENSION_MATCHED_NEURAL_RESPONSE_EFFICIENCY_AND_TASK_ADVANTAGE_NOT_SUPPORTED"

    report = {
        "scientific_status": status,
        "all_gates_pass": all_supported,
        "formal_interval_arithmetic": False,
        "dimension_matched_neural_response_information_efficiency_claimed": information_supported,
        "neural_network_task_advantage_claimed": advantage_supported,
        "protocol_sha256": protocol_hash,
        "claim_eligible_frozen_prospective_cohort": claim_eligible,
        "seeds_declared": len(seeds),
        "information_efficiency_seeds_passing": information_passing,
        "information_efficiency_successful_seed_fraction": information_fraction,
        "information_efficiency_cohort_gate_pass": information_cohort_pass,
        "task_advantage_seeds_passing": advantage_passing,
        "task_advantage_successful_seed_fraction": advantage_fraction,
        "task_advantage_cohort_gate_pass": advantage_cohort_pass,
        "seed_results": seed_results,
        "elapsed_seconds": time.time() - started,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scope": "small synthetic residual-CNN numerical audit; not an interval theorem, real-data result, or large-model claim",
    }
    report_hash = sha256_bytes(canonical_json(report))
    report["certificate_sha256_before_self_field"] = report_hash
    (output / "protocol.json").write_bytes(canonical_json(protocol) + b"\n")
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 112)
    print("FINAL RESULT")
    print("=" * 112)
    compact = {key: value for key, value in report.items() if key != "seed_results"}
    print(json.dumps(compact, indent=2))
    if args.quick:
        print("\nQUICK MODE: diagnostic only.")
    elif not claim_eligible:
        print("\nCUSTOM COHORT: diagnostic only; frozen defaults are required for claims.")
    elif all_supported:
        print("\nPASS: dimension-matched response efficiency and task advantage are prospectively supported.")
    elif information_supported:
        print("\nPARTIAL PASS: dimension-matched jet efficiency is supported; task advantage remains unresolved.")
    elif advantage_supported:
        print("\nPARTIAL PASS: task advantage is supported; jet information efficiency remains unresolved.")
    else:
        print("\nFAIL-CLOSED: neither preregistered claim reached the cohort gate.")
    return 0


if __name__ == "__main__":
    main()
