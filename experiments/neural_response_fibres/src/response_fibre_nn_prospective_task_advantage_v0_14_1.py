#!/usr/bin/env python3
"""Prospective fair-baseline neural task-advantage audit v0.14.1.

The baseline architecture was qualified on the exposed v0.13/v0.14.0
development seeds before the three seeds in this file were assigned. Every
non-geometric method uses the frozen declared-response backtracking line
search, and augmented-Lagrangian baselines are included.

The line search may inspect only the declared response and the secondary
objective.  Held-out responses and task labels are audit-only quantities and
never participate in step acceptance.  Results identify the best checkpoint
on the task-feasible Pareto set.

R_value72 and R_jet72 are co-primary response levels. Each level must support
intrinsic advantage on at least two of the three frozen prospective seeds;
the headline claim requires both cohort gates. Overrides and quick mode are
diagnostic only. This is a floating-point experiment, not an interval theorem
or a large-model claim.
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


VERSION = "0.14.1"
LEVELS = ("R_value72", "R_jet72")
DEVELOPMENT_SEEDS = (
    20260801, 20260817, 20260829,
    20260907, 20260919, 20261003,
    20261107, 20261119, 20261203,
    20270107, 20270119, 20270203,
    20270307, 20270319, 20270403,
)
PROSPECTIVE_SEEDS = (20270507, 20270519, 20270603)
FROZEN_CLAIM_ARGUMENTS = {
    "width": 8,
    "train_size": 512,
    "test_size": 512,
    "heldout_size": 96,
    "robust_size": 128,
    "teacher_epochs": 300,
    "teacher_lr": 2.0e-2,
    "steps": 40,
    "step_radius": 2.0e-3,
    "maximum_step_multiplier": 4.0,
    "noise_sigma": 0.12,
    "retract_iterations": 5,
    "response_absolute_tolerance": 1.0e-11,
    "backtracking_factor": 0.5,
    "maximum_backtracking_trials": 20,
    "minimum_objective_decrease": 1.0e-14,
}
SOFT_LAMBDAS = (1.0e2, 1.0e3, 1.0e4, 1.0e5)
AUGMENTED_RHOS = (1.0e2, 1.0e4)


def numbered_name(prefix: str, value: float) -> str:
    return f"{prefix}_{value:.0e}".replace("+", "")


SOFT_METHODS = tuple(
    numbered_name("soft_penalty_backtracking_lambda", value)
    for value in SOFT_LAMBDAS
)
AUGMENTED_METHODS = tuple(
    numbered_name("augmented_lagrangian_backtracking_rho", value)
    for value in AUGMENTED_RHOS
)
NON_GEOMETRIC_METHODS = (
    "unconstrained_backtracking",
    *SOFT_METHODS,
    *AUGMENTED_METHODS,
)
METHODS = (*NON_GEOMETRIC_METHODS, "hard_retract", "intrinsic")


@dataclass(frozen=True)
class Gates:
    minimum_teacher_clean_accuracy: float = 0.95
    minimum_teacher_noisy_accuracy: float = 0.85
    maximum_declared_response_relative_drift: float = 1.0e-3
    maximum_heldout_response_relative_drift: float = 1.0e-3
    minimum_secondary_objective_reduction: float = 1.0e-4
    minimum_diagnostic_advantage_ratio: float = 1.25
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
    clean = flat(theta, clean_inputs)
    noisy = flat(theta, noisy_inputs)
    return torch.mean((noisy - clean) ** 2)


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


def bounded_step(
    direction: torch.Tensor,
    learning_rate: float,
    maximum_radius: float,
) -> torch.Tensor:
    step = learning_rate * direction
    norm = torch.linalg.vector_norm(step)
    norm_value = float(norm.detach())
    if not math.isfinite(norm_value) or norm_value == 0.0:
        return torch.zeros_like(direction)
    if norm_value > maximum_radius:
        step = step * (maximum_radius / norm_value)
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


def method_hyperparameter(method: str) -> float:
    if method in SOFT_METHODS:
        return SOFT_LAMBDAS[SOFT_METHODS.index(method)]
    if method in AUGMENTED_METHODS:
        return AUGMENTED_RHOS[AUGMENTED_METHODS.index(method)]
    return 0.0


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
    hyperparameter: float,
    retract_iterations: int,
    response_absolute_tolerance: float,
    backtracking_factor: float,
    maximum_backtracking_trials: int,
    minimum_objective_decrease: float,
    gates: Gates,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    target = response_fn(theta0).detach()
    heldout_target = heldout_fn(theta0).detach()
    theta = theta0.detach().clone()
    dual = torch.zeros_like(target)
    trajectory: List[Dict[str, object]] = []
    total_retractions = 0
    total_backtracking_trials = 0
    accepted_steps = 0
    stalled = False

    def record(step: int, accepted_scale: float, trials: int) -> None:
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
                    "accepted_step_scale": accepted_scale,
                    "backtracking_trials": trials,
                }
            )

    record(0, 0.0, 0)
    for step_index in range(1, steps + 1):
        variable = theta.detach().requires_grad_(True)
        secondary = secondary_objective(flat, variable, x_robust, x_robust_noisy)
        residual = response_fn(variable) - target
        objective = secondary
        if method in SOFT_METHODS:
            objective = objective + 0.5 * hyperparameter * torch.mean(residual ** 2)
        elif method in AUGMENTED_METHODS:
            objective = (
                objective
                + torch.mean(dual * residual)
                + 0.5 * hyperparameter * torch.mean(residual ** 2)
            )
        gradient = torch.autograd.grad(objective, variable)[0].detach()
        if method == "intrinsic":
            jacobian = response_jacobian(response_fn, variable)
            projected, _ = svd_tangent_projection(jacobian, gradient)
            direction = -projected
        else:
            direction = -gradient
        base_step = bounded_step(direction, learning_rate, maximum_step_radius)
        current_secondary = float(secondary.detach())

        if method in NON_GEOMETRIC_METHODS:
            accepted = False
            accepted_scale = 0.0
            used_trials = 0
            proposal = theta
            for trial in range(maximum_backtracking_trials + 1):
                used_trials = trial + 1
                scale = backtracking_factor ** trial
                candidate = (theta + scale * base_step).detach()
                with torch.no_grad():
                    candidate_declared = relative_drift(
                        response_fn(candidate), target
                    )
                    candidate_secondary = float(
                        secondary_objective(
                            flat, candidate, x_robust, x_robust_noisy
                        )
                    )
                if (
                    candidate_declared
                    <= gates.maximum_declared_response_relative_drift
                    and candidate_secondary
                    <= current_secondary - minimum_objective_decrease
                ):
                    accepted = True
                    accepted_scale = scale
                    proposal = candidate
                    break
            total_backtracking_trials += used_trials
            if not accepted:
                stalled = True
                record(step_index, 0.0, used_trials)
                break
            theta = proposal
            accepted_steps += 1
            if method in AUGMENTED_METHODS:
                with torch.no_grad():
                    dual = dual + hyperparameter * (response_fn(theta) - target)
            record(step_index, accepted_scale, used_trials)
            continue

        proposal = (theta + base_step).detach()
        proposal, used, _ = retract_to_response(
            proposal,
            response_fn,
            target,
            retract_iterations,
            response_absolute_tolerance,
        )
        total_retractions += used
        theta = proposal
        accepted_steps += 1
        record(step_index, 1.0, 1)

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
    best_reduction = float(best["secondary_objective_reduction"])
    positive_feasible = best_reduction >= gates.minimum_secondary_objective_reduction
    return {
        "level": level,
        "method": method,
        "hyperparameter": hyperparameter if hyperparameter > 0.0 else None,
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
        "algorithm_uses_heldout_for_step_acceptance": False,
        "accepted_steps": accepted_steps,
        "stalled_before_declared_steps": stalled,
        "total_backtracking_trials": total_backtracking_trials,
        "mean_backtracking_trials_per_accepted_step": (
            total_backtracking_trials / accepted_steps
            if accepted_steps > 0 else None
        ),
        "retraction_iterations_total": total_retractions,
        "positive_task_feasible_checkpoint_exists": positive_feasible,
        "best_task_feasible_step": int(best["step"]),
        "best_task_feasible_secondary_reduction": best_reduction,
        "best_task_feasible_declared_drift": float(
            best["declared_response_relative_drift"]
        ),
        "best_task_feasible_heldout_drift": float(
            best["heldout_response_relative_drift"]
        ),
        "best_task_feasible_clean_accuracy_change": float(best["clean_accuracy"])
        - float(initial["clean_accuracy"]),
        "best_task_feasible_noisy_accuracy_change": float(best["noisy_accuracy"])
        - float(initial["noisy_accuracy"]),
        "maximum_declared_drift": max(
            float(row["declared_response_relative_drift"]) for row in trajectory
        ),
        "maximum_heldout_drift": max(
            float(row["heldout_response_relative_drift"]) for row in trajectory
        ),
    }, trajectory


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
    noise_generator = torch.Generator().manual_seed(seed + 107)
    direction_generator = torch.Generator().manual_seed(seed + 109)
    train_order = torch.randperm(args.train_size, generator=probe_generator)
    probe72 = x_train[train_order[:72]]
    probe24 = probe72[:24]
    heldout_order = torch.randperm(args.test_size, generator=probe_generator)
    x_heldout = x_test[heldout_order[: args.heldout_size]]
    robust_order = torch.randperm(args.train_size, generator=robust_generator)
    x_robust = x_train[robust_order[: args.robust_size]]
    x_robust_noisy = x_robust + args.noise_sigma * torch.randn(
        x_robust.shape, generator=robust_generator, dtype=dtype
    )
    x_test_noisy = x_test + args.noise_sigma * torch.randn(
        x_test.shape, generator=noise_generator, dtype=dtype
    )
    direction1, direction2 = normalised_directions(
        direction_generator, probe24.shape, dtype
    )
    heldout_fn = lambda theta: flat(theta, x_heldout)
    teacher_clean = accuracy(flat(theta0, x_test), y_test)
    teacher_noisy = accuracy(flat(theta0, x_test_noisy), y_test)

    all_rows: List[Dict[str, object]] = []
    levels: Dict[str, Dict[str, object]] = {}
    for level in LEVELS:
        response_fn = build_response_fn(
            level, flat, probe24, probe72, direction1, direction2
        )
        target = response_fn(theta0).detach()
        initial_jacobian = response_jacobian(response_fn, theta0)
        singular_values = torch.linalg.svdvals(initial_jacobian).detach()
        _, rank = svd_tangent_projection(
            initial_jacobian, torch.zeros_like(theta0)
        )
        variable = theta0.detach().requires_grad_(True)
        initial_secondary = secondary_objective(
            flat, variable, x_robust, x_robust_noisy
        )
        gradient = torch.autograd.grad(initial_secondary, variable)[0].detach()
        projected, _ = svd_tangent_projection(initial_jacobian, gradient)
        projected_norm = float(torch.linalg.vector_norm(projected))
        if projected_norm <= 0.0:
            raise ArithmeticError(f"{level}: zero projected gradient")
        learning_rate = args.step_radius / projected_norm
        maximum_radius = args.maximum_step_multiplier * args.step_radius

        summaries: Dict[str, Dict[str, object]] = {}
        for method in METHODS:
            print(f"  [{seed}] level={level} method={method}", flush=True)
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
                maximum_radius,
                method_hyperparameter(method),
                args.retract_iterations,
                args.response_absolute_tolerance,
                args.backtracking_factor,
                args.maximum_backtracking_trials,
                args.minimum_objective_decrease,
                gates,
            )
            summaries[method] = summary
            all_rows.extend(rows)

        eligible = [
            summaries[name] for name in NON_GEOMETRIC_METHODS
            if bool(summaries[name]["positive_task_feasible_checkpoint_exists"])
        ]
        best_baseline = max(
            [float(item["best_task_feasible_secondary_reduction"])
             for item in eligible],
            default=0.0,
        )
        intrinsic_reduction = float(
            summaries["intrinsic"]["best_task_feasible_secondary_reduction"]
        )
        ratio = (
            intrinsic_reduction / best_baseline if best_baseline > 0.0 else None
        )
        qualification_gates = {
            "teacher_clean_task_learned": teacher_clean
            >= gates.minimum_teacher_clean_accuracy,
            "teacher_noisy_task_nontrivial": teacher_noisy
            >= gates.minimum_teacher_noisy_accuracy,
            "response_is_full_rank_72": rank == 72 and int(target.numel()) == 72,
            "intrinsic_positive_task_feasible_checkpoint_exists": bool(
                summaries["intrinsic"]["positive_task_feasible_checkpoint_exists"]
            ),
            "positive_task_feasible_non_geometric_comparator_exists": bool(eligible),
            "heldout_never_used_by_any_algorithm": all(
                not bool(summary["algorithm_uses_heldout_for_step_acceptance"])
                for summary in summaries.values()
            ),
        }
        advantage_gates = {
            "fair_baseline_architecture_qualified_on_this_seed": all(
                qualification_gates.values()
            ),
            "intrinsic_beats_best_task_feasible_non_geometric_checkpoint": bool(
                ratio is not None
                and ratio >= gates.minimum_diagnostic_advantage_ratio
            ),
        }
        levels[level] = {
            "response_dimension": int(target.numel()),
            "response_jacobian_rank": rank,
            "response_jacobian_minimum_singular_value": float(singular_values[-1]),
            "response_jacobian_condition_number": float(
                singular_values[0] / singular_values[-1]
            ),
            "local_fibre_dimension": int(theta0.numel() - rank),
            "qualification_gates": qualification_gates,
            "baseline_qualification_pass": all(qualification_gates.values()),
            "eligible_non_geometric_methods": [item["method"] for item in eligible],
            "best_non_geometric_reduction": best_baseline,
            "intrinsic_reduction": intrinsic_reduction,
            "diagnostic_intrinsic_advantage_ratio": ratio,
            "diagnostic_ratio_reaches_1_25": bool(
                ratio is not None
                and ratio >= gates.minimum_diagnostic_advantage_ratio
            ),
            "task_advantage_gates": advantage_gates,
            "task_advantage_pass": all(advantage_gates.values()),
            "methods": summaries,
        }

    csv_path = output / f"trajectory_seed_{seed}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    return {
        "seed": seed,
        "parameter_dimension": int(theta0.numel()),
        "teacher_clean_accuracy": teacher_clean,
        "teacher_noisy_accuracy": teacher_noisy,
        "both_response_levels_baseline_qualified": all(
            bool(levels[level]["baseline_qualification_pass"])
            for level in LEVELS
        ),
        "both_response_levels_task_advantage_pass": all(
            bool(levels[level]["task_advantage_pass"])
            for level in LEVELS
        ),
        "levels": levels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        default="response_fibre_nn_prospective_task_advantage_v0_14_1_results",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in PROSPECTIVE_SEEDS),
        help="frozen prospective seeds; every override is diagnostic only",
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
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
    parser.add_argument("--backtracking-factor", type=float, default=0.5)
    parser.add_argument("--maximum-backtracking-trials", type=int, default=20)
    parser.add_argument("--minimum-objective-decrease", type=float, default=1.0e-14)
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
    if not (0.0 < args.backtracking_factor < 1.0):
        raise ValueError("backtracking_factor must lie strictly between zero and one")
    if args.maximum_backtracking_trials < 1:
        raise ValueError("maximum_backtracking_trials must be positive")
    if args.quick:
        args.teacher_epochs = min(args.teacher_epochs, 100)
        args.steps = min(args.steps, 5)
        args.train_size = min(args.train_size, 256)
        args.test_size = min(args.test_size, 256)
        args.heldout_size = min(args.heldout_size, 48)
        args.robust_size = min(args.robust_size, 64)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if args.quick:
        # Never consume a prospective seed in a smoke test.  Quick mode is
        # forcibly redirected to the last already-exposed development seed.
        seeds = [DEVELOPMENT_SEEDS[-1]]
    if not seeds:
        raise ValueError("at least one seed is required")
    claim_eligible = bool(
        not args.quick
        and tuple(seeds) == PROSPECTIVE_SEEDS
        and all(getattr(args, key) == value for key, value in FROZEN_CLAIM_ARGUMENTS.items())
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    gates = Gates()
    protocol = {
        "title": "RESPONSE-FIBRE PROSPECTIVE FAIR-BASELINE NEURAL TASK-ADVANTAGE AUDIT",
        "version": VERSION,
        "formal_interval_arithmetic": False,
        "purpose": "prospectively test intrinsic response-fibre task advantage against prequalified fair non-geometric comparators",
        "model": "teacher-trained residual CNN on a no-download synthetic 8x8 bar-orientation task",
        "response_levels": {
            "R_value72": "72 probe logits",
            "R_jet72": "24 logits plus two analytic input-direction derivatives per probe",
        },
        "methods": list(METHODS),
        "soft_penalty_lambda_grid": list(SOFT_LAMBDAS),
        "augmented_lagrangian_rho_grid": list(AUGMENTED_RHOS),
        "line_search_policy": "backtrack the non-geometric direction until declared response remains within budget and the secondary objective strictly decreases",
        "heldout_firewall": "held-out responses and labels are recorded only after step acceptance and never enter the line search",
        "comparison_unit": "best task-feasible checkpoint on each method trajectory",
        "co_primary_claim_policy": "R_value72 and R_jet72 each require at least two of three passing seeds; the headline claim requires both cohort gates",
        "development_seeds_excluded_from_claims": list(DEVELOPMENT_SEEDS),
        "prospective_seeds_frozen_before_run": list(PROSPECTIVE_SEEDS),
        "source_v0140_protocol_sha256": "83e88e44fd207be14950c9005f93a0ef2bb7cc97549dd38967e24a0cce44ffc4",
        "source_v0140_certificate_sha256": "ad71c665c30dd8e697da125c06c9e79cbf356be7ca8b59b00a7fbc288ee7908e",
        "fair_baseline_architecture_frozen_from_v0140": True,
        "claim_eligible_frozen_prospective_cohort": claim_eligible,
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
    print(f"RESPONSE-FIBRE PROSPECTIVE FAIR-BASELINE TASK-ADVANTAGE AUDIT v{VERSION}")
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
            f"  both_levels_advantage="
            f"{result['both_response_levels_task_advantage_pass']}",
            flush=True,
        )
        for level in LEVELS:
            item = result["levels"][level]
            print(
                f"    {level}: pass={item['task_advantage_pass']} "
                f"eligible={item['eligible_non_geometric_methods']} "
                f"ratio={item['diagnostic_intrinsic_advantage_ratio']}",
                flush=True,
            )

    level_counts = {
        level: sum(bool(item["levels"][level]["task_advantage_pass"]) for item in seed_results)
        for level in LEVELS
    }
    level_fractions = {
        level: level_counts[level] / len(seed_results) for level in LEVELS
    }
    level_cohort_gates = {
        level: level_fractions[level] >= gates.minimum_successful_seed_fraction
        for level in LEVELS
    }
    both_co_primary_gates = all(level_cohort_gates.values())
    supported = bool(claim_eligible and both_co_primary_gates)
    if supported:
        status = "PROSPECTIVE_NEURAL_RESPONSE_FIBRE_TASK_ADVANTAGE_SUPPORTED"
    elif claim_eligible:
        status = "PROSPECTIVE_NEURAL_RESPONSE_FIBRE_TASK_ADVANTAGE_NOT_SUPPORTED"
    else:
        status = "NEURAL_RESPONSE_FIBRE_TASK_ADVANTAGE_DIAGNOSTIC_ONLY"
    report = {
        "scientific_status": status,
        "all_gates_pass": supported,
        "formal_interval_arithmetic": False,
        "claim_eligible_frozen_prospective_cohort": claim_eligible,
        "neural_response_fibre_task_advantage_claimed": supported,
        "protocol_sha256": protocol_hash,
        "seeds_declared": len(seeds),
        "co_primary_level_seed_counts": level_counts,
        "co_primary_level_successful_seed_fractions": level_fractions,
        "co_primary_level_cohort_gates": level_cohort_gates,
        "both_co_primary_cohort_gates_pass": both_co_primary_gates,
        "seed_results": seed_results,
        "elapsed_seconds": time.time() - started,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "next_required_step": (
            "Freeze the report and publish the full v0.12-v0.14.1 positive/negative audit chain without rerunning or retuning."
            if supported
            else "Report the frozen prospective outcome without retuning on these seeds; any new hypothesis requires a separately preregistered cohort."
        ),
        "scope": "frozen prospective small synthetic residual-CNN numerical audit; not an interval theorem, real-data result, continuous-time neural ODE theorem, or large-model result",
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
    if supported:
        print("\nPASS: both co-primary neural response-fibre advantage gates close prospectively.")
    elif claim_eligible:
        print("\nFAIL-CLOSED: the frozen prospective task-advantage claim is not supported.")
    else:
        print("\nDIAGNOSTIC ONLY: quick mode or argument/seed overrides are not claim eligible.")
    return 0


if __name__ == "__main__":
    main()
