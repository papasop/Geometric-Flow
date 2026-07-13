"""Matched functional-step LoRA benchmark for Phase G.

Phase G asks whether the task gap shrinks when functional GeoFlow is compared
at matched functional displacement instead of equal raw parameter learning
rate. The original Phase F LoRA benchmark remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import GeometricOptimizer
from geometric_flow._tensor import assign_flat_update, get_flat_params, set_flat_params, trainable_params
from geometric_flow.functional_geometry import FunctionalMap, functional_projectors, projected_functional_geoflow_direction
from experiments.lora_reparameterization_benchmark import SmallLoRAMLP, make_data, make_transform


OPTIMIZERS = [
    "adamw",
    "diagonal_grad_square",
    "functional_geoflow_fixed_lr",
    "functional_geoflow_matched_step",
]


@dataclass
class StepRow:
    seed: int
    optimizer: str
    representation: int
    step: int
    train_scope: str
    functional_map: str
    train_loss_before: float
    train_loss_after: float
    functional_step_norm: float
    parameter_step_norm: float
    tangent_step_norm: float
    normal_step_norm: float
    tangent_normal_ratio: float
    accepted_step_scale: float
    requested_lr: float
    effective_lr: float
    calibration_target: float
    calibration_error: float
    descent_gate_passed: bool
    fallback: bool
    solver_residual: float
    null_leakage: float
    jvp_count: int
    vjp_count: int
    cache_hit: bool
    cache_age: int
    basis_rank: int
    cg_iterations: int
    wall_clock_step: float
    peak_memory_bytes: int


@dataclass
class RunRow:
    seed: int
    optimizer: str
    representation: int
    train_scope: str
    functional_map: str
    initial_equivalence_residual: float
    final_loss: float
    final_accuracy: float
    final_phi: str
    mean_functional_step: float
    median_functional_step: float
    mean_parameter_step: float
    mean_tangent_step: float
    mean_normal_step: float
    mean_calibration_error: float
    mean_null_leakage: float
    total_jvp: int
    total_vjp: int
    mean_wall_clock: float
    peak_memory_bytes: int
    tangent_drift: float
    near_null_amplification: float
    seconds: float


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or value == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_train_scope(model: SmallLoRAMLP, train_scope: str) -> None:
    if train_scope not in {"lora_only", "head_only", "lora_and_head"}:
        raise ValueError(f"unknown train_scope: {train_scope}")
    for param in model.parameters():
        param.requires_grad_(False)
    if train_scope in {"lora_only", "lora_and_head"}:
        model.lora.a.requires_grad_(True)
        model.lora.b.requires_grad_(True)
    if train_scope in {"head_only", "lora_and_head"}:
        model.head.weight.requires_grad_(True)
        model.head.bias.requires_grad_(True)


def representation_fn_for(mode: str):
    def representation_fn(model: SmallLoRAMLP, x: torch.Tensor, params: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if params is None:
            return model.functional_representation(x, mode)
        a = params.get("lora.a", model.lora.a)
        b = params.get("lora.b", model.lora.b)
        head_weight = params.get("head.weight", model.head.weight)
        head_bias = params.get("head.bias", model.head.bias)
        lora_output = F.linear(x, model.lora.base_weight + b @ a)
        hidden = torch.tanh(lora_output)
        logits = F.linear(hidden, head_weight, head_bias)
        if mode == "logits":
            return logits
        if mode == "lora_output":
            return lora_output
        if mode == "hidden":
            return hidden
        if mode == "logits_hidden":
            return torch.cat([logits.reshape(-1), hidden.reshape(-1)])
        raise ValueError(f"unknown functional map: {mode}")

    return representation_fn


def phi(model: SmallLoRAMLP, probe: torch.Tensor, mode: str) -> torch.Tensor:
    return model.functional_representation(probe, mode).reshape(-1).detach()


def make_fmap(model: SmallLoRAMLP, probe: torch.Tensor, mode: str) -> FunctionalMap:
    return FunctionalMap(model, probe, representation_fn=representation_fn_for(mode))


def make_projectors(model: SmallLoRAMLP, probe: torch.Tensor, mode: str):
    fmap = make_fmap(model, probe, mode)
    fjac = fmap.jacobian()
    return fmap, fjac, functional_projectors(fjac.jacobian, null_threshold_mode="spectral_gap")


def functional_step_norm_for(model: SmallLoRAMLP, probe: torch.Tensor, mode: str, params, direction: torch.Tensor, scale: float) -> float:
    before = get_flat_params(params)
    phi_before = phi(model, probe, mode)
    assign_flat_update(params, direction, scale=scale)
    value = float(torch.linalg.vector_norm(phi(model, probe, mode) - phi_before))
    set_flat_params(params, before)
    return value


def reference_functional_step(
    model: SmallLoRAMLP,
    xb: torch.Tensor,
    yb: torch.Tensor,
    probe: torch.Tensor,
    mode: str,
    train_scope: str,
    reference: str,
    lr: float,
    max_update_norm: float,
) -> float:
    ref_model = copy.deepcopy(model)
    configure_train_scope(ref_model, train_scope)
    ref_params = trainable_params(ref_model.parameters())
    phi_before = phi(ref_model, probe, mode)
    if reference == "adamw":
        opt = torch.optim.AdamW(ref_params, lr=lr)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ref_model(xb), yb)
        loss.backward()
        opt.step()
    elif reference == "diagonal_grad_square":
        opt = GeometricOptimizer(
            ref_params,
            lr=lr,
            lr_scale=1.0,
            mode="geometric",
            preconditioner="diagonal_grad_square",
            warmup_steps=0,
            max_update_norm=max_update_norm,
            grad_smoothing=0.0,
            adaptive_damping=False,
        )
        opt.step(lambda: F.cross_entropy(ref_model(xb), yb))
    else:
        raise ValueError(f"unknown calibration reference: {reference}")
    return float(torch.linalg.vector_norm(phi(ref_model, probe, mode) - phi_before))


def calibrate_scale(
    model: SmallLoRAMLP,
    probe: torch.Tensor,
    mode: str,
    params,
    direction: torch.Tensor,
    base_lr: float,
    target: float,
    args,
) -> tuple[float, float, float]:
    if target <= 1e-30 or direction.numel() == 0:
        achieved = functional_step_norm_for(model, probe, mode, params, direction, base_lr)
        return 1.0, achieved, 0.0
    low = float(args.lr_scale_min)
    high = float(args.lr_scale_max)
    best_scale = 1.0
    best_value = functional_step_norm_for(model, probe, mode, params, direction, base_lr)
    best_error = abs(best_value - target) / max(target, 1e-30)
    for _ in range(args.calibration_max_iters):
        mid = (low * high) ** 0.5
        value = functional_step_norm_for(model, probe, mode, params, direction, base_lr * mid)
        error = abs(value - target) / max(target, 1e-30)
        if error < best_error:
            best_scale = mid
            best_value = value
            best_error = error
        if value < target:
            low = mid
        else:
            high = mid
    return best_scale, best_value, best_error


def summarize(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(len(values), 1))


def median_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def pairwise_sensitivity(rows: list[RunRow], optimizer: str) -> float:
    selected = [row for row in rows if row.optimizer == optimizer]
    if len(selected) < 2:
        return 0.0
    phis = [torch.tensor([float(value) for value in row.final_phi.split(";")]) for row in selected]
    distances = [float(torch.linalg.vector_norm(left - right)) for left, right in itertools.combinations(phis, 2)]
    return summarize(distances)


def train_run(
    model: SmallLoRAMLP,
    optimizer_name: str,
    seed: int,
    representation: int,
    x: torch.Tensor,
    y: torch.Tensor,
    probe: torch.Tensor,
    batches: list[torch.Tensor],
    args,
) -> tuple[RunRow, list[StepRow]]:
    configure_train_scope(model, args.train_scope)
    params = trainable_params(model.parameters())
    start = time.perf_counter()
    step_rows: list[StepRow] = []
    scale_history: list[float] = []
    normal_basis = None
    basis_step = 0
    warm_start = None
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr)
    elif optimizer_name == "diagonal_grad_square":
        optimizer = GeometricOptimizer(
            params,
            lr=args.lr,
            lr_scale=1.0,
            mode="geometric",
            preconditioner="diagonal_grad_square",
            warmup_steps=0,
            max_update_norm=args.max_update_norm,
            grad_smoothing=0.0,
            adaptive_damping=False,
        )
    else:
        optimizer = None

    fmap0, fjac0, projectors0 = make_projectors(model, probe, args.functional_map)
    theta0 = fjac0.theta
    for step, indices in enumerate(batches, start=1):
        xb = x[indices]
        yb = y[indices]
        phi_before = phi(model, probe, args.functional_map)
        theta_before = get_flat_params(params)
        _, _, projectors = make_projectors(model, probe, args.functional_map)
        loss_before_tensor = F.cross_entropy(model(xb), yb)
        loss_before = float(loss_before_tensor.detach())
        wall_start = time.perf_counter()
        accepted_scale = 1.0
        calibration_target = 0.0
        calibration_error = 0.0
        descent = True
        fallback = False
        solver_residual = 0.0
        null_leakage = 0.0
        jvp_count = 0
        vjp_count = 0
        cache_hit = False
        cache_age = 0
        basis_rank = 0
        cg_iterations = 0
        peak_memory = 0
        effective_lr = args.lr

        if optimizer_name == "adamw":
            optimizer.zero_grad(set_to_none=True)
            loss_before_tensor.backward()
            optimizer.step()
        elif optimizer_name == "diagonal_grad_square":
            optimizer.step(lambda xb=xb, yb=yb: F.cross_entropy(model(xb), yb))
            log = optimizer.topography_log[-1]
            effective_lr = float(log["update_norm"] / max(log["direction_norm"], 1e-30))
            descent = bool(log["descent_gate_passed"])
            cg_iterations = int(log.get("cg_iterations", 0))
        else:
            use_cache = normal_basis is not None and (step - basis_step) < args.refresh_interval
            result = projected_functional_geoflow_direction(
                model,
                loss_before_tensor,
                probe,
                params=params,
                representation_fn=representation_fn_for(args.functional_map),
                damping=args.damping,
                max_update_norm=args.max_update_norm,
                response_solver="implicit_cg",
                production_mode=True,
                normal_basis=normal_basis if use_cache else None,
                warm_start=warm_start,
                max_basis_rank=args.max_basis_rank,
                max_vjp_probes=args.max_vjp_probes,
                vjp_probe_batch_size=args.vjp_probe_batch_size,
                cg_max_iter=args.cg_max_iter,
                cg_tolerance=args.cg_tol,
                functional_energy_fraction=1.0,
            )
            if result.normal_basis is not None and not result.basis_from_cache:
                normal_basis = result.normal_basis.detach()
                basis_step = step
            if result.cg_initial_guess is not None:
                warm_start = result.cg_initial_guess.detach()
            cache_hit = result.basis_from_cache
            cache_age = step - basis_step if cache_hit else 0
            basis_rank = result.retained_rank
            cg_iterations = result.cg_iterations
            descent = result.descent_gate_passed
            fallback = result.fallback
            solver_residual = result.solver_residual
            null_leakage = result.null_leakage
            jvp_count = result.jvp_count
            vjp_count = result.vjp_count
            peak_memory = result.peak_memory_bytes
            if optimizer_name == "functional_geoflow_matched_step":
                if step <= args.calibration_steps:
                    calibration_target = (
                        args.target_functional_step
                        if args.target_functional_step is not None
                        else reference_functional_step(
                            model,
                            xb,
                            yb,
                            probe,
                            args.functional_map,
                            args.train_scope,
                            args.calibration_reference,
                            args.lr,
                            args.max_update_norm,
                        )
                    )
                    accepted_scale, _, calibration_error = calibrate_scale(
                        model, probe, args.functional_map, params, result.direction, args.lr, calibration_target, args
                    )
                    scale_history.append(accepted_scale)
                else:
                    accepted_scale = median_or_zero(scale_history) if scale_history else 1.0
            effective_lr = args.lr * accepted_scale
            assign_flat_update(params, result.direction, scale=effective_lr)

        wall_clock = time.perf_counter() - wall_start
        loss_after = float(F.cross_entropy(model(xb), yb).detach())
        theta_after = get_flat_params(params)
        delta_theta = theta_after - theta_before
        phi_after = phi(model, probe, args.functional_map)
        functional_step = float(torch.linalg.vector_norm(phi_after - phi_before))
        parameter_step = float(torch.linalg.vector_norm(delta_theta))
        tangent_step = float(torch.linalg.vector_norm(projectors.tangent @ delta_theta)) if delta_theta.numel() else 0.0
        normal_step = float(torch.linalg.vector_norm(projectors.normal @ delta_theta)) if delta_theta.numel() else 0.0
        if optimizer_name == "functional_geoflow_matched_step" and calibration_target > 0:
            calibration_error = abs(functional_step - calibration_target) / max(calibration_target, 1e-30)
        step_rows.append(
            StepRow(
                seed=seed,
                optimizer=optimizer_name,
                representation=representation,
                step=step,
                train_scope=args.train_scope,
                functional_map=args.functional_map,
                train_loss_before=loss_before,
                train_loss_after=loss_after,
                functional_step_norm=functional_step,
                parameter_step_norm=parameter_step,
                tangent_step_norm=tangent_step,
                normal_step_norm=normal_step,
                tangent_normal_ratio=tangent_step / max(normal_step, 1e-30),
                accepted_step_scale=accepted_scale,
                requested_lr=args.lr,
                effective_lr=effective_lr,
                calibration_target=calibration_target,
                calibration_error=calibration_error,
                descent_gate_passed=descent,
                fallback=fallback,
                solver_residual=solver_residual,
                null_leakage=null_leakage,
                jvp_count=jvp_count,
                vjp_count=vjp_count,
                cache_hit=cache_hit,
                cache_age=cache_age,
                basis_rank=basis_rank,
                cg_iterations=cg_iterations,
                wall_clock_step=wall_clock,
                peak_memory_bytes=peak_memory,
            )
        )

    seconds = time.perf_counter() - start
    with torch.no_grad():
        logits = model(x)
        final_loss = float(F.cross_entropy(logits, y))
        final_accuracy = float((logits.argmax(dim=1) == y).float().mean())
        final_phi = phi(model, probe, args.functional_map)
    theta_final = get_flat_params(params)
    drift = theta_final - theta0
    tangent_drift = float(torch.linalg.vector_norm(projectors0.tangent @ drift)) if drift.numel() else 0.0
    normal_drift = float(torch.linalg.vector_norm(projectors0.normal @ drift)) if drift.numel() else 0.0
    run_row = RunRow(
        seed=seed,
        optimizer=optimizer_name,
        representation=representation,
        train_scope=args.train_scope,
        functional_map=args.functional_map,
        initial_equivalence_residual=0.0,
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        final_phi=";".join(f"{float(v):.9g}" for v in final_phi.reshape(-1)),
        mean_functional_step=summarize(row.functional_step_norm for row in step_rows),
        median_functional_step=median_or_zero(row.functional_step_norm for row in step_rows),
        mean_parameter_step=summarize(row.parameter_step_norm for row in step_rows),
        mean_tangent_step=summarize(row.tangent_step_norm for row in step_rows),
        mean_normal_step=summarize(row.normal_step_norm for row in step_rows),
        mean_calibration_error=summarize(row.calibration_error for row in step_rows if row.calibration_target > 0),
        mean_null_leakage=summarize(row.null_leakage for row in step_rows),
        total_jvp=sum(row.jvp_count for row in step_rows),
        total_vjp=sum(row.vjp_count for row in step_rows),
        mean_wall_clock=summarize(row.wall_clock_step for row in step_rows),
        peak_memory_bytes=max([row.peak_memory_bytes for row in step_rows] or [0]),
        tangent_drift=tangent_drift,
        near_null_amplification=tangent_drift / max(normal_drift, 1e-30),
        seconds=seconds,
    )
    return run_row, step_rows


def run_config(args) -> tuple[list[RunRow], list[StepRow], list[dict], dict]:
    run_rows: list[RunRow] = []
    step_rows: list[StepRow] = []
    for trial in range(args.trials):
        seed = args.seed + trial
        torch.manual_seed(seed)
        x, y = make_data(seed, args.samples, args.input_dim, args.output_dim)
        probe = x[: args.probe_size].clone()
        base = SmallLoRAMLP(args.input_dim, args.hidden_dim, args.output_dim, args.lora_rank)
        base_state = copy.deepcopy(base.state_dict())
        reference_model = SmallLoRAMLP(args.input_dim, args.hidden_dim, args.output_dim, args.lora_rank)
        reference_model.load_state_dict(base_state)
        reference_phi = phi(reference_model, probe, args.functional_map)
        batch_gen = torch.Generator().manual_seed(seed + 1009)
        batches = [torch.randint(0, x.shape[0], (args.batch_size,), generator=batch_gen) for _ in range(args.steps)]
        for optimizer_name in args.optimizers:
            for representation in range(args.representations):
                model = SmallLoRAMLP(args.input_dim, args.hidden_dim, args.output_dim, args.lora_rank)
                model.load_state_dict(base_state)
                model.lora.reparameterize(make_transform(args.lora_rank, representation))
                initial_residual = float(torch.linalg.vector_norm(phi(model, probe, args.functional_map) - reference_phi))
                run_row, rows = train_run(model, optimizer_name, seed, representation, x, y, probe, batches, args)
                run_row.initial_equivalence_residual = initial_residual
                run_rows.append(run_row)
                step_rows.extend(rows)

    aggregates = []
    for optimizer_name in sorted({row.optimizer for row in run_rows}):
        selected = [row for row in run_rows if row.optimizer == optimizer_name]
        aggregates.append(
            {
                "optimizer": optimizer_name,
                "mean_loss": summarize(row.final_loss for row in selected),
                "mean_accuracy": summarize(row.final_accuracy for row in selected),
                "reparameterization_sensitivity": pairwise_sensitivity(run_rows, optimizer_name),
                "mean_functional_step": summarize(row.mean_functional_step for row in selected),
                "functional_step_dispersion_across_representations": statistics.pstdev([row.mean_functional_step for row in selected])
                if len(selected) > 1
                else 0.0,
                "mean_tangent_drift": summarize(row.tangent_drift for row in selected),
                "mean_near_null_amplification": summarize(row.near_null_amplification for row in selected),
                "mean_calibration_error": summarize(row.mean_calibration_error for row in selected),
                "mean_null_leakage": summarize(row.mean_null_leakage for row in selected),
                "mean_seconds": summarize(row.seconds for row in selected),
            }
        )
    gates = compute_gates(run_rows, aggregates, args)
    return run_rows, step_rows, aggregates, gates


def aggregate_lookup(aggregates: list[dict], optimizer: str, field: str) -> float:
    for row in aggregates:
        if row["optimizer"] == optimizer:
            return float(row[field])
    return 0.0


def compute_gates(run_rows: list[RunRow], aggregates: list[dict], args) -> dict:
    max_initial = max([row.initial_equivalence_residual for row in run_rows] or [0.0])
    diag_sens = aggregate_lookup(aggregates, "diagonal_grad_square", "reparameterization_sensitivity")
    fixed_sens = aggregate_lookup(aggregates, "functional_geoflow_fixed_lr", "reparameterization_sensitivity")
    matched_sens = aggregate_lookup(aggregates, "functional_geoflow_matched_step", "reparameterization_sensitivity")
    diag_loss = aggregate_lookup(aggregates, "diagonal_grad_square", "mean_loss")
    fixed_loss = aggregate_lookup(aggregates, "functional_geoflow_fixed_lr", "mean_loss")
    matched_loss = aggregate_lookup(aggregates, "functional_geoflow_matched_step", "mean_loss")
    diag_tangent = aggregate_lookup(aggregates, "diagonal_grad_square", "mean_tangent_drift")
    matched_tangent = aggregate_lookup(aggregates, "functional_geoflow_matched_step", "mean_tangent_drift")
    matched_cal = aggregate_lookup(aggregates, "functional_geoflow_matched_step", "mean_calibration_error")
    matched_null = aggregate_lookup(aggregates, "functional_geoflow_matched_step", "mean_null_leakage")
    diag_seconds = aggregate_lookup(aggregates, "diagonal_grad_square", "mean_seconds")
    matched_seconds = aggregate_lookup(aggregates, "functional_geoflow_matched_step", "mean_seconds")
    fixed_gap = fixed_loss - diag_loss
    matched_gap = matched_loss - diag_loss
    by_seed_wins = 0
    by_seed_total = 0
    for seed in sorted({row.seed for row in run_rows}):
        diag = [row.final_loss for row in run_rows if row.seed == seed and row.optimizer == "diagonal_grad_square"]
        matched = [row.final_loss for row in run_rows if row.seed == seed and row.optimizer == "functional_geoflow_matched_step"]
        if diag and matched:
            by_seed_wins += int(summarize(matched) <= summarize(diag))
            by_seed_total += 1
    win_rate = by_seed_wins / max(by_seed_total, 1)
    return {
        "SOFTWARE_PASS": True,
        "INITIAL_EQUIVALENCE_PASS": max_initial < 1e-5,
        "FUNCTIONAL_STEP_MATCH_PASS": matched_cal < args.functional_step_tolerance,
        "STRUCTURAL_SENSITIVITY_PASS": matched_sens < diag_sens if diag_sens > 0 else False,
        "STRUCTURAL_WIN_RATE_PASS": win_rate >= 0.70,
        "NULL_LEAKAGE_PASS": matched_null < 1e-4,
        "TANGENT_SUPPRESSION_PASS": matched_tangent < diag_tangent if diag_tangent > 0 else False,
        "TASK_GAP_REDUCED_PASS": matched_gap < fixed_gap,
        "TASK_PARITY_PASS": matched_gap <= args.loss_parity_margin,
        "TASK_ADVANTAGE_PASS": matched_gap < 0.0,
        "COMPUTE_WARNING": (matched_seconds / max(diag_seconds, 1e-30)) > 10.0,
        "fixed_lr_sensitivity": fixed_sens,
        "matched_step_sensitivity": matched_sens,
        "diagonal_sensitivity": diag_sens,
        "fixed_lr_loss": fixed_loss,
        "matched_step_loss": matched_loss,
        "diagonal_loss": diag_loss,
        "functional_step_calibration_error": matched_cal,
        "matched_tangent_drift": matched_tangent,
        "matched_null_leakage": matched_null,
        "matched_vs_diagonal_loss_gap": matched_gap,
        "fixed_vs_diagonal_loss_gap": fixed_gap,
        "structural_win_rate": win_rate,
        "wall_clock_ratio_matched_vs_diagonal": matched_seconds / max(diag_seconds, 1e-30),
    }


def write_outputs(run_rows: list[RunRow], step_rows: list[StepRow], aggregates: list[dict], gates: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(run_rows[0]).keys()))
        writer.writeheader()
        for row in run_rows:
            writer.writerow(asdict(row))
    trajectory = out.with_name(out.stem + "_trajectory.csv")
    with trajectory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(step_rows[0]).keys()))
        writer.writeheader()
        for row in step_rows:
            writer.writerow(asdict(row))
    aggregate_path = out.with_name(out.stem + "_aggregate.csv")
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)
    gates_path = out.with_name(out.stem + "_gates.csv")
    with gates_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in gates.items():
            writer.writerow({"metric": key, "value": value})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--representations", type=int, default=5)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--probe-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=3)
    parser.add_argument("--train-scope", choices=["lora_only", "head_only", "lora_and_head"], default="lora_only")
    parser.add_argument("--functional-map", choices=["logits", "lora_output", "hidden", "logits_hidden"], default="hidden")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--max-update-norm", type=float, default=0.25)
    parser.add_argument("--refresh-interval", type=int, default=5)
    parser.add_argument("--max-basis-rank", type=int, default=16)
    parser.add_argument("--max-vjp-probes", type=int, default=24)
    parser.add_argument("--vjp-probe-batch-size", type=int, default=8)
    parser.add_argument("--cg-max-iter", type=int, default=24)
    parser.add_argument("--cg-tol", type=float, default=1e-5)
    parser.add_argument("--calibration-reference", choices=["adamw", "diagonal_grad_square"], default="diagonal_grad_square")
    parser.add_argument("--calibration-steps", type=int, default=10)
    parser.add_argument("--target-functional-step", type=float, default=None)
    parser.add_argument("--functional-step-tolerance", type=float, default=0.10)
    parser.add_argument("--lr-scale-min", type=float, default=1e-3)
    parser.add_argument("--lr-scale-max", type=float, default=1e3)
    parser.add_argument("--calibration-max-iters", type=int, default=12)
    parser.add_argument("--loss-parity-margin", type=float, default=0.02)
    parser.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    parser.add_argument("--functional-maps", default=None)
    parser.add_argument("--train-scopes", default=None)
    parser.add_argument("--lora-ranks", default=None)
    parser.add_argument("--probe-sizes", default=None)
    parser.add_argument("--calibration-references", default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts/lora_matched_step.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.optimizers = parse_csv(args.optimizers, OPTIMIZERS)
    maps = parse_csv(args.functional_maps, [args.functional_map])
    scopes = parse_csv(args.train_scopes, [args.train_scope])
    ranks = [int(value) for value in parse_csv(args.lora_ranks, [str(args.lora_rank)])]
    probes = [int(value) for value in parse_csv(args.probe_sizes, [str(args.probe_size)])]
    refs = parse_csv(args.calibration_references, [args.calibration_reference])
    all_run_rows: list[RunRow] = []
    all_step_rows: list[StepRow] = []
    all_aggregates: list[dict] = []
    last_gates: dict = {}
    for fmap, scope, rank, probe_size, ref in itertools.product(maps, scopes, ranks, probes, refs):
        local_args = copy.copy(args)
        local_args.functional_map = fmap
        local_args.train_scope = scope
        local_args.lora_rank = rank
        local_args.probe_size = probe_size
        local_args.calibration_reference = ref
        run_rows, step_rows, aggregates, gates = run_config(local_args)
        for row in aggregates:
            row.update({"functional_map": fmap, "train_scope": scope, "lora_rank": rank, "probe_size": probe_size, "calibration_reference": ref})
        all_run_rows.extend(run_rows)
        all_step_rows.extend(step_rows)
        all_aggregates.extend(aggregates)
        last_gates = gates
    write_outputs(all_run_rows, all_step_rows, all_aggregates, last_gates, args.out)
    for key, value in last_gates.items():
        print(f"{key}={value}")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_name(args.out.stem + '_trajectory.csv')}")


if __name__ == "__main__":
    main()
