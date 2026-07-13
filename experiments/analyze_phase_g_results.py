"""Offline reanalysis for Phase G matched-step LoRA artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lora_matched_step_benchmark import (
    compute_gates_from_statistics,
    gate_rows_for_config,
    phase_g_statistics,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_run_csvs(artifact_dir: Path) -> tuple[list[Path], list[str]]:
    candidates = []
    warnings = []
    for path in sorted(artifact_dir.glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
        except OSError as exc:
            warnings.append(f"warning: skipped {path}: {exc}")
            continue
        required = {
            "seed",
            "optimizer",
            "representation",
            "final_phi",
            "final_loss",
            "mean_functional_step",
            "mean_calibration_error",
            "mean_null_leakage",
            "tangent_drift",
            "train_scope",
            "functional_map",
        }
        if required.issubset(fields):
            candidates.append(path)
        else:
            warnings.append(f"warning: skipped non-run or incomplete CSV {path}")
    return candidates, warnings


def write_table(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_rows(rows: list[dict], loss_parity_margin: float, functional_step_tolerance: float) -> tuple[list[dict], dict]:
    aggregates, _ = phase_g_statistics(
        rows,
        loss_parity_margin=loss_parity_margin,
        functional_step_tolerance=functional_step_tolerance,
    )
    gates = compute_gates_from_statistics(
        rows,
        aggregates,
        loss_parity_margin=loss_parity_margin,
        functional_step_tolerance=functional_step_tolerance,
    )
    return aggregates, gates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--run-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--loss-parity-margin", type=float, default=0.02)
    parser.add_argument("--functional-step-tolerance", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_paths: list[Path] = []
    warnings: list[str] = []
    if args.run_csv is not None:
        run_paths.append(args.run_csv)
    if args.artifact_dir is not None:
        found, found_warnings = candidate_run_csvs(args.artifact_dir)
        run_paths.extend(found)
        warnings.extend(found_warnings)
    if not run_paths:
        raise SystemExit("no complete Phase G run CSV found")

    all_rows = []
    all_aggregates = []
    all_gate_rows = []
    for path in run_paths:
        rows = read_csv(path)
        aggregates, gates = analyze_rows(rows, args.loss_parity_margin, args.functional_step_tolerance)
        for row in aggregates:
            row["source_csv"] = str(path)
        all_rows.extend(rows)
        all_aggregates.extend(aggregates)
        first = rows[0]
        all_gate_rows.extend(
            gate_rows_for_config(
                gates,
                first.get("functional_map", ""),
                first.get("train_scope", ""),
                int(float(first.get("lora_rank", 0))) if first.get("lora_rank") else 0,
                int(float(first.get("probe_size", 0))) if first.get("probe_size") else 0,
                first.get("calibration_reference", ""),
            )
        )
    out_prefix = args.out
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_table(all_aggregates, out_prefix.with_name(out_prefix.name + "_aggregate.csv"))
    write_table(all_gate_rows, out_prefix.with_name(out_prefix.name + "_gates.csv"))
    for warning in warnings:
        print(warning)
    for row in all_gate_rows:
        if row["metric"] in {"STRUCTURAL_WIN_RATE_PASS", "TASK_GAP_REDUCED_PASS", "TASK_ADVANTAGE_PASS"}:
            print(f"{row['metric']}={row['value']}")
    print(f"wrote {out_prefix.with_name(out_prefix.name + '_aggregate.csv')}")
    print(f"wrote {out_prefix.with_name(out_prefix.name + '_gates.csv')}")


if __name__ == "__main__":
    main()
