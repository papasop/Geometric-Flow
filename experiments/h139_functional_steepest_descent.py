#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H13.9 direction audit: split flow vs net-product Frobenius optimum.

This script is a lightweight reproducibility audit for the H13.9 local
comparison in the README and variational foundation docs. It does not train a
model. Instead, it samples low-rank products M = B A and product gradients G,
then compares two local product-space directions:

    D_cur  = -(P_B G + G P_A)
    D_star = -(P_B G + G P_A - P_B G P_A)

Here D_cur is the product velocity induced by the implemented inverse-Gram
split executed-information direction, while D_star is the steepest direction
for the net full-product Frobenius metric on the fixed-rank tangent space.

The main check is the local efficiency ratio

    eta(D_cur) / eta(D_star),

where eta(D) = -<G,D>_F / ||D||_F. The audited theory gives the sharp lower
bound 2*sqrt(2)/3 for full-rank factors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


EPS = 1e-14
SHARP_BOUND = 2.0 * math.sqrt(2.0) / 3.0


@dataclass
class TrialResult:
    rank: int
    factor_kappa: float
    gauge_kappa: float
    efficiency_ratio: float
    cosine: float
    descent_current: float
    descent_star: float
    norm_current: float
    norm_star: float
    tangent_residual_star: float
    gauge_product_relerr: float


def fro_inner(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(x * y))


def fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def orthonormal_columns(
    rows: int,
    cols: int,
    rng: np.random.Generator,
) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


def invertible_matrix(rank: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    if kappa < 1:
        raise ValueError("kappa must be >= 1")
    u = orthonormal_columns(rank, rank, rng)
    v = orthonormal_columns(rank, rank, rng)
    s = np.geomspace(1.0, float(kappa), rank)
    return u @ np.diag(s) @ v.T


def make_factors(
    m: int,
    n: int,
    r: int,
    kappa: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not (1 <= r <= min(m, n)):
        raise ValueError("Require 1 <= r <= min(m,n)")
    u = orthonormal_columns(m, r, rng)
    v = orthonormal_columns(n, r, rng)
    core = invertible_matrix(r, math.sqrt(float(kappa)), rng)
    return u @ core, core @ v.T


def projectors(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pb = b @ np.linalg.inv(b.T @ b) @ b.T
    pa = a.T @ np.linalg.inv(a @ a.T) @ a
    return pb, pa


def directions(
    b: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pb, pa = projectors(b, a)
    d_cur = -(pb @ g + g @ pa)
    d_star = -(pb @ g + g @ pa - pb @ g @ pa)
    return d_cur, d_star


def efficiency(g: np.ndarray, d: np.ndarray) -> float:
    return -fro_inner(g, d) / max(fro_norm(d), EPS)


def tangent_residual(
    b: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
) -> float:
    pb, pa = projectors(b, a)
    projected = pb @ d + d @ pa - pb @ d @ pa
    return fro_norm(d - projected) / max(fro_norm(d), EPS)


def gauge_product_relerr(
    b: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> float:
    r = a.shape[0]
    s = invertible_matrix(r, kappa, rng)
    b_g = b @ np.linalg.inv(s)
    a_g = s @ a
    d_cur, _ = directions(b, a, g)
    d_g, _ = directions(b_g, a_g, g)
    return fro_norm(d_cur - d_g) / max(fro_norm(d_cur), EPS)


def run_trial(
    *,
    m: int,
    n: int,
    r: int,
    factor_kappa: float,
    gauge_kappa: float,
    rng: np.random.Generator,
) -> TrialResult:
    b, a = make_factors(m, n, r, factor_kappa, rng)
    g = rng.normal(size=(m, n))
    d_cur, d_star = directions(b, a, g)
    eta_cur = efficiency(g, d_cur)
    eta_star = efficiency(g, d_star)
    ratio = eta_cur / max(eta_star, EPS)
    cosine = fro_inner(d_cur, d_star) / max(fro_norm(d_cur) * fro_norm(d_star), EPS)
    return TrialResult(
        rank=r,
        factor_kappa=float(factor_kappa),
        gauge_kappa=float(gauge_kappa),
        efficiency_ratio=float(ratio),
        cosine=float(cosine),
        descent_current=float(-fro_inner(g, d_cur)),
        descent_star=float(-fro_inner(g, d_star)),
        norm_current=float(fro_norm(d_cur)),
        norm_star=float(fro_norm(d_star)),
        tangent_residual_star=float(tangent_residual(b, a, d_star)),
        gauge_product_relerr=float(gauge_product_relerr(b, a, g, gauge_kappa, rng)),
    )


def summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def write_csv(path: Path, rows: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--factor-kappas", default="1,10,100,1000,10000")
    parser.add_argument("--gauge-kappas", default="1,10,1000")
    parser.add_argument("--trials-per-setting", type=int, default=50)
    parser.add_argument("--seed", type=int, default=139)
    parser.add_argument("--out-dir", default="artifacts/h139_functional_steepest")
    parser.add_argument("--ratio-tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    ranks = parse_int_list(args.ranks)
    factor_kappas = parse_float_list(args.factor_kappas)
    gauge_kappas = parse_float_list(args.gauge_kappas)
    rng = np.random.default_rng(args.seed)

    rows: list[TrialResult] = []
    for r in ranks:
        for factor_kappa in factor_kappas:
            for gauge_kappa in gauge_kappas:
                for _ in range(args.trials_per_setting):
                    rows.append(
                        run_trial(
                            m=args.m,
                            n=args.n,
                            r=r,
                            factor_kappa=factor_kappa,
                            gauge_kappa=gauge_kappa,
                            rng=rng,
                        )
                    )

    ratios = [row.efficiency_ratio for row in rows]
    cosines = [row.cosine for row in rows]
    tangent_residuals = [row.tangent_residual_star for row in rows]
    gauge_residuals = [row.gauge_product_relerr for row in rows]
    min_ratio = min(ratios)
    pass_bound = min_ratio + args.ratio_tolerance >= SHARP_BOUND
    pass_tangent = max(tangent_residuals) < 1e-10
    # Extreme non-orthogonal gauges at large condition numbers produce ordinary
    # double-precision residuals. This is a numerical audit gate, not an
    # analytic covariance threshold.
    pass_gauge = np.quantile(gauge_residuals, 0.99) < 1e-6

    result = {
        "n_trials": len(rows),
        "matrix_shape": [args.m, args.n],
        "ranks": ranks,
        "factor_kappas": factor_kappas,
        "gauge_kappas": gauge_kappas,
        "sharp_bound": SHARP_BOUND,
        "efficiency_ratio": summary(ratios),
        "direction_cosine": summary(cosines),
        "tangent_residual_star": summary(tangent_residuals),
        "gauge_product_relerr": summary(gauge_residuals),
        "gauge_product_relerr_thresholds": {
            "p99": 1e-6,
            "interpretation": (
                "Finite-precision residual for product velocity covariance "
                "under extreme non-orthogonal gauges."
            ),
        },
        "gates": {
            "PASS_SHARP_BOUND": bool(pass_bound),
            "PASS_TANGENT_RESIDUAL": bool(pass_tangent),
            "PASS_GAUGE_NUMERICS": bool(pass_gauge),
        },
    }
    result["gates"]["PASS_ALL"] = all(result["gates"].values())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "h139_trials.csv", rows)
    with (out_dir / "h139_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Outputs: {out_dir}")
    return 0 if result["gates"]["PASS_ALL"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
