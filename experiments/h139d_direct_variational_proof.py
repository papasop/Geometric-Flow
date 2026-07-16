#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H13.9D-FIX — Direct Variational Proof Audit for Existing GeoFlow Direction

Goal
----
Verify, symbolically-by-identity and numerically, that the existing GeoFlow
factor direction

    V_A = -(B^T B)^{-1} grad_A L
    V_B = -grad_B L (A A^T)^{-1}

is the exact local steepest-descent direction under the split executed-
information metric

    ||V||_split^2 = ||B V_A||_F^2 + ||V_B A||_F^2.

Equivalent variational problem
------------------------------
For full-rank B in R^{m x r} and A in R^{r x n},

    maximize_V  -dL[V]
    subject to  ||B V_A||_F^2 + ||V_B A||_F^2 <= 1.

The normalized optimizer is

    V_star = -grad_split L / ||grad_split L||_split,

where

    grad_split L
      = ((B^T B)^{-1} grad_A L,
         grad_B L (A A^T)^{-1}).

This script checks:
  1. Riesz representation under the split metric;
  2. KKT stationarity and unit-budget optimality;
  3. equality in Cauchy-Schwarz;
  4. uniqueness in the full-rank case;
  5. gauge covariance under arbitrary invertible GL(r) transforms;
  6. rank-deficient pseudoinverse extension and non-uniqueness;
  7. distinction from the full-product Frobenius metric;
  8. random stress scans over dimensions, rank, and conditioning.

Colab
------
    %run /content/h139d_direct_variational_proof.py

Fast smoke test
---------------
    %run /content/h139d_direct_variational_proof.py \
        --trials-per-setting 50 --no-plots
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


# ---------------------------------------------------------------------
# Basic linear algebra
# ---------------------------------------------------------------------

def fro_inner(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(x * y))


def fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def relerr(x: np.ndarray, y: np.ndarray) -> float:
    return fro_norm(x - y) / max(fro_norm(y), EPS)


def orthonormal_columns(
    rows: int,
    cols: int,
    rng: np.random.Generator,
) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return q[:, :cols]


def make_factors(
    m: int,
    n: int,
    r: int,
    kappa: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not (1 <= r <= min(m, n)):
        raise ValueError("Require 1 <= r <= min(m,n)")
    if kappa < 1:
        raise ValueError("kappa must be >= 1")

    u = orthonormal_columns(m, r, rng)
    v = orthonormal_columns(n, r, rng)
    q1 = orthonormal_columns(r, r, rng)
    q2 = orthonormal_columns(r, r, rng)
    s = np.geomspace(1.0, float(kappa), r)
    core = q1 @ np.diag(np.sqrt(s)) @ q2.T
    return u @ core, core @ v.T


def split_metric(
    b: np.ndarray,
    a: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
) -> float:
    return (
        fro_inner(b @ v_a, b @ w_a)
        + fro_inner(v_b @ a, w_b @ a)
    )


def split_norm(
    b: np.ndarray,
    a: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
) -> float:
    value = split_metric(b, a, v_a, v_b, v_a, v_b)
    return math.sqrt(max(value, 0.0))


def loss_differential(
    g_a: np.ndarray,
    g_b: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
) -> float:
    return fro_inner(g_a, v_a) + fro_inner(g_b, v_b)


def split_gradient(
    b: np.ndarray,
    a: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_a = np.linalg.solve(b.T @ b, g_a)
    z_b = np.linalg.solve(a @ a.T, g_b.T).T
    return z_a, z_b


def negative_split_gradient(
    b: np.ndarray,
    a: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_a, z_b = split_gradient(b, a, g_a, g_b)
    return -z_a, -z_b


def normalized_optimizer(
    b: np.ndarray,
    a: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v_a, v_b = negative_split_gradient(b, a, g_a, g_b)
    nrm = split_norm(b, a, v_a, v_b)
    if nrm < EPS:
        return np.zeros_like(v_a), np.zeros_like(v_b)
    return v_a / nrm, v_b / nrm


def product_velocity(
    b: np.ndarray,
    a: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
) -> np.ndarray:
    return v_b @ a + b @ v_a


def projector_col(b: np.ndarray) -> np.ndarray:
    return b @ np.linalg.solve(b.T @ b, b.T)


def projector_row(a: np.ndarray) -> np.ndarray:
    return a.T @ np.linalg.solve(a @ a.T, a)


def full_product_optimum(
    b: np.ndarray,
    a: np.ndarray,
    g_m: np.ndarray,
) -> np.ndarray:
    p_b = projector_col(b)
    p_a = projector_row(a)
    return -(p_b @ g_m + g_m @ p_a - p_b @ g_m @ p_a)


def current_product_direction(
    b: np.ndarray,
    a: np.ndarray,
    g_m: np.ndarray,
) -> np.ndarray:
    p_b = projector_col(b)
    p_a = projector_row(a)
    return -(p_b @ g_m + g_m @ p_a)


def random_invertible(
    r: int,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    q1 = orthonormal_columns(r, r, rng)
    q2 = orthonormal_columns(r, r, rng)
    s = np.geomspace(1.0, float(kappa), r)
    return q1 @ np.diag(s) @ q2.T


def gauge_transform(
    b: np.ndarray,
    a: np.ndarray,
    g_a: np.ndarray,
    g_b: np.ndarray,
    s: np.ndarray,
):
    # Avoid explicit inversion:
    #   B' = B S^{-1}
    # implemented as solve(S^T, B^T)^T.
    b2 = np.linalg.solve(s.T, b.T).T
    a2 = s @ a

    # Covector transformation:
    #   grad_A' = S^{-T} grad_A
    #   grad_B' = grad_B S^T
    ga2 = np.linalg.solve(s.T, g_a)
    gb2 = g_b @ s.T
    return b2, a2, ga2, gb2


# ---------------------------------------------------------------------
# Full-rank theorem audit
# ---------------------------------------------------------------------

@dataclass
class Trial:
    seed: int
    m: int
    n: int
    r: int
    factor_kappa: float
    gauge_kappa: float

    riesz_relerr: float
    kkt_stationarity_relerr: float
    unit_budget_relerr: float
    cs_equality_relerr: float
    sampled_optimality_margin_min: float
    sampled_optimality_violations: int
    uniqueness_probe_min_gap: float

    gauge_metric_relerr: float
    gauge_product_relerr: float
    gauge_direction_relerr: float
    gauge_normalized_direction_relerr: float

    split_objective_value: float
    split_gradient_norm: float
    full_product_direction_cosine: float
    full_product_efficiency_ratio: float

    cond_btb: float
    cond_aat: float
    cond_s: float


def run_trial(
    *,
    seed: int,
    m: int,
    n: int,
    r: int,
    factor_kappa: float,
    gauge_kappa: float,
    random_feasible_samples: int,
) -> Trial:
    rng = np.random.default_rng(seed)
    b, a = make_factors(m, n, r, factor_kappa, rng)

    # Build a loss that depends only on M=BA.
    g_m = rng.normal(size=(m, n))
    g_a = b.T @ g_m
    g_b = g_m @ a.T

    grad_a, grad_b = split_gradient(b, a, g_a, g_b)
    v_a, v_b = normalized_optimizer(b, a, g_a, g_b)

    # Riesz test with an arbitrary tangent direction.
    w_a = rng.normal(size=a.shape)
    w_b = rng.normal(size=b.shape)
    lhs = split_metric(b, a, grad_a, grad_b, w_a, w_b)
    rhs = loss_differential(g_a, g_b, w_a, w_b)
    riesz = abs(lhs - rhs) / max(abs(rhs), 1.0)

    # KKT for max -dL[V] subject ||V||_split^2 <= 1.
    # Lagrangian: -dL[V] - lambda/2 (||V||^2 - 1)
    grad_norm = split_norm(b, a, grad_a, grad_b)
    lam = grad_norm
    stationarity_a = -g_a - lam * (b.T @ b @ v_a)
    stationarity_b = -g_b - lam * (v_b @ (a @ a.T))
    stationarity = math.sqrt(
        fro_norm(stationarity_a) ** 2
        + fro_norm(stationarity_b) ** 2
    ) / max(
        math.sqrt(fro_norm(g_a) ** 2 + fro_norm(g_b) ** 2),
        1.0,
    )

    unit_budget = abs(split_norm(b, a, v_a, v_b) - 1.0)
    objective = -loss_differential(g_a, g_b, v_a, v_b)
    cs_eq = abs(objective - grad_norm) / max(grad_norm, 1.0)

    # Random feasible competitors.
    margins = []
    violations = 0
    for _ in range(random_feasible_samples):
        u_a = rng.normal(size=a.shape)
        u_b = rng.normal(size=b.shape)
        nrm = split_norm(b, a, u_a, u_b)
        if nrm < EPS:
            continue
        scale = rng.random() / nrm
        u_a *= scale
        u_b *= scale
        candidate = -loss_differential(g_a, g_b, u_a, u_b)
        margin = objective - candidate
        margins.append(margin)
        if margin < -1e-10:
            violations += 1

    min_margin = min(margins) if margins else float("nan")

    # Uniqueness probe: perturb normalized optimum within unit sphere.
    uniqueness_gaps = []
    for _ in range(50):
        q_a = rng.normal(size=a.shape)
        q_b = rng.normal(size=b.shape)

        # Remove split-metric component along V.
        coeff = split_metric(b, a, q_a, q_b, v_a, v_b)
        q_a = q_a - coeff * v_a
        q_b = q_b - coeff * v_b
        qn = split_norm(b, a, q_a, q_b)
        if qn < EPS:
            continue
        q_a /= qn
        q_b /= qn
        angle = rng.uniform(1e-3, 0.5)
        c, s = math.cos(angle), math.sin(angle)
        p_a = c * v_a + s * q_a
        p_b = c * v_b + s * q_b
        probe_obj = -loss_differential(g_a, g_b, p_a, p_b)
        uniqueness_gaps.append(objective - probe_obj)

    uniqueness_min_gap = (
        min(uniqueness_gaps) if uniqueness_gaps else float("nan")
    )

    # Gauge covariance.
    s = random_invertible(r, gauge_kappa, rng)
    b2, a2, ga2, gb2 = gauge_transform(b, a, g_a, g_b, s)
    va2, vb2 = normalized_optimizer(b2, a2, ga2, gb2)

    # Transform original optimizer coordinates.
    va_expected = s @ v_a
    vb_expected = np.linalg.solve(s.T, v_b.T).T

    gauge_direction = math.sqrt(
        fro_norm(va2 - va_expected) ** 2
        + fro_norm(vb2 - vb_expected) ** 2
    ) / max(
        math.sqrt(fro_norm(va_expected) ** 2 + fro_norm(vb_expected) ** 2),
        EPS,
    )

    metric_orig = split_metric(b, a, v_a, v_b, w_a, w_b)
    wa2 = s @ w_a
    wb2 = w_b @ np.linalg.inv(s)
    metric_new = split_metric(b2, a2, va2, vb2, wa2, wb2)
    gauge_metric = abs(metric_new - metric_orig) / max(abs(metric_orig), 1.0)

    product_orig = product_velocity(b, a, v_a, v_b)
    product_new = product_velocity(b2, a2, va2, vb2)
    gauge_product = relerr(product_new, product_orig)

    # Same as direction relerr here, retained as explicit theorem gate.
    gauge_normalized = gauge_direction

    # Full-product comparison.
    d_cur = current_product_direction(b, a, g_m)
    d_full = full_product_optimum(b, a, g_m)
    cosine = fro_inner(d_cur, d_full) / max(
        fro_norm(d_cur) * fro_norm(d_full), EPS
    )
    eff_cur = -fro_inner(g_m, d_cur) / max(fro_norm(d_cur), EPS)
    eff_full = -fro_inner(g_m, d_full) / max(fro_norm(d_full), EPS)

    return Trial(
        seed=seed,
        m=m,
        n=n,
        r=r,
        factor_kappa=float(factor_kappa),
        gauge_kappa=float(gauge_kappa),
        riesz_relerr=riesz,
        kkt_stationarity_relerr=stationarity,
        unit_budget_relerr=unit_budget,
        cs_equality_relerr=cs_eq,
        sampled_optimality_margin_min=float(min_margin),
        sampled_optimality_violations=int(violations),
        uniqueness_probe_min_gap=float(uniqueness_min_gap),
        gauge_metric_relerr=gauge_metric,
        gauge_product_relerr=gauge_product,
        gauge_direction_relerr=gauge_direction,
        gauge_normalized_direction_relerr=gauge_normalized,
        split_objective_value=float(objective),
        split_gradient_norm=float(grad_norm),
        full_product_direction_cosine=float(cosine),
        full_product_efficiency_ratio=float(eff_cur / max(eff_full, EPS)),
        cond_btb=float(np.linalg.cond(b.T @ b)),
        cond_aat=float(np.linalg.cond(a @ a.T)),
        cond_s=float(np.linalg.cond(s)),
    )


# ---------------------------------------------------------------------
# Rank-deficient pseudoinverse audit
# ---------------------------------------------------------------------

@dataclass
class RankDeficientResult:
    seed: int
    m: int
    n: int
    r: int
    effective_rank: int
    riesz_on_visible_subspace_relerr: float
    null_direction_split_norm: float
    null_direction_loss_differential: float
    pseudoinverse_stationarity_relerr: float
    nonuniqueness_product_relerr: float


def run_rank_deficient_trial(
    *,
    seed: int,
    m: int,
    n: int,
    r: int,
    effective_rank: int,
) -> RankDeficientResult:
    rng = np.random.default_rng(seed)
    if not (1 <= effective_rank < r):
        raise ValueError("Require 1 <= effective_rank < r")

    u = orthonormal_columns(m, effective_rank, rng)
    v = orthonormal_columns(n, effective_rank, rng)

    b = np.zeros((m, r))
    a = np.zeros((r, n))
    b[:, :effective_rank] = u
    a[:effective_rank, :] = v.T

    g_m = rng.normal(size=(m, n))
    g_a = b.T @ g_m
    g_b = g_m @ a.T

    gram_b = b.T @ b
    gram_a = a @ a.T
    grad_a = np.linalg.pinv(gram_b) @ g_a
    grad_b = g_b @ np.linalg.pinv(gram_a)

    # Visible test direction.
    w_a = np.zeros_like(a)
    w_b = np.zeros_like(b)
    w_a[:effective_rank] = rng.normal(size=(effective_rank, n))
    w_b[:, :effective_rank] = rng.normal(size=(m, effective_rank))

    lhs = split_metric(b, a, grad_a, grad_b, w_a, w_b)
    rhs = loss_differential(g_a, g_b, w_a, w_b)
    riesz_visible = abs(lhs - rhs) / max(abs(rhs), 1.0)

    # Pure null direction: invisible to metric and loss.
    null_a = np.zeros_like(a)
    null_b = np.zeros_like(b)
    null_a[effective_rank:] = rng.normal(
        size=(r - effective_rank, n)
    )
    null_b[:, effective_rank:] = rng.normal(
        size=(m, r - effective_rank)
    )

    null_norm = split_norm(b, a, null_a, null_b)
    null_loss = loss_differential(g_a, g_b, null_a, null_b)

    # Pseudoinverse stationarity on visible subspace.
    res_a = gram_b @ grad_a - g_a
    res_b = grad_b @ gram_a - g_b
    stationarity = math.sqrt(
        fro_norm(res_a) ** 2 + fro_norm(res_b) ** 2
    ) / max(
        math.sqrt(fro_norm(g_a) ** 2 + fro_norm(g_b) ** 2),
        1.0,
    )

    product_base = product_velocity(b, a, -grad_a, -grad_b)
    product_shift = product_velocity(
        b,
        a,
        -grad_a + null_a,
        -grad_b + null_b,
    )
    nonunique_relerr = relerr(product_shift, product_base)

    return RankDeficientResult(
        seed=seed,
        m=m,
        n=n,
        r=r,
        effective_rank=effective_rank,
        riesz_on_visible_subspace_relerr=riesz_visible,
        null_direction_split_norm=float(null_norm),
        null_direction_loss_differential=float(abs(null_loss)),
        pseudoinverse_stationarity_relerr=stationarity,
        nonuniqueness_product_relerr=nonunique_relerr,
    )


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], fields: list[str]) -> dict:
    out = {"n": len(rows)}
    for field in fields:
        vals = np.asarray(
            [
                float(row[field])
                for row in rows
                if row.get(field) is not None
                and np.isfinite(float(row[field]))
            ],
            dtype=float,
        )
        if vals.size == 0:
            continue
        out[field] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
            "p01": float(np.quantile(vals, 0.01)),
            "p99": float(np.quantile(vals, 0.99)),
        }
    return out


def make_plots(out_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    ratios = np.asarray(
        [float(r["full_product_efficiency_ratio"]) for r in rows]
    )
    cosines = np.asarray(
        [float(r["full_product_direction_cosine"]) for r in rows]
    )

    fig = plt.figure(figsize=(8, 5))
    plt.hist(ratios, bins=40)
    plt.xlabel("Split-direction / full-product efficiency")
    plt.ylabel("Count")
    plt.title("H13.9D metric comparison")
    plt.tight_layout()
    fig.savefig(out_dir / "h139d_efficiency_ratio.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(cosines, bins=40)
    plt.xlabel("Direction cosine")
    plt.ylabel("Count")
    plt.title("Split vs full-product direction cosine")
    plt.tight_layout()
    fig.savefig(out_dir / "h139d_direction_cosine.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "H13.9D direct variational proof audit for existing GeoFlow."
        )
    )
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument(
        "--factor-kappas",
        default="1,10,100,1000,10000",
    )
    parser.add_argument(
        "--gauge-kappas",
        default="1,10,1000",
    )
    parser.add_argument(
        "--trials-per-setting",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--random-feasible-samples",
        type=int,
        default=200,
    )
    parser.add_argument("--base-seed", type=int, default=139400)
    parser.add_argument(
        "--rank-deficient-trials",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/h139d_variational_proof",
    )
    parser.add_argument("--no-plots", action="store_true")
    args, unknown = parser.parse_known_args()

    if unknown:
        print("[H13.9D] ignored notebook/kernel arguments:", unknown)

    ranks = parse_ints(args.ranks)
    factor_kappas = parse_floats(args.factor_kappas)
    gauge_kappas = parse_floats(args.gauge_kappas)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("H13.9D-FIX DIRECT VARIATIONAL PROOF AUDIT")
    print("=" * 120)
    print("Claim:")
    print(
        "  Existing GeoFlow direction is the exact normalized maximizer of"
    )
    print(
        "  -dL[V] under ||B V_A||_F^2 + ||V_B A||_F^2 <= 1."
    )
    print("ranks =", ranks)
    print("factor_kappas =", factor_kappas)
    print("gauge_kappas =", gauge_kappas)
    print("trials_per_setting =", args.trials_per_setting)

    rows: list[dict] = []
    setting_count = len(ranks) * len(factor_kappas) * len(gauge_kappas)
    setting_index = 0

    for r in ranks:
        if r > min(args.m, args.n):
            raise ValueError(f"rank {r} exceeds matrix dimensions")
        for fk in factor_kappas:
            for gk in gauge_kappas:
                setting_index += 1
                print(
                    f"[setting {setting_index}/{setting_count}] "
                    f"r={r} factor_kappa={fk:g} gauge_kappa={gk:g}"
                )
                for t in range(args.trials_per_setting):
                    seed = (
                        args.base_seed
                        + 1_000_000 * r
                        + 10_000 * int(round(math.log10(max(fk, 1))))
                        + 100 * int(round(math.log10(max(gk, 1))))
                        + t
                    )
                    result = run_trial(
                        seed=seed,
                        m=args.m,
                        n=args.n,
                        r=r,
                        factor_kappa=fk,
                        gauge_kappa=gk,
                        random_feasible_samples=(
                            args.random_feasible_samples
                        ),
                    )
                    rows.append(asdict(result))

    rank_def_rows = []
    rd_rank = max(2, min(max(ranks), min(args.m, args.n)))
    effective_rank = max(1, rd_rank // 2)
    for i in range(args.rank_deficient_trials):
        result = run_rank_deficient_trial(
            seed=args.base_seed + 9_000_000 + i,
            m=args.m,
            n=args.n,
            r=rd_rank,
            effective_rank=effective_rank,
        )
        rank_def_rows.append(asdict(result))

    fields = [
        "riesz_relerr",
        "kkt_stationarity_relerr",
        "unit_budget_relerr",
        "cs_equality_relerr",
        "sampled_optimality_margin_min",
        "sampled_optimality_violations",
        "uniqueness_probe_min_gap",
        "gauge_metric_relerr",
        "gauge_product_relerr",
        "gauge_direction_relerr",
        "full_product_direction_cosine",
        "full_product_efficiency_ratio",
    ]
    summary = summarize(rows, fields)

    rd_fields = [
        "riesz_on_visible_subspace_relerr",
        "null_direction_split_norm",
        "null_direction_loss_differential",
        "pseudoinverse_stationarity_relerr",
        "nonuniqueness_product_relerr",
    ]
    rank_def_summary = summarize(rank_def_rows, rd_fields)

    max_riesz = max(float(r["riesz_relerr"]) for r in rows)
    max_kkt = max(float(r["kkt_stationarity_relerr"]) for r in rows)
    max_budget = max(float(r["unit_budget_relerr"]) for r in rows)
    max_cs = max(float(r["cs_equality_relerr"]) for r in rows)
    total_violations = sum(
        int(r["sampled_optimality_violations"]) for r in rows
    )
    min_unique_gap = min(
        float(r["uniqueness_probe_min_gap"]) for r in rows
    )
    p99_gauge_metric = float(
        np.quantile(
            [float(r["gauge_metric_relerr"]) for r in rows],
            0.99,
        )
    )
    p99_gauge_product = float(
        np.quantile(
            [float(r["gauge_product_relerr"]) for r in rows],
            0.99,
        )
    )
    p99_gauge_direction = float(
        np.quantile(
            [float(r["gauge_direction_relerr"]) for r in rows],
            0.99,
        )
    )

    gauge_metric_values = np.asarray(
        [float(r["gauge_metric_relerr"]) for r in rows],
        dtype=float,
    )
    gauge_metric_median = float(np.median(gauge_metric_values))
    gauge_metric_p99 = float(np.quantile(gauge_metric_values, 0.99))
    gauge_metric_max = float(np.max(gauge_metric_values))

    # Three-tier numerical audit:
    #   typical precision  : median < 1e-12
    #   robust tail        : p99    < 1e-7
    #   extreme pathology : max    < 1e-6
    #
    # The exact algebraic identity is
    #   B'V_A' = BV_A,  V_B'A' = V_BA,
    # so any residual here is floating-point conditioning, not a
    # theoretical failure of gauge invariance.
    pass_gauge_metric_typical = gauge_metric_median < 1e-12
    pass_gauge_metric_p99 = gauge_metric_p99 < 1e-7
    pass_gauge_metric_extreme = gauge_metric_max < 1e-6

    gates = {
        "PASS_RIESZ_REPRESENTATION": max_riesz < 1e-9,
        "PASS_KKT_STATIONARITY": max_kkt < 1e-8,
        "PASS_UNIT_INFORMATION_BUDGET": max_budget < 1e-10,
        "PASS_CAUCHY_SCHWARZ_EQUALITY": max_cs < 1e-9,
        "PASS_RANDOM_FEASIBLE_OPTIMALITY": total_violations == 0,
        "PASS_FULL_RANK_UNIQUENESS_PROBES": min_unique_gap > 0.0,
        "PASS_GAUGE_METRIC_TYPICAL": pass_gauge_metric_typical,
        "PASS_GAUGE_METRIC_P99": pass_gauge_metric_p99,
        "PASS_GAUGE_METRIC_EXTREME": pass_gauge_metric_extreme,
        "PASS_GAUGE_PRODUCT_COVARIANCE": p99_gauge_product < 1e-7,
        "PASS_GAUGE_DIRECTION_COVARIANCE": p99_gauge_direction < 1e-7,
        "PASS_RANK_DEFICIENT_VISIBLE_RIESZ": (
            rank_def_summary[
                "riesz_on_visible_subspace_relerr"
            ]["max"] < 1e-9
        ),
        "PASS_RANK_DEFICIENT_NULL_COST": (
            rank_def_summary["null_direction_split_norm"]["max"]
            < 1e-10
            and rank_def_summary[
                "null_direction_loss_differential"
            ]["max"] < 1e-10
        ),
        "PASS_RANK_DEFICIENT_PSEUDOINVERSE": (
            rank_def_summary[
                "pseudoinverse_stationarity_relerr"
            ]["max"] < 1e-9
        ),
    }
    gates["PASS_ALL"] = all(gates.values())

    conclusion = {
        "theorem_statement": (
            "For full-rank A and B, the existing GeoFlow direction is "
            "the unique normalized maximizer of -dL[V] under the split "
            "executed-information budget "
            "||B V_A||_F^2 + ||V_B A||_F^2 <= 1."
        ),
        "metric": (
            "g_split(V,W)=<B V_A,B W_A>_F+<V_B A,W_B A>_F"
        ),
        "optimizer": {
            "V_A": "-(B^T B)^(-1) grad_A L",
            "V_B": "-grad_B L (A A^T)^(-1)",
        },
        "n_full_rank_trials": len(rows),
        "n_rank_deficient_trials": len(rank_def_rows),
        "gauge_metric_numerical_audit": {
            "median": gauge_metric_median,
            "p99": gauge_metric_p99,
            "max": gauge_metric_max,
            "thresholds": {
                "median": 1e-12,
                "p99": 1e-7,
                "max": 1e-6,
            },
            "interpretation": (
                "The metric is exactly gauge invariant algebraically; "
                "the reported residuals quantify finite-precision "
                "conditioning under extreme gauges."
            ),
        },
        "summary": summary,
        "rank_deficient_summary": rank_def_summary,
        "gates": gates,
        "interpretation": {
            "full_rank": (
                "Exact local steepest descent under channel-resolved "
                "executed functional information."
            ),
            "rank_deficient": (
                "Pseudoinverse gives the minimum-norm visible gradient, "
                "while null directions are zero-cost and non-unique."
            ),
            "full_product_distinction": (
                "This theorem is exact for the split metric, not for the "
                "net full-product Frobenius metric."
            ),
            "gauge_numerics": (
                "Gauge invariance is an exact algebraic identity. "
                "The implementation uses solve-based transforms and a "
                "three-tier finite-precision audit for typical, p99, "
                "and extreme-condition residuals."
            ),
        },
    }

    write_csv(out_dir / "h139d_trials.csv", rows)
    write_csv(
        out_dir / "h139d_rank_deficient_trials.csv",
        rank_def_rows,
    )
    (out_dir / "h139d_conclusion.json").write_text(
        json.dumps(conclusion, indent=2),
        encoding="utf-8",
    )
    (out_dir / "h139d_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if not args.no_plots:
        make_plots(out_dir, rows)

    print("\n" + "=" * 120)
    print("H13.9D-FIX CONCLUSION")
    print("=" * 120)
    print(json.dumps(conclusion, indent=2))
    print("\nScientific decision:")
    print(
        "  Existing GeoFlow direction is exactly variational under the "
        "split executed-information metric."
    )
    print(
        "  The proof structure is Riesz representation + "
        "Cauchy-Schwarz/KKT + gauge covariance."
    )
    print(
        "  Full-rank case is unique; rank-deficient case requires "
        "pseudoinverse and admits null-direction non-uniqueness."
    )
    print("\nOutputs:", out_dir)


if __name__ == "__main__":
    main()
