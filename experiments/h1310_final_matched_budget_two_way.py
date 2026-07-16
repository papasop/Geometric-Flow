#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H13.10-FINAL — MATCHED-BUDGET TWO-WAY STOCHASTIC VARIANCE DECOMPOSITION

Fixes relative to H13.10-v1
----------------------------
1. Exact gauge-covariance probes use the full-rank ordinary-inverse branch
   with ridge = 0. Regularized practical directions are audited separately.

2. Gauge residuals are relative and channel-resolved:
      eps_A = ||B' V_A' - B V_A|| / (||B' V_A'|| + ||B V_A|| + eps)
      eps_B = ||V_B' A' - V_B A|| / (||V_B' A'|| + ||V_B A|| + eps)
      eps_D = ||D' - D|| / (||D'|| + ||D|| + eps)

3. EMA probing uses the real pre-probe history state m_{t-1}. Each candidate
   minibatch forms m_t^(b) = beta m_{t-1} + (1-beta) g_b, while the full-batch
   reference uses the same history with g_full.

4. All training methods are matched to the same first-order product-step budget
   ||Delta M||_F = target_product_step, unless a numerical cap is hit.

5. Functional, gauge, and interaction variance use a two-way batch x gauge
   decomposition:
      D_{b,s} = mu + F_b + G_s + C_{b,s}
   with V_F, V_G, V_C reported separately.

Compared methods
----------------
- adamw_factor
- fixed_capacity_split
- legacy_k1_split
- full_product_corrected
- ema_geoflow_split

Outputs
-------
- per_trial.csv
- per_probe.csv
- method_summary.csv
- summary.json
- gates.json
- optional plots

Smoke test
----------
python h1310_final_matched_budget_two_way.py \
  --trials 2 --steps 30 --probe-batches 6 --probe-gauges 4 --no-plots

Default audit
-------------
python h1310_final_matched_budget_two_way.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

Tensor = torch.Tensor
EPS = 1e-15

METHODS = (
    "adamw_factor",
    "fixed_capacity_split",
    "legacy_k1_split",
    "full_product_corrected",
    "ema_geoflow_split",
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    seed: int = 1729
    trials: int = 6
    steps: int = 100
    samples: int = 1536
    input_dim: int = 24
    output_dim: int = 18
    rank: int = 4
    batch_size: int = 64

    probe_batches: int = 12
    probe_gauges: int = 6
    probe_steps: str = "0,25,50,75,99"

    gauge_kappa_min: float = 1.0
    gauge_kappa_max: float = 100.0
    factor_kappa: float = 20.0

    noise_std: float = 0.05
    weight_decay: float = 1e-4

    target_product_step: float = 0.05
    max_factor_step_norm: float = 100.0

    lr_adamw_shape: float = 1.0
    k1_gain: float = 0.30
    k1_min_scale: float = 0.35
    k1_max_scale: float = 2.0
    ema_beta: float = 0.92

    practical_ridge: float = 1e-8
    exact_condition_limit: float = 1e12

    dtype: str = "float64"
    device: str = "cpu"
    output_dir: str = "h1310_final_results"
    no_plots: bool = False


# =============================================================================
# Basic utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def ffloat(x: Tensor | float) -> float:
    if isinstance(x, Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def fnorm(x: Tensor) -> Tensor:
    return torch.linalg.norm(x)


def finner(x: Tensor, y: Tensor) -> Tensor:
    return torch.sum(x * y)


def relerr(x: Tensor, y: Tensor, eps: float = EPS) -> float:
    return ffloat(fnorm(x - y) / (fnorm(x) + fnorm(y) + eps))


def cosine(x: Tensor, y: Tensor, eps: float = EPS) -> float:
    return ffloat(finner(x, y) / (fnorm(x) * fnorm(y) + eps))


def qstats(values: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "p01", "p05", "median", "p95", "p99", "max")}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "min": float(a.min()),
        "p01": float(np.quantile(a, 0.01)),
        "p05": float(np.quantile(a, 0.05)),
        "median": float(np.quantile(a, 0.50)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def exact_left_solve(mat: Tensor, rhs: Tensor, cond_limit: float) -> Tensor:
    cond = ffloat(torch.linalg.cond(mat))
    if not math.isfinite(cond) or cond > cond_limit:
        raise RuntimeError(f"Exact branch condition too large: {cond:.3e}")
    return torch.linalg.solve(mat, rhs)


def exact_right_solve(rhs: Tensor, mat: Tensor, cond_limit: float) -> Tensor:
    return exact_left_solve(mat.T, rhs.T, cond_limit).T


def practical_left_solve(mat: Tensor, rhs: Tensor, ridge: float) -> Tensor:
    eye = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
    reg = mat + ridge * eye
    try:
        return torch.linalg.solve(reg, rhs)
    except RuntimeError:
        return torch.linalg.pinv(reg) @ rhs


def practical_right_solve(rhs: Tensor, mat: Tensor, ridge: float) -> Tensor:
    return practical_left_solve(mat.T, rhs.T, ridge).T


# =============================================================================
# Dataset and factorization
# =============================================================================

@dataclass
class Dataset:
    x: Tensor
    y: Tensor
    m_true: Tensor


def make_dataset(cfg: Config, seed: int) -> Dataset:
    dtype = get_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x = torch.randn(cfg.samples, cfg.input_dim, generator=gen, dtype=dtype, device=device)
    b_true = torch.randn(cfg.output_dim, cfg.rank, generator=gen, dtype=dtype, device=device)
    a_true = torch.randn(cfg.rank, cfg.input_dim, generator=gen, dtype=dtype, device=device)
    m_true = (b_true @ a_true) / math.sqrt(cfg.rank)
    y = x @ m_true.T
    y = y + cfg.noise_std * torch.randn(y.shape, generator=gen, dtype=dtype, device=device)
    return Dataset(x=x, y=y, m_true=m_true)


def batch_loss_grad_m(b: Tensor, a: Tensor, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    m = b @ a
    pred = x @ m.T
    err = pred - y
    loss = 0.5 * torch.mean(err ** 2)
    g_m = (err.T @ x) / (x.shape[0] * y.shape[1])
    return loss, g_m


def factor_grads(b: Tensor, a: Tensor, g_m: Tensor) -> Tuple[Tensor, Tensor]:
    return g_m @ a.T, b.T @ g_m


def full_loss(b: Tensor, a: Tensor, data: Dataset) -> float:
    loss, _ = batch_loss_grad_m(b, a, data.x, data.y)
    return ffloat(loss)


def rank_r_product_init(cfg: Config, seed: int) -> Tensor:
    dtype = get_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    m = 0.08 * torch.randn(cfg.output_dim, cfg.input_dim, generator=gen, dtype=dtype, device=device)
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    return u[:, :cfg.rank] @ torch.diag(s[:cfg.rank]) @ vh[:cfg.rank, :]


def factorize_product(m: Tensor, rank: int, kappa: float) -> Tuple[Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    u = u[:, :rank]
    s = torch.clamp(s[:rank], min=EPS)
    vh = vh[:rank, :]
    root = torch.sqrt(s)
    scales = torch.logspace(
        -0.5 * math.log10(max(kappa, 1.0)),
        0.5 * math.log10(max(kappa, 1.0)),
        rank,
        dtype=m.dtype,
        device=m.device,
    )
    b = u @ torch.diag(root * scales)
    a = torch.diag(root / scales) @ vh
    return b, a


def random_gauge(rank: int, kappa: float, dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> Tensor:
    q1, _ = torch.linalg.qr(torch.randn(rank, rank, generator=gen, dtype=dtype, device=device))
    q2, _ = torch.linalg.qr(torch.randn(rank, rank, generator=gen, dtype=dtype, device=device))
    sv = torch.logspace(0.0, math.log10(max(kappa, 1.0)), rank, dtype=dtype, device=device)
    return q1 @ torch.diag(sv) @ q2.T


def gauge_transform(b: Tensor, a: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
    return torch.linalg.solve(s.T, b.T).T, s @ a


# =============================================================================
# Directions
# =============================================================================

@dataclass
class Direction:
    v_b: Tensor
    v_a: Tensor
    channel_a: Tensor
    channel_b: Tensor
    d_product: Tensor
    split_norm: float
    product_norm: float
    descent: float


def build_split_direction(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    *,
    exact: bool,
    cond_limit: float,
    ridge: float,
    override_g_b: Optional[Tensor] = None,
    override_g_a: Optional[Tensor] = None,
) -> Direction:
    g_b, g_a = factor_grads(b, a, g_m)
    if override_g_b is not None:
        g_b = override_g_b
    if override_g_a is not None:
        g_a = override_g_a

    gb = b.T @ b
    ga = a @ a.T
    if exact:
        v_a = -exact_left_solve(gb, g_a, cond_limit)
        v_b = -exact_right_solve(g_b, ga, cond_limit)
    else:
        v_a = -practical_left_solve(gb, g_a, ridge)
        v_b = -practical_right_solve(g_b, ga, ridge)

    ch_a = b @ v_a
    ch_b = v_b @ a
    d = ch_a + ch_b
    return Direction(
        v_b=v_b,
        v_a=v_a,
        channel_a=ch_a,
        channel_b=ch_b,
        d_product=d,
        split_norm=ffloat(torch.sqrt(fnorm(ch_a) ** 2 + fnorm(ch_b) ** 2)),
        product_norm=ffloat(fnorm(d)),
        descent=ffloat(-finner(g_m, d)),
    )


def build_full_product_direction(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    *,
    exact: bool,
    cond_limit: float,
    ridge: float,
) -> Direction:
    gb = b.T @ b
    ga = a @ a.T

    if exact:
        pbg = b @ exact_left_solve(gb, b.T @ g_m, cond_limit)
        gpa = exact_right_solve(g_m @ a.T, ga, cond_limit) @ a
        pbgpa = b @ exact_left_solve(gb, b.T @ gpa, cond_limit)
    else:
        pbg = b @ practical_left_solve(gb, b.T @ g_m, ridge)
        gpa = practical_right_solve(g_m @ a.T, ga, ridge) @ a
        pbgpa = b @ practical_left_solve(gb, b.T @ gpa, ridge)

    d_star = -(pbg + gpa - pbgpa)

    # Minimum Euclidean factor lift via pseudoinverse of tangent map.
    m, r = b.shape
    _, n = a.shape
    eye_m = torch.eye(m, dtype=b.dtype, device=b.device)
    eye_n = torch.eye(n, dtype=b.dtype, device=b.device)
    j_b = torch.kron(a.T.contiguous(), eye_m)
    j_a = torch.kron(eye_n, b)
    j = torch.cat([j_b, j_a], dim=1)
    target = d_star.T.contiguous().reshape(-1, 1)
    sol = torch.linalg.pinv(j) @ target
    v_b = sol[: m * r].reshape(r, m).T
    v_a = sol[m * r :].reshape(n, r).T

    ch_a = b @ v_a
    ch_b = v_b @ a
    d = ch_a + ch_b
    return Direction(
        v_b=v_b,
        v_a=v_a,
        channel_a=ch_a,
        channel_b=ch_b,
        d_product=d,
        split_norm=ffloat(torch.sqrt(fnorm(ch_a) ** 2 + fnorm(ch_b) ** 2)),
        product_norm=ffloat(fnorm(d)),
        descent=ffloat(-finner(g_m, d)),
    )


# =============================================================================
# Optimizer state
# =============================================================================

@dataclass
class AdamState:
    m_b: Tensor
    v_b: Tensor
    m_a: Tensor
    v_a: Tensor
    t: int = 0


@dataclass
class EmaState:
    m_b: Tensor
    m_a: Tensor
    t: int = 0


@dataclass
class MethodState:
    b: Tensor
    a: Tensor
    adam: Optional[AdamState]
    ema: Optional[EmaState]
    k1_scale: float = 1.0
    previous_full_loss: Optional[float] = None


def init_method_state(b: Tensor, a: Tensor, method: str) -> MethodState:
    adam = None
    ema = None
    if method == "adamw_factor":
        adam = AdamState(
            m_b=torch.zeros_like(b),
            v_b=torch.zeros_like(b),
            m_a=torch.zeros_like(a),
            v_a=torch.zeros_like(a),
        )
    if method == "ema_geoflow_split":
        ema = EmaState(
            m_b=torch.zeros_like(b),
            m_a=torch.zeros_like(a),
        )
    return MethodState(b=b.clone(), a=a.clone(), adam=adam, ema=ema)


def adam_shape_direction(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    state: AdamState,
    weight_decay: float,
    update_state: bool = True,
) -> Direction:
    g_b, g_a = factor_grads(b, a, g_m)

    if update_state:
        state.t += 1
        state.m_b.mul_(0.9).add_(g_b, alpha=0.1)
        state.v_b.mul_(0.999).addcmul_(g_b, g_b, value=0.001)
        state.m_a.mul_(0.9).add_(g_a, alpha=0.1)
        state.v_a.mul_(0.999).addcmul_(g_a, g_a, value=0.001)

    t = max(state.t, 1)
    mb = state.m_b / (1.0 - 0.9 ** t)
    vb = state.v_b / (1.0 - 0.999 ** t)
    ma = state.m_a / (1.0 - 0.9 ** t)
    va = state.v_a / (1.0 - 0.999 ** t)

    v_b = -mb / (torch.sqrt(vb) + 1e-8) - weight_decay * b
    v_a = -ma / (torch.sqrt(va) + 1e-8) - weight_decay * a
    ch_a = b @ v_a
    ch_b = v_b @ a
    d = ch_a + ch_b
    return Direction(
        v_b=v_b,
        v_a=v_a,
        channel_a=ch_a,
        channel_b=ch_b,
        d_product=d,
        split_norm=ffloat(torch.sqrt(fnorm(ch_a) ** 2 + fnorm(ch_b) ** 2)),
        product_norm=ffloat(fnorm(d)),
        descent=ffloat(-finner(g_m, d)),
    )



def adam_candidate_direction(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    state: AdamState,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Direction:
    """
    Build the next AdamW direction from a frozen pre-probe history state.

    The real optimizer state is not mutated. Each candidate minibatch contributes
    through candidate next moments, making minibatch variance and alignment
    meaningful.
    """
    g_b, g_a = factor_grads(b, a, g_m)
    t = state.t + 1

    m_b = beta1 * state.m_b + (1.0 - beta1) * g_b
    v_b = beta2 * state.v_b + (1.0 - beta2) * (g_b * g_b)
    m_a = beta1 * state.m_a + (1.0 - beta1) * g_a
    v_a = beta2 * state.v_a + (1.0 - beta2) * (g_a * g_a)

    m_b_hat = m_b / (1.0 - beta1 ** t)
    v_b_hat = v_b / (1.0 - beta2 ** t)
    m_a_hat = m_a / (1.0 - beta1 ** t)
    v_a_hat = v_a / (1.0 - beta2 ** t)

    dir_b = -m_b_hat / (torch.sqrt(v_b_hat) + eps) - weight_decay * b
    dir_a = -m_a_hat / (torch.sqrt(v_a_hat) + eps) - weight_decay * a

    ch_a = b @ dir_a
    ch_b = dir_b @ a
    d = ch_a + ch_b
    return Direction(
        v_b=dir_b,
        v_a=dir_a,
        channel_a=ch_a,
        channel_b=ch_b,
        d_product=d,
        split_norm=ffloat(torch.sqrt(fnorm(ch_a) ** 2 + fnorm(ch_b) ** 2)),
        product_norm=ffloat(fnorm(d)),
        descent=ffloat(-finner(g_m, d)),
    )


def ema_candidate_direction(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    history: EmaState,
    beta: float,
    *,
    exact: bool,
    cond_limit: float,
    ridge: float,
) -> Direction:
    g_b, g_a = factor_grads(b, a, g_m)
    cand_b = beta * history.m_b + (1.0 - beta) * g_b
    cand_a = beta * history.m_a + (1.0 - beta) * g_a
    t = history.t + 1
    bias = 1.0 - beta ** t
    cand_b = cand_b / max(bias, EPS)
    cand_a = cand_a / max(bias, EPS)
    return build_split_direction(
        b, a, g_m,
        exact=exact,
        cond_limit=cond_limit,
        ridge=ridge,
        override_g_b=cand_b,
        override_g_a=cand_a,
    )


def update_ema_history(history: EmaState, b: Tensor, a: Tensor, g_m: Tensor, beta: float) -> None:
    g_b, g_a = factor_grads(b, a, g_m)
    history.m_b.mul_(beta).add_(g_b, alpha=1.0 - beta)
    history.m_a.mul_(beta).add_(g_a, alpha=1.0 - beta)
    history.t += 1


# =============================================================================
# Matched product-step update
# =============================================================================

@dataclass
class AppliedStep:
    scale: float
    exact_first_order_norm: float
    realized_product_norm: float
    cap_hit: bool


def apply_matched_product_step(
    b: Tensor,
    a: Tensor,
    direction: Direction,
    target: float,
    max_factor_step_norm: float,
    extra_scale: float = 1.0,
) -> Tuple[Tensor, Tensor, AppliedStep]:
    base = target / max(direction.product_norm, EPS)
    scale = base * extra_scale

    factor_norm = math.sqrt(ffloat(fnorm(direction.v_b) ** 2 + fnorm(direction.v_a) ** 2))
    cap_hit = False
    if scale * factor_norm > max_factor_step_norm:
        scale = max_factor_step_norm / max(factor_norm, EPS)
        cap_hit = True

    step_b = scale * direction.v_b
    step_a = scale * direction.v_a
    b2 = b + step_b
    a2 = a + step_a
    realized = ffloat(fnorm(b2 @ a2 - b @ a))
    return b2, a2, AppliedStep(
        scale=scale,
        exact_first_order_norm=scale * direction.product_norm,
        realized_product_norm=realized,
        cap_hit=cap_hit,
    )


# =============================================================================
# Two-way variance decomposition
# =============================================================================

@dataclass
class TwoWayMetrics:
    v_functional: float
    v_gauge: float
    v_interaction: float
    v_total: float
    decomposition_relerr: float

    gauge_channel_a_median: float
    gauge_channel_a_p99: float
    gauge_channel_b_median: float
    gauge_channel_b_p99: float
    gauge_product_median: float
    gauge_product_p99: float

    practical_gauge_product_median: float
    practical_gauge_product_p99: float

    full_batch_alignment_mean: float
    full_batch_alignment_min: float
    exact_probe_skips: int


def decompose_two_way(directions: Tensor) -> Tuple[float, float, float, float, float]:
    """
    directions shape: [B, S, m, n]
    D_bs = mu + F_b + G_s + C_bs
    """
    mu = directions.mean(dim=(0, 1), keepdim=True)
    mean_s = directions.mean(dim=1, keepdim=True)
    mean_b = directions.mean(dim=0, keepdim=True)

    f = mean_s - mu
    g = mean_b - mu
    c = directions - mu - f - g

    vf = f.pow(2).sum(dim=(-2, -1)).mean()
    vg = g.pow(2).sum(dim=(-2, -1)).mean()
    vc = c.pow(2).sum(dim=(-2, -1)).mean()
    vt = (directions - mu).pow(2).sum(dim=(-2, -1)).mean()
    rel = torch.abs(vt - (vf + vg + vc)) / (torch.abs(vt) + EPS)
    return ffloat(vf), ffloat(vg), ffloat(vc), ffloat(vt), ffloat(rel)


def probe_two_way(
    state: MethodState,
    method: str,
    data: Dataset,
    cfg: Config,
    seed: int,
) -> TwoWayMetrics:
    b = state.b
    a = state.a
    device = b.device
    dtype = b.dtype
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Fixed batch set.
    batch_indices = [
        torch.randint(0, data.x.shape[0], (cfg.batch_size,), generator=gen, device=device)
        for _ in range(cfg.probe_batches)
    ]

    # Fixed gauge set, including identity.
    gauges: List[Tensor] = [torch.eye(cfg.rank, dtype=dtype, device=device)]
    if cfg.probe_gauges > 1:
        kappas = np.geomspace(
            max(cfg.gauge_kappa_min, 1.0),
            max(cfg.gauge_kappa_max, cfg.gauge_kappa_min),
            cfg.probe_gauges - 1,
        )
        for kappa in kappas:
            gauges.append(random_gauge(cfg.rank, float(kappa), dtype, device, gen))

    # Full-batch reference in original gauge, using real history state for EMA.
    _, g_full = batch_loss_grad_m(b, a, data.x, data.y)
    if method == "ema_geoflow_split":
        assert state.ema is not None
        ref = ema_candidate_direction(
            b, a, g_full, state.ema, cfg.ema_beta,
            exact=True,
            cond_limit=cfg.exact_condition_limit,
            ridge=0.0,
        )
    elif method == "full_product_corrected":
        ref = build_full_product_direction(
            b, a, g_full,
            exact=True,
            cond_limit=cfg.exact_condition_limit,
            ridge=0.0,
        )
    elif method == "adamw_factor":
        assert state.adam is not None
        # Candidate next AdamW direction from the same frozen history.
        ref = adam_candidate_direction(
            b, a, g_full, state.adam, cfg.weight_decay
        )
    else:
        ref = build_split_direction(
            b, a, g_full,
            exact=True,
            cond_limit=cfg.exact_condition_limit,
            ridge=0.0,
        )

    grid: List[List[Tensor]] = []
    eps_a: List[float] = []
    eps_b: List[float] = []
    eps_d: List[float] = []
    practical_eps_d: List[float] = []
    aligns: List[float] = []
    skips = 0

    # Reference exact directions in identity gauge for each batch.
    identity_dirs: List[Direction] = []

    for idx in batch_indices:
        xb, yb = data.x[idx], data.y[idx]
        _, g_mb = batch_loss_grad_m(b, a, xb, yb)

        if method == "ema_geoflow_split":
            assert state.ema is not None
            d0 = ema_candidate_direction(
                b, a, g_mb, state.ema, cfg.ema_beta,
                exact=True,
                cond_limit=cfg.exact_condition_limit,
                ridge=0.0,
            )
        elif method == "full_product_corrected":
            d0 = build_full_product_direction(
                b, a, g_mb,
                exact=True,
                cond_limit=cfg.exact_condition_limit,
                ridge=0.0,
            )
        elif method == "adamw_factor":
            assert state.adam is not None
            d0 = adam_candidate_direction(
                b, a, g_mb, state.adam, cfg.weight_decay
            )
        else:
            d0 = build_split_direction(
                b, a, g_mb,
                exact=True,
                cond_limit=cfg.exact_condition_limit,
                ridge=0.0,
            )

        identity_dirs.append(d0)
        aligns.append(cosine(d0.d_product, ref.d_product))

    for bi, idx in enumerate(batch_indices):
        xb, yb = data.x[idx], data.y[idx]
        _, g_mb = batch_loss_grad_m(b, a, xb, yb)
        row: List[Tensor] = []

        for s in gauges:
            bg, ag = gauge_transform(b, a, s)

            try:
                if method == "ema_geoflow_split":
                    # Transform real history to the gauge coordinates.
                    assert state.ema is not None
                    hist_b_g = state.ema.m_b @ s.T
                    hist_a_g = torch.linalg.solve(s.T, state.ema.m_a)
                    hist_g = EmaState(m_b=hist_b_g, m_a=hist_a_g, t=state.ema.t)
                    dg = ema_candidate_direction(
                        bg, ag, g_mb, hist_g, cfg.ema_beta,
                        exact=True,
                        cond_limit=cfg.exact_condition_limit,
                        ridge=0.0,
                    )
                elif method == "full_product_corrected":
                    dg = build_full_product_direction(
                        bg, ag, g_mb,
                        exact=True,
                        cond_limit=cfg.exact_condition_limit,
                        ridge=0.0,
                    )
                elif method == "adamw_factor":
                    # Adam state is deliberately not transformed covariantly:
                    # this exposes representation dependence of coordinate moments.
                    assert state.adam is not None
                    dg = adam_candidate_direction(
                        bg, ag, g_mb, state.adam, cfg.weight_decay
                    )
                else:
                    dg = build_split_direction(
                        bg, ag, g_mb,
                        exact=True,
                        cond_limit=cfg.exact_condition_limit,
                        ridge=0.0,
                    )
            except RuntimeError:
                skips += 1
                # Keep grid rectangular using NaN marker; caller filters later.
                row.append(torch.full_like(identity_dirs[bi].d_product, float("nan")))
                continue

            row.append(dg.d_product)

            d0 = identity_dirs[bi]
            eps_a.append(relerr(dg.channel_a, d0.channel_a))
            eps_b.append(relerr(dg.channel_b, d0.channel_b))
            eps_d.append(relerr(dg.d_product, d0.d_product))

            # Practical regularized branch audit.
            if method in ("fixed_capacity_split", "legacy_k1_split", "ema_geoflow_split"):
                if method == "ema_geoflow_split":
                    assert state.ema is not None
                    hist_b_g = state.ema.m_b @ s.T
                    hist_a_g = torch.linalg.solve(s.T, state.ema.m_a)
                    hist_g = EmaState(m_b=hist_b_g, m_a=hist_a_g, t=state.ema.t)
                    dpr = ema_candidate_direction(
                        bg, ag, g_mb, hist_g, cfg.ema_beta,
                        exact=False,
                        cond_limit=cfg.exact_condition_limit,
                        ridge=cfg.practical_ridge,
                    )
                else:
                    dpr = build_split_direction(
                        bg, ag, g_mb,
                        exact=False,
                        cond_limit=cfg.exact_condition_limit,
                        ridge=cfg.practical_ridge,
                    )
                if method == "ema_geoflow_split":
                    dpr0 = ema_candidate_direction(
                        b, a, g_mb, state.ema, cfg.ema_beta,
                        exact=False,
                        cond_limit=cfg.exact_condition_limit,
                        ridge=cfg.practical_ridge,
                    )
                else:
                    dpr0 = build_split_direction(
                        b, a, g_mb,
                        exact=False,
                        cond_limit=cfg.exact_condition_limit,
                        ridge=cfg.practical_ridge,
                    )
                practical_eps_d.append(relerr(dpr.d_product, dpr0.d_product))

        grid.append(row)

    grid_t = torch.stack([torch.stack(r, dim=0) for r in grid], dim=0)

    # Remove gauge columns containing NaN so ANOVA stays valid.
    valid_cols = torch.isfinite(grid_t).all(dim=(0, 2, 3))
    grid_t = grid_t[:, valid_cols]
    if grid_t.shape[1] < 1:
        raise RuntimeError("No valid exact gauge columns remain")

    vf, vg, vc, vt, decomp_rel = decompose_two_way(grid_t)

    sa = qstats(eps_a)
    sb = qstats(eps_b)
    sd = qstats(eps_d)
    sp = qstats(practical_eps_d) if practical_eps_d else qstats([float("nan")])

    return TwoWayMetrics(
        v_functional=vf,
        v_gauge=vg,
        v_interaction=vc,
        v_total=vt,
        decomposition_relerr=decomp_rel,
        gauge_channel_a_median=sa["median"],
        gauge_channel_a_p99=sa["p99"],
        gauge_channel_b_median=sb["median"],
        gauge_channel_b_p99=sb["p99"],
        gauge_product_median=sd["median"],
        gauge_product_p99=sd["p99"],
        practical_gauge_product_median=sp["median"],
        practical_gauge_product_p99=sp["p99"],
        full_batch_alignment_mean=float(np.mean(aligns)),
        full_batch_alignment_min=float(np.min(aligns)),
        exact_probe_skips=skips,
    )


# =============================================================================
# Training
# =============================================================================

def parse_probe_steps(spec: str, steps: int) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v < 0:
            v = steps + v
        out.append(max(0, min(steps - 1, v)))
    return sorted(set(out))


def train_method(
    method: str,
    b0: Tensor,
    a0: Tensor,
    data: Dataset,
    cfg: Config,
    trial: int,
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    state = init_method_state(b0, a0, method)
    device = b0.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    initial_loss = full_loss(state.b, state.a, data)
    probes: List[TwoWayMetrics] = []
    probe_rows: List[Dict[str, object]] = []
    cap_hits = 0
    realized_steps: List[float] = []

    probe_steps = parse_probe_steps(cfg.probe_steps, cfg.steps)
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        idx = torch.randint(0, data.x.shape[0], (cfg.batch_size,), generator=gen, device=device)
        xb, yb = data.x[idx], data.y[idx]
        _, g_m = batch_loss_grad_m(state.b, state.a, xb, yb)

        if method == "adamw_factor":
            assert state.adam is not None
            direction = adam_shape_direction(
                state.b, state.a, g_m, state.adam, cfg.weight_decay, update_state=True
            )
            extra = cfg.lr_adamw_shape

        elif method == "ema_geoflow_split":
            assert state.ema is not None
            direction = ema_candidate_direction(
                state.b, state.a, g_m, state.ema, cfg.ema_beta,
                exact=False,
                cond_limit=cfg.exact_condition_limit,
                ridge=cfg.practical_ridge,
            )
            update_ema_history(state.ema, state.b, state.a, g_m, cfg.ema_beta)
            extra = 1.0

        elif method == "full_product_corrected":
            direction = build_full_product_direction(
                state.b, state.a, g_m,
                exact=False,
                cond_limit=cfg.exact_condition_limit,
                ridge=cfg.practical_ridge,
            )
            extra = 1.0

        else:
            direction = build_split_direction(
                state.b, state.a, g_m,
                exact=False,
                cond_limit=cfg.exact_condition_limit,
                ridge=cfg.practical_ridge,
            )
            if method == "legacy_k1_split":
                cur = full_loss(state.b, state.a, data)
                if state.previous_full_loss is not None:
                    improvement = state.previous_full_loss - cur
                    ratio = improvement / 1e-4
                    state.k1_scale *= math.exp(cfg.k1_gain * math.tanh(ratio - 1.0))
                    state.k1_scale = min(cfg.k1_max_scale, max(cfg.k1_min_scale, state.k1_scale))
                state.previous_full_loss = cur

                # Matched-budget protocol: K1's native requested scale is retained
                # as a diagnostic, but it does not multiply the final product-step
                # norm in the fair comparison.
                extra = 1.0
            else:
                extra = 1.0

        state.b, state.a, applied = apply_matched_product_step(
            state.b,
            state.a,
            direction,
            target=cfg.target_product_step,
            max_factor_step_norm=cfg.max_factor_step_norm,
            extra_scale=extra,
        )
        realized_steps.append(applied.realized_product_norm)
        cap_hits += int(applied.cap_hit)

        if step in probe_steps:
            probe = probe_two_way(
                state,
                method,
                data,
                cfg,
                seed + 900000 + step,
            )
            probes.append(probe)
            probe_rows.append({
                "trial": trial,
                "method": method,
                "step": step,
                "full_loss": full_loss(state.b, state.a, data),
                "product_error": ffloat(fnorm(state.b @ state.a - data.m_true) / (fnorm(data.m_true) + EPS)),
                **asdict(probe),
            })

    elapsed = time.perf_counter() - t0
    final = full_loss(state.b, state.a, data)

    def avg(name: str) -> float:
        vals = [float(getattr(p, name)) for p in probes]
        return float(np.mean(vals))

    summary = {
        "trial": trial,
        "seed": seed,
        "method": method,
        "initial_loss": initial_loss,
        "final_loss": final,
        "loss_improvement": initial_loss - final,
        "final_product_error": ffloat(fnorm(state.b @ state.a - data.m_true) / (fnorm(data.m_true) + EPS)),
        "elapsed_sec": elapsed,
        "mean_realized_product_step": float(np.mean(realized_steps)),
        "std_realized_product_step": float(np.std(realized_steps)),
        "cap_hits": cap_hits,
        "requested_k1_scale_final": state.k1_scale if method == "legacy_k1_split" else 1.0,
        "v_functional": avg("v_functional"),
        "v_gauge": avg("v_gauge"),
        "v_interaction": avg("v_interaction"),
        "v_total": avg("v_total"),
        "decomposition_relerr": avg("decomposition_relerr"),
        "gauge_channel_a_median": avg("gauge_channel_a_median"),
        "gauge_channel_a_p99": avg("gauge_channel_a_p99"),
        "gauge_channel_b_median": avg("gauge_channel_b_median"),
        "gauge_channel_b_p99": avg("gauge_channel_b_p99"),
        "gauge_product_median": avg("gauge_product_median"),
        "gauge_product_p99": avg("gauge_product_p99"),
        "practical_gauge_product_median": avg("practical_gauge_product_median"),
        "practical_gauge_product_p99": avg("practical_gauge_product_p99"),
        "full_batch_alignment_mean": avg("full_batch_alignment_mean"),
        "full_batch_alignment_min": avg("full_batch_alignment_min"),
        "exact_probe_skips": int(sum(p.exact_probe_skips for p in probes)),
    }
    return summary, probe_rows


# =============================================================================
# Aggregation and gates
# =============================================================================

def summarize_methods(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    keys = [
        "initial_loss", "final_loss", "loss_improvement", "final_product_error",
        "elapsed_sec", "mean_realized_product_step", "std_realized_product_step",
        "cap_hits", "requested_k1_scale_final",
        "v_functional", "v_gauge", "v_interaction", "v_total",
        "decomposition_relerr",
        "gauge_channel_a_median", "gauge_channel_a_p99",
        "gauge_channel_b_median", "gauge_channel_b_p99",
        "gauge_product_median", "gauge_product_p99",
        "practical_gauge_product_median", "practical_gauge_product_p99",
        "full_batch_alignment_mean", "full_batch_alignment_min",
        "exact_probe_skips",
    ]
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        row: Dict[str, object] = {"method": method, "n": len(subset)}
        for key in keys:
            st = qstats(float(x[key]) for x in subset)
            for sk, sv in st.items():
                row[f"{key}_{sk}"] = sv
        out.append(row)
    return out


def build_gates(summary: List[Dict[str, object]], cfg: Config) -> Dict[str, bool]:
    m = {str(r["method"]): r for r in summary}

    def val(method: str, key: str, default: float = float("nan")) -> float:
        return float(m.get(method, {}).get(key, default))

    exact_methods = ("fixed_capacity_split", "legacy_k1_split", "full_product_corrected", "ema_geoflow_split")

    gates = {
        "PASS_MATCHED_PRODUCT_STEP":
            max(abs(val(x, "mean_realized_product_step_mean") - cfg.target_product_step) for x in METHODS)
            < 0.25 * cfg.target_product_step,

        "PASS_SPLIT_CHANNEL_A_COVARIANCE":
            val("fixed_capacity_split", "gauge_channel_a_p99_mean", 1.0) < 1e-7,

        "PASS_SPLIT_CHANNEL_B_COVARIANCE":
            val("fixed_capacity_split", "gauge_channel_b_p99_mean", 1.0) < 1e-7,

        "PASS_SPLIT_PRODUCT_COVARIANCE":
            val("fixed_capacity_split", "gauge_product_p99_mean", 1.0) < 1e-7,

        "PASS_FULL_PRODUCT_COVARIANCE":
            val("full_product_corrected", "gauge_product_p99_mean", 1.0) < 1e-7,

        "PASS_TWO_WAY_DECOMPOSITION":
            max(val(x, "decomposition_relerr_max", 1.0) for x in METHODS) < 1e-10,

        "PASS_NO_EXACT_PROBE_SKIPS":
            all(val(x, "exact_probe_skips_max", 1.0) == 0.0 for x in exact_methods),

        "PASS_ALL_METHODS_IMPROVE_LOSS":
            all(val(x, "loss_improvement_mean", -1.0) > 0.0 for x in METHODS),

        "PASS_FINITE_RESULTS":
            all(
                math.isfinite(float(v))
                for row in summary
                for k, v in row.items()
                if (
                    isinstance(v, (float, int))
                    and k != "n"
                    and not k.startswith("practical_gauge_product_")
                )
            ),

        # Hypothesis gates: informative, not required for PASS_CORE.
        "HYPOTHESIS_EMA_REDUCES_FUNCTIONAL_VARIANCE":
            val("ema_geoflow_split", "v_functional_mean", float("inf"))
            < val("fixed_capacity_split", "v_functional_mean", -float("inf")),

        "HYPOTHESIS_EMA_IMPROVES_ALIGNMENT":
            val("ema_geoflow_split", "full_batch_alignment_mean_mean", -1.0)
            > val("fixed_capacity_split", "full_batch_alignment_mean_mean", 2.0),

        "HYPOTHESIS_FULL_PRODUCT_REDUCES_FUNCTIONAL_VARIANCE":
            val("full_product_corrected", "v_functional_mean", float("inf"))
            < val("fixed_capacity_split", "v_functional_mean", -float("inf")),

        "HYPOTHESIS_INTERACTION_IS_NONNEGLIGIBLE":
            val("fixed_capacity_split", "v_interaction_mean", 0.0)
            > 0.05 * max(val("fixed_capacity_split", "v_total_mean", 1.0), EPS),
    }

    required = [
        "PASS_MATCHED_PRODUCT_STEP",
        "PASS_SPLIT_CHANNEL_A_COVARIANCE",
        "PASS_SPLIT_CHANNEL_B_COVARIANCE",
        "PASS_SPLIT_PRODUCT_COVARIANCE",
        "PASS_FULL_PRODUCT_COVARIANCE",
        "PASS_TWO_WAY_DECOMPOSITION",
        "PASS_NO_EXACT_PROBE_SKIPS",
        "PASS_ALL_METHODS_IMPROVE_LOSS",
        "PASS_FINITE_RESULTS",
    ]
    gates["PASS_CORE"] = all(gates[k] for k in required)
    return gates


# =============================================================================
# Plotting
# =============================================================================

def make_plots(out_dir: Path, summary: List[Dict[str, object]]) -> None:
    if not summary:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping plots")
        return

    methods = [str(r["method"]) for r in summary]
    metrics = [
        ("final_loss_mean", "Final full-batch loss", "final_loss.png"),
        ("v_functional_mean", "Two-way functional variance", "v_functional.png"),
        ("v_gauge_mean", "Two-way gauge variance", "v_gauge.png"),
        ("v_interaction_mean", "Two-way interaction variance", "v_interaction.png"),
        ("full_batch_alignment_mean_mean", "Alignment to full-batch direction", "alignment.png"),
        ("gauge_product_p99_mean", "Exact product covariance p99 residual", "gauge_product_p99.png"),
    ]
    for key, title, filename in metrics:
        vals = [float(r[key]) for r in summary]
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(range(len(methods)), vals)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(key)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


# =============================================================================
# CLI and main
# =============================================================================

def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--samples", type=int, default=1536)
    p.add_argument("--input-dim", type=int, default=24)
    p.add_argument("--output-dim", type=int, default=18)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--probe-batches", type=int, default=12)
    p.add_argument("--probe-gauges", type=int, default=6)
    p.add_argument("--probe-steps", default="0,25,50,75,99")

    p.add_argument("--gauge-kappa-min", type=float, default=1.0)
    p.add_argument("--gauge-kappa-max", type=float, default=100.0)
    p.add_argument("--factor-kappa", type=float, default=20.0)

    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--target-product-step", type=float, default=0.05)
    p.add_argument("--max-factor-step-norm", type=float, default=100.0)
    p.add_argument("--lr-adamw-shape", type=float, default=1.0)

    p.add_argument("--k1-gain", type=float, default=0.30)
    p.add_argument("--k1-min-scale", type=float, default=0.35)
    p.add_argument("--k1-max-scale", type=float, default=2.0)
    p.add_argument("--ema-beta", type=float, default=0.92)

    p.add_argument("--practical-ridge", type=float, default=1e-8)
    p.add_argument("--exact-condition-limit", type=float, default=1e12)

    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="h1310_final_results")
    p.add_argument("--no-plots", action="store_true")

    ns, unknown = p.parse_known_args()
    if unknown:
        print(f"[H13.10-FINAL] ignored notebook/kernel arguments: {unknown}")

    return Config(
        seed=ns.seed,
        trials=ns.trials,
        steps=ns.steps,
        samples=ns.samples,
        input_dim=ns.input_dim,
        output_dim=ns.output_dim,
        rank=ns.rank,
        batch_size=ns.batch_size,
        probe_batches=ns.probe_batches,
        probe_gauges=ns.probe_gauges,
        probe_steps=ns.probe_steps,
        gauge_kappa_min=ns.gauge_kappa_min,
        gauge_kappa_max=ns.gauge_kappa_max,
        factor_kappa=ns.factor_kappa,
        noise_std=ns.noise_std,
        weight_decay=ns.weight_decay,
        target_product_step=ns.target_product_step,
        max_factor_step_norm=ns.max_factor_step_norm,
        lr_adamw_shape=ns.lr_adamw_shape,
        k1_gain=ns.k1_gain,
        k1_min_scale=ns.k1_min_scale,
        k1_max_scale=ns.k1_max_scale,
        ema_beta=ns.ema_beta,
        practical_ridge=ns.practical_ridge,
        exact_condition_limit=ns.exact_condition_limit,
        dtype=ns.dtype,
        device=ns.device,
        output_dir=ns.output_dir,
        no_plots=ns.no_plots,
    )


def validate(cfg: Config) -> None:
    if cfg.rank <= 0 or cfg.rank > min(cfg.input_dim, cfg.output_dim):
        raise ValueError("Invalid rank")
    if cfg.batch_size > cfg.samples:
        raise ValueError("batch_size must not exceed samples")
    if cfg.probe_batches < 2 or cfg.probe_gauges < 2:
        raise ValueError("probe_batches and probe_gauges must be at least 2")
    if cfg.target_product_step <= 0:
        raise ValueError("target_product_step must be positive")


def main() -> None:
    cfg = parse_args()
    validate(cfg)
    set_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("H13.10-FINAL MATCHED-BUDGET TWO-WAY VARIANCE DECOMPOSITION")
    print("=" * 120)
    print(json.dumps(asdict(cfg), indent=2))
    print(f"torch={torch.__version__} python={platform.python_version()}")

    dtype = get_dtype(cfg.dtype)
    device = torch.device(cfg.device)

    trial_rows: List[Dict[str, object]] = []
    probe_rows: List[Dict[str, object]] = []

    for trial in range(cfg.trials):
        seed = cfg.seed + 1009 * trial
        set_seed(seed)
        data = make_dataset(cfg, seed)
        m0 = rank_r_product_init(cfg, seed + 17)
        b0, a0 = factorize_product(m0, cfg.rank, cfg.factor_kappa)

        print(f"\n[trial {trial+1}/{cfg.trials}] seed={seed}")

        for mi, method in enumerate(METHODS):
            summary, probes = train_method(
                method,
                b0,
                a0,
                data,
                cfg,
                trial,
                seed + 100003 * (mi + 1),
            )
            trial_rows.append(summary)
            probe_rows.extend(probes)

            print(
                f"  {method:28s} "
                f"final={summary['final_loss']:.6e} "
                f"improve={summary['loss_improvement']:.3e} "
                f"VF={summary['v_functional']:.3e} "
                f"VG={summary['v_gauge']:.3e} "
                f"VC={summary['v_interaction']:.3e} "
                f"epsD99={summary['gauge_product_p99']:.3e} "
                f"align={summary['full_batch_alignment_mean']:.4f}"
            )

    method_summary = summarize_methods(trial_rows)
    gates = build_gates(method_summary, cfg)

    write_csv(out_dir / "per_trial.csv", trial_rows)
    write_csv(out_dir / "per_probe.csv", probe_rows)
    write_csv(out_dir / "method_summary.csv", method_summary)

    payload = {
        "title": "H13.10-FINAL matched-budget two-way stochastic variance decomposition",
        "config": asdict(cfg),
        "methods": list(METHODS),
        "method_summary": method_summary,
        "gates": gates,
        "interpretation": {
            "v_functional": "Batch main-effect variance in product-space directions.",
            "v_gauge": "Gauge main-effect variance in product-space directions.",
            "v_interaction": "Batch-by-gauge interaction variance.",
            "exact_covariance": "Ordinary-inverse full-rank branch with ridge=0.",
            "practical_covariance": "Ridge-regularized implementation branch, reported separately.",
            "matched_budget": "All methods are normalized to the same first-order product-step budget before update.",
            "ema_probe": "Uses the real pre-probe EMA history state.",
            "boundary": "Controlled mechanism audit; not a production GPT-2 optimizer ranking.",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with (out_dir / "gates.json").open("w", encoding="utf-8") as fh:
        json.dump(gates, fh, indent=2)

    if not cfg.no_plots:
        make_plots(out_dir, method_summary)

    print("\n" + "=" * 120)
    print("H13.10-FINAL METHOD SUMMARY")
    print("=" * 120)
    for row in method_summary:
        print(
            f"{row['method']:28s} "
            f"loss={row['final_loss_mean']:.6e} "
            f"step={row['mean_realized_product_step_mean']:.4e} "
            f"VF={row['v_functional_mean']:.3e} "
            f"VG={row['v_gauge_mean']:.3e} "
            f"VC={row['v_interaction_mean']:.3e} "
            f"epsD99={row['gauge_product_p99_mean']:.3e} "
            f"practical_epsD99={row['practical_gauge_product_p99_mean']:.3e} "
            f"align={row['full_batch_alignment_mean_mean']:.4f}"
        )

    print("\n" + "=" * 120)
    print("H13.10-FINAL GATES")
    print("=" * 120)
    print(json.dumps(gates, indent=2))

    print("\nDecision guide:")
    print("  1. Exact split eps_A/eps_B/eps_D near precision => H13.9D covariance survives stochastic probing.")
    print("  2. Practical ridge residual above exact residual => regularization, not theorem failure.")
    print("  3. EMA lower V_F and higher alignment => temporal filtering is a plausible missing ingredient.")
    print("  4. V_C / V_total quantifies genuine batch-by-gauge interaction.")
    print("  5. AdamW V_F uses candidate next moments from a common frozen history.")
    print("  6. Full-product lower V_F without better matched-budget loss => local net steepness is not sufficient.")

    print(f"\nOutputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
