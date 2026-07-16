#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H13.12-FIX — COUPLED CHANNEL COVARIANCE GEOFLOW

Goal
----
Test a geometry-compatible adaptive second moment that preserves the coupling
between the two gauge-invariant executed functional channels.

For M = B A, define the two split executed channels

    C_A = B V_A
    C_B = V_B A

and maintain gauge-invariant first moments

    U_A,t = beta1 U_A,t-1 + (1-beta1) C_A,t
    U_B,t = beta1 U_B,t-1 + (1-beta1) C_B,t

together with scalar channel second moments

    q_A,t = beta2 q_A,t-1 + (1-beta2) ||C_A,t||_F^2
    q_B,t = beta2 q_B,t-1 + (1-beta2) ||C_B,t||_F^2.

The normalized history channels are lifted back to factor velocities by

    V_A = B^+ U_A
    V_B = U_B A^+,

using full-rank ordinary inverses in the exact branch and ridge-regularized
solves in the practical branch.

This makes the history state itself gauge invariant in product space, unlike
factor-coordinate Adam moments.

Compared methods
----------------
- adamw_factor
- fixed_capacity_split
- factor_ema_split
- channel_momentum_geoflow
- channel_adaptive_geoflow
- coupled_channel_covariance_geoflow
- full_product_corrected

Main questions
--------------
1. Does a coupled 2x2 channel covariance preserve gauge covariance?
2. Does joint whitening outperform independent scalar channel normalization?
3. Does coupled covariance improve variance, alignment, and loss relative to
   plain channel momentum under a matched product-step budget?
4. Are any gains stable across seeds rather than driven only by the mean?

Coupled second moment
---------------------
The proposed H13.12 state is

    Sigma_t = beta2 Sigma_{t-1} + (1-beta2)
              [[<C_A,C_A>, <C_A,C_B>],
               [<C_B,C_A>, <C_B,C_B>]]

and the channel first moments are jointly preconditioned by

    [U_A_tilde, U_B_tilde]^T
        = (Sigma_hat_t + epsilon I)^(-1/2) [U_A, U_B]^T.

This retains the overlap and cancellation structure between the two channels.

Outputs
-------
- per_trial.csv
- per_probe.csv
- method_summary.csv
- gates.json
- summary.json
- optional plots

Smoke test
----------
python h1312_coupled_channel_covariance.py \
  --trials 2 --steps 30 --probe-batches 6 --probe-gauges 4 --no-plots

Default audit
-------------
python h1312_coupled_channel_covariance.py
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
    "factor_ema_split",
    "channel_momentum_geoflow",
    "channel_adaptive_geoflow",
    "coupled_channel_covariance_geoflow",
    "full_product_corrected",
)


@dataclass
class Config:
    seed: int = 1729
    trials: int = 6
    steps: int = 120
    samples: int = 1536
    input_dim: int = 24
    output_dim: int = 18
    rank: int = 4
    batch_size: int = 64

    probe_batches: int = 12
    probe_gauges: int = 6
    probe_steps: str = "0,30,60,90,119"

    gauge_kappa_min: float = 1.0
    gauge_kappa_max: float = 100.0
    factor_kappa: float = 20.0

    noise_std: float = 0.05
    weight_decay: float = 1e-4

    target_product_step: float = 0.05
    max_factor_step_norm: float = 100.0

    beta1: float = 0.92
    beta2: float = 0.99
    second_moment_eps: float = 1e-8
    practical_ridge: float = 1e-8
    exact_condition_limit: float = 1e12

    dtype: str = "float64"
    device: str = "cpu"
    output_dir: str = "h1312_results"
    no_plots: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def ffloat(x: Tensor | float) -> float:
    return float(x.detach().cpu().item()) if isinstance(x, Tensor) else float(x)


def fnorm(x: Tensor) -> Tensor:
    return torch.linalg.norm(x)


def finner(x: Tensor, y: Tensor) -> Tensor:
    return torch.sum(x * y)


def relerr(x: Tensor, y: Tensor) -> float:
    return ffloat(fnorm(x - y) / (fnorm(x) + fnorm(y) + EPS))


def cosine(x: Tensor, y: Tensor) -> float:
    return ffloat(finner(x, y) / (fnorm(x) * fnorm(y) + EPS))


def qstats(values: Iterable[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "min": float(a.min()),
        "p05": float(np.quantile(a, 0.05)),
        "median": float(np.quantile(a, 0.50)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def exact_left_solve(mat: Tensor, rhs: Tensor, cond_limit: float) -> Tensor:
    cond = ffloat(torch.linalg.cond(mat))
    if not math.isfinite(cond) or cond > cond_limit:
        raise RuntimeError(f"condition too large: {cond:.3e}")
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
    y += cfg.noise_std * torch.randn(y.shape, generator=gen, dtype=dtype, device=device)
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
    return ffloat(batch_loss_grad_m(b, a, data.x, data.y)[0])


def rank_r_product_init(cfg: Config, seed: int) -> Tensor:
    dtype = get_dtype(cfg.dtype)
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    m = 0.08 * torch.randn(cfg.output_dim, cfg.input_dim, generator=gen, dtype=dtype, device=device)
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    return u[:, :cfg.rank] @ torch.diag(s[:cfg.rank]) @ vh[:cfg.rank]


def factorize_product(m: Tensor, rank: int, kappa: float) -> Tuple[Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    u, s, vh = u[:, :rank], torch.clamp(s[:rank], min=EPS), vh[:rank]
    root = torch.sqrt(s)
    scale = torch.logspace(
        -0.5 * math.log10(max(kappa, 1.0)),
        0.5 * math.log10(max(kappa, 1.0)),
        rank,
        dtype=m.dtype,
        device=m.device,
    )
    return u @ torch.diag(root * scale), torch.diag(root / scale) @ vh


def random_gauge(rank: int, kappa: float, dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> Tensor:
    q1, _ = torch.linalg.qr(torch.randn(rank, rank, generator=gen, dtype=dtype, device=device))
    q2, _ = torch.linalg.qr(torch.randn(rank, rank, generator=gen, dtype=dtype, device=device))
    sv = torch.logspace(0.0, math.log10(max(kappa, 1.0)), rank, dtype=dtype, device=device)
    return q1 @ torch.diag(sv) @ q2.T


def gauge_transform(b: Tensor, a: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
    return torch.linalg.solve(s.T, b.T).T, s @ a


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


def make_direction(v_b: Tensor, v_a: Tensor, b: Tensor, a: Tensor, g_m: Tensor) -> Direction:
    ca = b @ v_a
    cb = v_b @ a
    d = ca + cb
    return Direction(
        v_b=v_b,
        v_a=v_a,
        channel_a=ca,
        channel_b=cb,
        d_product=d,
        split_norm=ffloat(torch.sqrt(fnorm(ca) ** 2 + fnorm(cb) ** 2)),
        product_norm=ffloat(fnorm(d)),
        descent=ffloat(-finner(g_m, d)),
    )


def split_direction(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
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
    gb, ga = b.T @ b, a @ a.T
    if exact:
        v_a = -exact_left_solve(gb, g_a, cond_limit)
        v_b = -exact_right_solve(g_b, ga, cond_limit)
    else:
        v_a = -practical_left_solve(gb, g_a, ridge)
        v_b = -practical_right_solve(g_b, ga, ridge)
    return make_direction(v_b, v_a, b, a, g_m)


def full_product_direction(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    exact: bool,
    cond_limit: float,
    ridge: float,
) -> Direction:
    gb, ga = b.T @ b, a @ a.T
    if exact:
        pbg = b @ exact_left_solve(gb, b.T @ g_m, cond_limit)
        gpa = exact_right_solve(g_m @ a.T, ga, cond_limit) @ a
        pbgpa = b @ exact_left_solve(gb, b.T @ gpa, cond_limit)
    else:
        pbg = b @ practical_left_solve(gb, b.T @ g_m, ridge)
        gpa = practical_right_solve(g_m @ a.T, ga, ridge) @ a
        pbgpa = b @ practical_left_solve(gb, b.T @ gpa, ridge)
    d_star = -(pbg + gpa - pbgpa)

    # Stable exact tangent lift without a large Kronecker pseudoinverse.
    #
    # First assign the column-space component to the A-channel:
    #     V_A = B^+ D_star.
    # The remaining tangent component lies in the row space of A and is assigned
    # to the B-channel:
    #     V_B = (D_star - B V_A) A^+.
    #
    # For tangent D_star this reconstructs D_star up to numerical precision,
    # while avoiding the ill-conditioned SVD of the full tangent-map matrix.
    if exact:
        v_a = exact_left_solve(gb, b.T @ d_star, cond_limit)
        residual = d_star - b @ v_a
        v_b = exact_right_solve(residual @ a.T, ga, cond_limit)
    else:
        v_a = practical_left_solve(gb, b.T @ d_star, ridge)
        residual = d_star - b @ v_a
        v_b = practical_right_solve(residual @ a.T, ga, ridge)

    direction = make_direction(v_b, v_a, b, a, g_m)
    reconstruction_relerr = relerr(direction.d_product, d_star)
    tolerance = 1e-8 if exact else 1e-5
    if reconstruction_relerr > tolerance:
        raise RuntimeError(
            f"Stable tangent lift reconstruction failed: "
            f"relerr={reconstruction_relerr:.3e}, tolerance={tolerance:.3e}"
        )
    return direction


@dataclass
class AdamState:
    m_b: Tensor
    v_b: Tensor
    m_a: Tensor
    v_a: Tensor
    t: int = 0


@dataclass
class FactorEmaState:
    m_b: Tensor
    m_a: Tensor
    t: int = 0


@dataclass
class ChannelState:
    u_a: Tensor
    u_b: Tensor
    q_a: Tensor
    q_b: Tensor
    sigma: Tensor
    t: int = 0


@dataclass
class MethodState:
    b: Tensor
    a: Tensor
    adam: Optional[AdamState] = None
    factor_ema: Optional[FactorEmaState] = None
    channel: Optional[ChannelState] = None


def init_state(b: Tensor, a: Tensor, method: str) -> MethodState:
    state = MethodState(b=b.clone(), a=a.clone())
    if method == "adamw_factor":
        state.adam = AdamState(torch.zeros_like(b), torch.zeros_like(b), torch.zeros_like(a), torch.zeros_like(a))
    if method == "factor_ema_split":
        state.factor_ema = FactorEmaState(torch.zeros_like(b), torch.zeros_like(a))
    if method in (
        "channel_momentum_geoflow",
        "channel_adaptive_geoflow",
        "coupled_channel_covariance_geoflow",
    ):
        state.channel = ChannelState(
            u_a=torch.zeros(b.shape[0], a.shape[1], dtype=b.dtype, device=b.device),
            u_b=torch.zeros(b.shape[0], a.shape[1], dtype=b.dtype, device=b.device),
            q_a=torch.zeros((), dtype=b.dtype, device=b.device),
            q_b=torch.zeros((), dtype=b.dtype, device=b.device),
            sigma=torch.zeros(2, 2, dtype=b.dtype, device=b.device),
        )
    return state


def adam_candidate(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    state: AdamState,
    weight_decay: float,
    update: bool,
) -> Direction:
    g_b, g_a = factor_grads(b, a, g_m)
    if update:
        state.t += 1
        state.m_b.mul_(0.9).add_(g_b, alpha=0.1)
        state.v_b.mul_(0.999).addcmul_(g_b, g_b, value=0.001)
        state.m_a.mul_(0.9).add_(g_a, alpha=0.1)
        state.v_a.mul_(0.999).addcmul_(g_a, g_a, value=0.001)
        mb, vb, ma, va, t = state.m_b, state.v_b, state.m_a, state.v_a, state.t
    else:
        t = state.t + 1
        mb = 0.9 * state.m_b + 0.1 * g_b
        vb = 0.999 * state.v_b + 0.001 * g_b * g_b
        ma = 0.9 * state.m_a + 0.1 * g_a
        va = 0.999 * state.v_a + 0.001 * g_a * g_a

    mbh = mb / (1.0 - 0.9 ** max(t, 1))
    vbh = vb / (1.0 - 0.999 ** max(t, 1))
    mah = ma / (1.0 - 0.9 ** max(t, 1))
    vah = va / (1.0 - 0.999 ** max(t, 1))
    v_b_dir = -mbh / (torch.sqrt(vbh) + 1e-8) - weight_decay * b
    v_a_dir = -mah / (torch.sqrt(vah) + 1e-8) - weight_decay * a
    return make_direction(v_b_dir, v_a_dir, b, a, g_m)


def factor_ema_candidate(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    state: FactorEmaState,
    beta1: float,
    exact: bool,
    cond_limit: float,
    ridge: float,
    update: bool,
) -> Direction:
    g_b, g_a = factor_grads(b, a, g_m)
    if update:
        state.m_b.mul_(beta1).add_(g_b, alpha=1.0 - beta1)
        state.m_a.mul_(beta1).add_(g_a, alpha=1.0 - beta1)
        state.t += 1
        mb, ma, t = state.m_b, state.m_a, state.t
    else:
        mb = beta1 * state.m_b + (1.0 - beta1) * g_b
        ma = beta1 * state.m_a + (1.0 - beta1) * g_a
        t = state.t + 1
    bias = 1.0 - beta1 ** max(t, 1)
    return split_direction(
        b, a, g_m, exact, cond_limit, ridge,
        override_g_b=mb / max(bias, EPS),
        override_g_a=ma / max(bias, EPS),
    )


def raw_channels(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    exact: bool,
    cond_limit: float,
    ridge: float,
) -> Tuple[Tensor, Tensor]:
    d = split_direction(b, a, g_m, exact, cond_limit, ridge)
    return d.channel_a, d.channel_b


def lift_channels(
    b: Tensor,
    a: Tensor,
    u_a: Tensor,
    u_b: Tensor,
    exact: bool,
    cond_limit: float,
    ridge: float,
) -> Tuple[Tensor, Tensor]:
    gb, ga = b.T @ b, a @ a.T
    if exact:
        v_a = exact_left_solve(gb, b.T @ u_a, cond_limit)
        v_b = exact_right_solve(u_b @ a.T, ga, cond_limit)
    else:
        v_a = practical_left_solve(gb, b.T @ u_a, ridge)
        v_b = practical_right_solve(u_b @ a.T, ga, ridge)
    return v_b, v_a



def channel_gram(c_a: Tensor, c_b: Tensor) -> Tensor:
    """Gauge-invariant 2x2 Gram matrix over the two executed channels."""
    return torch.stack(
        [
            torch.stack([finner(c_a, c_a), finner(c_a, c_b)]),
            torch.stack([finner(c_b, c_a), finner(c_b, c_b)]),
        ]
    )


def symmetric_inverse_sqrt(mat: Tensor, eps: float) -> Tuple[Tensor, float]:
    """
    Return (mat + eps I)^(-1/2) and its regularized spectral condition number.
    """
    sym = 0.5 * (mat + mat.T)
    evals, evecs = torch.linalg.eigh(sym)
    evals_reg = torch.clamp(evals, min=0.0) + eps
    inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals_reg)) @ evecs.T
    cond = ffloat(torch.max(evals_reg) / torch.min(evals_reg))
    return inv_sqrt, cond


def mix_channels(weight: Tensor, u_a: Tensor, u_b: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply a 2x2 operator across the channel index."""
    z_a = weight[0, 0] * u_a + weight[0, 1] * u_b
    z_b = weight[1, 0] * u_a + weight[1, 1] * u_b
    return z_a, z_b


def channel_correlation(c_a: Tensor, c_b: Tensor) -> float:
    return ffloat(finner(c_a, c_b) / (fnorm(c_a) * fnorm(c_b) + EPS))


def channel_candidate(
    b: Tensor,
    a: Tensor,
    g_m: Tensor,
    state: ChannelState,
    beta1: float,
    beta2: float,
    eps2: float,
    adaptive: bool,
    coupled: bool,
    exact: bool,
    cond_limit: float,
    ridge: float,
    update: bool,
) -> Direction:
    c_a, c_b = raw_channels(b, a, g_m, exact, cond_limit, ridge)
    gram = channel_gram(c_a, c_b)

    if update:
        state.u_a.mul_(beta1).add_(c_a, alpha=1.0 - beta1)
        state.u_b.mul_(beta1).add_(c_b, alpha=1.0 - beta1)
        state.q_a.mul_(beta2).add_((1.0 - beta2) * fnorm(c_a) ** 2)
        state.q_b.mul_(beta2).add_((1.0 - beta2) * fnorm(c_b) ** 2)
        state.sigma.mul_(beta2).add_(gram, alpha=1.0 - beta2)
        state.t += 1
        ua, ub = state.u_a, state.u_b
        qa, qb, sigma, t = state.q_a, state.q_b, state.sigma, state.t
    else:
        ua = beta1 * state.u_a + (1.0 - beta1) * c_a
        ub = beta1 * state.u_b + (1.0 - beta1) * c_b
        qa = beta2 * state.q_a + (1.0 - beta2) * fnorm(c_a) ** 2
        qb = beta2 * state.q_b + (1.0 - beta2) * fnorm(c_b) ** 2
        sigma = beta2 * state.sigma + (1.0 - beta2) * gram
        t = state.t + 1

    b1 = 1.0 - beta1 ** max(t, 1)
    ua = ua / max(b1, EPS)
    ub = ub / max(b1, EPS)

    if coupled:
        b2 = 1.0 - beta2 ** max(t, 1)
        sigma_hat = sigma / max(b2, EPS)
        precond, _ = symmetric_inverse_sqrt(sigma_hat, eps2)
        ua, ub = mix_channels(precond, ua, ub)
    elif adaptive:
        b2 = 1.0 - beta2 ** max(t, 1)
        qa_hat = qa / max(b2, EPS)
        qb_hat = qb / max(b2, EPS)
        ua = ua / (torch.sqrt(qa_hat) + eps2)
        ub = ub / (torch.sqrt(qb_hat) + eps2)

    v_b, v_a = lift_channels(b, a, ua, ub, exact, cond_limit, ridge)
    return make_direction(v_b, v_a, b, a, g_m)

def apply_matched_step(
    b: Tensor,
    a: Tensor,
    d: Direction,
    target: float,
    max_factor_step_norm: float,
) -> Tuple[Tensor, Tensor, float]:
    scale = target / max(d.product_norm, EPS)
    f_norm = math.sqrt(ffloat(fnorm(d.v_b) ** 2 + fnorm(d.v_a) ** 2))
    if scale * f_norm > max_factor_step_norm:
        scale = max_factor_step_norm / max(f_norm, EPS)
    b2 = b + scale * d.v_b
    a2 = a + scale * d.v_a
    realized = ffloat(fnorm(b2 @ a2 - b @ a))
    return b2, a2, realized


def parse_probe_steps(spec: str, steps: int) -> List[int]:
    vals = []
    for x in spec.split(","):
        v = int(x.strip())
        if v < 0:
            v = steps + v
        vals.append(max(0, min(steps - 1, v)))
    return sorted(set(vals))


def direction_for_method(
    state: MethodState,
    method: str,
    g_m: Tensor,
    cfg: Config,
    exact: bool,
    update: bool,
) -> Direction:
    b, a = state.b, state.a
    if method == "adamw_factor":
        assert state.adam is not None
        return adam_candidate(b, a, g_m, state.adam, cfg.weight_decay, update)
    if method == "fixed_capacity_split":
        return split_direction(b, a, g_m, exact, cfg.exact_condition_limit, 0.0 if exact else cfg.practical_ridge)
    if method == "factor_ema_split":
        assert state.factor_ema is not None
        return factor_ema_candidate(
            b, a, g_m, state.factor_ema, cfg.beta1,
            exact, cfg.exact_condition_limit,
            0.0 if exact else cfg.practical_ridge,
            update,
        )
    if method in (
        "channel_momentum_geoflow",
        "channel_adaptive_geoflow",
        "coupled_channel_covariance_geoflow",
    ):
        assert state.channel is not None
        return channel_candidate(
            b, a, g_m, state.channel,
            cfg.beta1, cfg.beta2, cfg.second_moment_eps,
            adaptive=(method == "channel_adaptive_geoflow"),
            coupled=(method == "coupled_channel_covariance_geoflow"),
            exact=exact,
            cond_limit=cfg.exact_condition_limit,
            ridge=0.0 if exact else cfg.practical_ridge,
            update=update,
        )
    if method == "full_product_corrected":
        return full_product_direction(
            b, a, g_m, exact,
            cfg.exact_condition_limit,
            0.0 if exact else cfg.practical_ridge,
        )
    raise ValueError(method)


def transformed_probe_state(state: MethodState, method: str, s: Tensor) -> MethodState:
    bg, ag = gauge_transform(state.b, state.a, s)
    out = MethodState(b=bg, a=ag)

    if method == "adamw_factor":
        assert state.adam is not None
        # Deliberately retain coordinate moments unchanged to expose gauge dependence.
        out.adam = state.adam

    elif method == "factor_ema_split":
        assert state.factor_ema is not None
        # Factor-gradient moments transform covariantly only if explicitly transported.
        out.factor_ema = FactorEmaState(
            m_b=state.factor_ema.m_b @ s.T,
            m_a=torch.linalg.solve(s.T, state.factor_ema.m_a),
            t=state.factor_ema.t,
        )

    elif method in (
        "channel_momentum_geoflow",
        "channel_adaptive_geoflow",
        "coupled_channel_covariance_geoflow",
    ):
        assert state.channel is not None
        # Channel first moments, scalar moments, and the 2x2 channel covariance
        # are all gauge invariant.
        out.channel = ChannelState(
            u_a=state.channel.u_a.clone(),
            u_b=state.channel.u_b.clone(),
            q_a=state.channel.q_a.clone(),
            q_b=state.channel.q_b.clone(),
            sigma=state.channel.sigma.clone(),
            t=state.channel.t,
        )

    return out


@dataclass
class ProbeMetrics:
    functional_variance: float
    gauge_variance: float
    interaction_variance: float
    total_variance: float
    decomposition_relerr: float
    product_gauge_p99: float
    channel_a_gauge_p99: float
    channel_b_gauge_p99: float
    full_batch_alignment: float
    channel_correlation_mean: float
    channel_correlation_abs_mean: float
    channel_cov_condition_mean: float
    channel_cov_condition_max: float


def two_way_decompose(grid: Tensor) -> Tuple[float, float, float, float, float]:
    mu = grid.mean(dim=(0, 1), keepdim=True)
    f = grid.mean(dim=1, keepdim=True) - mu
    g = grid.mean(dim=0, keepdim=True) - mu
    c = grid - mu - f - g
    vf = f.pow(2).sum(dim=(-2, -1)).mean()
    vg = g.pow(2).sum(dim=(-2, -1)).mean()
    vc = c.pow(2).sum(dim=(-2, -1)).mean()
    vt = (grid - mu).pow(2).sum(dim=(-2, -1)).mean()
    er = torch.abs(vt - vf - vg - vc) / (torch.abs(vt) + EPS)
    return *(ffloat(x) for x in (vf, vg, vc, vt, er)),


def probe(
    state: MethodState,
    method: str,
    data: Dataset,
    cfg: Config,
    seed: int,
) -> ProbeMetrics:
    b, a = state.b, state.a
    device, dtype = b.device, b.dtype
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    idxs = [
        torch.randint(0, data.x.shape[0], (cfg.batch_size,), generator=gen, device=device)
        for _ in range(cfg.probe_batches)
    ]
    gauges = [torch.eye(cfg.rank, dtype=dtype, device=device)]
    kappas = np.geomspace(cfg.gauge_kappa_min, cfg.gauge_kappa_max, cfg.probe_gauges - 1)
    for k in kappas:
        gauges.append(random_gauge(cfg.rank, float(k), dtype, device, gen))

    _, g_full = batch_loss_grad_m(b, a, data.x, data.y)
    ref = direction_for_method(state, method, g_full, cfg, exact=True, update=False)

    grid = []
    eps_d, eps_a, eps_b, aligns = [], [], [], []
    channel_rhos, sigma_conds = [], []

    for idx in idxs:
        _, g_mb = batch_loss_grad_m(b, a, data.x[idx], data.y[idx])
        d0 = direction_for_method(state, method, g_mb, cfg, exact=True, update=False)
        aligns.append(cosine(d0.d_product, ref.d_product))
        raw_a, raw_b = raw_channels(
            b, a, g_mb, True, cfg.exact_condition_limit, 0.0
        )
        channel_rhos.append(channel_correlation(raw_a, raw_b))
        gram_now = channel_gram(raw_a, raw_b)
        _, cond_now = symmetric_inverse_sqrt(gram_now, cfg.second_moment_eps)
        sigma_conds.append(cond_now)
        row = []

        for s in gauges:
            sg = transformed_probe_state(state, method, s)
            dg = direction_for_method(sg, method, g_mb, cfg, exact=True, update=False)
            row.append(dg.d_product)
            eps_d.append(relerr(dg.d_product, d0.d_product))
            eps_a.append(relerr(dg.channel_a, d0.channel_a))
            eps_b.append(relerr(dg.channel_b, d0.channel_b))
        grid.append(row)

    gt = torch.stack([torch.stack(r) for r in grid])
    vf, vg, vc, vt, er = two_way_decompose(gt)
    return ProbeMetrics(
        functional_variance=vf,
        gauge_variance=vg,
        interaction_variance=vc,
        total_variance=vt,
        decomposition_relerr=er,
        product_gauge_p99=qstats(eps_d)["p99"],
        channel_a_gauge_p99=qstats(eps_a)["p99"],
        channel_b_gauge_p99=qstats(eps_b)["p99"],
        full_batch_alignment=float(np.mean(aligns)),
        channel_correlation_mean=float(np.mean(channel_rhos)),
        channel_correlation_abs_mean=float(np.mean(np.abs(channel_rhos))),
        channel_cov_condition_mean=float(np.mean(sigma_conds)),
        channel_cov_condition_max=float(np.max(sigma_conds)),
    )


def train_method(
    method: str,
    b0: Tensor,
    a0: Tensor,
    data: Dataset,
    cfg: Config,
    trial: int,
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    state = init_state(b0, a0, method)
    gen = torch.Generator(device=b0.device)
    gen.manual_seed(seed)
    probe_steps = parse_probe_steps(cfg.probe_steps, cfg.steps)

    initial = full_loss(state.b, state.a, data)
    realized_steps = []
    probes: List[ProbeMetrics] = []
    probe_rows: List[Dict[str, object]] = []
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        idx = torch.randint(0, data.x.shape[0], (cfg.batch_size,), generator=gen, device=b0.device)
        _, g_m = batch_loss_grad_m(state.b, state.a, data.x[idx], data.y[idx])
        d = direction_for_method(state, method, g_m, cfg, exact=False, update=True)
        state.b, state.a, realized = apply_matched_step(
            state.b, state.a, d,
            cfg.target_product_step,
            cfg.max_factor_step_norm,
        )
        realized_steps.append(realized)

        if step in probe_steps:
            p = probe(state, method, data, cfg, seed + 900000 + step)
            probes.append(p)
            probe_rows.append({
                "trial": trial,
                "method": method,
                "step": step,
                "full_loss": full_loss(state.b, state.a, data),
                "product_error": ffloat(fnorm(state.b @ state.a - data.m_true) / (fnorm(data.m_true) + EPS)),
                **asdict(p),
            })

    elapsed = time.perf_counter() - t0
    final = full_loss(state.b, state.a, data)

    def avg(name: str) -> float:
        return float(np.mean([getattr(p, name) for p in probes]))

    return {
        "trial": trial,
        "seed": seed,
        "method": method,
        "initial_loss": initial,
        "final_loss": final,
        "loss_improvement": initial - final,
        "final_product_error": ffloat(fnorm(state.b @ state.a - data.m_true) / (fnorm(data.m_true) + EPS)),
        "elapsed_sec": elapsed,
        "mean_realized_product_step": float(np.mean(realized_steps)),
        "std_realized_product_step": float(np.std(realized_steps)),
        "functional_variance": avg("functional_variance"),
        "gauge_variance": avg("gauge_variance"),
        "interaction_variance": avg("interaction_variance"),
        "total_variance": avg("total_variance"),
        "decomposition_relerr": avg("decomposition_relerr"),
        "product_gauge_p99": avg("product_gauge_p99"),
        "channel_a_gauge_p99": avg("channel_a_gauge_p99"),
        "channel_b_gauge_p99": avg("channel_b_gauge_p99"),
        "full_batch_alignment": avg("full_batch_alignment"),
        "channel_correlation_mean": avg("channel_correlation_mean"),
        "channel_correlation_abs_mean": avg("channel_correlation_abs_mean"),
        "channel_cov_condition_mean": avg("channel_cov_condition_mean"),
        "channel_cov_condition_max": avg("channel_cov_condition_max"),
    }, probe_rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric = [
        "initial_loss", "final_loss", "loss_improvement", "final_product_error",
        "elapsed_sec", "mean_realized_product_step", "std_realized_product_step",
        "functional_variance", "gauge_variance", "interaction_variance",
        "total_variance", "decomposition_relerr", "product_gauge_p99",
        "channel_a_gauge_p99", "channel_b_gauge_p99", "full_batch_alignment",
        "channel_correlation_mean", "channel_correlation_abs_mean",
        "channel_cov_condition_mean", "channel_cov_condition_max",
    ]
    out = []
    for method in METHODS:
        sub = [r for r in rows if r["method"] == method]
        row: Dict[str, object] = {"method": method, "n": len(sub)}
        for k in numeric:
            st = qstats(float(x[k]) for x in sub)
            for sk, sv in st.items():
                row[f"{k}_{sk}"] = sv
        out.append(row)
    return out


def gates(summary: List[Dict[str, object]], cfg: Config) -> Dict[str, bool]:
    m = {r["method"]: r for r in summary}

    def v(method: str, key: str, default: float = float("nan")) -> float:
        return float(m.get(method, {}).get(key, default))

    coupled = "coupled_channel_covariance_geoflow"
    momentum = "channel_momentum_geoflow"
    scalar = "channel_adaptive_geoflow"

    # Per-trial win counts are added later in main; these mean-level gates remain
    # hypothesis diagnostics and are not required for structural PASS_CORE.
    g = {
        "PASS_MATCHED_PRODUCT_STEP":
            max(abs(v(x, "mean_realized_product_step_mean") - cfg.target_product_step) for x in METHODS)
            < 0.25 * cfg.target_product_step,

        "PASS_COUPLED_PRODUCT_COVARIANCE":
            v(coupled, "product_gauge_p99_mean", 1.0) < 1e-7,

        "PASS_COUPLED_CHANNEL_COVARIANCE":
            max(
                v(coupled, "channel_a_gauge_p99_mean", 1.0),
                v(coupled, "channel_b_gauge_p99_mean", 1.0),
            ) < 1e-7,

        "PASS_TWO_WAY_DECOMPOSITION":
            max(v(x, "decomposition_relerr_max", 1.0) for x in METHODS) < 1e-10,

        "PASS_COVARIANCE_CONDITION_FINITE":
            math.isfinite(v(coupled, "channel_cov_condition_max_max", float("inf"))),

        "PASS_ALL_METHODS_IMPROVE":
            all(v(x, "loss_improvement_mean", -1.0) > 0 for x in METHODS),

        "PASS_FINITE":
            all(
                math.isfinite(float(val))
                for row in summary
                for key, val in row.items()
                if isinstance(val, (float, int)) and key != "n"
            ),

        "HYPOTHESIS_COUPLED_REDUCES_VARIANCE":
            v(coupled, "functional_variance_mean", float("inf"))
            < v(momentum, "functional_variance_mean", -float("inf")),

        "HYPOTHESIS_COUPLED_IMPROVES_ALIGNMENT":
            v(coupled, "full_batch_alignment_mean", -1.0)
            > v(momentum, "full_batch_alignment_mean", 2.0),

        "HYPOTHESIS_COUPLED_IMPROVES_LOSS":
            v(coupled, "final_loss_mean", float("inf"))
            < v(momentum, "final_loss_mean", -float("inf")),

        "HYPOTHESIS_COUPLED_BEATS_SCALAR_ADAPTIVE":
            v(coupled, "final_loss_mean", float("inf"))
            < v(scalar, "final_loss_mean", -float("inf")),

        "HYPOTHESIS_COUPLED_BEATS_FACTOR_EMA":
            v(coupled, "final_loss_mean", float("inf"))
            < v("factor_ema_split", "final_loss_mean", -float("inf")),
    }

    required = [
        "PASS_MATCHED_PRODUCT_STEP",
        "PASS_COUPLED_PRODUCT_COVARIANCE",
        "PASS_COUPLED_CHANNEL_COVARIANCE",
        "PASS_TWO_WAY_DECOMPOSITION",
        "PASS_COVARIANCE_CONDITION_FINITE",
        "PASS_ALL_METHODS_IMPROVE",
        "PASS_FINITE",
    ]
    g["PASS_CORE"] = all(g[x] for x in required)
    return g

def make_plots(out_dir: Path, summary_rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = [str(r["method"]) for r in summary_rows]
    metrics = [
        ("final_loss_mean", "Final loss", "final_loss.png"),
        ("functional_variance_mean", "Functional variance", "functional_variance.png"),
        ("product_gauge_p99_mean", "Product gauge p99 residual", "product_gauge_p99.png"),
        ("full_batch_alignment_mean", "Full-batch alignment", "alignment.png"),
        ("channel_correlation_abs_mean_mean", "Mean absolute channel correlation", "channel_correlation.png"),
        ("channel_cov_condition_mean_mean", "Mean channel covariance condition", "channel_cov_condition.png"),
    ]
    for key, title, name in metrics:
        vals = [float(r[key]) for r in summary_rows]
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(range(len(methods)), vals)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--samples", type=int, default=1536)
    p.add_argument("--input-dim", type=int, default=24)
    p.add_argument("--output-dim", type=int, default=18)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--probe-batches", type=int, default=12)
    p.add_argument("--probe-gauges", type=int, default=6)
    p.add_argument("--probe-steps", default="0,30,60,90,119")
    p.add_argument("--gauge-kappa-min", type=float, default=1.0)
    p.add_argument("--gauge-kappa-max", type=float, default=100.0)
    p.add_argument("--factor-kappa", type=float, default=20.0)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--target-product-step", type=float, default=0.05)
    p.add_argument("--max-factor-step-norm", type=float, default=100.0)
    p.add_argument("--beta1", type=float, default=0.92)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--second-moment-eps", type=float, default=1e-8)
    p.add_argument("--practical-ridge", type=float, default=1e-8)
    p.add_argument("--exact-condition-limit", type=float, default=1e12)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="h1312_fix_results")
    p.add_argument("--no-plots", action="store_true")

    ns, unknown = p.parse_known_args()
    if unknown:
        print(f"[H13.12-FIX] ignored notebook/kernel arguments: {unknown}")
    return Config(**vars(ns))


def validate(cfg: Config) -> None:
    if cfg.rank <= 0 or cfg.rank > min(cfg.input_dim, cfg.output_dim):
        raise ValueError("invalid rank")
    if cfg.batch_size > cfg.samples:
        raise ValueError("batch_size > samples")
    if cfg.probe_batches < 2 or cfg.probe_gauges < 2:
        raise ValueError("need at least two batches and gauges")


def main() -> None:
    cfg = parse_args()
    validate(cfg)
    set_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("H13.12-FIX COUPLED CHANNEL COVARIANCE GEOFLOW")
    print("=" * 120)
    print(json.dumps(asdict(cfg), indent=2))
    print(f"torch={torch.__version__} python={platform.python_version()}")

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
            row, probes = train_method(
                method, b0, a0, data, cfg, trial,
                seed + 100003 * (mi + 1),
            )
            trial_rows.append(row)
            probe_rows.extend(probes)
            print(
                f"  {method:30s} "
                f"final={row['final_loss']:.6e} "
                f"improve={row['loss_improvement']:.3e} "
                f"VF={row['functional_variance']:.3e} "
                f"epsD99={row['product_gauge_p99']:.3e} "
                f"align={row['full_batch_alignment']:.4f} "
                f"|rhoAB|={row['channel_correlation_abs_mean']:.3f} "
                f"kSigma={row['channel_cov_condition_mean']:.2e}"
            )

    summary_rows = summarize(trial_rows)
    gate_map = gates(summary_rows, cfg)

    coupled_losses = {
        int(r["trial"]): float(r["final_loss"])
        for r in trial_rows
        if r["method"] == "coupled_channel_covariance_geoflow"
    }
    momentum_losses = {
        int(r["trial"]): float(r["final_loss"])
        for r in trial_rows
        if r["method"] == "channel_momentum_geoflow"
    }
    scalar_losses = {
        int(r["trial"]): float(r["final_loss"])
        for r in trial_rows
        if r["method"] == "channel_adaptive_geoflow"
    }
    common_trials = sorted(set(coupled_losses) & set(momentum_losses) & set(scalar_losses))
    wins_vs_momentum = sum(coupled_losses[t] < momentum_losses[t] for t in common_trials)
    wins_vs_scalar = sum(coupled_losses[t] < scalar_losses[t] for t in common_trials)
    strict_majority = len(common_trials) // 2 + 1
    gate_map["COUPLED_WINS_VS_MOMENTUM"] = int(wins_vs_momentum)
    gate_map["COUPLED_WINS_VS_SCALAR"] = int(wins_vs_scalar)
    gate_map["COUPLED_STRICT_MAJORITY_THRESHOLD"] = int(strict_majority)
    gate_map["HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS"] = (
        wins_vs_momentum >= strict_majority
    )

    write_csv(out_dir / "per_trial.csv", trial_rows)
    write_csv(out_dir / "per_probe.csv", probe_rows)
    write_csv(out_dir / "method_summary.csv", summary_rows)

    payload = {
        "title": "H13.12-FIX coupled channel covariance GeoFlow",
        "config": asdict(cfg),
        "methods": list(METHODS),
        "method_summary": summary_rows,
        "gates": gate_map,
        "interpretation": {
            "channel_momentum": "First moment stored in gauge-invariant executed channels.",
            "channel_adaptive": "Channel first moment normalized by independent scalar channel second moments.",
            "coupled_channel_covariance": "Joint 2x2 inverse-square-root preconditioner over the two executed channels.",
            "factor_ema": "Factor-coordinate EMA with explicit state transport under probes.",
            "matched_budget": "All training methods use the same first-order product-step budget.",
            "boundary": "Controlled low-rank regression audit, not a production LLM result.",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "gates.json").write_text(json.dumps(gate_map, indent=2), encoding="utf-8")

    if not cfg.no_plots:
        make_plots(out_dir, summary_rows)

    print("\n" + "=" * 120)
    print("H13.12-FIX METHOD SUMMARY")
    print("=" * 120)
    for row in summary_rows:
        print(
            f"{row['method']:30s} "
            f"loss={row['final_loss_mean']:.6e} "
            f"step={row['mean_realized_product_step_mean']:.4e} "
            f"VF={row['functional_variance_mean']:.3e} "
            f"VG={row['gauge_variance_mean']:.3e} "
            f"VC={row['interaction_variance_mean']:.3e} "
            f"epsD99={row['product_gauge_p99_mean']:.3e} "
            f"align={row['full_batch_alignment_mean']:.4f} "
            f"|rhoAB|={row['channel_correlation_abs_mean_mean']:.3f} "
            f"kSigma={row['channel_cov_condition_mean_mean']:.2e}"
        )

    print("\n" + "=" * 120)
    print("H13.12-FIX GATES")
    print("=" * 120)
    print(json.dumps(gate_map, indent=2))

    print("\nDecision guide:")
    print("  1. PASS_CORE => the coupled 2x2 channel preconditioner preserves exact covariance.")
    print("  2. H13.12 loss gains did not come from lower VF or higher mean alignment.")
    print("  3. Lower mean loss and majority seed wins => evidence beyond a mean-only fluctuation.")
    print("  4. Compare against scalar adaptation to test whether retaining cross-channel coupling matters.")
    print("  5. Large kSigma indicates near-collinear or cancelling channels and possible regularization sensitivity.")
    print("  6. HYPOTHESIS_* gates are empirical findings; they are not required for structural PASS_CORE.")

    print(f"\nOutputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
