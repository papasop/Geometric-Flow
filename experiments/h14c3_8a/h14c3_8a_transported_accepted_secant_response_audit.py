#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H14C3-8A — TRANSPORTED ACCEPTED-SECANT KLR RESPONSE AUDIT
=========================================================

Fixes over H14C3-8
------------------
1. Build secants only from accepted steps.
2. Use the actual executed displacement:
       s_t = Transport(dt_t * velocity_t)
3. Transport the previous gradient into the accepted new tangent frame:
       y_t = g_{t+1} - Transport(g_t)
4. Do not map negative eigenvalues with abs().
5. Do not force ||g_hat|| == ||g||.
6. Blend raw and response-preconditioned directions, with only an upper norm cap.
7. Emit mechanism diagnostics proving whether the operator actually changes
   the direction.

Compact state
-------------
    M = U diag(S) V^T

Compact tangent momentum
------------------------
    P = U K V^T + L V^T + U R
    U^T L = 0
    R V = 0

Accepted secant
---------------
For an accepted step from (U_t,V_t) to (U_{t+1},V_{t+1}):

    s_t = T_{t->t+1}(dt_t * velocity_t)
    y_t = g_{t+1} - T_{t->t+1}(g_t)

The history stores compact KLR secants together with their source frames.
At every use, all history entries are transported into the current tangent
frame before fitting the operator.

Methods
-------
1. plain_compact
2. diagonal_response_8a
3. full_core_response_8a
4. factor_adamw

Scalable response model
-----------------------
- K core: full signed r^2 x r^2 secant operator
- L/R wings: signed diagonal secant operator
- history stored compactly
- no persistent dense d_out x d_in optimizer state
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


Tensor = torch.Tensor
EPS = 1e-30


@dataclass
class Config:
    seed: int = 1414
    trials: int = 6
    conditions: str = "1,100,10000"

    n_train: int = 2048
    n_val: int = 1024
    d_in: int = 96
    d_out: int = 128
    rank: int = 4
    noise_std: float = 0.02

    steps: int = 400
    eval_interval: int = 25

    ham_dt: float = 0.12
    ham_damping: float = 0.08
    ham_mass: float = 1.0
    ham_max_momentum_norm: float = 50.0

    information_ridge: float = 1e-6
    information_max_condition: float = 1000.0

    response_history: int = 48
    response_beta: float = 0.95
    response_ridge: float = 1e-5
    response_eig_floor: float = 1e-4
    response_eig_ceiling: float = 1e3
    response_mix: float = 0.25
    response_norm_cap: float = 2.0
    response_start_accepted: int = 8
    response_min_secant_norm: float = 1e-12

    adaptive_dt_min: float = 1e-4
    adaptive_dt_max: float = 0.25
    adaptive_dt_shrink: float = 0.5
    adaptive_dt_expand: float = 1.05
    energy_tolerance: float = 1e-12
    adaptive_damping_min: float = 0.01
    adaptive_damping_max: float = 0.5

    adamw_lr_grid: str = "0.001,0.003,0.006"
    tune_trials: int = 3

    dtype: str = "float64"
    device: str = "cuda"
    output_dir: str = "h14c3_8a_transported_accepted_secant"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    for f in Config.__dataclass_fields__.values():
        p.add_argument(
            "--" + f.name.replace("_", "-"),
            type=type(f.default),
            default=f.default,
        )
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[H14C3-8A] ignored notebook/kernel arguments: {unknown}")
    return Config(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print(
            f"[H14C3-8A] GPU={torch.cuda.get_device_name(dev)} "
            f"memory={torch.cuda.get_device_properties(dev).total_memory / 2**30:.1f} GiB"
        )
        return dev
    print("[H14C3-8A] using CPU")
    return torch.device("cpu")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def orthonormal(rows: int, cols: int, device, dtype) -> Tensor:
    q, _ = torch.linalg.qr(
        torch.randn(rows, cols, device=device, dtype=dtype),
        mode="reduced",
    )
    return q


def lowrank_forward(
    x: Tensor,
    u: Tensor,
    s: Tensor,
    v: Tensor,
) -> Tensor:
    return ((x @ v) * s.unsqueeze(0)) @ u.T


def make_problem(
    cfg: Config,
    condition: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    set_seed(seed + 1)
    tu = orthonormal(cfg.d_out, cfg.rank, device, dtype)
    tv = orthonormal(cfg.d_in, cfg.rank, device, dtype)
    ts = torch.logspace(
        0.0,
        -math.log10(condition),
        cfg.rank,
        device=device,
        dtype=dtype,
    )

    set_seed(seed + 2)
    xtr = torch.randn(
        cfg.n_train, cfg.d_in, device=device, dtype=dtype
    ) / math.sqrt(cfg.d_in)
    xva = torch.randn(
        cfg.n_val, cfg.d_in, device=device, dtype=dtype
    ) / math.sqrt(cfg.d_in)

    ytr = lowrank_forward(xtr, tu, ts, tv)
    yva = lowrank_forward(xva, tu, ts, tv)
    if cfg.noise_std > 0:
        ytr = ytr + cfg.noise_std * torch.randn_like(ytr)
        yva = yva + cfg.noise_std * torch.randn_like(yva)

    set_seed(seed + 3)
    init = (
        orthonormal(cfg.d_out, cfg.rank, device, dtype),
        0.05 * torch.ones(cfg.rank, device=device, dtype=dtype),
        orthonormal(cfg.d_in, cfg.rank, device, dtype),
    )
    return xtr, ytr, xva, yva, init


def clipped_information_inverse(
    x: Tensor,
    ridge: float,
    max_condition: float,
) -> Tensor:
    c = x.T @ x / x.shape[0]
    c = 0.5 * (c + c.T)
    evals, evecs = torch.linalg.eigh(c)
    evals = torch.clamp(evals, min=ridge)
    floor = evals.max() / max_condition
    evals = torch.clamp(evals, min=floor)
    return evecs @ torch.diag(1.0 / evals) @ evecs.T


@dataclass
class KLR:
    k: Tensor
    l: Tensor
    r: Tensor


def zero_klr(u: Tensor, v: Tensor) -> KLR:
    rank = u.shape[1]
    return KLR(
        torch.zeros(rank, rank, device=u.device, dtype=u.dtype),
        torch.zeros(u.shape[0], rank, device=u.device, dtype=u.dtype),
        torch.zeros(rank, v.shape[0], device=u.device, dtype=u.dtype),
    )


def copy_klr(p: KLR) -> KLR:
    return KLR(p.k.clone(), p.l.clone(), p.r.clone())


def scale_klr(p: KLR, a: float) -> KLR:
    return KLR(a * p.k, a * p.l, a * p.r)


def add_klr(a: KLR, b: KLR, alpha: float) -> KLR:
    return KLR(
        a.k + alpha * b.k,
        a.l + alpha * b.l,
        a.r + alpha * b.r,
    )


def subtract_klr(a: KLR, b: KLR) -> KLR:
    return add_klr(a, b, -1.0)


def klr_dot(a: KLR, b: KLR) -> Tensor:
    return (
        torch.sum(a.k * b.k)
        + torch.sum(a.l * b.l)
        + torch.sum(a.r * b.r)
    )


def klr_norm_sq(p: KLR) -> Tensor:
    return klr_dot(p, p)


def klr_norm(p: KLR) -> Tensor:
    return torch.sqrt(torch.clamp(klr_norm_sq(p), min=0.0))


def project_klr_constraints(
    u: Tensor,
    v: Tensor,
    p: KLR,
) -> KLR:
    l = p.l - u @ (u.T @ p.l)
    r = p.r - (p.r @ v) @ v.T
    return KLR(p.k, l, r)


def klr_constraint_residual(u: Tensor, v: Tensor, p: KLR) -> float:
    num = torch.linalg.norm(u.T @ p.l) + torch.linalg.norm(p.r @ v)
    den = (
        torch.linalg.norm(p.k)
        + torch.linalg.norm(p.l)
        + torch.linalg.norm(p.r)
        + EPS
    )
    return float((num / den).detach().cpu())


def compact_loss_grad_klr(
    x: Tensor,
    y: Tensor,
    u: Tensor,
    s: Tensor,
    v: Tensor,
) -> Tuple[Tensor, KLR]:
    xv = x @ v
    pred = (xv * s.unsqueeze(0)) @ u.T
    residual = pred - y
    denom = residual.numel()

    eu = residual @ u
    gv = residual.T @ xv / denom
    utg = eu.T @ x / denom

    k = eu.T @ xv / denom
    l = gv - u @ k
    r = utg - k @ v.T

    grad = project_klr_constraints(u, v, KLR(k, l, r))
    loss = 0.5 * torch.mean(residual * residual)
    return loss, grad


def right_metric_action(
    u: Tensor,
    v: Tensor,
    p: KLR,
    c_inv: Tensor,
    mass: float,
) -> KLR:
    vciv = v.T @ c_inv @ v
    rciv = p.r @ c_inv @ v

    k = (p.k @ vciv + rciv) / mass
    l = (p.l @ vciv) / mass

    utz = (p.k @ v.T @ c_inv + p.r @ c_inv) / mass
    r = utz - k @ v.T

    return project_klr_constraints(u, v, KLR(k, l, r))


def klr_kinetic(
    u: Tensor,
    v: Tensor,
    p: KLR,
    c_inv: Tensor,
    mass: float,
) -> Tensor:
    a = torch.cat([u, p.l], dim=1)
    b = torch.cat([p.k @ v.T + p.r, v.T], dim=0)
    return 0.5 * torch.sum(
        (a.T @ a) * (((b @ c_inv) @ b.T).T)
    ) / mass


def retract_klr(
    u: Tensor,
    s: Tensor,
    v: Tensor,
    eta: KLR,
    rank: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    ql, rl = torch.linalg.qr(eta.l, mode="reduced")
    qr, rr = torch.linalg.qr(eta.r.T, mode="reduced")

    lmask = torch.abs(torch.diag(rl)) > 1e-13
    rmask = torch.abs(torch.diag(rr)) > 1e-13
    ql, rl = ql[:, lmask], rl[lmask, :]
    qr, rr = qr[:, rmask], rr[rmask, :]

    nl, nr = ql.shape[1], qr.shape[1]
    core = torch.zeros(
        rank + nl,
        rank + nr,
        device=u.device,
        dtype=u.dtype,
    )
    core[:rank, :rank] = torch.diag(s) + eta.k
    if nr:
        core[:rank, rank:] = rr.T
    if nl:
        core[rank:, :rank] = rl

    uc, sc, vhc = torch.linalg.svd(core, full_matrices=False)
    ub = torch.cat([u, ql], dim=1)
    vb = torch.cat([v, qr], dim=1)

    un = ub @ uc[:, :rank]
    vn = vb @ vhc[:rank, :].T
    sn = sc[:rank]

    qu, ru = torch.linalg.qr(un, mode="reduced")
    qv, rv = torch.linalg.qr(vn, mode="reduced")
    small = ru @ torch.diag(sn) @ rv.T
    us, ss, vhs = torch.linalg.svd(small, full_matrices=False)
    return qu @ us, ss, qv @ vhs.T


def transport_klr(
    ou: Tensor,
    ov: Tensor,
    nu: Tensor,
    nv: Tensor,
    p: KLR,
) -> KLR:
    vo = ov.T @ nv
    uo = nu.T @ ou

    pnv = ou @ (p.k @ vo + p.r @ nv) + p.l @ vo
    nutp = (
        uo @ (p.k @ ov.T + p.r)
        + (nu.T @ p.l) @ ov.T
    )

    k = nu.T @ pnv
    l = pnv - nu @ k
    r = nutp - k @ nv.T
    return project_klr_constraints(nu, nv, KLR(k, l, r))


@dataclass
class SecantRecord:
    u: Tensor
    v: Tensor
    s: KLR
    y: KLR


class SecantHistory:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.records: List[SecantRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def append(
        self,
        u: Tensor,
        v: Tensor,
        s: KLR,
        y: KLR,
    ) -> None:
        self.records.append(
            SecantRecord(
                u=u.detach().clone(),
                v=v.detach().clone(),
                s=copy_klr(s),
                y=copy_klr(y),
            )
        )
        if len(self.records) > self.maxlen:
            self.records.pop(0)

    def transported(
        self,
        u: Tensor,
        v: Tensor,
    ) -> List[Tuple[KLR, KLR, float]]:
        out = []
        n = len(self.records)
        for i, rec in enumerate(self.records):
            weight = 1.0 if n <= 1 else (i + 1) / n
            st = transport_klr(rec.u, rec.v, u, v, rec.s)
            yt = transport_klr(rec.u, rec.v, u, v, rec.y)
            out.append((st, yt, weight))
        return out


@dataclass
class ResponseDiagnostics:
    active: bool = False
    history_size: int = 0
    cosine_raw_pre: float = 1.0
    relative_direction_change: float = 0.0
    raw_to_pre_norm_ratio: float = 1.0
    response_condition: float = 1.0
    negative_eigen_fraction: float = 0.0
    floor_fraction: float = 0.0
    ceiling_fraction: float = 0.0
    secant_fit_residual: float = 1.0


def safe_cosine(a: KLR, b: KLR) -> float:
    denom = klr_norm(a) * klr_norm(b) + EPS
    return float((klr_dot(a, b) / denom).detach().cpu())


def clip_preconditioned_norm(
    raw: KLR,
    pre: KLR,
    cap: float,
) -> KLR:
    raw_n = float(klr_norm(raw).detach().cpu())
    pre_n = float(klr_norm(pre).detach().cpu())
    max_n = cap * max(raw_n, 1e-30)
    if pre_n > max_n:
        return scale_klr(pre, max_n / pre_n)
    return pre


def build_response_operator(
    method: str,
    cfg: Config,
    history: SecantHistory,
    u: Tensor,
    v: Tensor,
) -> Tuple[Optional[Tensor], Tensor, Tensor, ResponseDiagnostics]:
    diag = ResponseDiagnostics(history_size=len(history))
    transported = history.transported(u, v)

    dk_num = torch.zeros(
        cfg.rank, cfg.rank, device=u.device, dtype=u.dtype
    )
    dk_den = torch.zeros_like(dk_num)
    dl_num = torch.zeros_like(u)
    dl_den = torch.zeros_like(u)
    dr_num = torch.zeros(
        cfg.rank, v.shape[0], device=u.device, dtype=u.dtype
    )
    dr_den = torch.zeros_like(dr_num)

    q = cfg.rank * cfg.rank
    xx = torch.zeros(q, q, device=u.device, dtype=u.dtype)
    yx = torch.zeros_like(xx)

    used = 0
    fit_num = torch.tensor(0.0, device=u.device, dtype=u.dtype)
    fit_den = torch.tensor(0.0, device=u.device, dtype=u.dtype)

    for index, (st, yt, w_linear) in enumerate(transported):
        sn = float(klr_norm(st).detach().cpu())
        if sn < cfg.response_min_secant_norm:
            continue

        age = len(transported) - index - 1
        age_weight = cfg.response_beta ** max(0, age)
        w = float(age_weight * w_linear)

        sk = st.k.reshape(-1)
        yk = yt.k.reshape(-1)
        xx += w * torch.outer(sk, sk)
        yx += w * torch.outer(yk, sk)

        dk_num += w * yt.k * st.k
        dk_den += w * st.k * st.k
        dl_num += w * yt.l * st.l
        dl_den += w * st.l * st.l
        dr_num += w * yt.r * st.r
        dr_den += w * st.r * st.r
        used += 1

    if used == 0:
        ones_k = torch.ones(
            cfg.rank, cfg.rank, device=u.device, dtype=u.dtype
        )
        return None, ones_k, torch.cat(
            [torch.ones_like(u).reshape(-1),
             torch.ones_like(dr_num).reshape(-1)]
        ), diag

    # Signed diagonal responses: clamp, do not abs.
    dk_raw = dk_num / (dk_den + cfg.response_ridge)
    dl_raw = dl_num / (dl_den + cfg.response_ridge)
    dr_raw = dr_num / (dr_den + cfg.response_ridge)

    dk = torch.clamp(
        dk_raw,
        min=cfg.response_eig_floor,
        max=cfg.response_eig_ceiling,
    )
    dl = torch.clamp(
        dl_raw,
        min=cfg.response_eig_floor,
        max=cfg.response_eig_ceiling,
    )
    dr = torch.clamp(
        dr_raw,
        min=cfg.response_eig_floor,
        max=cfg.response_eig_ceiling,
    )

    all_raw = torch.cat([
        dk_raw.reshape(-1),
        dl_raw.reshape(-1),
        dr_raw.reshape(-1),
    ])
    all_clamped = torch.cat([
        dk.reshape(-1),
        dl.reshape(-1),
        dr.reshape(-1),
    ])

    diag.negative_eigen_fraction = float(
        (all_raw < 0).to(torch.float64).mean().detach().cpu()
    )
    diag.floor_fraction = float(
        (all_raw <= cfg.response_eig_floor)
        .to(torch.float64).mean().detach().cpu()
    )
    diag.ceiling_fraction = float(
        (all_raw >= cfg.response_eig_ceiling)
        .to(torch.float64).mean().detach().cpu()
    )

    a_core = None
    if method == "full_core_response_8a":
        eye = torch.eye(q, device=u.device, dtype=u.dtype)
        a_raw = yx @ torch.linalg.inv(xx + cfg.response_ridge * eye)
        a_sym = 0.5 * (a_raw + a_raw.T)
        evals, evecs = torch.linalg.eigh(a_sym)

        diag.negative_eigen_fraction = max(
            diag.negative_eigen_fraction,
            float(
                (evals < 0)
                .to(torch.float64)
                .mean()
                .detach()
                .cpu()
            ),
        )

        clamped = torch.clamp(
            evals,
            min=cfg.response_eig_floor,
            max=cfg.response_eig_ceiling,
        )
        a_core = evecs @ torch.diag(clamped) @ evecs.T
        diag.response_condition = float(
            (clamped.max() / clamped.min()).detach().cpu()
        )
        diag.floor_fraction = max(
            diag.floor_fraction,
            float(
                (evals <= cfg.response_eig_floor)
                .to(torch.float64)
                .mean()
                .detach()
                .cpu()
            ),
        )
        diag.ceiling_fraction = max(
            diag.ceiling_fraction,
            float(
                (evals >= cfg.response_eig_ceiling)
                .to(torch.float64)
                .mean()
                .detach()
                .cpu()
            ),
        )

        # Core secant fit diagnostic.
        for st, yt, _ in transported:
            sk = st.k.reshape(-1)
            yk = yt.k.reshape(-1)
            fit_num += torch.sum((yk - a_core @ sk) ** 2)
            fit_den += torch.sum(yk ** 2)
    else:
        diag.response_condition = float(
            (
                all_clamped.max()
                / torch.clamp(all_clamped.min(), min=EPS)
            ).detach().cpu()
        )
        for st, yt, _ in transported:
            pred = KLR(dk * st.k, dl * st.l, dr * st.r)
            fit_num += klr_norm_sq(subtract_klr(yt, pred))
            fit_den += klr_norm_sq(yt)

    diag.secant_fit_residual = float(
        torch.sqrt(fit_num / (fit_den + EPS)).detach().cpu()
    )
    diag.active = True
    wing_diag = torch.cat([dl.reshape(-1), dr.reshape(-1)])
    return a_core, dk, wing_diag, diag


def precondition_gradient(
    method: str,
    cfg: Config,
    history: SecantHistory,
    u: Tensor,
    v: Tensor,
    g: KLR,
    accepted_count: int,
) -> Tuple[KLR, ResponseDiagnostics]:
    diag = ResponseDiagnostics(history_size=len(history))

    if (
        method == "plain_compact"
        or accepted_count < cfg.response_start_accepted
        or len(history) < 2
    ):
        return g, diag

    a_core, dk, wing_diag, diag = build_response_operator(
        method, cfg, history, u, v
    )

    if method == "diagonal_response_8a":
        pk = g.k / (dk + cfg.response_ridge)
    elif method == "full_core_response_8a":
        if a_core is None:
            return g, diag
        q = cfg.rank * cfg.rank
        eye = torch.eye(q, device=u.device, dtype=u.dtype)
        pk = torch.linalg.solve(
            a_core + cfg.response_ridge * eye,
            g.k.reshape(-1),
        ).reshape_as(g.k)
    else:
        raise ValueError(method)

    nl = g.l.numel()
    dl = wing_diag[:nl].reshape_as(g.l)
    dr = wing_diag[nl:].reshape_as(g.r)

    p_response = KLR(
        pk,
        g.l / (dl + cfg.response_ridge),
        g.r / (dr + cfg.response_ridge),
    )

    alpha = cfg.response_mix
    mixed = KLR(
        (1.0 - alpha) * g.k + alpha * p_response.k,
        (1.0 - alpha) * g.l + alpha * p_response.l,
        (1.0 - alpha) * g.r + alpha * p_response.r,
    )
    mixed = clip_preconditioned_norm(
        g, mixed, cfg.response_norm_cap
    )

    diag.cosine_raw_pre = safe_cosine(g, mixed)
    diag.relative_direction_change = float(
        (
            klr_norm(subtract_klr(mixed, g))
            / (klr_norm(g) + EPS)
        ).detach().cpu()
    )
    diag.raw_to_pre_norm_ratio = float(
        (klr_norm(mixed) / (klr_norm(g) + EPS))
        .detach()
        .cpu()
    )
    return mixed, diag


def append_accepted_secant(
    cfg: Config,
    history: SecantHistory,
    ou: Tensor,
    ov: Tensor,
    nu: Tensor,
    nv: Tensor,
    executed_eta_old: KLR,
    old_grad: KLR,
    new_grad: KLR,
) -> Tuple[float, float]:
    s_new = transport_klr(
        ou, ov, nu, nv, executed_eta_old
    )
    old_grad_new = transport_klr(
        ou, ov, nu, nv, old_grad
    )
    y_new = subtract_klr(new_grad, old_grad_new)

    s_norm = float(klr_norm(s_new).detach().cpu())
    y_norm = float(klr_norm(y_new).detach().cpu())

    if s_norm >= cfg.response_min_secant_norm:
        history.append(nu, nv, s_new, y_new)
    return s_norm, y_norm


def run_intrinsic(
    method: str,
    cfg: Config,
    xtr: Tensor,
    ytr: Tensor,
    xva: Tensor,
    yva: Tensor,
    init,
):
    u, s, v = (z.clone() for z in init)
    p = zero_klr(u, v)
    history = SecantHistory(cfg.response_history)
    c_inv = clipped_information_inverse(
        xtr,
        cfg.information_ridge,
        cfg.information_max_condition,
    )

    dt = cfg.ham_dt
    damping = cfg.ham_damping
    accepted = 0
    max_constraint = 0.0
    max_response_cond = 1.0
    max_direction_change = 0.0
    min_cosine = 1.0
    max_floor_fraction = 0.0
    max_negative_fraction = 0.0
    max_abs_log_norm_ratio = 0.0
    sum_abs_log_norm_ratio = 0.0
    norm_ratio_changed_count = 0
    norm_ratio_count = 0
    active_steps = 0
    traces = []
    start = time.perf_counter()

    for step in range(cfg.steps):
        ou, os, ov, op = (
            u.clone(),
            s.clone(),
            v.clone(),
            copy_klr(p),
        )
        dt_used = dt

        old_loss, raw_grad = compact_loss_grad_klr(
            xtr, ytr, u, s, v
        )
        grad_pre, response_diag = precondition_gradient(
            method,
            cfg,
            history,
            u,
            v,
            raw_grad,
            accepted,
        )

        if response_diag.active:
            active_steps += 1
            max_response_cond = max(
                max_response_cond,
                response_diag.response_condition,
            )
            max_direction_change = max(
                max_direction_change,
                response_diag.relative_direction_change,
            )
            min_cosine = min(
                min_cosine,
                response_diag.cosine_raw_pre,
            )
            max_floor_fraction = max(
                max_floor_fraction,
                response_diag.floor_fraction,
            )
            max_negative_fraction = max(
                max_negative_fraction,
                response_diag.negative_eigen_fraction,
            )
            abs_log_norm_ratio = abs(
                math.log(max(response_diag.raw_to_pre_norm_ratio, 1e-300))
            )
            max_abs_log_norm_ratio = max(
                max_abs_log_norm_ratio,
                abs_log_norm_ratio,
            )
            sum_abs_log_norm_ratio += abs_log_norm_ratio
            norm_ratio_count += 1
            if abs_log_norm_ratio > 1e-3:
                norm_ratio_changed_count += 1

        old_energy = old_loss + klr_kinetic(
            u, v, p, c_inv, cfg.ham_mass
        )
        damp = math.exp(-0.5 * damping * dt_used)
        pt = add_klr(
            scale_klr(p, damp),
            grad_pre,
            -0.5 * dt_used,
        )

        pnorm = float(klr_norm(pt).detach().cpu())
        if pnorm > cfg.ham_max_momentum_norm:
            pt = scale_klr(
                pt,
                cfg.ham_max_momentum_norm / pnorm,
            )

        vel = right_metric_action(
            u, v, pt, c_inv, cfg.ham_mass
        )
        executed_eta_old = scale_klr(vel, dt_used)
        un, sn, vn = retract_klr(
            u, s, v, executed_eta_old, cfg.rank
        )

        ptrans = transport_klr(u, v, un, vn, pt)
        new_loss, new_raw_grad = compact_loss_grad_klr(
            xtr, ytr, un, sn, vn
        )

        # Candidate second kick uses the current fitted history only.
        new_grad_pre, new_diag = precondition_gradient(
            method,
            cfg,
            history,
            un,
            vn,
            new_raw_grad,
            accepted,
        )
        pn = scale_klr(
            add_klr(
                ptrans,
                new_grad_pre,
                -0.5 * dt_used,
            ),
            damp,
        )

        new_energy = new_loss + klr_kinetic(
            un, vn, pn, c_inv, cfg.ham_mass
        )

        old_e = float(old_energy.detach().cpu())
        energy_ok = float(new_energy.detach().cpu()) <= (
            old_e
            + cfg.energy_tolerance
            + 1e-8 * max(abs(old_e), 1.0)
        )
        loss_ok = float(new_loss.detach().cpu()) <= (
            float(old_loss.detach().cpu()) + 1e-12
        )
        finite = bool(
            torch.isfinite(sn).all()
            and torch.isfinite(pn.k).all()
            and torch.isfinite(pn.l).all()
            and torch.isfinite(pn.r).all()
        )
        accept = finite and (energy_ok or loss_ok)

        secant_s_norm = 0.0
        secant_y_norm = 0.0

        if accept:
            # Update response only after the step is truly accepted.
            secant_s_norm, secant_y_norm = append_accepted_secant(
                cfg,
                history,
                ou,
                ov,
                un,
                vn,
                executed_eta_old,
                raw_grad,
                new_raw_grad,
            )

            u, s, v, p = un, sn, vn, pn
            accepted += 1
            dt = min(
                cfg.adaptive_dt_max,
                dt * cfg.adaptive_dt_expand,
            )
            damping = (
                max(
                    cfg.adaptive_damping_min,
                    damping * 0.995,
                )
                if energy_ok
                else min(
                    cfg.adaptive_damping_max,
                    damping * 1.05,
                )
            )
        else:
            u, s, v, p = ou, os, ov, op
            dt = max(
                cfg.adaptive_dt_min,
                dt * cfg.adaptive_dt_shrink,
            )
            damping = min(
                cfg.adaptive_damping_max,
                damping * 1.1,
            )

        max_constraint = max(
            max_constraint,
            klr_constraint_residual(u, v, p),
        )

        if (
            step == 0
            or step == cfg.steps - 1
            or (step + 1) % cfg.eval_interval == 0
        ):
            val = 0.5 * torch.mean(
                (lowrank_forward(xva, u, s, v) - yva) ** 2
            )
            traces.append({
                "method": method,
                "step": step + 1,
                "val_loss": float(val.detach().cpu()),
                "accept_rate": accepted / (step + 1),
                "accepted_count": accepted,
                "history_size": len(history),
                "dt": dt,
                "damping": damping,
                "response_active": response_diag.active,
                "cosine_raw_pre": response_diag.cosine_raw_pre,
                "relative_direction_change":
                    response_diag.relative_direction_change,
                "raw_to_pre_norm_ratio":
                    response_diag.raw_to_pre_norm_ratio,
                "response_condition":
                    response_diag.response_condition,
                "negative_eigen_fraction":
                    response_diag.negative_eigen_fraction,
                "floor_fraction":
                    response_diag.floor_fraction,
                "ceiling_fraction":
                    response_diag.ceiling_fraction,
                "secant_fit_residual":
                    response_diag.secant_fit_residual,
                "accepted_secant_s_norm": secant_s_norm,
                "accepted_secant_y_norm": secant_y_norm,
                "klr_constraint_residual": max_constraint,
            })

    final_val = 0.5 * torch.mean(
        (lowrank_forward(xva, u, s, v) - yva) ** 2
    )

    return {
        "method": method,
        "final_val_loss": float(final_val.detach().cpu()),
        "best_val_loss": min(r["val_loss"] for r in traces),
        "accept_rate": accepted / cfg.steps,
        "accepted_steps": accepted,
        "response_active_steps": active_steps,
        "max_klr_constraint_residual": max_constraint,
        "max_response_condition": max_response_cond,
        "max_relative_direction_change": max_direction_change,
        "min_cosine_raw_pre": min_cosine,
        "max_abs_log_norm_ratio": max_abs_log_norm_ratio,
        "mean_abs_log_norm_ratio": (
            sum_abs_log_norm_ratio / max(norm_ratio_count, 1)
        ),
        "fraction_norm_ratio_changed": (
            norm_ratio_changed_count / max(norm_ratio_count, 1)
        ),
        "max_floor_fraction": max_floor_fraction,
        "max_negative_eigen_fraction": max_negative_fraction,
        "final_history_size": len(history),
        "wall_seconds": time.perf_counter() - start,
    }, traces


def run_adamw(
    cfg: Config,
    xtr: Tensor,
    ytr: Tensor,
    xva: Tensor,
    yva: Tensor,
    init,
    lr: float,
):
    u, s, v = init
    root = torch.sqrt(s)
    b = torch.nn.Parameter(u * root.unsqueeze(0))
    a = torch.nn.Parameter(root.unsqueeze(1) * v.T)

    opt = torch.optim.AdamW(
        [a, b],
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    start = time.perf_counter()
    best = float("inf")
    for _ in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        pred = (xtr @ a.T) @ b.T
        loss = 0.5 * torch.mean((pred - ytr) ** 2)
        loss.backward()
        opt.step()

        with torch.no_grad():
            val = 0.5 * torch.mean(
                (((xva @ a.T) @ b.T) - yva) ** 2
            )
            best = min(best, float(val.detach().cpu()))

    with torch.no_grad():
        final_val = 0.5 * torch.mean(
            (((xva @ a.T) @ b.T) - yva) ** 2
        )

    return {
        "method": "factor_adamw",
        "adamw_lr": lr,
        "final_val_loss": float(final_val.detach().cpu()),
        "best_val_loss": best,
        "accept_rate": 1.0,
        "accepted_steps": cfg.steps,
        "response_active_steps": 0,
        "max_klr_constraint_residual": 0.0,
        "max_response_condition": 1.0,
        "max_relative_direction_change": 0.0,
        "min_cosine_raw_pre": 1.0,
        "max_floor_fraction": 0.0,
        "max_negative_eigen_fraction": 0.0,
        "final_history_size": 0,
        "wall_seconds": time.perf_counter() - start,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean_ci(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, 0.0
    std = float(arr.std(ddof=1))
    return mean, 1.96 * std / math.sqrt(len(arr))


def derive_response_verdict(
    raw_gates: Dict[str, Any],
    *,
    norm_ratio_threshold: float = 1e-3,
    direction_threshold: float = 1e-3,
) -> Dict[str, bool]:
    """Separate post-hoc mechanism interpretation from raw audit gates."""
    return {
        "DERIVED_MAGNITUDE_RESPONSE_ACTIVE": bool(
            raw_gates["PASS_DIAGONAL_BEATS_PLAIN_ON_MEAN"]
            and raw_gates.get("MAX_ABS_LOG_NORM_RATIO", 0.0)
            > norm_ratio_threshold
        ),
        "DERIVED_FULL_CORE_DIRECTION_ACTIVE": bool(
            raw_gates["PASS_FULL_CORE_BEATS_PLAIN_ON_MEAN"]
            and raw_gates["MAX_RELATIVE_DIRECTION_CHANGE"]
            > direction_threshold
            and raw_gates["MIN_COSINE_RAW_PRE"] < 0.999999
        ),
    }


def main() -> int:
    cfg = parse_args()
    device = choose_device(cfg.device)
    dtype = dtype_from_name(cfg.dtype)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("H14C3-8A TRANSPORTED ACCEPTED-SECANT RESPONSE AUDIT")
    print("=" * 120)
    print(json.dumps(asdict(cfg), indent=2))

    methods = [
        "plain_compact",
        "diagonal_response_8a",
        "full_core_response_8a",
    ]
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    condition_summary: List[Dict[str, Any]] = []

    conditions = parse_floats(cfg.conditions)
    lr_grid = parse_floats(cfg.adamw_lr_grid)

    for ci, condition in enumerate(conditions):
        tune_scores = []
        for lr in lr_grid:
            vals = []
            for ti in range(cfg.tune_trials):
                seed = cfg.seed + ci * 100000 + ti * 1009
                problem = make_problem(
                    cfg, condition, device, dtype, seed
                )
                row = run_adamw(
                    cfg, *problem[:-1], problem[-1], lr
                )
                vals.append(row["final_val_loss"])
            tune_scores.append((float(np.mean(vals)), lr))
        best_lr = min(tune_scores)[1]

        print(
            f"\n[condition={condition:g}] tuned AdamW lr={best_lr}"
        )

        per_method_values: Dict[str, List[float]] = {
            m: [] for m in methods + ["factor_adamw"]
        }

        for trial in range(cfg.trials):
            seed = (
                cfg.seed
                + ci * 100000
                + 50000
                + trial * 1009
            )
            xtr, ytr, xva, yva, init = make_problem(
                cfg, condition, device, dtype, seed
            )

            trial_rows = []
            for method in methods:
                row, tr = run_intrinsic(
                    method,
                    cfg,
                    xtr,
                    ytr,
                    xva,
                    yva,
                    init,
                )
                row.update({
                    "condition": condition,
                    "trial": trial,
                    "seed": seed,
                })
                rows.append(row)
                trial_rows.append(row)
                per_method_values[method].append(
                    row["final_val_loss"]
                )
                for trow in tr:
                    trow.update({
                        "condition": condition,
                        "trial": trial,
                        "seed": seed,
                    })
                    traces.append(trow)

            adam = run_adamw(
                cfg,
                xtr,
                ytr,
                xva,
                yva,
                init,
                best_lr,
            )
            adam.update({
                "condition": condition,
                "trial": trial,
                "seed": seed,
            })
            rows.append(adam)
            per_method_values["factor_adamw"].append(
                adam["final_val_loss"]
            )

            print(
                f"  trial {trial+1}/{cfg.trials} "
                f"plain={trial_rows[0]['final_val_loss']:.8e} "
                f"diag8A={trial_rows[1]['final_val_loss']:.8e} "
                f"full8A={trial_rows[2]['final_val_loss']:.8e} "
                f"adamw={adam['final_val_loss']:.8e}\n"
                f"      diag change={trial_rows[1]['max_relative_direction_change']:.3e} "
                f"cos_min={trial_rows[1]['min_cosine_raw_pre']:.9f} "
                f"full change={trial_rows[2]['max_relative_direction_change']:.3e} "
                f"cos_min={trial_rows[2]['min_cosine_raw_pre']:.9f}"
            )

        adam_vals = per_method_values["factor_adamw"]
        for method in methods + ["factor_adamw"]:
            vals = per_method_values[method]
            mean, ci95 = mean_ci(vals)
            if method == "factor_adamw":
                wins = 0
                paired_diff = 0.0
            else:
                diffs = [
                    a - b for a, b in zip(vals, adam_vals)
                ]
                wins = sum(d < 0 for d in diffs)
                paired_diff = float(np.mean(diffs))

            condition_summary.append({
                "condition": condition,
                "method": method,
                "mean_final_val_loss": mean,
                "ci95_halfwidth": ci95,
                "wins_vs_adamw": wins,
                "trials": cfg.trials,
                "mean_paired_diff_vs_adamw": paired_diff,
                "selected_adamw_lr": best_lr,
            })

    summary_lookup = {
        (r["condition"], r["method"]): r
        for r in condition_summary
    }

    response_rows = [
        r for r in rows
        if r["method"] in (
            "diagonal_response_8a",
            "full_core_response_8a",
        )
    ]
    full_diffs = [
        r["mean_paired_diff_vs_adamw"]
        for r in condition_summary
        if r["method"] == "full_core_response_8a"
    ]
    diag_diffs = [
        r["mean_paired_diff_vs_adamw"]
        for r in condition_summary
        if r["method"] == "diagonal_response_8a"
    ]

    raw_gates = {
        "PASS_RESPONSE_MECHANISM_ACTIVE": all(
            r["response_active_steps"] > 0
            and r["max_relative_direction_change"] > 1e-3
            and r["min_cosine_raw_pre"] < 0.999999
            for r in response_rows
        ),
        "PASS_FULL_CORE_BEATS_PLAIN_ON_MEAN": all(
            summary_lookup[(c, "full_core_response_8a")][
                "mean_final_val_loss"
            ]
            <
            summary_lookup[(c, "plain_compact")][
                "mean_final_val_loss"
            ]
            for c in conditions
        ),
        "PASS_DIAGONAL_BEATS_PLAIN_ON_MEAN": all(
            summary_lookup[(c, "diagonal_response_8a")][
                "mean_final_val_loss"
            ]
            <
            summary_lookup[(c, "plain_compact")][
                "mean_final_val_loss"
            ]
            for c in conditions
        ),
        "PASS_FULL_CORE_BEATS_ADAMW_ALL_CONDITIONS": all(
            d < 0 for d in full_diffs
        ),
        "PASS_DIAGONAL_BEATS_ADAMW_ALL_CONDITIONS": all(
            d < 0 for d in diag_diffs
        ),
        "MEAN_FULL_CORE_PAIRED_DIFF_VS_ADAMW": float(
            np.mean(full_diffs)
        ),
        "MEAN_DIAGONAL_PAIRED_DIFF_VS_ADAMW": float(
            np.mean(diag_diffs)
        ),
        "MAX_RELATIVE_DIRECTION_CHANGE": max(
            r["max_relative_direction_change"]
            for r in response_rows
        ),
        "MIN_COSINE_RAW_PRE": min(
            r["min_cosine_raw_pre"]
            for r in response_rows
        ),
        "MAX_FLOOR_FRACTION": max(
            r["max_floor_fraction"]
            for r in response_rows
        ),
        "MAX_ABS_LOG_NORM_RATIO": max(
            r["max_abs_log_norm_ratio"]
            for r in response_rows
        ),
        "MEAN_ABS_LOG_NORM_RATIO": float(np.mean([
            r["mean_abs_log_norm_ratio"]
            for r in response_rows
        ])),
        "FRACTION_NORM_RATIO_CHANGED": float(np.mean([
            r["fraction_norm_ratio_changed"]
            for r in response_rows
        ])),
        "MAX_NEGATIVE_EIGEN_FRACTION": max(
            r["max_negative_eigen_fraction"]
            for r in response_rows
        ),
        "MAX_KLR_CONSTRAINT_RESIDUAL": max(
            r["max_klr_constraint_residual"]
            for r in rows
            if r["method"] != "factor_adamw"
        ),
        "MAX_RESPONSE_CONDITION": max(
            r["max_response_condition"]
            for r in response_rows
        ),
    }
    derived_verdict = derive_response_verdict(raw_gates)
    gates = {**raw_gates, **derived_verdict}

    write_csv(out / "summary.csv", rows)
    write_csv(out / "traces.csv", traces)
    write_csv(
        out / "condition_summary.csv",
        condition_summary,
    )
    (out / "gates.json").write_text(
        json.dumps(gates, indent=2),
        encoding="utf-8",
    )
    (out / "raw_gates.json").write_text(
        json.dumps(raw_gates, indent=2),
        encoding="utf-8",
    )
    (out / "derived_verdict.json").write_text(
        json.dumps(derived_verdict, indent=2),
        encoding="utf-8",
    )
    (out / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 120)
    print("H14C3-8A VERDICT")
    print("=" * 120)
    print(json.dumps(gates, indent=2))

    print("\nCondition summary:")
    for row in condition_summary:
        print(
            f"cond={row['condition']:g} "
            f"{row['method']:<24} "
            f"mean={row['mean_final_val_loss']:.8e} "
            f"diff_vs_adamw={row['mean_paired_diff_vs_adamw']:+.2e} "
            f"wins={row['wins_vs_adamw']}/{row['trials']}"
        )

    print(f"\nOutputs: {out.resolve()}")
    return 0


if __name__ == "__main__":
    main()
