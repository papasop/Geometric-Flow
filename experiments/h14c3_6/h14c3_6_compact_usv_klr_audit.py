#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H14C3-6 — COMPACT USV-KLR INTRINSIC HAMILTONIAN AUDIT
=====================================================

Compact intrinsic state:
    M = U diag(S) V^T

Compact tangent momentum:
    P = U K V^T + L V^T + U R

with gauge conditions:
    U^T L = 0
    R V = 0

No persistent dense M or dense P state is stored.

This audit compares:
1. compact intrinsic information Hamiltonian
2. dense reference intrinsic information Hamiltonian
3. factor AdamW baseline

It validates:
- compact/dense trajectory agreement
- rank preservation
- tangent constraints
- true gauge-invariant initialization
- memory-state accounting
- runtime and final validation loss

The controlled task is low-rank matrix regression.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


Tensor = torch.Tensor
EPS = 1e-30


@dataclass
class Config:
    seed: int = 1414
    trials: int = 6

    n_train: int = 2048
    n_val: int = 1024
    d_in: int = 96
    d_out: int = 128
    rank: int = 4
    noise_std: float = 0.02
    target_condition: float = 100.0

    steps: int = 400
    eval_interval: int = 20

    ham_dt: float = 0.12
    ham_damping: float = 0.08
    ham_mass: float = 1.0
    ham_max_momentum_norm: float = 50.0

    information_ridge: float = 1e-6
    information_max_condition: float = 1000.0

    adaptive_dt_min: float = 1e-4
    adaptive_dt_max: float = 0.25
    adaptive_dt_shrink: float = 0.5
    adaptive_dt_expand: float = 1.05
    energy_tolerance: float = 1e-12
    adaptive_damping_min: float = 0.01
    adaptive_damping_max: float = 0.5

    adamw_lr: float = 3e-3
    adamw_weight_decay: float = 0.0
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

    gauge_conditions: str = "1,10,100,1000"
    gauge_tolerance: float = 2e-10
    compact_dense_tolerance: float = 5e-9
    rank_tolerance: float = 1e-9
    tangent_tolerance: float = 1e-9

    dtype: str = "float64"
    device: str = "cuda"
    output_dir: str = "h14c3_6_compact_usv_klr_audit"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    for f in Config.__dataclass_fields__.values():
        name = "--" + f.name.replace("_", "-")
        p.add_argument(name, type=type(f.default), default=f.default)
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[H14C3-6] ignored notebook/kernel arguments: {unknown}")
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
            f"[H14C3-6] GPU={torch.cuda.get_device_name(dev)} "
            f"memory={torch.cuda.get_device_properties(dev).total_memory / 2**30:.1f} GiB"
        )
        return dev
    print("[H14C3-6] using CPU")
    return torch.device("cpu")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def fro_inner(a: Tensor, b: Tensor) -> Tensor:
    return torch.sum(a * b)


def fro_norm(a: Tensor) -> Tensor:
    return torch.linalg.norm(a)


def relerr(a: Tensor, b: Tensor) -> float:
    return float(
        (fro_norm(a - b) / (fro_norm(a) + fro_norm(b) + EPS))
        .detach().cpu()
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def orthonormal(rows: int, cols: int, device, dtype) -> Tensor:
    q, _ = torch.linalg.qr(
        torch.randn(rows, cols, device=device, dtype=dtype),
        mode="reduced",
    )
    return q


def make_target(cfg: Config, device, dtype, seed: int) -> Tensor:
    set_seed(seed)
    u = orthonormal(cfg.d_out, cfg.rank, device, dtype)
    v = orthonormal(cfg.d_in, cfg.rank, device, dtype)
    s = torch.logspace(
        0.0,
        -math.log10(cfg.target_condition),
        cfg.rank,
        device=device,
        dtype=dtype,
    )
    return u @ torch.diag(s) @ v.T


def make_data(cfg: Config, device, dtype, target: Tensor, seed: int):
    set_seed(seed)
    xtr = torch.randn(
        cfg.n_train, cfg.d_in, device=device, dtype=dtype
    ) / math.sqrt(cfg.d_in)
    xva = torch.randn(
        cfg.n_val, cfg.d_in, device=device, dtype=dtype
    ) / math.sqrt(cfg.d_in)

    ytr = xtr @ target.T
    yva = xva @ target.T

    if cfg.noise_std > 0:
        ytr = ytr + cfg.noise_std * torch.randn_like(ytr)
        yva = yva + cfg.noise_std * torch.randn_like(yva)

    return xtr, ytr, xva, yva


def product(u: Tensor, s: Tensor, v: Tensor) -> Tensor:
    return u @ torch.diag(s) @ v.T


def loss_and_grad(m: Tensor, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    residual = x @ m.T - y
    loss = 0.5 * torch.mean(residual * residual)
    grad = residual.T @ x / residual.numel()
    return loss, grad


def compact_svd(m: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    return u[:, :rank], s[:rank], vh[:rank, :].T


def tangent_project(u: Tensor, v: Tensor, z: Tensor) -> Tensor:
    puz = u @ (u.T @ z)
    zpv = (z @ v) @ v.T
    puzpv = u @ ((u.T @ z) @ v) @ v.T
    return puz + zpv - puzpv


def clipped_information_metric(
    x: Tensor,
    ridge: float,
    max_condition: float,
) -> Tuple[Tensor, float, float]:
    c = x.T @ x / x.shape[0]
    c = 0.5 * (c + c.T)
    evals, evecs = torch.linalg.eigh(c)
    evals = torch.clamp(evals, min=ridge)
    raw_cond = float((evals.max() / evals.min()).detach().cpu())
    floor = evals.max() / max_condition
    clipped = torch.clamp(evals, min=floor)
    clip_cond = float((clipped.max() / clipped.min()).detach().cpu())
    c_inv = evecs @ torch.diag(1.0 / clipped) @ evecs.T
    return c_inv, raw_cond, clip_cond


# ---------------------------------------------------------------------
# Compact tangent KLR representation
# ---------------------------------------------------------------------

@dataclass
class CompactMomentum:
    k: Tensor       # (r, r)
    l: Tensor       # (d_out, r), U^T L = 0
    rpart: Tensor   # (r, d_in), R V = 0


def zero_momentum(u: Tensor, v: Tensor) -> CompactMomentum:
    rank = u.shape[1]
    return CompactMomentum(
        k=torch.zeros(rank, rank, device=u.device, dtype=u.dtype),
        l=torch.zeros(u.shape[0], rank, device=u.device, dtype=u.dtype),
        rpart=torch.zeros(rank, v.shape[0], device=u.device, dtype=u.dtype),
    )


def dense_from_klr(
    u: Tensor,
    v: Tensor,
    p: CompactMomentum,
) -> Tensor:
    return (
        u @ p.k @ v.T
        + p.l @ v.T
        + u @ p.rpart
    )


def klr_from_dense(
    u: Tensor,
    v: Tensor,
    z: Tensor,
) -> CompactMomentum:
    zt = tangent_project(u, v, z)
    k = u.T @ zt @ v
    l = zt @ v - u @ k
    rpart = u.T @ zt - k @ v.T

    # Cleanup constraints.
    l = l - u @ (u.T @ l)
    rpart = rpart - (rpart @ v) @ v.T
    return CompactMomentum(k=k, l=l, rpart=rpart)


def klr_add(
    a: CompactMomentum,
    b: CompactMomentum,
    alpha: float = 1.0,
) -> CompactMomentum:
    return CompactMomentum(
        k=a.k + alpha * b.k,
        l=a.l + alpha * b.l,
        rpart=a.rpart + alpha * b.rpart,
    )


def klr_scale(a: CompactMomentum, alpha: float) -> CompactMomentum:
    return CompactMomentum(
        k=alpha * a.k,
        l=alpha * a.l,
        rpart=alpha * a.rpart,
    )


def klr_norm_sq(a: CompactMomentum) -> Tensor:
    # Orthogonal tangent blocks under the KLR gauge constraints.
    return (
        fro_inner(a.k, a.k)
        + fro_inner(a.l, a.l)
        + fro_inner(a.rpart, a.rpart)
    )


def klr_constraint_residual(
    u: Tensor,
    v: Tensor,
    p: CompactMomentum,
) -> float:
    a = fro_norm(u.T @ p.l)
    b = fro_norm(p.rpart @ v)
    den = (
        fro_norm(p.k)
        + fro_norm(p.l)
        + fro_norm(p.rpart)
        + EPS
    )
    return float(((a + b) / den).detach().cpu())


def tangent_grad_to_klr(
    u: Tensor,
    v: Tensor,
    grad: Tensor,
) -> CompactMomentum:
    return klr_from_dense(u, v, grad)


def information_velocity_dense_from_klr(
    u: Tensor,
    v: Tensor,
    p: CompactMomentum,
    c_inv: Tensor,
    mass: float,
) -> Tensor:
    dense_p = dense_from_klr(u, v, p)
    return tangent_project(u, v, (dense_p @ c_inv) / mass)


def information_kinetic_klr(
    u: Tensor,
    v: Tensor,
    p: CompactMomentum,
    c_inv: Tensor,
    mass: float,
) -> Tensor:
    dense_p = dense_from_klr(u, v, p)
    return 0.5 * fro_inner(dense_p @ c_inv, dense_p) / mass


def retraction_small_core(
    u: Tensor,
    s: Tensor,
    v: Tensor,
    eta: Tensor,
    rank: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    m = product(u, s, v)

    left_raw = eta @ v - u @ (u.T @ eta @ v)
    right_raw = eta.T @ u - v @ (v.T @ eta.T @ u)

    ql, rl = torch.linalg.qr(left_raw, mode="reduced")
    qr, rr = torch.linalg.qr(right_raw, mode="reduced")

    lmask = torch.abs(torch.diag(rl)) > 1e-12
    rmask = torch.abs(torch.diag(rr)) > 1e-12
    ql = ql[:, lmask]
    qr = qr[:, rmask]

    ub = torch.cat([u, ql], dim=1)
    vb = torch.cat([v, qr], dim=1)

    core = ub.T @ (m + eta) @ vb
    uc, sc, vhc = torch.linalg.svd(core, full_matrices=False)

    ur = ub @ uc[:, :rank]
    vr = vb @ vhc[:rank, :].T
    sr = sc[:rank]

    qu, ru = torch.linalg.qr(ur, mode="reduced")
    qv, rv = torch.linalg.qr(vr, mode="reduced")
    small = ru @ torch.diag(sr) @ rv.T
    us, ss, vhs = torch.linalg.svd(small, full_matrices=False)

    return qu @ us[:, :rank], ss[:rank], qv @ vhs[:rank, :].T


def transport_klr(
    old_u: Tensor,
    old_v: Tensor,
    new_u: Tensor,
    new_v: Tensor,
    p: CompactMomentum,
) -> CompactMomentum:
    # Current implementation reconstructs only transient dense tangent,
    # projects to the new tangent space, then compresses back to KLR.
    dense = dense_from_klr(old_u, old_v, p)
    transported = tangent_project(new_u, new_v, dense)
    return klr_from_dense(new_u, new_v, transported)


# ---------------------------------------------------------------------
# Compact and dense runs
# ---------------------------------------------------------------------

def compact_hamiltonian_run(
    cfg: Config,
    xtr: Tensor,
    ytr: Tensor,
    xva: Tensor,
    yva: Tensor,
    init_u: Tensor,
    init_s: Tensor,
    init_v: Tensor,
):
    u, s, v = init_u.clone(), init_s.clone(), init_v.clone()
    p = zero_momentum(u, v)

    c_inv, cond_raw, cond_clip = clipped_information_metric(
        xtr,
        cfg.information_ridge,
        cfg.information_max_condition,
    )

    dt = cfg.ham_dt
    damping = cfg.ham_damping
    accepted = 0
    rejected = 0
    max_constraint = 0.0
    max_rank_tail = 0.0
    traces = []
    path = [product(u, s, v).detach().cpu()]
    start = time.perf_counter()

    for step in range(cfg.steps):
        old_u, old_s, old_v = u.clone(), s.clone(), v.clone()
        old_p = CompactMomentum(
            p.k.clone(), p.l.clone(), p.rpart.clone()
        )

        m = product(u, s, v)
        old_loss, grad = loss_and_grad(m, xtr, ytr)
        old_energy = old_loss + information_kinetic_klr(
            u, v, p, c_inv, cfg.ham_mass
        )

        g = tangent_grad_to_klr(u, v, grad)
        damp = math.exp(-0.5 * damping * dt)

        p_trial = klr_scale(p, damp)
        p_trial = klr_add(p_trial, g, alpha=-0.5 * dt)

        pnorm = float(torch.sqrt(klr_norm_sq(p_trial)).detach().cpu())
        if pnorm > cfg.ham_max_momentum_norm:
            p_trial = klr_scale(
                p_trial,
                cfg.ham_max_momentum_norm / pnorm,
            )

        velocity = information_velocity_dense_from_klr(
            u, v, p_trial, c_inv, cfg.ham_mass
        )
        eta = dt * velocity

        u_new, s_new, v_new = retraction_small_core(
            u, s, v, eta, cfg.rank
        )

        p_trans = transport_klr(
            u, v, u_new, v_new, p_trial
        )

        m_new = product(u_new, s_new, v_new)
        new_loss, new_grad = loss_and_grad(m_new, xtr, ytr)
        g_new = tangent_grad_to_klr(
            u_new, v_new, new_grad
        )

        p_new = klr_add(p_trans, g_new, alpha=-0.5 * dt)
        p_new = klr_scale(p_new, damp)

        new_energy = new_loss + information_kinetic_klr(
            u_new, v_new, p_new, c_inv, cfg.ham_mass
        )

        finite = bool(
            torch.isfinite(m_new).all()
            and torch.isfinite(p_new.k).all()
            and torch.isfinite(p_new.l).all()
            and torch.isfinite(p_new.rpart).all()
            and torch.isfinite(new_energy)
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
        accept = finite and (energy_ok or loss_ok)

        if accept:
            u, s, v, p = u_new, s_new, v_new, p_new
            accepted += 1
            dt = min(cfg.adaptive_dt_max, dt * cfg.adaptive_dt_expand)
            if energy_ok:
                damping = max(
                    cfg.adaptive_damping_min,
                    damping * 0.995,
                )
            else:
                damping = min(
                    cfg.adaptive_damping_max,
                    damping * 1.05,
                )
        else:
            u, s, v, p = old_u, old_s, old_v, old_p
            rejected += 1
            dt = max(cfg.adaptive_dt_min, dt * cfg.adaptive_dt_shrink)
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
            m_now = product(u, s, v)
            val, _ = loss_and_grad(m_now, xva, yva)
            sv = torch.linalg.svdvals(m_now)
            tail = float(
                torch.linalg.norm(sv[cfg.rank:]).detach().cpu()
                if sv.numel() > cfg.rank else 0.0
            )
            max_rank_tail = max(max_rank_tail, tail)
            traces.append({
                "method": "compact_usv_klr_hamiltonian",
                "step": step + 1,
                "val_loss": float(val.detach().cpu()),
                "accept_rate_so_far": accepted / (step + 1),
                "dt": dt,
                "damping": damping,
                "klr_constraint_residual": max_constraint,
                "rank_tail_norm": tail,
            })
            path.append(m_now.detach().cpu())

    final_m = product(u, s, v)
    final_val, _ = loss_and_grad(final_m, xva, yva)

    persistent_elements = (
        u.numel()
        + s.numel()
        + v.numel()
        + p.k.numel()
        + p.l.numel()
        + p.rpart.numel()
    )

    return {
        "method": "compact_usv_klr_hamiltonian",
        "final_val_loss": float(final_val.detach().cpu()),
        "best_val_loss": min(r["val_loss"] for r in traces),
        "accept_rate": accepted / cfg.steps,
        "max_klr_constraint_residual": max_constraint,
        "max_rank_tail_norm": max_rank_tail,
        "persistent_state_elements": persistent_elements,
        "wall_seconds": time.perf_counter() - start,
        "finite": bool(torch.isfinite(final_m).all()),
    }, traces, path


def dense_hamiltonian_run(
    cfg: Config,
    xtr: Tensor,
    ytr: Tensor,
    xva: Tensor,
    yva: Tensor,
    init_u: Tensor,
    init_s: Tensor,
    init_v: Tensor,
):
    u, s, v = init_u.clone(), init_s.clone(), init_v.clone()
    p = torch.zeros(
        cfg.d_out, cfg.d_in,
        device=u.device,
        dtype=u.dtype,
    )

    c_inv, _, _ = clipped_information_metric(
        xtr,
        cfg.information_ridge,
        cfg.information_max_condition,
    )

    dt = cfg.ham_dt
    damping = cfg.ham_damping
    accepted = 0
    max_rank_tail = 0.0
    traces = []
    path = [product(u, s, v).detach().cpu()]
    start = time.perf_counter()

    for step in range(cfg.steps):
        old_u, old_s, old_v = u.clone(), s.clone(), v.clone()
        old_p = p.clone()

        m = product(u, s, v)
        old_loss, grad = loss_and_grad(m, xtr, ytr)
        old_energy = old_loss + 0.5 * fro_inner(
            p @ c_inv, p
        ) / cfg.ham_mass

        grad_t = tangent_project(u, v, grad)
        damp = math.exp(-0.5 * damping * dt)

        p_trial = damp * p - 0.5 * dt * grad_t
        p_trial = tangent_project(u, v, p_trial)

        pnorm = float(fro_norm(p_trial).detach().cpu())
        if pnorm > cfg.ham_max_momentum_norm:
            p_trial = p_trial * (
                cfg.ham_max_momentum_norm / pnorm
            )

        velocity = tangent_project(
            u, v, (p_trial @ c_inv) / cfg.ham_mass
        )
        eta = dt * velocity

        u_new, s_new, v_new = retraction_small_core(
            u, s, v, eta, cfg.rank
        )

        p_trans = tangent_project(u_new, v_new, p_trial)
        m_new = product(u_new, s_new, v_new)
        new_loss, new_grad = loss_and_grad(m_new, xtr, ytr)
        new_grad_t = tangent_project(
            u_new, v_new, new_grad
        )
        p_new = damp * (
            p_trans - 0.5 * dt * new_grad_t
        )
        p_new = tangent_project(u_new, v_new, p_new)

        new_energy = new_loss + 0.5 * fro_inner(
            p_new @ c_inv, p_new
        ) / cfg.ham_mass

        finite = bool(
            torch.isfinite(m_new).all()
            and torch.isfinite(p_new).all()
            and torch.isfinite(new_energy)
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
        accept = finite and (energy_ok or loss_ok)

        if accept:
            u, s, v, p = u_new, s_new, v_new, p_new
            accepted += 1
            dt = min(cfg.adaptive_dt_max, dt * cfg.adaptive_dt_expand)
            if energy_ok:
                damping = max(
                    cfg.adaptive_damping_min,
                    damping * 0.995,
                )
            else:
                damping = min(
                    cfg.adaptive_damping_max,
                    damping * 1.05,
                )
        else:
            u, s, v, p = old_u, old_s, old_v, old_p
            dt = max(cfg.adaptive_dt_min, dt * cfg.adaptive_dt_shrink)
            damping = min(
                cfg.adaptive_damping_max,
                damping * 1.1,
            )

        if (
            step == 0
            or step == cfg.steps - 1
            or (step + 1) % cfg.eval_interval == 0
        ):
            m_now = product(u, s, v)
            val, _ = loss_and_grad(m_now, xva, yva)
            sv = torch.linalg.svdvals(m_now)
            tail = float(
                torch.linalg.norm(sv[cfg.rank:]).detach().cpu()
                if sv.numel() > cfg.rank else 0.0
            )
            max_rank_tail = max(max_rank_tail, tail)
            traces.append({
                "method": "dense_reference_hamiltonian",
                "step": step + 1,
                "val_loss": float(val.detach().cpu()),
                "accept_rate_so_far": accepted / (step + 1),
                "rank_tail_norm": tail,
            })
            path.append(m_now.detach().cpu())

    final_m = product(u, s, v)
    final_val, _ = loss_and_grad(final_m, xva, yva)

    persistent_elements = (
        u.numel()
        + s.numel()
        + v.numel()
        + p.numel()
    )

    return {
        "method": "dense_reference_hamiltonian",
        "final_val_loss": float(final_val.detach().cpu()),
        "best_val_loss": min(r["val_loss"] for r in traces),
        "accept_rate": accepted / cfg.steps,
        "max_rank_tail_norm": max_rank_tail,
        "persistent_state_elements": persistent_elements,
        "wall_seconds": time.perf_counter() - start,
        "finite": bool(torch.isfinite(final_m).all()),
    }, traces, path


def factor_adamw_run(
    cfg: Config,
    xtr: Tensor,
    ytr: Tensor,
    xva: Tensor,
    yva: Tensor,
    init_u: Tensor,
    init_s: Tensor,
    init_v: Tensor,
):
    root = torch.sqrt(init_s)
    b = torch.nn.Parameter(init_u * root.unsqueeze(0))
    a = torch.nn.Parameter(root.unsqueeze(1) * init_v.T)

    opt = torch.optim.AdamW(
        [a, b],
        lr=cfg.adamw_lr,
        betas=(cfg.adamw_beta1, cfg.adamw_beta2),
        weight_decay=cfg.adamw_weight_decay,
    )

    traces = []
    start = time.perf_counter()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        m = b @ a
        residual = xtr @ m.T - ytr
        loss = 0.5 * torch.mean(residual * residual)
        loss.backward()
        opt.step()

        if (
            step == 0
            or step == cfg.steps - 1
            or (step + 1) % cfg.eval_interval == 0
        ):
            with torch.no_grad():
                val, _ = loss_and_grad(b @ a, xva, yva)
            traces.append({
                "method": "factor_adamw",
                "step": step + 1,
                "val_loss": float(val.detach().cpu()),
            })

    final_m = (b @ a).detach()
    final_val, _ = loss_and_grad(final_m, xva, yva)

    # Parameter + gradient + two Adam moments approximation.
    factor_elements = a.numel() + b.numel()
    persistent_elements = 4 * factor_elements

    return {
        "method": "factor_adamw",
        "final_val_loss": float(final_val.detach().cpu()),
        "best_val_loss": min(r["val_loss"] for r in traces),
        "persistent_state_elements": persistent_elements,
        "wall_seconds": time.perf_counter() - start,
        "finite": bool(torch.isfinite(final_m).all()),
    }, traces


def random_gauge(rank: int, condition: float, device, dtype) -> Tensor:
    q1 = orthonormal(rank, rank, device, dtype)
    q2 = orthonormal(rank, rank, device, dtype)
    vals = torch.logspace(
        -0.5 * math.log10(condition),
        0.5 * math.log10(condition),
        rank,
        device=device,
        dtype=dtype,
    )
    return q1 @ torch.diag(vals) @ q2.T


def gauge_initialization_audit(
    cfg: Config,
    u: Tensor,
    s: Tensor,
    v: Tensor,
) -> List[Dict[str, Any]]:
    root = torch.sqrt(s)
    b = u * root.unsqueeze(0)
    a = root.unsqueeze(1) * v.T
    canonical = b @ a

    rows = []
    for i, kappa in enumerate(parse_floats(cfg.gauge_conditions)):
        set_seed(cfg.seed + 900000 + i)
        if kappa == 1.0:
            q = torch.eye(
                cfg.rank, device=u.device, dtype=u.dtype
            )
        else:
            q = random_gauge(
                cfg.rank, kappa, u.device, u.dtype
            )
        ag = torch.linalg.solve(q, a)
        bg = b @ q
        represented = bg @ ag
        ug, sg, vg = compact_svd(represented, cfg.rank)
        reconstructed = product(ug, sg, vg)
        rows.append({
            "gauge_condition": kappa,
            "factor_product_residual": relerr(
                canonical, represented
            ),
            "intrinsic_reconstruction_residual": relerr(
                canonical, reconstructed
            ),
        })
    return rows


def path_residual(a: List[Tensor], b: List[Tensor]) -> float:
    return max(relerr(x, y) for x, y in zip(a, b))


def main() -> int:
    cfg = parse_args()
    device = choose_device(cfg.device)
    dtype = dtype_from_name(cfg.dtype)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("H14C3-6 COMPACT USV-KLR INTRINSIC HAMILTONIAN AUDIT")
    print("=" * 120)
    print(json.dumps(asdict(cfg), indent=2))

    summary_rows = []
    trace_rows = []
    gauge_rows = []

    max_compact_dense_path = 0.0
    max_klr_constraint = 0.0
    max_rank_tail = 0.0
    max_gauge_residual = 0.0

    for trial in range(cfg.trials):
        seed = cfg.seed + 1009 * trial
        target = make_target(cfg, device, dtype, seed + 1)
        xtr, ytr, xva, yva = make_data(
            cfg, device, dtype, target, seed + 2
        )

        set_seed(seed + 3)
        u = orthonormal(cfg.d_out, cfg.rank, device, dtype)
        v = orthonormal(cfg.d_in, cfg.rank, device, dtype)
        s = 0.05 * torch.ones(
            cfg.rank, device=device, dtype=dtype
        )

        gr = gauge_initialization_audit(cfg, u, s, v)
        for row in gr:
            row["trial"] = trial
            gauge_rows.append(row)
            max_gauge_residual = max(
                max_gauge_residual,
                row["factor_product_residual"],
                row["intrinsic_reconstruction_residual"],
            )

        compact_row, compact_trace, compact_path = (
            compact_hamiltonian_run(
                cfg, xtr, ytr, xva, yva, u, s, v
            )
        )
        dense_row, dense_trace, dense_path = (
            dense_hamiltonian_run(
                cfg, xtr, ytr, xva, yva, u, s, v
            )
        )
        adam_row, adam_trace = factor_adamw_run(
            cfg, xtr, ytr, xva, yva, u, s, v
        )

        traj_res = path_residual(compact_path, dense_path)
        max_compact_dense_path = max(
            max_compact_dense_path, traj_res
        )
        max_klr_constraint = max(
            max_klr_constraint,
            compact_row["max_klr_constraint_residual"],
        )
        max_rank_tail = max(
            max_rank_tail,
            compact_row["max_rank_tail_norm"],
            dense_row["max_rank_tail_norm"],
        )

        for row in (compact_row, dense_row, adam_row):
            row["trial"] = trial
            row["seed"] = seed
            row["compact_dense_path_residual"] = traj_res
            summary_rows.append(row)

        for row in compact_trace + dense_trace + adam_trace:
            row["trial"] = trial
            row["seed"] = seed
            trace_rows.append(row)

        print(
            f"[trial {trial+1}/{cfg.trials}] "
            f"compact={compact_row['final_val_loss']:.8e} "
            f"dense={dense_row['final_val_loss']:.8e} "
            f"adamw={adam_row['final_val_loss']:.8e} "
            f"path={traj_res:.2e} "
            f"KLR={compact_row['max_klr_constraint_residual']:.2e} "
            f"state_ratio="
            f"{compact_row['persistent_state_elements'] / dense_row['persistent_state_elements']:.3f}"
        )

    compact_states = [
        r["persistent_state_elements"]
        for r in summary_rows
        if r["method"] == "compact_usv_klr_hamiltonian"
    ]
    dense_states = [
        r["persistent_state_elements"]
        for r in summary_rows
        if r["method"] == "dense_reference_hamiltonian"
    ]
    adam_states = [
        r["persistent_state_elements"]
        for r in summary_rows
        if r["method"] == "factor_adamw"
    ]

    gates = {
        "PASS_COMPACT_DENSE_TRAJECTORY": (
            max_compact_dense_path <= cfg.compact_dense_tolerance
        ),
        "MAX_COMPACT_DENSE_TRAJECTORY_RESIDUAL": max_compact_dense_path,
        "PASS_KLR_CONSTRAINTS": (
            max_klr_constraint <= cfg.tangent_tolerance
        ),
        "MAX_KLR_CONSTRAINT_RESIDUAL": max_klr_constraint,
        "PASS_RANK_PRESERVED": (
            max_rank_tail <= cfg.rank_tolerance
        ),
        "MAX_RANK_TAIL": max_rank_tail,
        "PASS_GAUGE_INITIALIZATION": (
            max_gauge_residual <= cfg.gauge_tolerance
        ),
        "MAX_GAUGE_RESIDUAL": max_gauge_residual,
        "MEAN_COMPACT_STATE_ELEMENTS": float(np.mean(compact_states)),
        "MEAN_DENSE_STATE_ELEMENTS": float(np.mean(dense_states)),
        "MEAN_ADAMW_APPROX_STATE_ELEMENTS": float(np.mean(adam_states)),
        "COMPACT_OVER_DENSE_STATE_RATIO": float(
            np.mean(compact_states) / np.mean(dense_states)
        ),
        "COMPACT_OVER_ADAMW_STATE_RATIO": float(
            np.mean(compact_states) / np.mean(adam_states)
        ),
    }
    gates["PASS_CORE"] = all([
        gates["PASS_COMPACT_DENSE_TRAJECTORY"],
        gates["PASS_KLR_CONSTRAINTS"],
        gates["PASS_RANK_PRESERVED"],
        gates["PASS_GAUGE_INITIALIZATION"],
    ])

    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "traces.csv", trace_rows)
    write_csv(out / "gauge_audit.csv", gauge_rows)
    (out / "gates.json").write_text(
        json.dumps(gates, indent=2),
        encoding="utf-8",
    )
    (out / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 120)
    print("H14C3-6 GATES")
    print("=" * 120)
    print(json.dumps(gates, indent=2))
    print(f"\nOutputs: {out.resolve()}")
    return 0


if __name__ == "__main__":
    main()
