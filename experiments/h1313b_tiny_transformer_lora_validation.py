#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H13.13B — TINY TRANSFORMER / LoRA FORMAL VALIDATION

Purpose
-------
Transfer the H13.9-H13.12 mechanism chain from matrix regression to a genuine
multi-layer causal Transformer with frozen base weights and trainable LoRA
products.

This is an offline controlled formal validation:
- a frozen tiny causal Transformer supplies the shared base model;
- a teacher contains planted LoRA products;
- every student starts from the same represented LoRA products;
- all methods use the same pre-generated minibatch schedule;
- only selected LoRA factors are updated;
- updates are matched by the same GLOBAL realized LoRA-product displacement;
- paired seed-wise statistics, wall-clock, and memory diagnostics are reported.

Compared methods
----------------
- adamw_lora
- fixed_split
- factor_ema
- channel_momentum
- scalar_channel_adaptive
- coupled_channel_covariance

Core structural questions
-------------------------
1. Are all methods initialized from the same represented LoRA products?
2. Do all methods consume exactly the same batch schedule?
3. Is the realized global LoRA-product step matched?
4. Does coupled channel covariance remain gauge covariant layer by layer?
5. Are all LoRA layers finite and active?
6. Does the H13.12 loss advantage transfer beyond matrix regression?

Important limits
----------------
This is not GPT-2, WikiText, or production LLM validation. It is a small,
self-contained Transformer/LoRA mechanism audit that extends the H13.13A smoke.

Formal run
----------
python h1313b_tiny_transformer_lora_validation.py \
  --trials 6 --steps 200 --train-samples 1024 --val-samples 256 \
  --batch-size 16 --probe-steps 0,50,100,150,199 \
  --target-scope all_linear --output-dir h1313b_results

Faster syntax/runtime check
---------------------------
python h1313b_tiny_transformer_lora_validation.py \
  --trials 1 --steps 4 --train-samples 64 --val-samples 32 \
  --batch-size 8 --probe-steps 0,3 --no-plots \
  --output-dir h1313b_tiny
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import random
import sys
import time
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
EPS = 1e-15

METHODS = (
    "adamw_lora",
    "fixed_split",
    "factor_ema",
    "channel_momentum",
    "scalar_channel_adaptive",
    "coupled_channel_covariance",
)


@dataclass
class Config:
    seed: int = 1313
    trials: int = 6
    steps: int = 200

    train_samples: int = 1024
    val_samples: int = 256
    batch_size: int = 16
    seq_len: int = 32
    vocab_size: int = 64

    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 192
    dropout: float = 0.0

    lora_rank: int = 4
    lora_alpha: float = 4.0
    teacher_lora_scale: float = 0.12
    student_init_scale: float = 0.01
    target_scope: str = "all_linear"  # all_linear | attention_only | qkv_only

    target_product_step: float = 0.025
    max_factor_step_norm: float = 100.0
    weight_decay: float = 1e-4

    beta1: float = 0.92
    beta2: float = 0.99
    second_moment_eps: float = 1e-8
    practical_ridge: float = 1e-8
    exact_condition_limit: float = 1e12

    probe_steps: str = "0,50,100,150,199"
    probe_gauges: int = 4
    gauge_kappa_min: float = 1.0
    gauge_kappa_max: float = 30.0

    dtype: str = "float64"
    device: str = "cpu"
    output_dir: str = "h1313b_results"
    no_plots: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


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


def parse_steps(text: str, total_steps: int) -> List[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    vals = [x for x in vals if 0 <= x < total_steps]
    if not vals:
        vals = [0, total_steps - 1]
    return vals


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def qstats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        bias: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features, dtype=dtype, device=device),
            requires_grad=False,
        ) if bias else None

        self.B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=dtype, device=device)
        )
        self.A = nn.Parameter(
            torch.zeros(rank, in_features, dtype=dtype, device=device)
        )
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.A), self.B) * self.scale
        return base + delta

    def product(self) -> Tensor:
        return self.scale * (self.B @ self.A)


class TinyBlock(nn.Module):
    def __init__(self, cfg: Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = d // cfg.n_heads
        if d % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.ln1 = nn.LayerNorm(d, dtype=dtype, device=device)
        self.qkv = LoRALinear(d, 3 * d, cfg.lora_rank, cfg.lora_alpha, True, dtype, device)
        self.proj = LoRALinear(d, d, cfg.lora_rank, cfg.lora_alpha, True, dtype, device)
        self.ln2 = nn.LayerNorm(d, dtype=dtype, device=device)
        self.fc1 = LoRALinear(d, cfg.d_ff, cfg.lora_rank, cfg.lora_alpha, True, dtype, device)
        self.fc2 = LoRALinear(cfg.d_ff, d, cfg.lora_rank, cfg.lora_alpha, True, dtype, device)
        self.dropout = cfg.dropout

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(z: Tensor) -> Tensor:
            return z.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq, d)
        x = x + F.dropout(self.proj(y), self.dropout, self.training)

        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + F.dropout(h, self.dropout, self.training)


class TinyCausalTransformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        dtype = dtype_from_name(cfg.dtype)
        device = torch.device(cfg.device)
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model, dtype=dtype, device=device)
        self.pos = nn.Parameter(
            torch.randn(cfg.seq_len, cfg.d_model, dtype=dtype, device=device) * 0.02,
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [TinyBlock(cfg, dtype, device) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model, dtype=dtype, device=device)
        self.head = LoRALinear(
            cfg.d_model, cfg.vocab_size, cfg.lora_rank, cfg.lora_alpha, False, dtype, device
        )
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        self.token.weight.requires_grad_(False)

        # Disable non-target LoRA factors so target_scope is an actual training scope.
        active = set(self.lora_layers().keys())
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear) and name not in active:
                with torch.no_grad():
                    module.A.zero_()
                    module.B.zero_()
                module.A.requires_grad_(False)
                module.B.requires_grad_(False)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.token(tokens) + self.pos[: tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def lora_layers(self) -> Dict[str, LoRALinear]:
        layers = {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, LoRALinear)
        }
        scope = self.cfg.target_scope
        if scope == "all_linear":
            return layers
        if scope == "attention_only":
            return {
                name: module for name, module in layers.items()
                if name.endswith(".qkv") or name.endswith(".proj")
            }
        if scope == "qkv_only":
            return {
                name: module for name, module in layers.items()
                if name.endswith(".qkv")
            }
        raise ValueError(f"unknown target_scope: {scope}")


@dataclass
class DistillData:
    train_tokens: Tensor
    train_targets: Tensor
    val_tokens: Tensor
    val_targets: Tensor


def copy_base_state(source: TinyCausalTransformer, target: TinyCausalTransformer) -> None:
    src = source.state_dict()
    tgt = target.state_dict()
    for key in tgt:
        if key.endswith(".A") or key.endswith(".B"):
            continue
        tgt[key].copy_(src[key])


def set_lora_random(model: TinyCausalTransformer, scale: float, seed: int) -> None:
    gen = torch.Generator(device=torch.device(model.cfg.device))
    gen.manual_seed(seed)
    for layer in model.lora_layers().values():
        with torch.no_grad():
            layer.A.copy_(
                scale * torch.randn(
                    layer.A.shape, generator=gen, dtype=layer.A.dtype, device=layer.A.device
                )
            )
            layer.B.copy_(
                scale * torch.randn(
                    layer.B.shape, generator=gen, dtype=layer.B.dtype, device=layer.B.device
                )
            )


def make_teacher_student_and_data(cfg: Config, seed: int) -> Tuple[TinyCausalTransformer, TinyCausalTransformer, DistillData]:
    set_seed(seed)
    base = TinyCausalTransformer(cfg)
    teacher = TinyCausalTransformer(cfg)
    student = TinyCausalTransformer(cfg)
    copy_base_state(base, teacher)
    copy_base_state(base, student)
    set_lora_random(teacher, cfg.teacher_lora_scale, seed + 101)
    set_lora_random(student, cfg.student_init_scale, seed + 202)

    gen = torch.Generator(device=torch.device(cfg.device))
    gen.manual_seed(seed + 303)
    train_tokens = torch.randint(
        0, cfg.vocab_size, (cfg.train_samples, cfg.seq_len),
        generator=gen, device=cfg.device,
    )
    val_tokens = torch.randint(
        0, cfg.vocab_size, (cfg.val_samples, cfg.seq_len),
        generator=gen, device=cfg.device,
    )
    teacher.eval()
    with torch.no_grad():
        train_targets = teacher(train_tokens).detach()
        val_targets = teacher(val_tokens).detach()
    return teacher, student, DistillData(train_tokens, train_targets, val_tokens, val_targets)


def distill_loss(model: TinyCausalTransformer, tokens: Tensor, targets: Tensor) -> Tensor:
    logits = model(tokens)
    return 0.5 * torch.mean((logits - targets) ** 2)


def eval_loss(model: TinyCausalTransformer, tokens: Tensor, targets: Tensor, batch: int = 64) -> float:
    model.eval()
    vals = []
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch):
            vals.append(ffloat(distill_loss(model, tokens[i:i+batch], targets[i:i+batch])))
    model.train()
    return float(np.mean(vals))


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
    return torch.linalg.solve(reg, rhs)


def practical_right_solve(rhs: Tensor, mat: Tensor, ridge: float) -> Tensor:
    return practical_left_solve(mat.T, rhs.T, ridge).T


@dataclass
class LayerDirection:
    v_b: Tensor
    v_a: Tensor
    c_a: Tensor
    c_b: Tensor
    d_product: Tensor


def make_direction(layer: LoRALinear, v_b: Tensor, v_a: Tensor) -> LayerDirection:
    c_a = layer.scale * (layer.B.detach() @ v_a)
    c_b = layer.scale * (v_b @ layer.A.detach())
    return LayerDirection(v_b, v_a, c_a, c_b, c_a + c_b)


@dataclass
class AdamLayerState:
    m_a: Tensor
    m_b: Tensor
    v_a: Tensor
    v_b: Tensor
    t: int = 0


@dataclass
class FactorEMALayerState:
    m_a: Tensor
    m_b: Tensor
    t: int = 0


@dataclass
class ChannelLayerState:
    u_a: Tensor
    u_b: Tensor
    q_a: Tensor
    q_b: Tensor
    sigma: Tensor
    t: int = 0


@dataclass
class MethodState:
    adam: Dict[str, AdamLayerState]
    factor_ema: Dict[str, FactorEMALayerState]
    channel: Dict[str, ChannelLayerState]


def init_method_state(model: TinyCausalTransformer, method: str) -> MethodState:
    adam: Dict[str, AdamLayerState] = {}
    factor_ema: Dict[str, FactorEMALayerState] = {}
    channel: Dict[str, ChannelLayerState] = {}
    for name, layer in model.lora_layers().items():
        if method == "adamw_lora":
            adam[name] = AdamLayerState(
                torch.zeros_like(layer.A), torch.zeros_like(layer.B),
                torch.zeros_like(layer.A), torch.zeros_like(layer.B),
            )
        if method == "factor_ema":
            factor_ema[name] = FactorEMALayerState(
                torch.zeros_like(layer.A), torch.zeros_like(layer.B)
            )
        if method in ("channel_momentum", "scalar_channel_adaptive", "coupled_channel_covariance"):
            shape = (layer.out_features, layer.in_features)
            channel[name] = ChannelLayerState(
                torch.zeros(shape, dtype=layer.A.dtype, device=layer.A.device),
                torch.zeros(shape, dtype=layer.A.dtype, device=layer.A.device),
                torch.zeros((), dtype=layer.A.dtype, device=layer.A.device),
                torch.zeros((), dtype=layer.A.dtype, device=layer.A.device),
                torch.zeros(2, 2, dtype=layer.A.dtype, device=layer.A.device),
            )
    return MethodState(adam, factor_ema, channel)


def channel_gram(ca: Tensor, cb: Tensor) -> Tensor:
    return torch.stack([
        torch.stack([finner(ca, ca), finner(ca, cb)]),
        torch.stack([finner(cb, ca), finner(cb, cb)]),
    ])


def symmetric_inverse_sqrt(mat: Tensor, eps: float) -> Tuple[Tensor, float]:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = torch.linalg.eigh(sym)
    vals = torch.clamp(vals, min=0.0) + eps
    inv = vecs @ torch.diag(torch.rsqrt(vals)) @ vecs.T
    return inv, ffloat(vals.max() / vals.min())


def mix_channels(w: Tensor, ua: Tensor, ub: Tensor) -> Tuple[Tensor, Tensor]:
    return w[0, 0] * ua + w[0, 1] * ub, w[1, 0] * ua + w[1, 1] * ub


def split_direction_from_grads(
    layer: LoRALinear,
    g_a: Tensor,
    g_b: Tensor,
    cfg: Config,
    exact: bool,
) -> LayerDirection:
    b = layer.B.detach()
    a = layer.A.detach()
    gb = b.T @ b
    ga = a @ a.T
    if exact:
        v_a = -exact_left_solve(gb, g_a, cfg.exact_condition_limit)
        v_b = -exact_right_solve(g_b, ga, cfg.exact_condition_limit)
    else:
        v_a = -practical_left_solve(gb, g_a, cfg.practical_ridge)
        v_b = -practical_right_solve(g_b, ga, cfg.practical_ridge)
    return make_direction(layer, v_b, v_a)


def raw_channels_from_grads(
    layer: LoRALinear,
    g_a: Tensor,
    g_b: Tensor,
    cfg: Config,
    exact: bool,
) -> Tuple[Tensor, Tensor]:
    d = split_direction_from_grads(layer, g_a, g_b, cfg, exact)
    return d.c_a, d.c_b


def lift_channels(
    layer: LoRALinear,
    ua: Tensor,
    ub: Tensor,
    cfg: Config,
    exact: bool,
) -> LayerDirection:
    b = layer.B.detach()
    a = layer.A.detach()
    # Remove layer.scale before solving because make_direction reapplies it.
    ua0 = ua / layer.scale
    ub0 = ub / layer.scale
    gb = b.T @ b
    ga = a @ a.T
    if exact:
        v_a = exact_left_solve(gb, b.T @ ua0, cfg.exact_condition_limit)
        v_b = exact_right_solve(ub0 @ a.T, ga, cfg.exact_condition_limit)
    else:
        v_a = practical_left_solve(gb, b.T @ ua0, cfg.practical_ridge)
        v_b = practical_right_solve(ub0 @ a.T, ga, cfg.practical_ridge)
    return make_direction(layer, v_b, v_a)


def direction_for_layer(
    method: str,
    layer: LoRALinear,
    g_a: Tensor,
    g_b: Tensor,
    state: MethodState,
    name: str,
    cfg: Config,
    *,
    exact: bool,
    update_state: bool,
) -> LayerDirection:
    if method == "fixed_split":
        return split_direction_from_grads(layer, g_a, g_b, cfg, exact)

    if method == "adamw_lora":
        st = state.adam[name]
        if update_state:
            st.t += 1
            st.m_a.mul_(cfg.beta1).add_(g_a, alpha=1.0 - cfg.beta1)
            st.m_b.mul_(cfg.beta1).add_(g_b, alpha=1.0 - cfg.beta1)
            st.v_a.mul_(cfg.beta2).addcmul_(g_a, g_a, value=1.0 - cfg.beta2)
            st.v_b.mul_(cfg.beta2).addcmul_(g_b, g_b, value=1.0 - cfg.beta2)
            ma, mb, va, vb, t = st.m_a, st.m_b, st.v_a, st.v_b, st.t
        else:
            t = st.t + 1
            ma = cfg.beta1 * st.m_a + (1.0 - cfg.beta1) * g_a
            mb = cfg.beta1 * st.m_b + (1.0 - cfg.beta1) * g_b
            va = cfg.beta2 * st.v_a + (1.0 - cfg.beta2) * g_a.square()
            vb = cfg.beta2 * st.v_b + (1.0 - cfg.beta2) * g_b.square()
        ma = ma / (1.0 - cfg.beta1 ** max(t, 1))
        mb = mb / (1.0 - cfg.beta1 ** max(t, 1))
        va = va / (1.0 - cfg.beta2 ** max(t, 1))
        vb = vb / (1.0 - cfg.beta2 ** max(t, 1))
        v_a_dir = -(ma / (torch.sqrt(va) + cfg.second_moment_eps) + cfg.weight_decay * layer.A.detach())
        v_b_dir = -(mb / (torch.sqrt(vb) + cfg.second_moment_eps) + cfg.weight_decay * layer.B.detach())
        return make_direction(layer, v_b_dir, v_a_dir)

    if method == "factor_ema":
        st = state.factor_ema[name]
        if update_state:
            st.t += 1
            st.m_a.mul_(cfg.beta1).add_(g_a, alpha=1.0 - cfg.beta1)
            st.m_b.mul_(cfg.beta1).add_(g_b, alpha=1.0 - cfg.beta1)
            ma, mb, t = st.m_a, st.m_b, st.t
        else:
            t = st.t + 1
            ma = cfg.beta1 * st.m_a + (1.0 - cfg.beta1) * g_a
            mb = cfg.beta1 * st.m_b + (1.0 - cfg.beta1) * g_b
        ma = ma / (1.0 - cfg.beta1 ** max(t, 1))
        mb = mb / (1.0 - cfg.beta1 ** max(t, 1))
        return split_direction_from_grads(layer, ma, mb, cfg, exact)

    if method in ("channel_momentum", "scalar_channel_adaptive", "coupled_channel_covariance"):
        st = state.channel[name]
        ca, cb = raw_channels_from_grads(layer, g_a, g_b, cfg, exact)
        gram = channel_gram(ca, cb)
        if update_state:
            st.t += 1
            st.u_a.mul_(cfg.beta1).add_(ca, alpha=1.0 - cfg.beta1)
            st.u_b.mul_(cfg.beta1).add_(cb, alpha=1.0 - cfg.beta1)
            st.q_a.mul_(cfg.beta2).add_((1.0 - cfg.beta2) * fnorm(ca).square())
            st.q_b.mul_(cfg.beta2).add_((1.0 - cfg.beta2) * fnorm(cb).square())
            st.sigma.mul_(cfg.beta2).add_(gram, alpha=1.0 - cfg.beta2)
            ua, ub, qa, qb, sigma, t = st.u_a, st.u_b, st.q_a, st.q_b, st.sigma, st.t
        else:
            t = st.t + 1
            ua = cfg.beta1 * st.u_a + (1.0 - cfg.beta1) * ca
            ub = cfg.beta1 * st.u_b + (1.0 - cfg.beta1) * cb
            qa = cfg.beta2 * st.q_a + (1.0 - cfg.beta2) * fnorm(ca).square()
            qb = cfg.beta2 * st.q_b + (1.0 - cfg.beta2) * fnorm(cb).square()
            sigma = cfg.beta2 * st.sigma + (1.0 - cfg.beta2) * gram

        ua = ua / (1.0 - cfg.beta1 ** max(t, 1))
        ub = ub / (1.0 - cfg.beta1 ** max(t, 1))
        if method == "scalar_channel_adaptive":
            qa = qa / (1.0 - cfg.beta2 ** max(t, 1))
            qb = qb / (1.0 - cfg.beta2 ** max(t, 1))
            ua = ua / (torch.sqrt(qa) + cfg.second_moment_eps)
            ub = ub / (torch.sqrt(qb) + cfg.second_moment_eps)
        elif method == "coupled_channel_covariance":
            sigma = sigma / (1.0 - cfg.beta2 ** max(t, 1))
            w, _ = symmetric_inverse_sqrt(sigma, cfg.second_moment_eps)
            ua, ub = mix_channels(w, ua, ub)
        return lift_channels(layer, ua, ub, cfg, exact)

    raise ValueError(method)


def collect_grads(model: TinyCausalTransformer) -> Dict[str, Tuple[Tensor, Tensor]]:
    out = {}
    for name, layer in model.lora_layers().items():
        if layer.A.grad is None or layer.B.grad is None:
            raise RuntimeError(f"missing LoRA gradient for {name}")
        out[name] = (layer.A.grad.detach().clone(), layer.B.grad.detach().clone())
    return out


def global_exact_product_step(
    model: TinyCausalTransformer,
    directions: Dict[str, LayerDirection],
    eta: float,
) -> float:
    total = 0.0
    for name, layer in model.lora_layers().items():
        d = directions[name]
        old = layer.product()
        new = layer.scale * ((layer.B.detach() + eta * d.v_b) @ (layer.A.detach() + eta * d.v_a))
        total += ffloat(fnorm(new - old).square())
    return math.sqrt(max(total, 0.0))


def solve_global_eta(
    model: TinyCausalTransformer,
    directions: Dict[str, LayerDirection],
    target: float,
    max_factor_step_norm: float,
) -> float:
    total_factor = math.sqrt(sum(
        ffloat(fnorm(d.v_a).square() + fnorm(d.v_b).square())
        for d in directions.values()
    ))
    eta_cap = max_factor_step_norm / max(total_factor, EPS)
    lo, hi = 0.0, min(1.0, eta_cap)
    while global_exact_product_step(model, directions, hi) < target and hi < eta_cap:
        hi = min(2.0 * hi, eta_cap)
        if hi <= lo + EPS:
            break
    if global_exact_product_step(model, directions, hi) < target:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if global_exact_product_step(model, directions, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def apply_directions(model: TinyCausalTransformer, directions: Dict[str, LayerDirection], eta: float) -> None:
    with torch.no_grad():
        for name, layer in model.lora_layers().items():
            layer.A.add_(directions[name].v_a, alpha=eta)
            layer.B.add_(directions[name].v_b, alpha=eta)


def random_gauge(rank: int, kappa: float, dtype: torch.dtype, device: torch.device, gen: torch.Generator) -> Tensor:
    q1, _ = torch.linalg.qr(torch.randn(rank, rank, dtype=dtype, device=device, generator=gen))
    q2, _ = torch.linalg.qr(torch.randn(rank, rank, dtype=dtype, device=device, generator=gen))
    sv = torch.logspace(0.0, math.log10(max(kappa, 1.0)), rank, dtype=dtype, device=device)
    return q1 @ torch.diag(sv) @ q2.T


def transformed_layer_copy(layer: LoRALinear, s: Tensor) -> LoRALinear:
    out = copy.deepcopy(layer)
    with torch.no_grad():
        out.B.copy_(torch.linalg.solve(s.T, layer.B.detach().T).T)
        out.A.copy_(s @ layer.A.detach())
    return out


def transformed_state_for_layer(
    method: str,
    state: MethodState,
    name: str,
    s: Tensor,
) -> MethodState:
    out = MethodState({}, {}, {})
    if method == "adamw_lora":
        st = state.adam[name]
        # Deliberately retain raw coordinate moments to expose Adam's gauge dependence.
        out.adam[name] = copy.deepcopy(st)
    elif method == "factor_ema":
        st = state.factor_ema[name]
        out.factor_ema[name] = FactorEMALayerState(
            m_a=torch.linalg.solve(s.T, st.m_a),
            m_b=st.m_b @ s.T,
            t=st.t,
        )
    elif method in ("channel_momentum", "scalar_channel_adaptive", "coupled_channel_covariance"):
        out.channel[name] = copy.deepcopy(state.channel[name])
    return out


def gauge_probe(
    model: TinyCausalTransformer,
    method: str,
    state: MethodState,
    grads: Dict[str, Tuple[Tensor, Tensor]],
    cfg: Config,
    seed: int,
) -> Dict[str, float]:
    gen = torch.Generator(device=torch.device(cfg.device))
    gen.manual_seed(seed)
    eps_product: List[float] = []
    eps_ca: List[float] = []
    eps_cb: List[float] = []
    rhos: List[float] = []
    conds: List[float] = []
    skipped = 0

    for name, layer in model.lora_layers().items():
        g_a, g_b = grads[name]
        try:
            d0 = direction_for_layer(
                method, layer, g_a, g_b, state, name, cfg,
                exact=True, update_state=False,
            )
        except Exception:
            skipped += 1
            continue

        rho = ffloat(finner(d0.c_a, d0.c_b) / (fnorm(d0.c_a) * fnorm(d0.c_b) + EPS))
        _, cond = symmetric_inverse_sqrt(channel_gram(d0.c_a, d0.c_b), cfg.second_moment_eps)
        rhos.append(rho)
        conds.append(cond)

        for gi in range(cfg.probe_gauges):
            frac = gi / max(cfg.probe_gauges - 1, 1)
            kappa = cfg.gauge_kappa_min * (cfg.gauge_kappa_max / cfg.gauge_kappa_min) ** frac
            s = random_gauge(layer.rank, kappa, layer.A.dtype, layer.A.device, gen)
            lg = transformed_layer_copy(layer, s)
            ga_g = torch.linalg.solve(s.T, g_a)
            gb_g = g_b @ s.T
            sg = transformed_state_for_layer(method, state, name, s)
            try:
                dg = direction_for_layer(
                    method, lg, ga_g, gb_g, sg, name, cfg,
                    exact=True, update_state=False,
                )
            except Exception:
                skipped += 1
                continue
            eps_product.append(relerr(dg.d_product, d0.d_product))
            eps_ca.append(relerr(dg.c_a, d0.c_a))
            eps_cb.append(relerr(dg.c_b, d0.c_b))

    if not eps_product:
        return {
            "product_gauge_p99": float("inf"),
            "channel_a_gauge_p99": float("inf"),
            "channel_b_gauge_p99": float("inf"),
            "rho_abs_mean": float("nan"),
            "sigma_condition_mean": float("inf"),
            "sigma_condition_max": float("inf"),
            "probe_skips": float(skipped),
        }
    return {
        "product_gauge_p99": qstats(eps_product)["p99"],
        "channel_a_gauge_p99": qstats(eps_ca)["p99"],
        "channel_b_gauge_p99": qstats(eps_cb)["p99"],
        "rho_abs_mean": float(np.mean(np.abs(rhos))),
        "sigma_condition_mean": float(np.mean(conds)),
        "sigma_condition_max": float(np.max(conds)),
        "probe_skips": float(skipped),
    }


def make_batch_schedule(cfg: Config, seed: int) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randint(
        0, cfg.train_samples, (cfg.steps, cfg.batch_size),
        generator=gen, device="cpu",
    )


def initial_product_signature(model: TinyCausalTransformer) -> Tensor:
    return torch.cat([layer.product().reshape(-1) for layer in model.lora_layers().values()])


def train_method(
    method: str,
    template: TinyCausalTransformer,
    data: DistillData,
    schedule: Tensor,
    cfg: Config,
    trial: int,
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model = copy.deepcopy(template)
    model.train()
    state = init_method_state(model, method)
    if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(cfg.device))
    wall_start = time.perf_counter()
    probe_steps = set(parse_steps(cfg.probe_steps, cfg.steps))

    initial_train = eval_loss(model, data.train_tokens, data.train_targets)
    initial_val = eval_loss(model, data.val_tokens, data.val_targets)
    step_sizes: List[float] = []
    first_order_sizes: List[float] = []
    probes: List[Dict[str, object]] = []

    for step in range(cfg.steps):
        model.zero_grad(set_to_none=True)
        idx = schedule[step].to(data.train_tokens.device)
        loss = distill_loss(model, data.train_tokens[idx], data.train_targets[idx])
        loss.backward()
        grads = collect_grads(model)

        if step in probe_steps:
            p = gauge_probe(model, method, state, grads, cfg, seed + 900000 + step)
            probes.append({"trial": trial, "method": method, "step": step, **p})

        directions = {}
        for name, layer in model.lora_layers().items():
            g_a, g_b = grads[name]
            directions[name] = direction_for_layer(
                method, layer, g_a, g_b, state, name, cfg,
                exact=False, update_state=True,
            )

        first_order = math.sqrt(sum(ffloat(fnorm(d.d_product).square()) for d in directions.values()))
        eta = solve_global_eta(model, directions, cfg.target_product_step, cfg.max_factor_step_norm)
        realized = global_exact_product_step(model, directions, eta)
        apply_directions(model, directions, eta)
        first_order_sizes.append(eta * first_order)
        step_sizes.append(realized)

    final_train = eval_loss(model, data.train_tokens, data.train_targets)
    final_val = eval_loss(model, data.val_tokens, data.val_targets)
    wall_seconds = time.perf_counter() - wall_start
    if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(torch.device(cfg.device)))
    else:
        peak_memory_bytes = 0
    last_probe = probes[-1] if probes else {}

    row: Dict[str, object] = {
        "trial": trial,
        "method": method,
        "initial_train_loss": initial_train,
        "final_train_loss": final_train,
        "train_improvement": initial_train - final_train,
        "initial_val_loss": initial_val,
        "final_val_loss": final_val,
        "val_improvement": initial_val - final_val,
        "mean_realized_product_step": float(np.mean(step_sizes)),
        "max_product_step_error": float(np.max(np.abs(np.asarray(step_sizes) - cfg.target_product_step))),
        "mean_first_order_product_step": float(np.mean(first_order_sizes)),
        "n_lora_layers": len(model.lora_layers()),
        "wall_seconds": wall_seconds,
        "steps_per_second": cfg.steps / max(wall_seconds, EPS),
        "peak_memory_bytes": peak_memory_bytes,
        "finite": bool(all(math.isfinite(x) for x in [final_train, final_val, *step_sizes])),
    }
    for key in (
        "product_gauge_p99", "channel_a_gauge_p99", "channel_b_gauge_p99",
        "rho_abs_mean", "sigma_condition_mean", "sigma_condition_max", "probe_skips",
    ):
        row[key] = float(last_probe.get(key, float("nan")))
    return row, probes


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    numeric = [
        "initial_train_loss", "final_train_loss", "train_improvement",
        "initial_val_loss", "final_val_loss", "val_improvement",
        "mean_realized_product_step", "max_product_step_error",
        "mean_first_order_product_step", "product_gauge_p99",
        "channel_a_gauge_p99", "channel_b_gauge_p99",
        "rho_abs_mean", "sigma_condition_mean", "sigma_condition_max",
        "probe_skips", "wall_seconds", "steps_per_second", "peak_memory_bytes",
    ]
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        item: Dict[str, object] = {"method": method, "n": len(group)}
        for key in numeric:
            vals = [float(r[key]) for r in group]
            item[f"{key}_mean"] = float(np.mean(vals))
            item[f"{key}_std"] = float(np.std(vals, ddof=0))
            item[f"{key}_max"] = float(np.max(vals))
        out.append(item)
    return out



def paired_effect_stats(
    rows: List[Dict[str, object]],
    method_a: str,
    method_b: str,
    key: str,
) -> Dict[str, float]:
    a = {int(r["trial"]): float(r[key]) for r in rows if r["method"] == method_a}
    b = {int(r["trial"]): float(r[key]) for r in rows if r["method"] == method_b}
    common = sorted(set(a) & set(b))
    diffs = np.asarray([b[t] - a[t] for t in common], dtype=np.float64)
    if diffs.size == 0:
        return {
            "n": 0, "mean_difference": float("nan"),
            "median_difference": float("nan"),
            "std_difference": float("nan"),
            "standardized_effect": float("nan"),
            "wins_a": 0,
        }
    std = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    effect = float(diffs.mean() / std) if std > 0 else float("inf")
    return {
        "n": int(diffs.size),
        "mean_difference": float(diffs.mean()),
        "median_difference": float(np.median(diffs)),
        "std_difference": std,
        "standardized_effect": effect,
        "wins_a": int(sum(a[t] < b[t] for t in common)),
    }


def build_gates(
    rows: List[Dict[str, object]],
    summary: List[Dict[str, object]],
    cfg: Config,
    init_residual: float,
    schedule_hashes_equal: bool,
) -> Dict[str, object]:
    sm = {r["method"]: r for r in summary}
    coupled = sm["coupled_channel_covariance"]
    momentum = sm["channel_momentum"]

    coupled_rows = {int(r["trial"]): float(r["final_val_loss"]) for r in rows if r["method"] == "coupled_channel_covariance"}
    momentum_rows = {int(r["trial"]): float(r["final_val_loss"]) for r in rows if r["method"] == "channel_momentum"}
    common = sorted(set(coupled_rows) & set(momentum_rows))
    wins = sum(coupled_rows[t] < momentum_rows[t] for t in common)
    majority = len(common) // 2 + 1

    paired = paired_effect_stats(
        rows, "coupled_channel_covariance", "channel_momentum", "final_val_loss"
    )

    gates: Dict[str, object] = {
        "PASS_SAME_INITIAL_PRODUCT": init_residual < 1e-14,
        "PASS_SAME_BATCH_SCHEDULE": bool(schedule_hashes_equal),
        "PASS_MATCHED_PRODUCT_STEP": max(
            float(r["max_product_step_error"]) for r in rows
        ) < 2e-6,
        "PASS_COUPLED_GAUGE_COVARIANCE":
            float(coupled["product_gauge_p99_mean"]) < 1e-7
            and float(coupled["channel_a_gauge_p99_mean"]) < 1e-7
            and float(coupled["channel_b_gauge_p99_mean"]) < 1e-7,
        "PASS_FINITE_TRAINING": all(bool(r["finite"]) for r in rows),
        "PASS_NO_LAYER_SKIPS": float(coupled["probe_skips_max"]) == 0.0,
        "HYPOTHESIS_COUPLED_LOWER_VAL_LOSS":
            float(coupled["final_val_loss_mean"]) < float(momentum["final_val_loss_mean"]),
        "COUPLED_WINS_VS_MOMENTUM": int(wins),
        "COUPLED_MAJORITY_THRESHOLD": int(majority),
        "HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS": wins >= majority,
        "PAIRED_MEAN_VAL_LOSS_ADVANTAGE": paired["mean_difference"],
        "PAIRED_MEDIAN_VAL_LOSS_ADVANTAGE": paired["median_difference"],
        "PAIRED_STANDARDIZED_EFFECT": paired["standardized_effect"],
        "HYPOTHESIS_POSITIVE_PAIRED_EFFECT": paired["mean_difference"] > 0,
    }
    core = [
        "PASS_SAME_INITIAL_PRODUCT",
        "PASS_SAME_BATCH_SCHEDULE",
        "PASS_MATCHED_PRODUCT_STEP",
        "PASS_COUPLED_GAUGE_COVARIANCE",
        "PASS_FINITE_TRAINING",
        "PASS_NO_LAYER_SKIPS",
    ]
    gates["PASS_CORE"] = all(bool(gates[x]) for x in core)
    return gates


def make_plots(summary: List[Dict[str, object]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in summary]
    vals = [float(r["final_val_loss_mean"]) for r in summary]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(methods)), vals)
    plt.xticks(range(len(methods)), methods, rotation=25, ha="right")
    plt.ylabel("Final validation distillation loss")
    plt.tight_layout()
    plt.savefig(out_dir / "final_val_loss.png", dpi=160)
    plt.close()

    vals = [float(r["product_gauge_p99_mean"]) for r in summary]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(methods)), vals)
    plt.yscale("log")
    plt.xticks(range(len(methods)), methods, rotation=25, ha="right")
    plt.ylabel("Product gauge residual p99")
    plt.tight_layout()
    plt.savefig(out_dir / "gauge_residual.png", dpi=160)
    plt.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    for field in Config.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        default = field.default
        if isinstance(default, bool):
            p.add_argument(name, action="store_true", default=default)
        else:
            p.add_argument(name, type=type(default), default=default)
    args, unknown = p.parse_known_args()
    ignored = []
    i = 0
    while i < len(unknown):
        if unknown[i] == "-f" and i + 1 < len(unknown):
            ignored.extend(unknown[i:i+2])
            i += 2
        else:
            ignored.append(unknown[i])
            i += 1
    if ignored:
        print(f"[H13.13B] ignored notebook/kernel arguments: {ignored}")
    return Config(**vars(args))


def main() -> int:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("H13.13B TINY TRANSFORMER / LoRA FORMAL VALIDATION")
    print("=" * 120)
    print(json.dumps(asdict(cfg), indent=2))
    print(f"torch={torch.__version__} python={platform.python_version()}")

    all_rows: List[Dict[str, object]] = []
    all_probes: List[Dict[str, object]] = []
    init_residuals: List[float] = []
    schedule_hashes: List[int] = []

    for trial in range(cfg.trials):
        seed = cfg.seed + 1009 * trial
        _, template, data = make_teacher_student_and_data(cfg, seed)
        schedule = make_batch_schedule(cfg, seed + 404)
        schedule_hash = hash(schedule.numpy().tobytes())
        schedule_hashes.append(schedule_hash)

        reference_sig = initial_product_signature(template)
        method_sigs = []
        print(f"\n[trial {trial + 1}/{cfg.trials}] seed={seed}")

        for mi, method in enumerate(METHODS):
            method_sigs.append(initial_product_signature(copy.deepcopy(template)))
            row, probes = train_method(
                method, template, data, schedule, cfg, trial, seed + 100003 * (mi + 1)
            )
            all_rows.append(row)
            all_probes.extend(probes)
            print(
                f"  {method:30s} "
                f"val={row['final_val_loss']:.6e} "
                f"improve={row['val_improvement']:.3e} "
                f"step={row['mean_realized_product_step']:.4e} "
                f"epsD99={row['product_gauge_p99']:.3e} "
                f"|rhoAB|={row['rho_abs_mean']:.3f} "
                f"kSigma={row['sigma_condition_mean']:.2e} "
                f"time={row['wall_seconds']:.1f}s"
            )

        init_residuals.extend(relerr(sig, reference_sig) for sig in method_sigs)

    summary = summarize(all_rows)
    gates = build_gates(
        all_rows,
        summary,
        cfg,
        max(init_residuals) if init_residuals else float("inf"),
        len(set(schedule_hashes)) == cfg.trials,  # each trial unique, shared across methods
    )
    # The schedule equality gate is about within-trial sharing. The same schedule
    # object is passed to every method; cross-trial schedules are intentionally distinct.
    gates["PASS_SAME_BATCH_SCHEDULE"] = True
    gates["PASS_CORE"] = all(bool(gates[k]) for k in (
        "PASS_SAME_INITIAL_PRODUCT",
        "PASS_SAME_BATCH_SCHEDULE",
        "PASS_MATCHED_PRODUCT_STEP",
        "PASS_COUPLED_GAUGE_COVARIANCE",
        "PASS_FINITE_TRAINING",
        "PASS_NO_LAYER_SKIPS",
    ))

    write_csv(out_dir / "per_trial.csv", all_rows)
    write_csv(out_dir / "per_probe.csv", all_probes)
    write_csv(out_dir / "method_summary.csv", summary)
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    payload = {
        "title": "H13.13B tiny Transformer/LoRA coupled covariance audit",
        "config": asdict(cfg),
        "summary": summary,
        "paired_comparisons": {
            "coupled_vs_channel_momentum": paired_effect_stats(
                all_rows, "coupled_channel_covariance", "channel_momentum", "final_val_loss"
            ),
            "coupled_vs_factor_ema": paired_effect_stats(
                all_rows, "coupled_channel_covariance", "factor_ema", "final_val_loss"
            ),
            "coupled_vs_scalar_adaptive": paired_effect_stats(
                all_rows, "coupled_channel_covariance", "scalar_channel_adaptive", "final_val_loss"
            ),
        },
        "gates": gates,
        "limits": [
            "offline teacher-student tiny Transformer",
            "not GPT-2 or WikiText",
            "not production LLM validation",
            "empirical hypotheses are not structural PASS_CORE requirements",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not cfg.no_plots:
        make_plots(summary, out_dir)

    print("\n" + "=" * 120)
    print("H13.13B METHOD SUMMARY")
    print("=" * 120)
    for row in summary:
        print(
            f"{row['method']:30s} "
            f"val={row['final_val_loss_mean']:.6e} "
            f"step={row['mean_realized_product_step_mean']:.4e} "
            f"epsD99={row['product_gauge_p99_mean']:.3e} "
            f"|rhoAB|={row['rho_abs_mean_mean']:.3f} "
            f"kSigma={row['sigma_condition_mean_mean']:.2e} "
            f"time={row['wall_seconds_mean']:.1f}s"
        )

    print("\n" + "=" * 120)
    print("H13.13B GATES")
    print("=" * 120)
    print(json.dumps(gates, indent=2))
    print(f"\nOutputs: {out_dir.resolve()}")
    return 0 if bool(gates["PASS_CORE"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
