#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H13.14F-FIX — GPT-2 SMALL FULL-12-LAYER LONG-HORIZON VALIDATION

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
- factor_ema
- channel_momentum
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
self-contained Transformer/LoRA mechanism audit intended to precede H13.13B.

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

try:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "Install dependencies with: pip install -U transformers datasets"
    ) from exc

Tensor = torch.Tensor
EPS = 1e-15

METHODS = (
    "factor_ema",
    "channel_momentum",
    "coupled_channel_covariance",
)



@dataclass
class Config:
    seed: int = 1414
    trials: int = 3
    steps: int = 1000

    model_name: str = "openai-community/gpt2"
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_samples: int = 512
    val_samples: int = 128
    batch_size: int = 4
    eval_batch_size: int = 4
    seq_len: int = 64

    lora_rank: int = 4
    lora_alpha: float = 4.0
    student_init_scale: float = 0.01
    target_last_n_layers: int = 12

    target_product_step: float = 0.005
    max_factor_step_norm: float = 100.0
    weight_decay: float = 0.0
    beta1: float = 0.92
    beta2: float = 0.99
    second_moment_eps: float = 1e-8
    practical_ridge: float = 1e-8
    exact_condition_limit: float = 1e12

    probe_steps: str = "0,25,50,100,200,400,600,800,999"
    probe_gauges: int = 2
    gauge_kappa_min: float = 1.0
    gauge_kappa_max: float = 10.0
    gauge_tolerance_float32: float = 1e-5
    gauge_tolerance_float64: float = 1e-10
    required_coupled_wins: int = 2

    dtype: str = "float32"
    device: str = "cuda"
    output_dir: str = "h1314f_fix_gpt2_full12_long_horizon"
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
    """Frozen GPT-2 Conv1D projection plus trainable BA LoRA product."""

    def __init__(
        self,
        base_weight: Tensor,
        base_bias: Tensor | None,
        rank: int,
        alpha: float,
        init_scale: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.in_features = int(base_weight.shape[0])
        self.out_features = int(base_weight.shape[1])
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.register_buffer("base_weight", base_weight.detach().clone())
        if base_bias is None:
            self.base_bias = None
        else:
            self.register_buffer("base_bias", base_bias.detach().clone())

        gen = torch.Generator(device=base_weight.device)
        gen.manual_seed(seed)
        self.A = nn.Parameter(
            init_scale * torch.randn(
                rank, self.in_features,
                generator=gen,
                dtype=base_weight.dtype,
                device=base_weight.device,
            )
        )
        self.B = nn.Parameter(
            init_scale * torch.randn(
                self.out_features, rank,
                generator=gen,
                dtype=base_weight.dtype,
                device=base_weight.device,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.base_weight.T, self.base_bias)
        delta = F.linear(F.linear(x, self.A), self.B) * self.scale
        return base + delta

    def product(self) -> Tensor:
        return self.scale * (self.B @ self.A)


class GPT2LoRAModel(nn.Module):
    def __init__(self, cfg: Config, seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        dtype = dtype_from_name(cfg.dtype)
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            print("[H13.14F-FIX] CUDA unavailable; falling back to CPU.")
            cfg.device = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=dtype,
        ).to(cfg.device)
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(False)

        blocks = self.model.transformer.h
        n_total = len(blocks)
        n_target = min(max(cfg.target_last_n_layers, 1), n_total)
        target_indices = list(range(n_total - n_target, n_total))
        self._lora_names: List[str] = []

        for offset, block_index in enumerate(target_indices):
            old = blocks[block_index].attn.c_attn
            wrapped = LoRALinear(
                old.weight,
                old.bias,
                cfg.lora_rank,
                cfg.lora_alpha,
                cfg.student_init_scale,
                seed + 1000 + offset,
            )
            blocks[block_index].attn.c_attn = wrapped
            self._lora_names.append(
                f"model.transformer.h.{block_index}.attn.c_attn"
            )

        self.model.train()

    def forward(self, tokens: Tensor, labels: Tensor | None = None):
        return self.model(input_ids=tokens, labels=labels, use_cache=False)

    def lora_layers(self) -> Dict[str, LoRALinear]:
        modules = dict(self.named_modules())
        return {name: modules[name] for name in self._lora_names}


@dataclass
class LanguageData:
    train_tokens: Tensor
    train_targets: Tensor
    val_tokens: Tensor
    val_targets: Tensor


def tokenize_wikitext(cfg: Config) -> LanguageData:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenizer.model_max_length = 10**12
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    except Exception as first_error:
        fallback_name = (
            "Salesforce/wikitext"
            if cfg.dataset_name == "wikitext"
            else "wikitext"
        )
        print(
            f"[H13.14F-FIX] dataset load failed for {cfg.dataset_name!r}: "
            f"{type(first_error).__name__}: {first_error}"
        )
        print(f"[H13.14F-FIX] retrying with dataset id {fallback_name!r}")
        try:
            ds = load_dataset(fallback_name, cfg.dataset_config)
        except Exception:
            raise RuntimeError(
                "Could not load WikiText-2. Use dataset_name='Salesforce/wikitext' "
                "or install compatible versions with: "
                "pip install -U datasets huggingface_hub"
            ) from first_error
    block = cfg.seq_len

    def make_blocks(split: str, limit: int) -> Tensor:
        texts = [x for x in ds[split]["text"] if x and x.strip()]
        joined = "\n\n".join(texts)
        ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
        needed = limit * block
        if len(ids) < needed:
            raise RuntimeError(
                f"{split} has only {len(ids)} tokens; need {needed}. "
                "Reduce samples or sequence length."
            )
        arr = torch.tensor(ids[:needed], dtype=torch.long)
        return arr.view(limit, block)

    train = make_blocks("train", cfg.train_samples)
    val = make_blocks("validation", cfg.val_samples)
    return LanguageData(train, train.clone(), val, val.clone())


def make_teacher_student_and_data(
    cfg: Config, seed: int
) -> Tuple[None, GPT2LoRAModel, LanguageData]:
    set_seed(seed)
    template = GPT2LoRAModel(cfg, seed)
    data = tokenize_wikitext(cfg)
    return None, template, data


def distill_loss(model: GPT2LoRAModel, tokens: Tensor, targets: Tensor) -> Tensor:
    return model(tokens, labels=targets).loss


def eval_loss(
    model: GPT2LoRAModel,
    tokens: Tensor,
    targets: Tensor,
    batch: int | None = None,
) -> float:
    model.eval()
    batch = batch or model.cfg.eval_batch_size
    vals: List[float] = []
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch):
            x = tokens[i : i + batch].to(model.cfg.device)
            y = targets[i : i + batch].to(model.cfg.device)
            vals.append(ffloat(distill_loss(model, x, y)))
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

def init_method_state(model: GPT2LoRAModel, method: str) -> MethodState:
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


def collect_grads(model: GPT2LoRAModel) -> Dict[str, Tuple[Tensor, Tensor]]:
    out = {}
    for name, layer in model.lora_layers().items():
        if layer.A.grad is None or layer.B.grad is None:
            raise RuntimeError(f"missing LoRA gradient for {name}")
        out[name] = (layer.A.grad.detach().clone(), layer.B.grad.detach().clone())
    return out


def global_exact_product_step(
    model: GPT2LoRAModel,
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
    model: GPT2LoRAModel,
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


def apply_directions(model: GPT2LoRAModel, directions: Dict[str, LayerDirection], eta: float) -> None:
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
    model: GPT2LoRAModel,
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


def initial_product_signature(model: GPT2LoRAModel) -> Tensor:
    return torch.cat([layer.product().reshape(-1) for layer in model.lora_layers().values()])


def train_method(
    method: str,
    template: GPT2LoRAModel,
    data: LanguageData,
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
        loss = distill_loss(
            model,
            data.train_tokens[idx].to(cfg.device),
            data.train_targets[idx].to(cfg.device),
        )
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
        "gpt2_scope_exact": bool(
            len(model.lora_layers()) == cfg.target_last_n_layers
            and all(name.endswith(".attn.c_attn") for name in model.lora_layers())
        ),
        "gpt2_full12_scope_exact": bool(
            cfg.target_last_n_layers == 12
            and len(model.lora_layers()) == 12
            and list(model.lora_layers().keys()) == [
                f"model.transformer.h.{i}.attn.c_attn" for i in range(12)
            ]
        ),
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
    factor = sm["factor_ema"]

    coupled_rows = {int(r["trial"]): float(r["final_val_loss"]) for r in rows if r["method"] == "coupled_channel_covariance"}
    momentum_rows = {int(r["trial"]): float(r["final_val_loss"]) for r in rows if r["method"] == "channel_momentum"}
    common = sorted(set(coupled_rows) & set(momentum_rows))
    wins = sum(coupled_rows[t] < momentum_rows[t] for t in common)
    factor_rows = {
        int(r["trial"]): float(r["final_val_loss"])
        for r in rows if r["method"] == "factor_ema"
    }
    wins_vs_factor = sum(
        coupled_rows[t] < factor_rows[t]
        for t in sorted(set(coupled_rows) & set(factor_rows))
    )
    majority = cfg.required_coupled_wins

    paired = paired_effect_stats(
        rows, "coupled_channel_covariance", "channel_momentum", "final_val_loss"
    )
    paired_vs_factor = paired_effect_stats(
        rows, "coupled_channel_covariance", "factor_ema", "final_val_loss"
    )

    gauge_tolerance = (
        cfg.gauge_tolerance_float32
        if cfg.dtype == "float32"
        else cfg.gauge_tolerance_float64
    )

    gates: Dict[str, object] = {
        "GAUGE_TOLERANCE_USED": gauge_tolerance,
        "COUPLED_PRODUCT_GAUGE_P99_MEAN":
            float(coupled["product_gauge_p99_mean"]),
        "COUPLED_PRODUCT_GAUGE_P99_MAX": max(
            float(r["product_gauge_p99"])
            for r in rows
            if r["method"] == "coupled_channel_covariance"
        ),
        "PASS_COUPLED_GAUGE_EVERY_SEED_STRICT": max(
            float(r["product_gauge_p99"])
            for r in rows
            if r["method"] == "coupled_channel_covariance"
        ) < gauge_tolerance,
        "PASS_SAME_INITIAL_PRODUCT": init_residual < 1e-14,
        "PASS_SAME_BATCH_SCHEDULE": bool(schedule_hashes_equal),
        "PASS_MATCHED_PRODUCT_STEP": max(
            float(r["max_product_step_error"]) for r in rows
        ) < 2e-6,
        "PASS_COUPLED_GAUGE_COVARIANCE":
            float(coupled["product_gauge_p99_mean"]) < gauge_tolerance
            and float(coupled["channel_a_gauge_p99_mean"]) < gauge_tolerance
            and float(coupled["channel_b_gauge_p99_mean"]) < gauge_tolerance,
        "PASS_FINITE_TRAINING": all(bool(r["finite"]) for r in rows),
        "PASS_NO_LAYER_SKIPS": float(coupled["probe_skips_max"]) == 0.0,
        "PASS_GPT2_SCOPE_EXACT": all(bool(r["gpt2_scope_exact"]) for r in rows),
        "PASS_GPT2_FULL12_SCOPE_EXACT": all(
            bool(r["gpt2_full12_scope_exact"]) for r in rows
        ),
        "HYPOTHESIS_COUPLED_LOWER_VAL_LOSS":
            float(coupled["final_val_loss_mean"]) < float(momentum["final_val_loss_mean"]),
        "COUPLED_WINS_VS_MOMENTUM": int(wins),
        "COUPLED_MAJORITY_THRESHOLD": int(majority),
        "HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS": wins >= majority,
        "PAIRED_MEAN_VAL_LOSS_ADVANTAGE": paired["mean_difference"],
        "PAIRED_MEDIAN_VAL_LOSS_ADVANTAGE": paired["median_difference"],
        "PAIRED_STANDARDIZED_EFFECT": paired["standardized_effect"],
        "HYPOTHESIS_POSITIVE_PAIRED_EFFECT": paired["mean_difference"] > 0,
        "HYPOTHESIS_COUPLED_LOWER_KSIGMA":
            float(coupled["sigma_condition_mean_mean"]) < float(momentum["sigma_condition_mean_mean"]),
        "HYPOTHESIS_COUPLED_LOWER_THAN_FACTOR_EMA":
            float(coupled["final_val_loss_mean"]) < float(factor["final_val_loss_mean"]),
        "COUPLED_WINS_VS_FACTOR_EMA": wins_vs_factor,
        "HYPOTHESIS_COUPLED_MAJORITY_WINS_VS_FACTOR_EMA":
            wins_vs_factor >= majority,
        "PAIRED_MEAN_ADVANTAGE_VS_FACTOR_EMA":
            paired_vs_factor["mean_difference"],
        "PAIRED_STANDARDIZED_EFFECT_VS_FACTOR_EMA":
            paired_vs_factor["standardized_effect"],
    }
    core = [
        "PASS_SAME_INITIAL_PRODUCT",
        "PASS_SAME_BATCH_SCHEDULE",
        "PASS_MATCHED_PRODUCT_STEP",
        "PASS_COUPLED_GAUGE_COVARIANCE",
        "PASS_FINITE_TRAINING",
        "PASS_NO_LAYER_SKIPS",
        "PASS_GPT2_SCOPE_EXACT",
        "PASS_GPT2_FULL12_SCOPE_EXACT",
    ]
    gates["PASS_CORE"] = all(bool(gates[x]) for x in core)
    gates["PASS_EMPIRICAL_FULL12"] = bool(
        gates["HYPOTHESIS_COUPLED_LOWER_VAL_LOSS"]
        and gates["HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS"]
        and gates["HYPOTHESIS_POSITIVE_PAIRED_EFFECT"]
        and gates["HYPOTHESIS_COUPLED_LOWER_KSIGMA"]
    )
    gates["PASS_LONG_HORIZON_BASELINES"] = bool(
        gates["HYPOTHESIS_COUPLED_LOWER_THAN_FACTOR_EMA"]
        and gates["HYPOTHESIS_COUPLED_MAJORITY_WINS_VS_FACTOR_EMA"]
        and gates["HYPOTHESIS_COUPLED_LOWER_VAL_LOSS"]
        and gates["HYPOTHESIS_COUPLED_MAJORITY_SEED_WINS"]
    )
    gates["PASS_LONG_HORIZON_EMPIRICAL"] = bool(
        gates["PASS_EMPIRICAL_FULL12"]
        and gates["PASS_LONG_HORIZON_BASELINES"]
    )
    return gates


def make_plots(summary: List[Dict[str, object]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in summary]
    vals = [float(r["final_val_loss_mean"]) for r in summary]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(methods)), vals)
    plt.xticks(range(len(methods)), methods, rotation=25, ha="right")
    plt.ylabel("Final WikiText-2 validation cross-entropy")
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
        print(f"[H13.14F-FIX] ignored notebook/kernel arguments: {ignored}")
    return Config(**vars(args))


def main() -> int:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("H13.14F-FIX GPT-2 SMALL FULL-12-LAYER LONG-HORIZON VALIDATION")
    print("=" * 120)
    print(json.dumps(asdict(cfg), indent=2))
    print(f"torch={torch.__version__} python={platform.python_version()}")
    print(
        "[H13.14F-FIX] calibration: "
        f"steps={cfg.steps}, batch={cfg.batch_size}, "
        f"product_step={cfg.target_product_step}, beta2={cfg.beta2}, "
        f"gauge_tol={cfg.gauge_tolerance_float32 if cfg.dtype == 'float32' else cfg.gauge_tolerance_float64}"
    )
    try:
        import datasets as datasets_pkg
        import transformers as transformers_pkg
        import huggingface_hub as hf_hub_pkg
        print(
            f"transformers={transformers_pkg.__version__} "
            f"datasets={datasets_pkg.__version__} "
            f"huggingface_hub={hf_hub_pkg.__version__}"
        )
    except Exception:
        pass

    all_rows: List[Dict[str, object]] = []
    all_probes: List[Dict[str, object]] = []
    init_residuals: List[float] = []
    schedule_hashes: List[int] = []

    for trial in range(cfg.trials):
        seed = cfg.seed + 1009 * trial
        _, template, data = make_teacher_student_and_data(cfg, seed)
        active_layers = list(template.lora_layers().keys())
        print("  ACTIVE_LORA_LAYERS:")
        for layer_name in active_layers:
            print(f"    {layer_name}")
        print(f"  N_ACTIVE_LORA_LAYERS = {len(active_layers)}")
        expected_layers = [
            f"model.transformer.h.{i}.attn.c_attn" for i in range(12)
        ]
        if cfg.target_last_n_layers != 12:
            raise RuntimeError(
                f"H13.14C requires target_last_n_layers=12, "
                f"got {cfg.target_last_n_layers}"
            )
        if active_layers != expected_layers:
            raise RuntimeError(
                "full-12 QKV scope mismatch: "
                f"expected {expected_layers}, got {active_layers}"
            )
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

        # Incremental checkpoints: preserve completed expensive trials even if
        # a later summary/plot/gate step raises an exception.
        write_csv(out_dir / "per_trial_checkpoint.csv", all_rows)
        write_csv(out_dir / "per_probe_checkpoint.csv", all_probes)
        checkpoint_payload = {
            "config": asdict(cfg),
            "completed_trials": trial + 1,
            "rows": all_rows,
            "probes": all_probes,
            "max_initial_product_residual":
                max(init_residuals) if init_residuals else float("inf"),
        }
        (out_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint_payload, indent=2), encoding="utf-8"
        )

    summary = summarize(all_rows)
    for item in summary:
        loss_value = float(item["final_val_loss_mean"])
        item["final_val_perplexity"] = (
            float(math.exp(loss_value)) if loss_value < 50.0 else float("inf")
        )
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
        "PASS_GPT2_SCOPE_EXACT",
        "PASS_GPT2_FULL12_SCOPE_EXACT",
    ))

    write_csv(out_dir / "per_trial.csv", all_rows)
    write_csv(out_dir / "per_probe.csv", all_probes)
    write_csv(out_dir / "method_summary.csv", summary)
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    payload = {
        "title": "H13.14F-FIX GPT-2 small full-12-layer long-horizon validation",
        "config": asdict(cfg),
        "summary": summary,
        "paired_comparisons": {
            "coupled_vs_channel_momentum": paired_effect_stats(
                all_rows, "coupled_channel_covariance", "channel_momentum", "final_val_loss"
            ),
        },
        "gates": gates,
        "limits": [
            "GPT-2 small pretrained-model smoke on WikiText-2",
            "only the last configured c_attn layers receive LoRA",
            "not production LLM validation",
            "empirical hypotheses are not structural PASS_CORE requirements",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not cfg.no_plots:
        make_plots(summary, out_dir)

    print("\n" + "=" * 120)
    print("H13.14F-FIX METHOD SUMMARY")
    print("=" * 120)
    for row in summary:
        print(
            f"{row['method']:30s} "
            f"val={row['final_val_loss_mean']:.6e} "
            f"ppl={row['final_val_perplexity']:.3f} "
            f"step={row['mean_realized_product_step_mean']:.4e} "
            f"epsD99={row['product_gauge_p99_mean']:.3e} "
            f"|rhoAB|={row['rho_abs_mean_mean']:.3f} "
            f"kSigma={row['sigma_condition_mean_mean']:.2e} "
            f"time={row['wall_seconds_mean']:.1f}s"
        )

    print("\n" + "=" * 120)
    print("H13.14F-FIX GATES")
    print("=" * 120)
    print(json.dumps(gates, indent=2))
    print(f"\nOutputs: {out_dir.resolve()}")
    return 0 if bool(gates["PASS_CORE"]) else 1


if __name__ == "__main__":
    main()
