"""D7 fixed-rank tangent benchmark.

This experiment is a scientific regression benchmark, not a production
optimizer implementation. It compares factor-space Adam, ambient product-space
Adam, fixed-rank tangent SGD, fixed-rank tangent Adam, and held-out
trust-calibrated tangent Adam on a small synthetic Transformer task with
gauge-equivalent LoRA representations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometric_flow import (
    FixedRankFunctionalAdam,
    FixedRankManifold,
    HeldOutTrustRegion,
    ProductParameter,
    ProductState,
)


MODES = [
    "factor_adam",
    "explicit_product_adam",
    "rank_tangent_sgd",
    "rank_tangent_adam",
    "rank_tangent_trust",
]


@dataclass
class Config:
    seeds: list[int]
    representations: int
    steps: int
    vocab_size: int
    seq_len: int
    train_samples: int
    calib_samples: int
    eval_samples: int
    grad_batch_size: int
    fixed_calib_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    lora_rank: int
    factor_lr: float
    product_adam_lr: float
    rank_tangent_sgd_lr: float
    rank_tangent_adam_lr: float
    max_factor_update_norm: float
    max_product_step_norm: float
    trust_scale_grid: tuple[float, ...]
    armijo_relative_decrease: float
    gauge_condition_max: float
    out_dir: Path
    device: torch.device


@dataclass
class DatasetBundle:
    train_x: torch.Tensor
    train_y: torch.Tensor
    calib_x: torch.Tensor
    calib_y: torch.Tensor
    eval_x: torch.Tensor
    eval_y: torch.Tensor


@dataclass
class RunResult:
    seed: int
    representation: int
    mode: str
    initial_loss: float
    final_loss: float
    final_accuracy: float
    mean_applied_dm_norm: float
    mean_selected_scale: float
    fraction_scale_zero: float
    fraction_scale_max: float
    acceptance_rate: float
    mean_tangent_residual: float
    max_tangent_residual: float
    mean_rank_violation: float
    mean_retraction_relative_error: float
    mean_gradient_batch_loss_change: float
    mean_fixed_calibration_loss_change: float
    elapsed_seconds: float
    final_logits: str
    final_products: str


class LoRALinear(nn.Module):
    """LoRA factor mode or explicit product-state mode."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.A = nn.Parameter(0.04 * torch.randn(rank, in_features))
        self.B = nn.Parameter(0.04 * torch.randn(out_features, rank))
        self.M_state = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.use_explicit_product = False
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.M_state if self.use_explicit_product else self.B @ self.A
        return F.linear(x, self.weight + update)

    @torch.no_grad()
    def apply_gauge(self, transform: torch.Tensor) -> None:
        if self.use_explicit_product:
            raise RuntimeError("apply gauge before explicit-product activation")
        self.A.copy_(transform @ self.A)
        self.B.copy_(self.B @ torch.linalg.inv(transform))

    @torch.no_grad()
    def activate_explicit_product(self) -> None:
        self.M_state.copy_(self.B @ self.A)
        self.M_state.requires_grad_(True)
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)
        self.use_explicit_product = True

    def current_product(self) -> torch.Tensor:
        return self.M_state if self.use_explicit_product else self.B @ self.A


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rank: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = LoRALinear(d_model, d_model, rank)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = LoRALinear(d_model, d_model, rank)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        for module in [self.k_proj, self.out_proj]:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        attention = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        output = attention @ v
        output = output.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rank: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rank)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff_in = LoRALinear(d_model, d_ff, rank)
        self.ff_out = LoRALinear(d_ff, d_model, rank)
        for layer_norm in [self.ln1, self.ln2]:
            for parameter in layer_norm.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ff_out(F.gelu(self.ff_in(self.ln2(x))))


class SmallLoRATransformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Parameter(0.02 * torch.randn(cfg.seq_len, cfg.d_model), requires_grad=False)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.lora_rank) for _ in range(cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.token_embedding.weight.requires_grad = False
        self.lm_head.weight.requires_grad = False
        for parameter in self.final_ln.parameters():
            parameter.requires_grad = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq = tokens.shape[1]
        x = self.token_embedding(tokens) + self.position_embedding[:seq]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_ln(x))

    def lora_modules(self):
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                yield name, module

    @torch.no_grad()
    def activate_explicit_products(self) -> None:
        for _, module in self.lora_modules():
            module.activate_explicit_product()

    def factor_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        for _, module in self.lora_modules():
            params.extend([module.A, module.B])
        return [param for param in params if param.requires_grad]


class TensorAdam:
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_number = 0
        self.m: dict[object, torch.Tensor] = {}
        self.v: dict[object, torch.Tensor] = {}

    def begin_step(self) -> None:
        self.step_number += 1

    @torch.no_grad()
    def update(self, key, gradient: torch.Tensor) -> torch.Tensor:
        if key not in self.m:
            self.m[key] = torch.zeros_like(gradient)
            self.v[key] = torch.zeros_like(gradient)
        self.m[key].mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
        self.v[key].mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
        m_hat = self.m[key] / (1.0 - self.beta1**self.step_number)
        v_hat = self.v[key] / (1.0 - self.beta2**self.step_number)
        return -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


class ScalarEMA:
    def __init__(self, beta2: float = 0.98, eps: float = 1e-8) -> None:
        self.beta2 = beta2
        self.eps = eps
        self.step_number = 0
        self.state: dict[object, torch.Tensor] = {}

    def begin_step(self) -> None:
        self.step_number += 1

    @torch.no_grad()
    def normalize(self, key, direction: torch.Tensor, lr: float) -> torch.Tensor:
        squared_norm = direction.pow(2).sum()
        if key not in self.state:
            self.state[key] = torch.zeros_like(squared_norm)
        self.state[key].mul_(self.beta2).add_(squared_norm, alpha=1.0 - self.beta2)
        corrected = self.state[key] / (1.0 - self.beta2**self.step_number)
        return -lr * direction / (torch.sqrt(corrected) + self.eps)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_well_conditioned_matrix(rank: int, generator: torch.Generator, condition_max: float) -> torch.Tensor:
    for _ in range(100):
        raw = torch.randn(rank, rank, generator=generator)
        q, _ = torch.linalg.qr(raw)
        scales = torch.exp(torch.empty(rank).uniform_(-0.55, 0.55, generator=generator))
        transform = q @ torch.diag(scales)
        if torch.linalg.cond(transform).item() <= condition_max:
            return transform
    raise RuntimeError("could not generate a well-conditioned gauge matrix")


def make_gauge_copy(base_model: SmallLoRATransformer, seed: int, representation: int, cfg: Config):
    model = copy.deepcopy(base_model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 10_000 + representation * 101 + 17)
    with torch.no_grad():
        for _, module in model.lora_modules():
            transform = random_well_conditioned_matrix(module.rank, generator, cfg.gauge_condition_max).to(
                device=module.A.device,
                dtype=module.A.dtype,
            )
            module.apply_gauge(transform)
    return model


@torch.no_grad()
def make_teacher_task(base_model: SmallLoRATransformer, seed: int, cfg: Config) -> DatasetBundle:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 50_000)
    teacher = copy.deepcopy(base_model).to(cfg.device)
    for _, module in teacher.lora_modules():
        module.A.add_(0.20 * torch.randn(module.A.shape, generator=generator).to(module.A))
        module.B.add_(0.20 * torch.randn(module.B.shape, generator=generator).to(module.B))
    train_x = torch.randint(0, cfg.vocab_size, (cfg.train_samples, cfg.seq_len), generator=generator)
    calib_x = torch.randint(0, cfg.vocab_size, (cfg.calib_samples, cfg.seq_len), generator=generator)
    eval_x = torch.randint(0, cfg.vocab_size, (cfg.eval_samples, cfg.seq_len), generator=generator)
    train_y = teacher(train_x.to(cfg.device)).argmax(dim=-1).cpu()
    calib_y = teacher(calib_x.to(cfg.device)).argmax(dim=-1).cpu()
    eval_y = teacher(eval_x.to(cfg.device)).argmax(dim=-1).cpu()
    return DatasetBundle(train_x, train_y, calib_x, calib_y, eval_x, eval_y)


def product_state_for(model: SmallLoRATransformer) -> ProductState:
    return ProductState([ProductParameter(name, module.M_state, module.rank) for name, module in model.lora_modules()])


def clip_tensor(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = tensor.norm()
    if float(norm.detach().cpu()) <= max_norm:
        return tensor
    return tensor * (max_norm / norm.clamp_min(torch.finfo(tensor.dtype).tiny))


def clip_direction_dict(direction: dict, max_norm: float) -> dict:
    first = next(iter(direction.values()))
    norm = torch.sqrt(sum(value.pow(2).sum() for value in direction.values()).to(device=first.device, dtype=first.dtype))
    scale = min(1.0, max_norm / max(float(norm.detach().cpu()), 1e-30))
    return {key: scale * value for key, value in direction.items()}


@torch.no_grad()
def apply_factor_direction(model: SmallLoRATransformer, direction: dict) -> None:
    for name, module in model.lora_modules():
        module.A.add_(direction[(name, "A")])
        module.B.add_(direction[(name, "B")])


@torch.no_grad()
def apply_ambient_product_direction(model: SmallLoRATransformer, direction: dict) -> tuple[list[float], list[float]]:
    residuals = []
    retraction_errors = []
    for name, module in model.lora_modules():
        manifold = FixedRankManifold(module.rank)
        before = module.M_state.detach().clone()
        delta = direction[name]
        residuals.append(manifold.tangent_residual(module.M_state.detach(), delta))
        new_matrix, diag = manifold.retract(module.M_state.detach(), delta)
        module.M_state.copy_(new_matrix)
        retraction_errors.append(diag.retraction_relative_error)
        assert before.shape == module.M_state.shape
    return residuals, retraction_errors


@torch.no_grad()
def evaluate_batch_loss(model: SmallLoRATransformer, x_batch: torch.Tensor, y_batch: torch.Tensor, cfg: Config) -> float:
    logits = model(x_batch)
    return float(F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y_batch.reshape(-1)).item())


@torch.no_grad()
def evaluate(model: SmallLoRATransformer, x: torch.Tensor, y: torch.Tensor, cfg: Config) -> dict:
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
    accuracy = (logits.argmax(dim=-1) == y).float().mean()
    products = [module.current_product().detach().cpu().reshape(-1) for _, module in model.lora_modules()]
    product_vector = torch.cat(products).numpy().tolist()
    model.train()
    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy.item()),
        "logits": logits.detach().cpu().reshape(-1).numpy().tolist(),
        "products": product_vector,
    }


def encoded(values: list[float]) -> str:
    return json.dumps(values, separators=(",", ":"))


def normalized_distance(first: list[float], second: list[float]) -> float:
    if not first:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)) / len(first))


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def train_one(
    model: SmallLoRATransformer,
    data: DatasetBundle,
    grad_batches: list[torch.Tensor],
    fixed_calib_x: torch.Tensor,
    fixed_calib_y: torch.Tensor,
    seed: int,
    representation: int,
    mode: str,
    cfg: Config,
) -> RunResult:
    model = model.to(cfg.device)
    model.train()
    if mode != "factor_adam":
        model.activate_explicit_products()
    eval_x, eval_y = data.eval_x.to(cfg.device), data.eval_y.to(cfg.device)
    fixed_calib_x, fixed_calib_y = fixed_calib_x.to(cfg.device), fixed_calib_y.to(cfg.device)
    initial = evaluate(model, eval_x, eval_y, cfg)
    factor_optimizer = TensorAdam(cfg.factor_lr)
    product_adam = TensorAdam(cfg.product_adam_lr)
    tangent_scalar = ScalarEMA()
    product_state = product_state_for(model) if mode in {"rank_tangent_adam", "rank_tangent_trust"} else None
    trust = (
        HeldOutTrustRegion(cfg.trust_scale_grid, cfg.armijo_relative_decrease)
        if mode == "rank_tangent_trust"
        else None
    )
    tangent_optimizer = (
        FixedRankFunctionalAdam(
            product_state,
            lr=cfg.rank_tangent_adam_lr,
            max_update_norm=cfg.max_product_step_norm,
            trust_region=trust,
        )
        if product_state is not None
        else None
    )
    applied_norms: list[float] = []
    selected_scales: list[float] = []
    zero_flags: list[bool] = []
    max_flags: list[bool] = []
    accepted_flags: list[bool] = []
    tangent_residuals: list[float] = []
    rank_violations: list[float] = []
    retraction_errors: list[float] = []
    gradient_changes: list[float] = []
    calibration_changes: list[float] = []
    started = time.perf_counter()

    for step in range(cfg.steps):
        indices = grad_batches[step]
        grad_x = data.train_x[indices].to(cfg.device)
        grad_y = data.train_y[indices].to(cfg.device)
        model.zero_grad(set_to_none=True)
        logits = model(grad_x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), grad_y.reshape(-1))
        pre_gradient_loss = float(loss.item())
        pre_calibration_loss = evaluate_batch_loss(model, fixed_calib_x, fixed_calib_y, cfg)
        selected_scale = 1.0
        accepted = True
        hit_max = False
        before_products = {name: module.current_product().detach().clone() for name, module in model.lora_modules()}

        if mode == "factor_adam":
            loss.backward()
            factor_optimizer.begin_step()
            direction = {}
            for name, module in model.lora_modules():
                direction[(name, "A")] = factor_optimizer.update((name, "A"), module.A.grad.detach())
                direction[(name, "B")] = factor_optimizer.update((name, "B"), module.B.grad.detach())
            apply_factor_direction(model, clip_direction_dict(direction, cfg.max_factor_update_norm))
            step_tangent_residuals = [float("nan")]
            step_retraction_errors = [float("nan")]
        elif mode == "explicit_product_adam":
            loss.backward()
            product_adam.begin_step()
            direction = {}
            for name, module in model.lora_modules():
                direction[name] = clip_tensor(product_adam.update(name, module.M_state.grad.detach()), cfg.max_product_step_norm)
            step_tangent_residuals, step_retraction_errors = apply_ambient_product_direction(model, direction)
        elif mode == "rank_tangent_sgd":
            loss.backward()
            tangent_scalar.begin_step()
            direction = {}
            step_tangent_residuals = []
            step_retraction_errors = []
            for name, module in model.lora_modules():
                manifold = FixedRankManifold(module.rank)
                tangent_gradient = manifold.project_tangent(module.M_state.detach(), module.M_state.grad.detach())
                delta = clip_tensor(tangent_scalar.normalize(name, tangent_gradient, cfg.rank_tangent_sgd_lr), cfg.max_product_step_norm)
                step_tangent_residuals.append(manifold.tangent_residual(module.M_state.detach(), delta))
                new_matrix, diag = manifold.retract(module.M_state.detach(), delta)
                with torch.no_grad():
                    module.M_state.copy_(new_matrix)
                step_retraction_errors.append(diag.retraction_relative_error)
        elif mode in {"rank_tangent_adam", "rank_tangent_trust"}:
            loss.backward()

            def calibration_closure() -> torch.Tensor:
                return F.cross_entropy(model(fixed_calib_x).reshape(-1, cfg.vocab_size), fixed_calib_y.reshape(-1))

            tangent_optimizer.step(calibration_closure=calibration_closure if mode == "rank_tangent_trust" else None)
            aggregate = tangent_optimizer.last_diagnostics.get("aggregate", {})
            selected_scale = float(aggregate.get("mean_selected_scale", 1.0))
            accepted = bool(aggregate.get("mean_accepted", 1.0))
            hit_max = bool(aggregate.get("mean_hit_max_scale", 0.0))
            step_tangent_residuals = [
                float(entry["tangent_residual"])
                for entry in tangent_optimizer.last_diagnostics["products"].values()
            ]
            step_retraction_errors = [
                float(entry["retraction_relative_error"])
                for entry in tangent_optimizer.last_diagnostics["products"].values()
            ]
        else:
            raise ValueError(f"unknown mode: {mode}")

        after_products = {name: module.current_product().detach().clone() for name, module in model.lora_modules()}
        applied_norms.append(finite_mean([(after_products[name] - before_products[name]).norm().item() for name in before_products]))
        selected_scales.append(selected_scale)
        zero_flags.append(selected_scale == 0.0)
        max_flags.append(hit_max)
        accepted_flags.append(accepted)
        tangent_residuals.append(finite_mean(step_tangent_residuals))
        rank_violations.append(
            statistics.fmean(
                float(FixedRankManifold(module.rank).numerical_rank(module.current_product().detach()) > module.rank)
                for _, module in model.lora_modules()
            )
        )
        retraction_errors.append(finite_mean(step_retraction_errors))
        post_gradient_loss = evaluate_batch_loss(model, grad_x, grad_y, cfg)
        post_calibration_loss = evaluate_batch_loss(model, fixed_calib_x, fixed_calib_y, cfg)
        gradient_changes.append(post_gradient_loss - pre_gradient_loss)
        calibration_changes.append(post_calibration_loss - pre_calibration_loss)

    final = evaluate(model, eval_x, eval_y, cfg)
    return RunResult(
        seed=seed,
        representation=representation,
        mode=mode,
        initial_loss=initial["loss"],
        final_loss=final["loss"],
        final_accuracy=final["accuracy"],
        mean_applied_dm_norm=finite_mean(applied_norms),
        mean_selected_scale=finite_mean(selected_scales),
        fraction_scale_zero=statistics.fmean(zero_flags),
        fraction_scale_max=statistics.fmean(max_flags),
        acceptance_rate=statistics.fmean(accepted_flags),
        mean_tangent_residual=finite_mean(tangent_residuals),
        max_tangent_residual=max([v for v in tangent_residuals if math.isfinite(v)], default=float("nan")),
        mean_rank_violation=finite_mean(rank_violations),
        mean_retraction_relative_error=finite_mean(retraction_errors),
        mean_gradient_batch_loss_change=finite_mean(gradient_changes),
        mean_fixed_calibration_loss_change=finite_mean(calibration_changes),
        elapsed_seconds=time.perf_counter() - started,
        final_logits=encoded(final["logits"]),
        final_products=encoded(final["products"]),
    )


def bootstrap_mean_ci(values: list[float], seed: int = 12345, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(0, len(values), (samples, len(values)), generator=generator)
    means = tensor[indices].mean(dim=1)
    return float(torch.quantile(means, 0.025).item()), float(torch.quantile(means, 0.975).item())


def within_seed_distance(rows: list[dict], seed: int, mode: str, column: str) -> float:
    vectors = [json.loads(row[column]) for row in rows if int(row["seed"]) == seed and row["mode"] == mode]
    distances = [
        normalized_distance(vectors[first], vectors[second])
        for first, second in itertools.combinations(range(len(vectors)), 2)
    ]
    return finite_mean(distances)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows: list[dict], cfg: Config) -> tuple[list[dict], list[dict], dict, str]:
    seed_rows = []
    for seed in cfg.seeds:
        for mode in MODES:
            selected = [row for row in rows if int(row["seed"]) == seed and row["mode"] == mode]
            seed_rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "logit_sensitivity": within_seed_distance(rows, seed, mode, "final_logits"),
                    "product_sensitivity": within_seed_distance(rows, seed, mode, "final_products"),
                    "mean_loss": finite_mean([float(row["final_loss"]) for row in selected]),
                    "mean_accuracy": finite_mean([float(row["final_accuracy"]) for row in selected]),
                    "mean_applied_dm_norm": finite_mean([float(row["mean_applied_dm_norm"]) for row in selected]),
                    "mean_selected_scale": finite_mean([float(row["mean_selected_scale"]) for row in selected]),
                    "fraction_scale_zero": finite_mean([float(row["fraction_scale_zero"]) for row in selected]),
                    "fraction_scale_max": finite_mean([float(row["fraction_scale_max"]) for row in selected]),
                    "acceptance_rate": finite_mean([float(row["acceptance_rate"]) for row in selected]),
                    "mean_tangent_residual": finite_mean([float(row["mean_tangent_residual"]) for row in selected]),
                    "max_tangent_residual": max([float(row["max_tangent_residual"]) for row in selected], default=float("nan")),
                    "mean_rank_violation": finite_mean([float(row["mean_rank_violation"]) for row in selected]),
                    "mean_retraction_relative_error": finite_mean(
                        [float(row["mean_retraction_relative_error"]) for row in selected]
                    ),
                    "mean_gradient_batch_loss_change": finite_mean(
                        [float(row["mean_gradient_batch_loss_change"]) for row in selected]
                    ),
                    "mean_fixed_calibration_loss_change": finite_mean(
                        [float(row["mean_fixed_calibration_loss_change"]) for row in selected]
                    ),
                    "mean_seconds": finite_mean([float(row["elapsed_seconds"]) for row in selected]),
                }
            )

    factor_logit = {row["seed"]: row["logit_sensitivity"] for row in seed_rows if row["mode"] == "factor_adam"}
    factor_product = {row["seed"]: row["product_sensitivity"] for row in seed_rows if row["mode"] == "factor_adam"}
    factor_loss = {row["seed"]: row["mean_loss"] for row in seed_rows if row["mode"] == "factor_adam"}
    summary_rows = []
    for mode in MODES:
        selected = [row for row in seed_rows if row["mode"] == mode]
        logit_ratios = [row["logit_sensitivity"] / max(factor_logit[row["seed"]], 1e-30) for row in selected]
        product_ratios = [row["product_sensitivity"] / max(factor_product[row["seed"]], 1e-30) for row in selected]
        loss_differences = [row["mean_loss"] - factor_loss[row["seed"]] for row in selected]
        logit_ci = bootstrap_mean_ci(logit_ratios)
        product_ci = bootstrap_mean_ci(product_ratios)
        loss_ci = bootstrap_mean_ci(loss_differences)
        summary_rows.append(
            {
                "mode": mode,
                "mean_loss": finite_mean([row["mean_loss"] for row in selected]),
                "mean_accuracy": finite_mean([row["mean_accuracy"] for row in selected]),
                "mean_logit_sensitivity": finite_mean([row["logit_sensitivity"] for row in selected]),
                "mean_product_sensitivity": finite_mean([row["product_sensitivity"] for row in selected]),
                "logit_ratio_vs_factor_mean": finite_mean(logit_ratios),
                "logit_ratio_ci_low": logit_ci[0],
                "logit_ratio_ci_high": logit_ci[1],
                "product_ratio_vs_factor_mean": finite_mean(product_ratios),
                "product_ratio_ci_low": product_ci[0],
                "product_ratio_ci_high": product_ci[1],
                "structural_win_rate": statistics.fmean(ratio < 1.0 for ratio in logit_ratios),
                "loss_minus_factor_mean": finite_mean(loss_differences),
                "loss_minus_factor_ci_low": loss_ci[0],
                "loss_minus_factor_ci_high": loss_ci[1],
                "task_win_rate": statistics.fmean(value < 0.0 for value in loss_differences),
                "mean_applied_dm_norm": finite_mean([row["mean_applied_dm_norm"] for row in selected]),
                "mean_selected_scale": finite_mean([row["mean_selected_scale"] for row in selected]),
                "fraction_scale_zero": finite_mean([row["fraction_scale_zero"] for row in selected]),
                "fraction_scale_max": finite_mean([row["fraction_scale_max"] for row in selected]),
                "acceptance_rate": finite_mean([row["acceptance_rate"] for row in selected]),
                "mean_tangent_residual": finite_mean([row["mean_tangent_residual"] for row in selected]),
                "max_tangent_residual": max([row["max_tangent_residual"] for row in selected], default=float("nan")),
                "mean_rank_violation": finite_mean([row["mean_rank_violation"] for row in selected]),
                "mean_retraction_relative_error": finite_mean(
                    [row["mean_retraction_relative_error"] for row in selected]
                ),
                "mean_gradient_batch_loss_change": finite_mean(
                    [row["mean_gradient_batch_loss_change"] for row in selected]
                ),
                "mean_fixed_calibration_loss_change": finite_mean(
                    [row["mean_fixed_calibration_loss_change"] for row in selected]
                ),
                "mean_seconds": finite_mean([row["mean_seconds"] for row in selected]),
            }
        )
    summary_rows.sort(key=lambda row: (row["mean_loss"], row["mean_logit_sensitivity"]))

    by_mode = {row["mode"]: row for row in summary_rows}
    product_adam = by_mode["explicit_product_adam"]
    tangent_adam = by_mode["rank_tangent_adam"]
    tangent_trust = by_mode["rank_tangent_trust"]
    gates = {
        "D7_TANGENT_RESIDUAL_PASS": bool(tangent_trust["max_tangent_residual"] < 1e-4),
        "D7_RANK_PRESERVATION_PASS": bool(tangent_trust["mean_rank_violation"] == 0.0),
        "D7_NEAR_EXACT_GAUGE_INVARIANCE": bool(tangent_trust["logit_ratio_ci_high"] < 0.01),
        "D7_TANGENT_TRUST_STRUCTURAL_PASS": bool(tangent_trust["logit_ratio_ci_high"] < 1.0),
        "D7_TANGENT_ADAM_BEATS_PRODUCT_ADAM": bool(tangent_adam["mean_loss"] < product_adam["mean_loss"]),
        "D7_TANGENT_TRUST_BEATS_PRODUCT_ADAM": bool(tangent_trust["mean_loss"] < product_adam["mean_loss"]),
        "D7_TANGENT_TRUST_TASK_PARITY_PASS": bool(tangent_trust["loss_minus_factor_ci_high"] <= 0.02),
        "D7_TANGENT_TRUST_TASK_ADVANTAGE_PASS": bool(tangent_trust["loss_minus_factor_ci_high"] < 0.0),
        "D7_TRUST_REJECTION_ACTIVE": bool(tangent_trust["fraction_scale_zero"] > 0.0),
        "D7_TRUST_ACCEPTS_SOME_STEPS": bool(tangent_trust["acceptance_rate"] > 0.0),
        "D7_TRUST_NOT_ALWAYS_MAX": bool(tangent_trust["fraction_scale_max"] < 0.95),
    }
    if (
        gates["D7_TANGENT_TRUST_STRUCTURAL_PASS"]
        and gates["D7_TANGENT_TRUST_TASK_ADVANTAGE_PASS"]
        and gates["D7_TANGENT_RESIDUAL_PASS"]
        and gates["D7_RANK_PRESERVATION_PASS"]
    ):
        verdict = "D7 STRONG SUCCESS: fixed-rank tangent optimization preserves gauge invariance and beats factor Adam."
    elif (
        gates["D7_TANGENT_TRUST_STRUCTURAL_PASS"]
        and gates["D7_TANGENT_TRUST_TASK_PARITY_PASS"]
        and gates["D7_TANGENT_RESIDUAL_PASS"]
        and gates["D7_RANK_PRESERVATION_PASS"]
    ):
        verdict = "D7 SUCCESS: fixed-rank tangent optimization preserves gauge invariance and reaches task parity."
    elif gates["D7_TANGENT_TRUST_STRUCTURAL_PASS"] and gates["D7_TANGENT_TRUST_BEATS_PRODUCT_ADAM"]:
        verdict = "D7 PARTIAL SUCCESS: fixed-rank tangent geometry improves over ambient product Adam."
    elif gates["D7_TANGENT_TRUST_STRUCTURAL_PASS"]:
        verdict = "D7 STRUCTURAL ONLY: fixed-rank tangent optimization remains gauge invariant."
    else:
        verdict = "D7 FAILED: fixed-rank tangent implementation did not preserve the expected structural result."
    return seed_rows, summary_rows, gates, verdict


def run(cfg: Config) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    for seed_index, seed in enumerate(cfg.seeds, start=1):
        print(f"\nSEED {seed} ({seed_index}/{len(cfg.seeds)})")
        seed_all(seed)
        base_model = SmallLoRATransformer(cfg)
        data = make_teacher_task(base_model, seed, cfg)
        grad_generator = torch.Generator(device="cpu")
        grad_generator.manual_seed(seed + 80_000)
        grad_batches = [
            torch.randint(0, cfg.train_samples, (cfg.grad_batch_size,), generator=grad_generator)
            for _ in range(cfg.steps)
        ]
        fixed_calib_generator = torch.Generator(device="cpu")
        fixed_calib_generator.manual_seed(seed + 90_000)
        fixed_calib_indices = torch.randperm(cfg.calib_samples, generator=fixed_calib_generator)[: cfg.fixed_calib_size]
        fixed_calib_x = data.calib_x[fixed_calib_indices]
        fixed_calib_y = data.calib_y[fixed_calib_indices]
        representation_models = [make_gauge_copy(base_model, seed, rep, cfg) for rep in range(cfg.representations)]
        with torch.no_grad():
            probe = data.eval_x[:16].to(cfg.device)
            initial_logits = [
                model.to(cfg.device)(probe).detach().cpu().reshape(-1).numpy().tolist()
                for model in representation_models
            ]
            initial_residual = max(normalized_distance(initial_logits[0], vector) for vector in initial_logits[1:])
        print(f"initial gauge-equivalence residual = {initial_residual:.3e}")
        if initial_residual > 2e-6:
            raise RuntimeError("initial gauge equivalence failed")
        for representation in range(cfg.representations):
            for mode in MODES:
                result = train_one(
                    model=copy.deepcopy(representation_models[representation]),
                    data=data,
                    grad_batches=grad_batches,
                    fixed_calib_x=fixed_calib_x,
                    fixed_calib_y=fixed_calib_y,
                    seed=seed,
                    representation=representation,
                    mode=mode,
                    cfg=cfg,
                )
                row = asdict(result)
                all_results.append(row)
                print(
                    f"{mode:<24} rep={representation} loss={result.final_loss:.6f} "
                    f"acc={result.final_accuracy:.4f} tan={result.mean_tangent_residual:.2e} "
                    f"rankv={result.mean_rank_violation:.3f} time={result.elapsed_seconds:.1f}s"
                )
                write_csv(cfg.out_dir / "d7_raw.csv", all_results)
    seed_rows, summary_rows, gates, verdict = analyze(all_results, cfg)
    write_csv(cfg.out_dir / "d7_raw.csv", all_results)
    write_csv(cfg.out_dir / "d7_per_seed.csv", seed_rows)
    write_csv(cfg.out_dir / "d7_summary.csv", summary_rows)
    report = {
        "device": str(cfg.device),
        "configuration": {
            **{key: value for key, value in asdict(cfg).items() if key not in {"device", "out_dir"}},
            "device": str(cfg.device),
            "out_dir": str(cfg.out_dir),
        },
        "gates": gates,
        "summary": summary_rows,
        "verdict": verdict,
    }
    with (cfg.out_dir / "d7_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
    print("\nSUMMARY")
    for row in summary_rows:
        print(row)
    print("\nFINAL D7 GATES")
    for key, value in gates.items():
        print(f"{key:<45} = {value}")
    print("\nVERDICT:")
    print(verdict)
    print("\nArtifacts:")
    for name in ["d7_raw.csv", "d7_per_seed.csv", "d7_summary.csv", "d7_report.json"]:
        print(cfg.out_dir / name)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="101,211,307")
    parser.add_argument("--representations", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/d7_fixed_rank"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-samples", type=int, default=384)
    parser.add_argument("--calib-samples", type=int, default=192)
    parser.add_argument("--eval-samples", type=int, default=192)
    parser.add_argument("--grad-batch-size", type=int, default=24)
    parser.add_argument("--fixed-calib-size", type=int, default=48)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        seeds=[int(item) for item in args.seeds.split(",") if item.strip()],
        representations=args.representations,
        steps=args.steps,
        vocab_size=32,
        seq_len=12,
        train_samples=args.train_samples,
        calib_samples=args.calib_samples,
        eval_samples=args.eval_samples,
        grad_batch_size=args.grad_batch_size,
        fixed_calib_size=args.fixed_calib_size,
        d_model=24,
        n_heads=4,
        n_layers=2,
        d_ff=48,
        lora_rank=3,
        factor_lr=0.025,
        product_adam_lr=0.010,
        rank_tangent_sgd_lr=0.020,
        rank_tangent_adam_lr=0.010,
        max_factor_update_norm=0.30,
        max_product_step_norm=0.20,
        trust_scale_grid=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0),
        armijo_relative_decrease=1e-5,
        gauge_condition_max=5.0,
        out_dir=args.out_dir,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    run(config_from_args(parse_args()))
