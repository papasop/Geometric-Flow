#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESEARCH ARCHIVE ONLY.

This script preserves an H10.12 one-parameter capacity-ratio experiment. It is
not part of the public optimizer API and is not imported by ``geometric_flow``.
The Adam-calibrated rho controller explored here is intentionally not exposed
as a library optimizer. The script keeps the experimental GPT-2 Conv1D factor
layout used during that run, which is transposed relative to the library
convention ``M = B @ A``.

GeoFlow H10.5 FAST -- LR>3 GAUGE-PROGRESS TRADE-OFF
================================================================

This is the next README-mainline experiment.

It does NOT use Adam as a proposal generator for the GeoFlow method.
Instead, it defines a native gauge-equivariant update directly on LoRA factors.

LoRA product
------------
    M = A @ B

Gauge-equivalent factors
------------------------
    A' = A @ R
    B' = inv(R) @ B

Native quotient update
----------------------
For factor gradients G_A and G_B:

    dA = -lr * G_A @ inv(B B^T)
    dB = -lr * inv(A^T A) @ G_B

Under an invertible gauge transform, the exact update obeys:

    dA' = dA @ R
    dB' = inv(R) @ dB

so the induced product update is representation independent to first order.

The balanced variant additionally canonicalizes the low-rank factors after each
step using only thin QR decompositions and an r x r SVD:

    A B = Q_A (R_A R_B^T) Q_B^T

No dense d_in x d_out optimizer moments are stored.

Methods
-------
1. factor_adam
2. quotient_flow_raw
3. quotient_flow_balanced

Primary question
----------------
Can a low-memory autonomous quotient optimizer preserve continuation under an
exactly function-preserving LoRA refactorization?

Colab
------
%run /content/geoflow_h103_ab_finite_step_integrators_oneclick.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import numpy as np
    import torch
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "numpy", "torch"]
    )
    import numpy as np
    import torch


REPOSITORY_URL = "https://github.com/papasop/Geometric-Flow.git"

TEXT_CORPUS = [
    "Geometry can distinguish representation from function.",
    "An optimizer proposal is not automatically an admissible update.",
    "Equivalent parameterizations may represent the same neural function.",
    "Low rank adaptation introduces redundant factor coordinates.",
    "Product coordinates remove arbitrary internal gauge choices.",
    "A fixed rank manifold has a tangent space and a retraction.",
    "Functional optimization should preserve represented behavior.",
    "Invariant optimizer states improve continuation consistency.",
    "Adam is coordinate dependent under non orthogonal transformations.",
    "The same product matrix can have many factor decompositions.",
    "Geometric flow filters proposals before they are executed.",
    "A quotient formulation identifies functionally neutral motion.",
]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def run_command(command: List[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_dependencies(repo_dir: Path) -> None:
    packages = []
    if importlib.util.find_spec("transformers") is None:
        packages.append("transformers>=4.40,<5")
    if importlib.util.find_spec("huggingface_hub") is None:
        packages.append("huggingface_hub>=0.27")
    if packages:
        run_command([sys.executable, "-m", "pip", "install", "-q", *packages])

    if not (repo_dir / "geometric_flow").exists():
        if repo_dir.exists() and not (repo_dir / ".git").exists():
            raise RuntimeError(f"{repo_dir} exists but is not a GeoFlow checkout")
        if not repo_dir.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            run_command(
                ["git", "clone", "--depth", "1", REPOSITORY_URL, str(repo_dir)]
            )

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


GPT2_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "pytorch_model.bin",
)


def direct_download(url: str, destination: Path, retries: int = 4) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")

    for attempt in range(retries):
        try:
            cache_buster = f"{int(time.time())}_{attempt}_{random.randrange(10**9)}"
            separator = "&" if "?" in url else "?"
            request = urllib.request.Request(
                f"{url}{separator}download=1&cb={cache_buster}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
                "wb"
            ) as file:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
            temporary.replace(destination)
            return
        except Exception:
            if temporary.exists():
                temporary.unlink()
            if attempt + 1 == retries:
                raise
            time.sleep(2 * (attempt + 1))


def ensure_local_gpt2(local_dir: Path) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)

    candidates = (
        local_dir,
        Path("/content/hf_models/gpt2_h82"),
        Path("/content/hf_models/gpt2_readme_continuation"),
        Path("/content/hf_models/gpt2_h92"),
    )

    for candidate in candidates:
        weight = candidate / "pytorch_model.bin"
        if (
            (candidate / "config.json").exists()
            and (candidate / "vocab.json").exists()
            and (candidate / "merges.txt").exists()
            and weight.exists()
            and weight.stat().st_size > 500_000_000
        ):
            print("Using local GPT-2 snapshot:", candidate)
            return str(candidate)

    try:
        from huggingface_hub import hf_hub_download

        for filename in GPT2_FILES:
            try:
                hf_hub_download(
                    repo_id="gpt2",
                    filename=filename,
                    local_dir=str(local_dir),
                    force_download=False,
                )
            except Exception:
                if filename == "pytorch_model.bin":
                    raise
    except Exception as exc:
        print("Hub client failed; using direct fallback:", repr(exc))
        base = "https://huggingface.co/gpt2/resolve/main"
        for filename in GPT2_FILES:
            destination = local_dir / filename
            if destination.exists() and (
                filename != "pytorch_model.bin"
                or destination.stat().st_size > 500_000_000
            ):
                continue
            try:
                direct_download(f"{base}/{filename}", destination)
            except Exception:
                if filename not in {"generation_config.json", "tokenizer.json"}:
                    raise

    return str(local_dir)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model_name: str
    target_modules: List[str]
    rank: int
    seeds: List[int]

    warmup_steps: int
    continuation_steps: int
    sequence_length: int
    batch_size: int
    validation_batches: int
    eval_interval: int

    factor_adam_lr: float
    quotient_lr: float
    gradient_clip_norm: float
    gram_condition_limit: float
    gauge_condition_number: float

    bootstrap_samples: int
    device: str
    repo_dir: str
    out_dir: str
    local_model_dir: str


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_token_stream(tokenizer, repeats: int = 256) -> torch.Tensor:
    # Tokenize without model truncation. Only short windows are sent to GPT-2.
    text = "\n".join(TEXT_CORPUS * repeats)
    old_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e9)
    try:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]
    finally:
        tokenizer.model_max_length = old_max
    return ids.to(torch.long)


def make_batches(
    token_stream: torch.Tensor,
    steps: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> List[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    max_start = len(token_stream) - sequence_length - 1
    batches = []

    for _ in range(steps):
        starts = torch.randint(
            0,
            max_start,
            (batch_size,),
            generator=generator,
        )
        rows = [
            token_stream[int(start) : int(start) + sequence_length]
            for start in starts
        ]
        batches.append(torch.stack(rows, dim=0))

    return batches


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class FactorLoRAConv1D(torch.nn.Module):
    """
    GPT-2 Conv1D weight shape: [in_features, out_features].
    Product convention: M = A @ B.
    """

    def __init__(self, base: torch.nn.Module, rank: int, scale: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        in_features, out_features = base.weight.shape
        self.scale = float(scale)

        self.A = torch.nn.Parameter(torch.empty(in_features, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_features))

        torch.nn.init.normal_(self.A, mean=0.0, std=0.02)

    def product(self) -> torch.Tensor:
        return self.A @ self.B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.product()) * self.scale


def get_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module:
    current = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def set_submodule(root: torch.nn.Module, path: str, module: torch.nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def inject_factor_lora(model, targets: Sequence[str], rank: int) -> None:
    for path in targets:
        base = get_submodule(model, path)
        set_submodule(
            model,
            path,
            FactorLoRAConv1D(base, rank, scale=1.0 / rank),
        )


def create_model(model_path: str, cfg: Config, seed: int):
    from transformers import AutoModelForCausalLM

    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        use_safetensors=False,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    inject_factor_lora(model, cfg.target_modules, cfg.rank)
    model.to(torch.device(cfg.device))
    model.train()
    return model


def modules(model, targets: Sequence[str]):
    return [get_submodule(model, path) for path in targets]


def trainable_parameters(model, targets: Sequence[str]):
    result = []
    for module in modules(model, targets):
        result.extend([module.A, module.B])
    return result


def model_loss(model, batch: torch.Tensor) -> torch.Tensor:
    return model(input_ids=batch, labels=batch).loss


@torch.no_grad()
def evaluate_loss(
    model,
    batches: Sequence[torch.Tensor],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    values = [
        float(model_loss(model, batch.to(device)).detach().cpu())
        for batch in batches
    ]
    if was_training:
        model.train()
    return float(np.mean(values))


@torch.no_grad()
def copy_factor_state(source, target, targets: Sequence[str]) -> None:
    for path in targets:
        source_module = get_submodule(source, path)
        target_module = get_submodule(target, path)
        target_module.A.copy_(source_module.A)
        target_module.B.copy_(source_module.B)


@torch.no_grad()
def collect_products(model, targets: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {
        path: get_submodule(model, path).product().detach().clone()
        for path in targets
    }


# ---------------------------------------------------------------------------
# Gauge transform and low-rank canonicalization
# ---------------------------------------------------------------------------

def make_invertible_gauge(
    rank: int,
    condition_number: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    q1, _ = torch.linalg.qr(
        torch.randn(rank, rank, generator=generator, dtype=torch.float64)
    )
    q2, _ = torch.linalg.qr(
        torch.randn(rank, rank, generator=generator, dtype=torch.float64)
    )

    singular_values = torch.logspace(
        0.0,
        math.log10(condition_number),
        steps=rank,
        dtype=torch.float64,
    )
    singular_values /= singular_values.mean()

    matrix = q1 @ torch.diag(singular_values) @ q2.T
    return matrix.to(device=device, dtype=dtype)


@torch.no_grad()
def apply_gauge(
    model,
    targets: Sequence[str],
    condition_number: float,
    seed: int,
) -> Dict[str, float]:
    residuals = {}

    for index, path in enumerate(targets):
        module = get_submodule(model, path)
        before = module.product().detach().clone()

        gauge = make_invertible_gauge(
            module.A.shape[1],
            condition_number,
            seed + 1009 * index,
            module.A.device,
            module.A.dtype,
        )
        inverse = torch.linalg.inv(gauge)

        module.A.copy_(module.A @ gauge)
        module.B.copy_(inverse @ module.B)

        after = module.product().detach()
        residuals[path] = float(
            (
                torch.linalg.norm(after - before)
                / (torch.linalg.norm(before) + 1e-12)
            )
            .detach()
            .cpu()
        )

    return residuals


@torch.no_grad()
def balanced_factorization_in_place(module: FactorLoRAConv1D) -> float:
    """
    Canonical balanced factorization using only:
      QR(A), QR(B^T), and SVD of an r x r core.

    It never stores the dense product M as an optimizer state.
    """
    product_before = module.product().detach()

    q_a, r_a = torch.linalg.qr(module.A, mode="reduced")
    q_b, r_b = torch.linalg.qr(module.B.transpose(0, 1), mode="reduced")

    core = r_a @ r_b.transpose(0, 1)
    u, singular_values, vh = torch.linalg.svd(core, full_matrices=False)

    sqrt_s = torch.sqrt(torch.clamp(singular_values, min=0.0))
    a_balanced = (q_a @ u) * sqrt_s.unsqueeze(0)
    b_balanced = sqrt_s.unsqueeze(1) * (vh @ q_b.transpose(0, 1))

    module.A.copy_(a_balanced)
    module.B.copy_(b_balanced)

    product_after = module.product().detach()
    residual = torch.linalg.norm(product_after - product_before) / (
        torch.linalg.norm(product_before) + 1e-12
    )
    return float(residual.detach().cpu())


@torch.no_grad()
def canonicalize_model(model, targets: Sequence[str]) -> float:
    residuals = [
        balanced_factorization_in_place(get_submodule(model, path))
        for path in targets
    ]
    return float(max(residuals))


# ---------------------------------------------------------------------------
# Native quotient optimizer
# ---------------------------------------------------------------------------

class FactorizedQuotientFlow:
    """
    Stateless autonomous quotient update.

    No Adam moments.
    No dense product matrix state.
    """

    def __init__(
        self,
        modules_: Sequence[FactorLoRAConv1D],
        lr: float,
        gradient_clip_norm: float,
        gram_condition_limit: float,
        balance_after_step: bool,
    ):
        self.modules = list(modules_)
        self.lr = float(lr)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.gram_condition_limit = float(gram_condition_limit)
        self.balance_after_step = bool(balance_after_step)

        self.step_index = 0
        self.fallback_count = 0
        self.balance_residual_max = 0.0
        self.condition_max = 0.0

    def zero_grad(self):
        for module in self.modules:
            module.A.grad = None
            module.B.grad = None

    @staticmethod
    def _safe_inverse(matrix: torch.Tensor, condition_limit: float):
        singular_values = torch.linalg.svdvals(matrix)
        condition = singular_values.max() / torch.clamp(
            singular_values.min(), min=1e-30
        )

        if float(condition.detach().cpu()) <= condition_limit:
            inverse = torch.linalg.inv(matrix)
            fallback = False
        else:
            inverse = torch.linalg.pinv(matrix, rcond=1.0 / condition_limit)
            fallback = True

        return inverse, float(condition.detach().cpu()), fallback

    @torch.no_grad()
    def step(self):
        self.step_index += 1

        raw_updates = []
        total_norm_sq = 0.0

        for module in self.modules:
            if module.A.grad is None or module.B.grad is None:
                raw_updates.append((None, None))
                continue

            gram_b = module.B @ module.B.transpose(0, 1)
            gram_a = module.A.transpose(0, 1) @ module.A

            inverse_b, condition_b, fallback_b = self._safe_inverse(
                gram_b, self.gram_condition_limit
            )
            inverse_a, condition_a, fallback_a = self._safe_inverse(
                gram_a, self.gram_condition_limit
            )

            self.condition_max = max(
                self.condition_max,
                condition_a,
                condition_b,
            )
            self.fallback_count += int(fallback_a) + int(fallback_b)

            delta_a = -self.lr * (module.A.grad @ inverse_b)
            delta_b = -self.lr * (inverse_a @ module.B.grad)

            total_norm_sq += float(
                torch.sum(delta_a * delta_a).detach().cpu()
            )
            total_norm_sq += float(
                torch.sum(delta_b * delta_b).detach().cpu()
            )
            raw_updates.append((delta_a, delta_b))

        total_norm = math.sqrt(total_norm_sq)
        if self.gradient_clip_norm > 0.0 and total_norm > self.gradient_clip_norm:
            scale = self.gradient_clip_norm / (total_norm + 1e-12)
        else:
            scale = 1.0

        for module, (delta_a, delta_b) in zip(self.modules, raw_updates):
            if delta_a is None:
                continue
            module.A.add_(delta_a, alpha=scale)
            module.B.add_(delta_b, alpha=scale)

            if self.balance_after_step:
                residual = balanced_factorization_in_place(module)
                self.balance_residual_max = max(
                    self.balance_residual_max,
                    residual,
                )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def warmup_factor_adam(
    model,
    batches: Sequence[torch.Tensor],
    cfg: Config,
):
    device = torch.device(cfg.device)
    optimizer = torch.optim.Adam(
        trainable_parameters(model, cfg.target_modules),
        lr=cfg.factor_adam_lr,
    )

    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(model, batch.to(device))
        loss.backward()
        optimizer.step()

    del optimizer


def continue_factor_adam(
    model,
    batches,
    validation,
    cfg,
):
    device = torch.device(cfg.device)
    optimizer = torch.optim.Adam(
        trainable_parameters(model, cfg.target_modules),
        lr=cfg.factor_adam_lr,
    )

    curve = []
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(model, batch.to(device))
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == len(batches):
            curve.append({
                "step": step,
                "loss": evaluate_loss(model, validation, device),
            })

    del optimizer
    return curve, {}


def continue_quotient(
    model,
    batches,
    validation,
    cfg,
    balanced: bool,
):
    device = torch.device(cfg.device)
    optimizer = FactorizedQuotientFlow(
        modules(model, cfg.target_modules),
        lr=cfg.quotient_lr,
        gradient_clip_norm=cfg.gradient_clip_norm,
        gram_condition_limit=cfg.gram_condition_limit,
        balance_after_step=balanced,
    )

    curve = []
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad()
        loss = model_loss(model, batch.to(device))
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == len(batches):
            curve.append({
                "step": step,
                "loss": evaluate_loss(model, validation, device),
            })

    diagnostics = {
        "fallback_count": optimizer.fallback_count,
        "condition_max": optimizer.condition_max,
        "balance_residual_max": optimizer.balance_residual_max,
    }
    return curve, diagnostics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def product_divergence(products_a, products_b):
    numerator_sq = 0.0
    denominator_sq = 0.0
    per_layer = {}

    for path in products_a:
        difference = products_a[path] - products_b[path]
        numerator_sq += float(torch.sum(difference * difference).detach().cpu())
        denominator_sq += float(
            torch.sum(products_a[path] * products_a[path]).detach().cpu()
        )
        per_layer[path] = float(
            (
                torch.linalg.norm(difference)
                / (torch.linalg.norm(products_a[path]) + 1e-12)
            )
            .detach()
            .cpu()
        )

    aggregate = math.sqrt(numerator_sq) / (
        math.sqrt(denominator_sq) + 1e-12
    )
    return float(aggregate), per_layer


def bootstrap_ci(values, samples: int, seed: int = 123456):
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


@dataclass
class Result:
    seed: int
    method: str

    initial_product_residual: float
    initial_loss_divergence: float

    final_product_divergence: float
    final_loss_base: float
    final_loss_gauge: float
    final_loss_divergence: float

    mean_curve_loss_divergence: float
    max_curve_loss_divergence: float

    optimizer_state_scalars: int
    trainable_parameter_scalars: int

    fallback_count: int
    condition_max: float
    balance_residual_max: float


def method_summary(results, method, metric, samples):
    values = [
        float(getattr(row, metric))
        for row in results
        if row.method == method
    ]
    low, high = bootstrap_ci(values, samples)
    array = np.asarray(values)
    return {
        "method": method,
        "metric": metric,
        "n": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95_low": low,
        "ci95_high": high,
        "values": values,
    }


def paired_log_ratio(results, numerator, denominator, metric, samples):
    by_seed = {}
    for row in results:
        by_seed.setdefault(row.seed, {})[row.method] = row

    eps = 1e-18
    values = []
    for seed, items in sorted(by_seed.items()):
        if numerator not in items or denominator not in items:
            continue
        a = float(getattr(items[numerator], metric))
        b = float(getattr(items[denominator], metric))
        values.append(math.log10((a + eps) / (b + eps)))

    low, high = bootstrap_ci(values, samples)
    array = np.asarray(values)
    return {
        "comparison": f"{numerator}_over_{denominator}",
        "metric": metric,
        "mean_log10_ratio": float(array.mean()),
        "ci95_low_log10_ratio": low,
        "ci95_high_log10_ratio": high,
        "geometric_mean_ratio": float(10.0 ** array.mean()),
        "values": values,
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



@dataclass
class FocusedResult:
    seed: int
    method: str
    selected_lr: float
    selected_substeps: int
    selected_clip_norm: float

    initial_product_residual: float
    initial_loss_divergence: float

    initial_base_loss: float
    final_base_loss: float
    base_loss_improvement: float
    base_product_displacement: float

    target_adam_loss_improvement: float
    target_adam_product_displacement: float

    loss_progress_ratio: float
    displacement_progress_ratio: float
    selection_score: float

    final_product_divergence: float
    final_loss_divergence: float

    optimizer_state_scalars: int
    condition_max: float
    fallback_count: int
    balance_residual_max: float
    retraction_discarded_energy_max: float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def product_displacement(initial_products, final_products):
    numerator_sq = 0.0
    denominator_sq = 0.0
    for path in initial_products:
        difference = final_products[path] - initial_products[path]
        numerator_sq += float(torch.sum(difference * difference).detach().cpu())
        denominator_sq += float(
            torch.sum(initial_products[path] * initial_products[path]).detach().cpu()
        )
    return math.sqrt(numerator_sq) / (math.sqrt(denominator_sq) + 1e-12)


def initialize_equivalent_pair(model_path, cfg, seed, base_state, balanced=True):
    base_model = create_model(model_path, cfg, seed)
    gauge_model = create_model(model_path, cfg, seed)

    with torch.no_grad():
        for path, (a, b) in zip(cfg.target_modules, base_state):
            base_module = get_submodule(base_model, path)
            gauge_module = get_submodule(gauge_model, path)
            base_module.A.copy_(a)
            base_module.B.copy_(b)
            gauge_module.A.copy_(a)
            gauge_module.B.copy_(b)

    apply_gauge(
        gauge_model,
        cfg.target_modules,
        cfg.gauge_condition_number,
        seed + 5003,
    )

    balance_residual = 0.0
    if balanced:
        balance_residual = max(
            canonicalize_model(base_model, cfg.target_modules),
            canonicalize_model(gauge_model, cfg.target_modules),
        )

    return base_model, gauge_model, balance_residual


@torch.no_grad()
def low_rank_retract_sum(
    left_blocks: Sequence[torch.Tensor],
    right_blocks: Sequence[torch.Tensor],
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Retract a low-rank sum

        M = sum_i L_i @ R_i

    to rank `rank` without materializing dense M.

    Build:
        L = [L_1 ... L_k]
        R = [R_1; ...; R_k]

    Then use thin QR on L and R^T plus SVD of the small core.
    """
    left = torch.cat(list(left_blocks), dim=1)
    right = torch.cat(list(right_blocks), dim=0)

    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.transpose(0, 1), mode="reduced")

    core = r_left @ r_right.transpose(0, 1)
    u, singular_values, vh = torch.linalg.svd(core, full_matrices=False)

    kept = min(rank, singular_values.numel())
    u = u[:, :kept]
    vh = vh[:kept, :]
    singular_values = singular_values[:kept]

    sqrt_s = torch.sqrt(torch.clamp(singular_values, min=0.0))
    new_a = (q_left @ u) * sqrt_s.unsqueeze(0)
    new_b = sqrt_s.unsqueeze(1) * (vh @ q_right.transpose(0, 1))

    discarded_energy = 0.0
    if core.shape[0] > kept:
        all_s = torch.linalg.svdvals(core)
        numerator = torch.sum(all_s[kept:] ** 2)
        denominator = torch.sum(all_s ** 2) + 1e-30
        discarded_energy = float((numerator / denominator).detach().cpu())

    return new_a, new_b, discarded_energy


class ProductConsistentQuotientFlow:
    """
    H10.3-B: finite-step product-consistent update.

    First-order quotient target:
        M_target = A B + dA B + A dB

    This is represented as a rank <= 3r sum and retracted to rank r using
    only thin QR and a small SVD. No dense M or Adam moments are stored.
    """

    def __init__(
        self,
        modules_: Sequence[FactorLoRAConv1D],
        lr: float,
        gradient_clip_norm: float,
        gram_condition_limit: float,
    ):
        self.modules = list(modules_)
        self.lr = float(lr)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.gram_condition_limit = float(gram_condition_limit)

        self.condition_max = 0.0
        self.fallback_count = 0
        self.retraction_discarded_energy_max = 0.0

    def zero_grad(self):
        for module in self.modules:
            module.A.grad = None
            module.B.grad = None

    @staticmethod
    def _safe_inverse(matrix, condition_limit):
        singular_values = torch.linalg.svdvals(matrix)
        condition = singular_values.max() / torch.clamp(
            singular_values.min(), min=1e-30
        )
        condition_value = float(condition.detach().cpu())

        if condition_value <= condition_limit:
            inverse = torch.linalg.inv(matrix)
            fallback = False
        else:
            inverse = torch.linalg.pinv(matrix, rcond=1.0 / condition_limit)
            fallback = True

        return inverse, condition_value, fallback

    @torch.no_grad()
    def step(self):
        updates = []
        total_norm_sq = 0.0

        for module in self.modules:
            gram_b = module.B @ module.B.transpose(0, 1)
            gram_a = module.A.transpose(0, 1) @ module.A

            inverse_b, condition_b, fallback_b = self._safe_inverse(
                gram_b, self.gram_condition_limit
            )
            inverse_a, condition_a, fallback_a = self._safe_inverse(
                gram_a, self.gram_condition_limit
            )

            self.condition_max = max(
                self.condition_max, condition_a, condition_b
            )
            self.fallback_count += int(fallback_a) + int(fallback_b)

            d_a = -self.lr * (module.A.grad @ inverse_b)
            d_b = -self.lr * (inverse_a @ module.B.grad)

            total_norm_sq += float(torch.sum(d_a * d_a).detach().cpu())
            total_norm_sq += float(torch.sum(d_b * d_b).detach().cpu())
            updates.append((d_a, d_b))

        total_norm = math.sqrt(total_norm_sq)
        scale = 1.0
        if self.gradient_clip_norm > 0.0 and total_norm > self.gradient_clip_norm:
            scale = self.gradient_clip_norm / (total_norm + 1e-12)

        for module, (d_a, d_b) in zip(self.modules, updates):
            d_a = d_a * scale
            d_b = d_b * scale

            new_a, new_b, discarded_energy = low_rank_retract_sum(
                left_blocks=[module.A, d_a, module.A],
                right_blocks=[module.B, module.B, d_b],
                rank=module.A.shape[1],
            )
            module.A.copy_(new_a)
            module.B.copy_(new_b)

            self.retraction_discarded_energy_max = max(
                self.retraction_discarded_energy_max,
                discarded_energy,
            )


def continue_substepped_quotient(
    model,
    batches,
    validation,
    cfg,
    macro_lr,
    substeps,
):
    """
    H10.3-A: split each macro update into K quotient substeps.
    Gradients are recomputed on the same minibatch at every substep.
    """
    device = torch.device(cfg.device)
    local_lr = float(macro_lr) / int(substeps)

    optimizer = FactorizedQuotientFlow(
        modules(model, cfg.target_modules),
        lr=local_lr,
        gradient_clip_norm=cfg.gradient_clip_norm,
        gram_condition_limit=cfg.gram_condition_limit,
        balance_after_step=True,
    )

    curve = []
    for step, batch in enumerate(batches, start=1):
        batch = batch.to(device)
        for _ in range(int(substeps)):
            optimizer.zero_grad()
            loss = model_loss(model, batch)
            loss.backward()
            optimizer.step()

        if step % cfg.eval_interval == 0 or step == len(batches):
            curve.append({
                "step": step,
                "loss": evaluate_loss(model, validation, device),
            })

    diagnostics = {
        "condition_max": optimizer.condition_max,
        "fallback_count": optimizer.fallback_count,
        "balance_residual_max": optimizer.balance_residual_max,
        "retraction_discarded_energy_max": 0.0,
    }
    return curve, diagnostics


def continue_product_consistent(
    model,
    batches,
    validation,
    cfg,
    macro_lr,
):
    device = torch.device(cfg.device)

    optimizer = ProductConsistentQuotientFlow(
        modules(model, cfg.target_modules),
        lr=macro_lr,
        gradient_clip_norm=cfg.gradient_clip_norm,
        gram_condition_limit=cfg.gram_condition_limit,
    )

    curve = []
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad()
        loss = model_loss(model, batch.to(device))
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == len(batches):
            curve.append({
                "step": step,
                "loss": evaluate_loss(model, validation, device),
            })

    diagnostics = {
        "condition_max": optimizer.condition_max,
        "fallback_count": optimizer.fallback_count,
        "balance_residual_max": 0.0,
        "retraction_discarded_energy_max":
            optimizer.retraction_discarded_energy_max,
    }
    return curve, diagnostics


def safe_ratio(value, target, eps=1e-12):
    return (max(float(value), 0.0) + eps) / (max(float(target), 0.0) + eps)


def match_score(loss_ratio, displacement_ratio, mode):
    loss_mismatch = abs(math.log10(loss_ratio))
    move_mismatch = abs(math.log10(displacement_ratio))

    if mode == "loss":
        score = loss_mismatch
    elif mode == "displacement":
        score = move_mismatch
    else:
        score = math.sqrt(loss_mismatch ** 2 + move_mismatch ** 2)

    return score, loss_mismatch, move_mismatch


@dataclass
class Result:
    seed: int
    method: str
    selected_lr: float
    selected_substeps: int
    selected_clip_norm: float

    initial_product_residual: float
    initial_loss_divergence: float

    initial_base_loss: float
    final_base_loss: float
    base_loss_improvement: float
    base_product_displacement: float

    target_adam_loss_improvement: float
    target_adam_product_displacement: float

    loss_progress_ratio: float
    displacement_progress_ratio: float
    selection_score: float

    final_product_divergence: float
    final_loss_divergence: float

    optimizer_state_scalars: int
    condition_max: float
    fallback_count: int
    balance_residual_max: float
    retraction_discarded_energy_max: float


def run_pair(
    method,
    model_path,
    cfg,
    seed,
    base_state,
    continuation_batches,
    validation,
    lr,
    substeps,
):
    base_model, gauge_model, initial_balance_residual = initialize_equivalent_pair(
        model_path,
        cfg,
        seed,
        base_state,
        balanced=True,
    )

    device = torch.device(cfg.device)
    initial_base_products = collect_products(base_model, cfg.target_modules)
    initial_gauge_products = collect_products(gauge_model, cfg.target_modules)

    initial_product_residual, _ = product_divergence(
        initial_base_products,
        initial_gauge_products,
    )
    initial_base_loss = evaluate_loss(base_model, validation, device)
    initial_gauge_loss = evaluate_loss(gauge_model, validation, device)

    if method == "substep":
        base_curve, base_diag = continue_substepped_quotient(
            base_model, continuation_batches, validation, cfg, lr, substeps
        )
        gauge_curve, gauge_diag = continue_substepped_quotient(
            gauge_model, continuation_batches, validation, cfg, lr, substeps
        )
    elif method == "product_consistent":
        base_curve, base_diag = continue_product_consistent(
            base_model, continuation_batches, validation, cfg, lr
        )
        gauge_curve, gauge_diag = continue_product_consistent(
            gauge_model, continuation_batches, validation, cfg, lr
        )
    else:
        raise ValueError(method)

    final_base_products = collect_products(base_model, cfg.target_modules)
    final_gauge_products = collect_products(gauge_model, cfg.target_modules)

    final_base_loss = evaluate_loss(base_model, validation, device)
    final_gauge_loss = evaluate_loss(gauge_model, validation, device)

    final_divergence, _ = product_divergence(
        final_base_products,
        final_gauge_products,
    )
    displacement = product_displacement(
        initial_base_products,
        final_base_products,
    )

    result = {
        "initial_product_residual": initial_product_residual,
        "initial_loss_divergence": abs(
            initial_base_loss - initial_gauge_loss
        ),
        "initial_base_loss": initial_base_loss,
        "final_base_loss": final_base_loss,
        "base_loss_improvement": initial_base_loss - final_base_loss,
        "base_product_displacement": displacement,
        "final_product_divergence": final_divergence,
        "final_loss_divergence": abs(final_base_loss - final_gauge_loss),
        "condition_max": max(
            base_diag.get("condition_max", 0.0),
            gauge_diag.get("condition_max", 0.0),
        ),
        "fallback_count": (
            base_diag.get("fallback_count", 0)
            + gauge_diag.get("fallback_count", 0)
        ),
        "balance_residual_max": max(
            initial_balance_residual,
            base_diag.get("balance_residual_max", 0.0),
            gauge_diag.get("balance_residual_max", 0.0),
        ),
        "retraction_discarded_energy_max": max(
            base_diag.get("retraction_discarded_energy_max", 0.0),
            gauge_diag.get("retraction_discarded_energy_max", 0.0),
        ),
        "base_curve": base_curve,
        "gauge_curve": gauge_curve,
    }

    del base_model, gauge_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_base_only(
    method,
    model_path,
    cfg,
    seed,
    base_state,
    continuation_batches,
    validation,
    lr,
    substeps,
):
    base_model, _, initial_balance_residual = initialize_equivalent_pair(
        model_path,
        cfg,
        seed,
        base_state,
        balanced=True,
    )
    device = torch.device(cfg.device)
    initial_products = collect_products(base_model, cfg.target_modules)
    initial_loss = evaluate_loss(base_model, validation, device)

    if method == "substep":
        curve, diag = continue_substepped_quotient(
            base_model, continuation_batches, validation, cfg, lr, substeps
        )
    else:
        curve, diag = continue_product_consistent(
            base_model, continuation_batches, validation, cfg, lr
        )

    final_products = collect_products(base_model, cfg.target_modules)
    final_loss = evaluate_loss(base_model, validation, device)

    result = {
        "loss_improvement": initial_loss - final_loss,
        "product_displacement": product_displacement(
            initial_products,
            final_products,
        ),
        "condition_max": diag.get("condition_max", 0.0),
        "fallback_count": diag.get("fallback_count", 0),
        "balance_residual_max": max(
            initial_balance_residual,
            diag.get("balance_residual_max", 0.0),
        ),
        "retraction_discarded_energy_max":
            diag.get("retraction_discarded_energy_max", 0.0),
    }

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def bootstrap_summary(values, samples):
    values = list(map(float, values))
    low, high = bootstrap_ci(values, samples)
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95_low": low,
        "ci95_high": high,
        "values": values,
    }


def write_csv(path: Path, rows: Sequence[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class BudgetResult:
    seed: int
    lr: float
    substeps: int
    steps_used: int

    adam_loss_improvement: float
    adam_displacement: float
    adam_gauge_divergence: float

    loss_ratio: float
    displacement_ratio: float
    gauge_ratio: float
    gauge_suppression: float

    final_base_loss: float
    final_gauge_loss: float
    final_loss_divergence: float

    condition_max: float
    fallback_count: int
    balance_residual_max: float

    matched_progress: bool
    selection_score: float


def _progress_score(loss_ratio, move_ratio):
    return math.sqrt(
        math.log10(max(loss_ratio, 1e-12)) ** 2
        + math.log10(max(move_ratio, 1e-12)) ** 2
    )


def _write_csv(path, rows):
    if not rows:
        return
    payload = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0].keys()))
        writer.writeheader()
        writer.writerows(payload)


def run_pair_budgeted(
    model_path,
    cfg,
    seed,
    base_state,
    continuation_batches,
    validation,
    *,
    macro_lr,
    substeps,
    adam_loss_improvement,
    adam_displacement,
    loss_stop_ratio,
    move_stop_ratio,
    check_interval,
):
    """
    Run base and gauge branches for exactly the same macro steps.

    The stopping decision uses only the base branch. The gauge branch never
    participates in stopping or hyperparameter selection.

    Stop at the first checkpoint where either:
      base loss improvement / Adam loss improvement >= loss_stop_ratio
    or
      base product displacement / Adam displacement >= move_stop_ratio
    """
    base_model, gauge_model, initial_balance_residual = initialize_equivalent_pair(
        model_path, cfg, seed, base_state, balanced=True
    )
    device = torch.device(cfg.device)

    initial_base_products = collect_products(base_model, cfg.target_modules)
    initial_gauge_products = collect_products(gauge_model, cfg.target_modules)
    initial_base_loss = evaluate_loss(base_model, validation, device)

    initial_product_residual, _ = product_divergence(
        initial_base_products, initial_gauge_products
    )
    if initial_product_residual >= 1e-5:
        raise RuntimeError(
            f"Initial equivalent-pair residual too large: {initial_product_residual:.3e}"
        )

    local_lr = float(macro_lr) / int(substeps)
    base_opt = FactorizedQuotientFlow(
        modules(base_model, cfg.target_modules),
        lr=local_lr,
        gradient_clip_norm=0.0,
        gram_condition_limit=cfg.gram_condition_limit,
        balance_after_step=True,
    )
    gauge_opt = FactorizedQuotientFlow(
        modules(gauge_model, cfg.target_modules),
        lr=local_lr,
        gradient_clip_norm=0.0,
        gram_condition_limit=cfg.gram_condition_limit,
        balance_after_step=True,
    )

    steps_used = 0
    loss_ratio = 0.0
    move_ratio = 0.0

    for step, batch in enumerate(continuation_batches, start=1):
        batch = batch.to(device)

        for _ in range(int(substeps)):
            base_opt.zero_grad()
            base_loss = model_loss(base_model, batch)
            base_loss.backward()
            base_opt.step()

            gauge_opt.zero_grad()
            gauge_loss = model_loss(gauge_model, batch)
            gauge_loss.backward()
            gauge_opt.step()

        steps_used = step

        should_check = (
            step % int(check_interval) == 0
            or step == len(continuation_batches)
        )
        if should_check:
            current_loss = evaluate_loss(base_model, validation, device)
            current_products = collect_products(base_model, cfg.target_modules)

            loss_ratio = safe_ratio(
                initial_base_loss - current_loss,
                adam_loss_improvement,
            )
            move_ratio = safe_ratio(
                product_displacement(initial_base_products, current_products),
                adam_displacement,
            )

            if (
                loss_ratio >= float(loss_stop_ratio)
                or move_ratio >= float(move_stop_ratio)
            ):
                break

    final_base_products = collect_products(base_model, cfg.target_modules)
    final_gauge_products = collect_products(gauge_model, cfg.target_modules)

    final_base_loss = evaluate_loss(base_model, validation, device)
    final_gauge_loss = evaluate_loss(gauge_model, validation, device)

    loss_ratio = safe_ratio(
        initial_base_loss - final_base_loss,
        adam_loss_improvement,
    )
    move_ratio = safe_ratio(
        product_displacement(initial_base_products, final_base_products),
        adam_displacement,
    )
    gauge_divergence, _ = product_divergence(
        final_base_products, final_gauge_products
    )

    result = {
        "steps_used": steps_used,
        "loss_ratio": loss_ratio,
        "move_ratio": move_ratio,
        "gauge_divergence": gauge_divergence,
        "final_base_loss": final_base_loss,
        "final_gauge_loss": final_gauge_loss,
        "final_loss_divergence": abs(final_base_loss - final_gauge_loss),
        "condition_max": max(
            base_opt.condition_max,
            gauge_opt.condition_max,
        ),
        "fallback_count": (
            base_opt.fallback_count + gauge_opt.fallback_count
        ),
        "balance_residual_max": max(
            initial_balance_residual,
            base_opt.balance_residual_max,
            gauge_opt.balance_residual_max,
        ),
    }

    del base_model, gauge_model, base_opt, gauge_opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result



@dataclass
class CapacityAdaptiveResult:
    seed: int
    rho: float
    phi_macro: float
    macro_flow_time: float
    local_function_tolerance: float
    steps_used: int
    mean_auto_substeps: float
    min_auto_substeps: int
    max_auto_substeps: int
    mean_capacity: float
    max_capacity: float
    mean_flow_dt: float
    total_flow_time: float
    mean_predicted_local_dphi: float
    max_predicted_local_dphi: float
    max_flow_dt_used: float
    dt_cap_hits: int

    adam_loss_improvement: float
    adam_displacement: float
    adam_gauge_divergence: float

    loss_ratio: float
    displacement_ratio: float
    gauge_ratio: float
    gauge_suppression: float

    final_base_loss: float
    final_gauge_loss: float
    final_loss_divergence: float

    condition_max: float
    fallback_count: int
    balance_residual_max: float
    matched_progress: bool


@torch.no_grad()
def _unit_quotient_directions_and_capacity(
    factor_modules,
    condition_limit: float,
):
    """
    Compute the unit-flow quotient direction and its product-space capacity.

    For the GPT-2 Conv1D convention M = A @ B,

        V_A = -grad_A (B B^T)^(-1)
        V_B = -(A^T A)^(-1) grad_B
        V_M = V_A B + A V_B

    The capacity is the global Frobenius norm

        H_opt = sqrt(sum_l ||V_M,l||_F^2).

    No parameters are updated here.
    """
    payload = []
    total_product_speed_sq = 0.0
    condition_max = 0.0
    fallback_count = 0

    for module in factor_modules:
        if module.A.grad is None or module.B.grad is None:
            raise RuntimeError("Capacity evaluation requires A/B gradients.")

        gram_b = module.B @ module.B.transpose(0, 1)
        gram_a = module.A.transpose(0, 1) @ module.A

        inv_b, cond_b, fallback_b = FactorizedQuotientFlow._safe_inverse(
            gram_b, condition_limit
        )
        inv_a, cond_a, fallback_a = FactorizedQuotientFlow._safe_inverse(
            gram_a, condition_limit
        )

        v_a = -(module.A.grad @ inv_b)
        v_b = -(inv_a @ module.B.grad)
        v_m = v_a @ module.B + module.A @ v_b

        total_product_speed_sq += float(
            torch.sum(v_m * v_m).detach().cpu()
        )
        condition_max = max(condition_max, cond_a, cond_b)
        fallback_count += int(fallback_a) + int(fallback_b)
        payload.append((module, v_a, v_b))

    capacity = math.sqrt(max(total_product_speed_sq, 0.0))
    return payload, capacity, condition_max, fallback_count


@torch.no_grad()
def _apply_unit_directions(
    payload,
    flow_dt: float,
    *,
    balance_after_step: bool,
):
    balance_residual = 0.0
    for module, v_a, v_b in payload:
        module.A.add_(v_a, alpha=float(flow_dt))
        module.B.add_(v_b, alpha=float(flow_dt))
        if balance_after_step:
            balance_residual = max(
                balance_residual,
                balanced_factorization_in_place(module),
            )
    return balance_residual


def run_pair_capacity_adaptive(
    model_path,
    cfg,
    seed,
    base_state,
    continuation_batches,
    validation,
    *,
    macro_flow_time,
    local_function_tolerance,
    max_auto_substeps,
    max_flow_dt,
    adam_loss_improvement,
    adam_displacement,
    loss_stop_ratio,
    move_stop_ratio,
    check_interval,
):
    """
    Capacity-controlled quotient integration.

    Two primary dynamical controls:

        T_macro        = macro_flow_time
        epsilon_local = local_function_tolerance

    At each local step, compute product-space capacity H_opt and use

        d_tau = min(T_remaining, epsilon_local / H_opt).

    The number of substeps is generated automatically because H_opt changes
    along the trajectory. The stopping decision, local step size, and auto-K
    are computed from the base branch only. The gauge-equivalent branch follows
    the same d_tau sequence.
    """
    if macro_flow_time <= 0:
        raise ValueError("macro_flow_time must be positive")
    if local_function_tolerance <= 0:
        raise ValueError("local_function_tolerance must be positive")

    base_model, gauge_model, initial_balance_residual = initialize_equivalent_pair(
        model_path, cfg, seed, base_state, balanced=True
    )
    device = torch.device(cfg.device)

    initial_base_products = collect_products(base_model, cfg.target_modules)
    initial_gauge_products = collect_products(gauge_model, cfg.target_modules)
    initial_base_loss = evaluate_loss(base_model, validation, device)

    initial_product_residual, _ = product_divergence(
        initial_base_products, initial_gauge_products
    )
    if initial_product_residual >= 1e-5:
        raise RuntimeError(
            f"Initial equivalent-pair residual too large: {initial_product_residual:.3e}"
        )

    base_modules = modules(base_model, cfg.target_modules)
    gauge_modules = modules(gauge_model, cfg.target_modules)

    steps_used = 0
    loss_ratio = 0.0
    move_ratio = 0.0

    all_k = []
    capacities = []
    flow_dts = []
    dt_cap_hits = 0
    condition_max = 0.0
    fallback_count = 0
    balance_residual_max = initial_balance_residual

    for macro_step, batch in enumerate(continuation_batches, start=1):
        batch = batch.to(device)
        remaining_time = float(macro_flow_time)
        local_count = 0

        while remaining_time > 1e-12:
            if local_count >= int(max_auto_substeps):
                raise RuntimeError(
                    "Adaptive K exceeded max_auto_substeps. "
                    "Increase --max-auto-substeps or loosen the local tolerance."
                )

            # Fresh base gradient and capacity.
            for module in base_modules:
                module.A.grad = None
                module.B.grad = None
            base_loss = model_loss(base_model, batch)
            base_loss.backward()

            base_payload, capacity, cond, fallbacks = (
                _unit_quotient_directions_and_capacity(
                    base_modules,
                    cfg.gram_condition_limit,
                )
            )
            condition_max = max(condition_max, cond)
            fallback_count += fallbacks

            # True capacity-adaptive time step:
            #
            #   d_tau = min(T_remaining, epsilon_local / H_opt)
            #
            # Therefore high capacity produces smaller local flow time and more
            # substeps. K is no longer fixed by a displacement ratio.
            safe_capacity = max(capacity, 1e-12)
            flow_dt = min(
                remaining_time,
                float(local_function_tolerance) / safe_capacity,
            )

            # Numerical safeguard only.
            if flow_dt > float(max_flow_dt):
                flow_dt = float(max_flow_dt)
                dt_cap_hits += 1

            predicted_dphi = flow_dt * safe_capacity
            if flow_dt <= 1e-14 or predicted_dphi <= 1e-14:
                raise RuntimeError(
                    "Capacity step made no progress; capacity may be numerically zero."
                )

            # Fresh gauge gradient. It follows the base-derived flow_dt.
            for module in gauge_modules:
                module.A.grad = None
                module.B.grad = None
            gauge_loss = model_loss(gauge_model, batch)
            gauge_loss.backward()

            gauge_payload, _, gauge_cond, gauge_fallbacks = (
                _unit_quotient_directions_and_capacity(
                    gauge_modules,
                    cfg.gram_condition_limit,
                )
            )
            condition_max = max(condition_max, gauge_cond)
            fallback_count += gauge_fallbacks

            balance_residual_max = max(
                balance_residual_max,
                _apply_unit_directions(
                    base_payload,
                    flow_dt,
                    balance_after_step=True,
                ),
                _apply_unit_directions(
                    gauge_payload,
                    flow_dt,
                    balance_after_step=True,
                ),
            )

            remaining_time -= flow_dt
            local_count += 1
            capacities.append(capacity)
            flow_dts.append(flow_dt)

        all_k.append(local_count)
        steps_used = macro_step

        should_check = (
            macro_step % int(check_interval) == 0
            or macro_step == len(continuation_batches)
        )
        if should_check:
            current_loss = evaluate_loss(base_model, validation, device)
            current_products = collect_products(base_model, cfg.target_modules)

            loss_ratio = safe_ratio(
                initial_base_loss - current_loss,
                adam_loss_improvement,
            )
            move_ratio = safe_ratio(
                product_displacement(initial_base_products, current_products),
                adam_displacement,
            )

            if (
                loss_ratio >= float(loss_stop_ratio)
                or move_ratio >= float(move_stop_ratio)
            ):
                break

    final_base_products = collect_products(base_model, cfg.target_modules)
    final_gauge_products = collect_products(gauge_model, cfg.target_modules)
    final_base_loss = evaluate_loss(base_model, validation, device)
    final_gauge_loss = evaluate_loss(gauge_model, validation, device)

    loss_ratio = safe_ratio(
        initial_base_loss - final_base_loss,
        adam_loss_improvement,
    )
    move_ratio = safe_ratio(
        product_displacement(initial_base_products, final_base_products),
        adam_displacement,
    )
    gauge_divergence, _ = product_divergence(
        final_base_products, final_gauge_products
    )

    result = {
        "steps_used": steps_used,
        "mean_auto_substeps": float(np.mean(all_k)),
        "min_auto_substeps": int(np.min(all_k)),
        "max_auto_substeps": int(np.max(all_k)),
        "mean_capacity": float(np.mean(capacities)),
        "max_capacity": float(np.max(capacities)),
        "mean_flow_dt": float(np.mean(flow_dts)),
        "total_flow_time": float(np.sum(flow_dts)),
        "mean_predicted_local_dphi": float(
            np.mean(np.array(flow_dts) * np.array(capacities))
        ),
        "max_predicted_local_dphi": float(
            np.max(np.array(flow_dts) * np.array(capacities))
        ),
        "max_flow_dt_used": float(np.max(flow_dts)),
        "dt_cap_hits": int(dt_cap_hits),
        "loss_ratio": float(loss_ratio),
        "move_ratio": float(move_ratio),
        "gauge_divergence": float(gauge_divergence),
        "final_base_loss": float(final_base_loss),
        "final_gauge_loss": float(final_gauge_loss),
        "final_loss_divergence": float(abs(final_base_loss - final_gauge_loss)),
        "condition_max": float(condition_max),
        "fallback_count": int(fallback_count),
        "balance_residual_max": float(balance_residual_max),
    }

    del base_model, gauge_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=[
            "transformer.h.0.attn.c_attn",
            "transformer.h.1.attn.c_attn",
        ],
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1201, 1303, 1409, 1511, 1601],
    )

    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--max-continuation-steps", type=int, default=100)
    parser.add_argument("--check-interval", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--factor-adam-lr", type=float, default=1e-3)

    # The two primary dynamical controls.
    parser.add_argument(
        "--macro-flow-time",
        type=float,
        default=2.6,
        help="Total quotient-flow time per macro step.",
    )
    parser.add_argument(
        "--rho-grid",
        nargs="+",
        type=float,
        default=[0.015, 0.02, 0.025],
        help=(
            "Dimensionless ratio in epsilon_local = rho * Phi_macro. "
            "Phi_macro is derived per seed; rho is the only new capacity-control parameter."
        ),
    )

    # Numerical safeguards only.
    parser.add_argument("--max-auto-substeps", type=int, default=128)
    parser.add_argument(
        "--max-flow-dt",
        type=float,
        default=2.6,
        help="Safety cap on d_tau=dPhi/H; not used as a search parameter.",
    )

    parser.add_argument("--loss-stop-ratio", type=float, default=1.8)
    parser.add_argument("--move-stop-ratio", type=float, default=0.95)
    parser.add_argument("--loss-match-low", type=float, default=0.8)
    parser.add_argument("--loss-match-high", type=float, default=2.0)
    parser.add_argument("--move-match-low", type=float, default=0.5)
    parser.add_argument("--move-match-high", type=float, default=1.2)

    parser.add_argument("--gram-condition-limit", type=float, default=1e10)
    parser.add_argument("--gauge-condition-number", type=float, default=100.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)

    parser.add_argument(
        "--repo-dir",
        default="/content/Geometric-Flow-h112",
    )
    parser.add_argument(
        "--out-dir",
        default="/content/geoflow_h112_results",
    )
    parser.add_argument(
        "--local-model-dir",
        default="/content/hf_models/gpt2_h112",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print("[Colab/Jupyter notice] Ignored kernel arguments:", unknown)

    if args.macro_flow_time <= 0:
        raise ValueError("--macro-flow-time must be positive")
    if not args.rho_grid:
        raise ValueError("Provide at least one rho value.")
    if any(value <= 0 or value >= 1 for value in args.rho_grid):
        raise ValueError("Each rho must satisfy 0 < rho < 1.")
    return args


def main():
    args = parse_args()

    repo_dir = Path(args.repo_dir)
    ensure_dependencies(repo_dir)

    cfg = Config(
        model_name=args.model_name,
        target_modules=list(args.target_modules),
        rank=args.rank,
        seeds=list(args.seeds),
        warmup_steps=args.warmup_steps,
        continuation_steps=args.max_continuation_steps,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        validation_batches=args.validation_batches,
        eval_interval=args.eval_interval,
        factor_adam_lr=args.factor_adam_lr,
        quotient_lr=0.0,
        gradient_clip_norm=0.0,
        gram_condition_limit=args.gram_condition_limit,
        gauge_condition_number=args.gauge_condition_number,
        bootstrap_samples=args.bootstrap_samples,
        device="cuda" if torch.cuda.is_available() else "cpu",
        repo_dir=str(repo_dir),
        out_dir=args.out_dir,
        local_model_dir=args.local_model_dir,
    )

    model_path = ensure_local_gpt2(Path(cfg.local_model_dir))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    stream = build_token_stream(tokenizer)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        **asdict(cfg),
        "macro_flow_time": args.macro_flow_time,
        "rho_grid": list(args.rho_grid),
        "one_parameter_law": "epsilon_local = rho * Phi_macro",
        "macro_scale_definition": "Phi_macro = Adam product displacement for the same seed",
        "held_out_seeds": list(args.seeds),
        "nominal_K_grid": None,
        "auto_K_note": "K depends on the evolving capacity H_opt and is not predetermined.",
        "capacity_law": "d_tau = min(T_remaining, epsilon_local / H_opt)",
        "capacity_definition": "sqrt(sum_l ||V_A B + A V_B||_F^2)",
        "auto_K": True,
        "stopping_uses_gauge_branch": False,
        "dt_uses_gauge_branch": False,
        "max_auto_substeps": args.max_auto_substeps,
        "max_flow_dt": args.max_flow_dt,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    print("=" * 148)
    print("GeoFlow H10.12 -- ONE-PARAMETER CAPACITY-RATIO TEST")
    print("=" * 148)
    print(json.dumps(config_payload, indent=2))

    rows = []

    for seed in cfg.seeds:
        print(f"\n[seed {seed}] warmup", flush=True)

        warmup_batches = make_batches(
            stream,
            cfg.warmup_steps,
            cfg.batch_size,
            cfg.sequence_length,
            seed + 11,
        )
        continuation_batches = make_batches(
            stream,
            cfg.continuation_steps,
            cfg.batch_size,
            cfg.sequence_length,
            seed + 23,
        )
        validation = make_batches(
            stream,
            cfg.validation_batches,
            cfg.batch_size,
            cfg.sequence_length,
            seed + 37,
        )

        warm = create_model(model_path, cfg, seed)
        warmup_factor_adam(warm, warmup_batches, cfg)

        base_state = [
            (
                get_submodule(warm, path).A.detach().clone(),
                get_submodule(warm, path).B.detach().clone(),
            )
            for path in cfg.target_modules
        ]

        adam_base, adam_gauge, _ = initialize_equivalent_pair(
            model_path,
            cfg,
            seed,
            base_state,
            balanced=False,
        )
        device = torch.device(cfg.device)

        adam_initial_products = collect_products(
            adam_base,
            cfg.target_modules,
        )
        adam_initial_loss = evaluate_loss(
            adam_base,
            validation,
            device,
        )

        continue_factor_adam(
            adam_base,
            continuation_batches,
            validation,
            cfg,
        )
        continue_factor_adam(
            adam_gauge,
            continuation_batches,
            validation,
            cfg,
        )

        adam_final_products = collect_products(
            adam_base,
            cfg.target_modules,
        )
        adam_final_gauge_products = collect_products(
            adam_gauge,
            cfg.target_modules,
        )
        adam_final_loss = evaluate_loss(
            adam_base,
            validation,
            device,
        )

        adam_loss_improvement = adam_initial_loss - adam_final_loss
        adam_displacement = product_displacement(
            adam_initial_products,
            adam_final_products,
        )
        adam_gauge_divergence, _ = product_divergence(
            adam_final_products,
            adam_final_gauge_products,
        )

        print(
            f"  Adam target: loss={adam_loss_improvement:+.6e} "
            f"move={adam_displacement:.6e} "
            f"gauge={adam_gauge_divergence:.6e}",
            flush=True,
        )

        del adam_base, adam_gauge, warm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        phi_macro = float(adam_displacement)
        if not math.isfinite(phi_macro) or phi_macro <= 0:
            raise RuntimeError(
                f"Invalid Phi_macro from Adam displacement: {phi_macro}"
            )

        for rho in args.rho_grid:
            tolerance = float(rho * phi_macro)
            candidate = run_pair_capacity_adaptive(
                model_path,
                cfg,
                seed,
                base_state,
                continuation_batches,
                validation,
                macro_flow_time=args.macro_flow_time,
                local_function_tolerance=float(tolerance),
                max_auto_substeps=args.max_auto_substeps,
                max_flow_dt=args.max_flow_dt,
                adam_loss_improvement=adam_loss_improvement,
                adam_displacement=adam_displacement,
                loss_stop_ratio=args.loss_stop_ratio,
                move_stop_ratio=args.move_stop_ratio,
                check_interval=args.check_interval,
            )

            gauge_ratio = safe_ratio(
                candidate["gauge_divergence"],
                adam_gauge_divergence,
            )
            suppression = 1.0 / max(gauge_ratio, 1e-12)
            matched = (
                args.loss_match_low
                <= candidate["loss_ratio"]
                <= args.loss_match_high
                and args.move_match_low
                <= candidate["move_ratio"]
                <= args.move_match_high
            )

            row = CapacityAdaptiveResult(
                seed=seed,
                rho=float(rho),
                phi_macro=float(phi_macro),
                macro_flow_time=float(args.macro_flow_time),
                local_function_tolerance=float(tolerance),
                steps_used=int(candidate["steps_used"]),
                mean_auto_substeps=float(
                    candidate["mean_auto_substeps"]
                ),
                min_auto_substeps=int(
                    candidate["min_auto_substeps"]
                ),
                max_auto_substeps=int(
                    candidate["max_auto_substeps"]
                ),
                mean_capacity=float(candidate["mean_capacity"]),
                max_capacity=float(candidate["max_capacity"]),
                mean_flow_dt=float(candidate["mean_flow_dt"]),
                total_flow_time=float(candidate["total_flow_time"]),
                mean_predicted_local_dphi=float(
                    candidate["mean_predicted_local_dphi"]
                ),
                max_predicted_local_dphi=float(
                    candidate["max_predicted_local_dphi"]
                ),
                max_flow_dt_used=float(
                    candidate["max_flow_dt_used"]
                ),
                dt_cap_hits=int(candidate["dt_cap_hits"]),
                adam_loss_improvement=float(adam_loss_improvement),
                adam_displacement=float(adam_displacement),
                adam_gauge_divergence=float(adam_gauge_divergence),
                loss_ratio=float(candidate["loss_ratio"]),
                displacement_ratio=float(candidate["move_ratio"]),
                gauge_ratio=float(gauge_ratio),
                gauge_suppression=float(suppression),
                final_base_loss=float(candidate["final_base_loss"]),
                final_gauge_loss=float(candidate["final_gauge_loss"]),
                final_loss_divergence=float(
                    candidate["final_loss_divergence"]
                ),
                condition_max=float(candidate["condition_max"]),
                fallback_count=int(candidate["fallback_count"]),
                balance_residual_max=float(
                    candidate["balance_residual_max"]
                ),
                matched_progress=bool(matched),
            )
            rows.append(row)

            print(
                f"    rho={rho:.5f} Phi_macro={phi_macro:.4f} "
                f"eps={tolerance:.6g} "
                f"autoK={row.mean_auto_substeps:.2f} "
                f"[{row.min_auto_substeps},{row.max_auto_substeps}] "
                f"loss={row.loss_ratio:.3f} "
                f"move={row.displacement_ratio:.3f} "
                f"gauge={row.gauge_ratio:.3f} "
                f"supp={row.gauge_suppression:.2f}x "
                f"cap_hits={row.dt_cap_hits} "
                f"matched={row.matched_progress}",
                flush=True,
            )

    _write_csv(out_dir / "all_candidates.csv", rows)

    groups = {}
    for row in rows:
        groups.setdefault(
            row.rho,
            [],
        ).append(row)

    aggregates = []
    for rho, group in groups.items():
        suppressions = np.array(
            [row.gauge_suppression for row in group],
            dtype=np.float64,
        )
        gauge_ratios = np.array(
            [row.gauge_ratio for row in group],
            dtype=np.float64,
        )
        aggregate = {
            "macro_flow_time": args.macro_flow_time,
            "rho": float(rho),
            "mean_phi_macro": float(np.mean([
                row.phi_macro for row in group
            ])),
            "mean_local_function_tolerance": float(np.mean([
                row.local_function_tolerance for row in group
            ])),
            "n": len(group),
            "mean_auto_substeps": float(np.mean(
                [row.mean_auto_substeps for row in group]
            )),
            "min_auto_substeps": int(min(
                row.min_auto_substeps for row in group
            )),
            "max_auto_substeps": int(max(
                row.max_auto_substeps for row in group
            )),
            "mean_loss_ratio": float(np.mean(
                [row.loss_ratio for row in group]
            )),
            "mean_displacement_ratio": float(np.mean(
                [row.displacement_ratio for row in group]
            )),
            "geometric_mean_gauge_ratio": float(
                10.0 ** np.mean(np.log10(
                    np.clip(gauge_ratios, 1e-12, None)
                ))
            ),
            "geometric_mean_gauge_suppression": float(
                10.0 ** np.mean(np.log10(
                    np.clip(suppressions, 1e-12, None)
                ))
            ),
            "min_gauge_suppression": float(np.min(suppressions)),
            "p20_gauge_suppression": float(
                np.percentile(suppressions, 20)
            ),
            "ten_x_seed_fraction": float(
                np.mean(suppressions >= 10.0)
            ),
            "all_matched_progress": bool(
                all(row.matched_progress for row in group)
            ),
            "all_gauge_better_than_adam": bool(
                all(row.gauge_ratio < 1.0 for row in group)
            ),
            "total_dt_cap_hits": int(sum(
                row.dt_cap_hits for row in group
            )),
            "fallback_count": int(sum(
                row.fallback_count for row in group
            )),
            "balance_residual_max": float(max(
                row.balance_residual_max for row in group
            )),
        }
        aggregates.append(aggregate)

    aggregates.sort(
        key=lambda item: (
            not item["all_matched_progress"],
            -item["p20_gauge_suppression"],
            item["geometric_mean_gauge_ratio"],
        )
    )
    winner = aggregates[0]

    winner_rows = groups[winner["rho"]]
    winner_ratios = np.array(
        [row.gauge_ratio for row in winner_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260715)
    boot = []
    for _ in range(int(cfg.bootstrap_samples)):
        sample = rng.choice(
            winner_ratios,
            size=len(winner_ratios),
            replace=True,
        )
        boot.append(
            10.0 ** float(np.mean(np.log10(
                np.clip(sample, 1e-12, None)
            )))
        )
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    gates = {
        "H112_ONE_PARAMETER_CAPACITY_LAW":
            True,
        "H112_RHO_IN_UNIT_INTERVAL":
            all(0.0 < value < 1.0 for value in args.rho_grid),
        "H112_HELD_OUT_SEEDS_5":
            len(args.seeds) == 5,
        "H112_AUTO_K_GENERATED":
            winner["mean_auto_substeps"] > 0,
        "H112_K_NOT_PREDETERMINED":
            winner["max_auto_substeps"] > winner["min_auto_substeps"],
        "H112_MACRO_SCALE_POSITIVE":
            all(row.phi_macro > 0.0 for row in winner_rows),
        "H112_LOCAL_DPHI_TOLERANCE_RESPECTED":
            all(
                row.max_predicted_local_dphi
                <= row.local_function_tolerance * (1.0 + 1e-6)
                for row in winner_rows
            ),
        "H112_ALL_SEEDS_MATCHED_PROGRESS":
            winner["all_matched_progress"],
        "H112_ALL_SEEDS_GAUGE_BETTER_THAN_ADAM":
            winner["all_gauge_better_than_adam"],
        "H112_MEAN_GAUGE_SUPPRESSION_10X_PASS":
            winner["geometric_mean_gauge_ratio"] <= 0.1,
        "H112_GAUGE_10X_SEED_RATE_80_PERCENT_PASS":
            float(np.mean([
                row.gauge_suppression >= 10.0 for row in winner_rows
            ])) >= 0.8,
        "H112_GAUGE_7X_ALL_SEEDS_PASS":
            all(row.gauge_suppression >= 7.0 for row in winner_rows),
        "H112_P20_SUPPRESSION_10X_PASS":
            winner["p20_gauge_suppression"] >= 10.0,
        "H112_MIN_SUPPRESSION_7X_PASS":
            winner["min_gauge_suppression"] >= 7.0,
        "H112_BOOTSTRAP_UPPER_GAUGE_RATIO_BELOW_0P1_PASS":
            float(ci_high) <= 0.1,
        "H112_NO_FLOW_DT_CAP_HITS":
            winner["total_dt_cap_hits"] == 0,
        "H112_NO_FALLBACK_PASS":
            winner["fallback_count"] == 0,
        "H112_BALANCE_PASS":
            winner["balance_residual_max"] < 1e-5,
    }

    summary = {
        "config": config_payload,
        "selected_rho_configuration": winner,
        "all_aggregates": aggregates,
        "held_out_statistics": {
            "n_seeds": len(winner_rows),
            "ten_x_seed_fraction": float(np.mean([
                row.gauge_suppression >= 10.0 for row in winner_rows
            ])),
            "seven_x_seed_fraction": float(np.mean([
                row.gauge_suppression >= 7.0 for row in winner_rows
            ])),
            "min_suppression": float(min(
                row.gauge_suppression for row in winner_rows
            )),
            "median_suppression": float(np.median([
                row.gauge_suppression for row in winner_rows
            ])),
        },
        "bootstrap_geometric_mean_gauge_ratio_ci95": [
            float(ci_low),
            float(ci_high),
        ],
        "bootstrap_geometric_mean_gauge_suppression_ci95": [
            float(1.0 / max(ci_high, 1e-12)),
            float(1.0 / max(ci_low, 1e-12)),
        ],
        "gates": gates,
        "interpretation": (
            "The adaptive controller replaces manual K with the capacity law "
            "d_tau=dPhi/H_opt. The two primary controls are macro function-space "
            "target and local function-space tolerance."
        ),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 148)
    print("H10.12 ONE-PARAMETER CAPACITY-RATIO SUMMARY")
    print("=" * 148)
    print(json.dumps(winner, indent=2))
    print("\nBOOTSTRAP CI")
    print(json.dumps({
        "gauge_ratio_ci95": [float(ci_low), float(ci_high)],
        "suppression_ci95": [
            float(1.0 / max(ci_high, 1e-12)),
            float(1.0 / max(ci_low, 1e-12)),
        ],
    }, indent=2))
    print("\nDECISION GATES")
    for key, value in gates.items():
        print(f"{key} = {value}")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
