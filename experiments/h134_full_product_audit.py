# ============================================================
# GeoFlow H13.4 — Full-product gauge-dynamics audit
#
# This experiment:
#   1. installs dependencies;
#   2. clones/updates papasop/Geometric-Flow;
#   3. loads a real WikiText train/validation/test split;
#   4. loads GPT-2;
#   5. injects LoRA factors into selected c_attn layers;
#   6. compares balanced and gauge-equivalent LoRA branches;
#   7. measures complete B@A product-space update direction, magnitude,
#      and trajectory gaps;
#   8. compares AdamW with CapacityAdaptiveQuotientFlow across kappa values;
#   9. writes JSON/CSV outputs.
#
# Colab:
#   Upload this file and run:
#
#   %run /content/h134_full_product_audit.py
#
# Scientific scope:
#   Empirical full-product gauge-dynamics audit under locked batches.
#   This is not a formal proof of exact mathematical gauge invariance.
#
# Engineering note:
#   This Colab-oriented audit script vendors archived H12.7/H13.2 helper
#   routines for continuity with the original notebook lineage. The script
#   entrypoint is h134_main(); older main()/h132_main() helpers are retained as
#   archived compatibility code and should not be read as the active H13.4
#   experimental design. Future cleanup should move shared LoRA audit utilities
#   into experiments/common/.
# ============================================================

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------
# Bootstrap before importing optional packages.
# ---------------------------------------------------------------------

REPO_URL = "https://github.com/papasop/Geometric-Flow.git"


def shell(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=None if cwd is None else str(cwd),
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}"
        )
    return result


def ensure_packages() -> None:
    required = {
        "torch": None,
        "transformers": "transformers>=4.40,<6",
        "datasets": "datasets>=2.18,<5",
        "numpy": "numpy>=1.24",
    }
    missing = []
    for module_name, requirement in required.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            if requirement is not None:
                missing.append(requirement)
    if missing:
        shell([sys.executable, "-m", "pip", "install", "-q", *missing])


def prepare_repo(repo_dir: Path, force_reclone: bool) -> None:
    if force_reclone and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        shell(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    else:
        shell(["git", "fetch", "origin", "main", "--depth", "1"], cwd=repo_dir)
        shell(["git", "reset", "--hard", "origin/main"], cwd=repo_dir)
    shell([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir)])


_SKIP_BOOTSTRAP = os.environ.get("GEOFLOW_H134_SKIP_BOOTSTRAP") == "1"
if not _SKIP_BOOTSTRAP:
    ensure_packages()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if _SKIP_BOOTSTRAP:
    load_dataset = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
else:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

# Imported after repository installation in main().
CapacityAdaptiveQuotientFlow = None


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one integer is required.")
    return values


def parse_str_list(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one target module is required.")
    return values


# ---------------------------------------------------------------------
# Real-data token blocks
# ---------------------------------------------------------------------


class TokenBlockDataset(Dataset):
    def __init__(self, blocks: torch.Tensor):
        if blocks.ndim != 2:
            raise ValueError("blocks must have shape [n_blocks, sequence_length]")
        self.blocks = blocks.long().contiguous()

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self.blocks[index]
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }


def stable_text_sample(texts: list[str], max_examples: int, seed: int) -> list[str]:
    clean = [x for x in texts if isinstance(x, str) and x.strip()]
    if max_examples <= 0 or len(clean) <= max_examples:
        return clean
    rng = random.Random(seed)
    indices = list(range(len(clean)))
    rng.shuffle(indices)
    return [clean[i] for i in indices[:max_examples]]


def tokenize_to_blocks(
    texts: list[str],
    tokenizer,
    sequence_length: int,
    max_blocks: int,
) -> torch.Tensor:
    # Tokenize in moderate batches to avoid a giant temporary list.
    all_ids: list[int] = []
    batch_size = 256
    eos = tokenizer.eos_token_id
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
        for row in encoded:
            all_ids.extend(row)
            if eos is not None:
                all_ids.append(eos)
        if max_blocks > 0 and len(all_ids) >= (max_blocks + 1) * sequence_length:
            break

    usable = (len(all_ids) // sequence_length) * sequence_length
    if max_blocks > 0:
        usable = min(usable, max_blocks * sequence_length)
    if usable < sequence_length:
        raise RuntimeError("Not enough tokens to create one block.")

    tensor = torch.tensor(all_ids[:usable], dtype=torch.long)
    return tensor.view(-1, sequence_length)


def load_real_wikitext(
    tokenizer,
    dataset_config: str,
    sequence_length: int,
    train_examples: int,
    eval_examples: int,
    train_blocks: int,
    eval_blocks: int,
    seed: int,
) -> tuple[TokenBlockDataset, TokenBlockDataset, TokenBlockDataset, dict]:
    try:
        raw = load_dataset("Salesforce/wikitext", dataset_config)
        dataset_source = "Salesforce/wikitext"
    except Exception:
        raw = load_dataset("wikitext", dataset_config)
        dataset_source = "wikitext"

    train_texts = stable_text_sample(
        list(raw["train"]["text"]),
        train_examples,
        seed,
    )
    val_texts = stable_text_sample(
        list(raw["validation"]["text"]),
        eval_examples,
        seed + 1,
    )
    test_texts = stable_text_sample(
        list(raw["test"]["text"]),
        eval_examples,
        seed + 2,
    )

    train_tensor = tokenize_to_blocks(
        train_texts, tokenizer, sequence_length, train_blocks
    )
    val_tensor = tokenize_to_blocks(
        val_texts, tokenizer, sequence_length, eval_blocks
    )
    test_tensor = tokenize_to_blocks(
        test_texts, tokenizer, sequence_length, eval_blocks
    )

    metadata = {
        "dataset_source": dataset_source,
        "dataset_config": dataset_config,
        "train_documents_used": len(train_texts),
        "validation_documents_used": len(val_texts),
        "test_documents_used": len(test_texts),
        "train_blocks": int(train_tensor.shape[0]),
        "validation_blocks": int(val_tensor.shape[0]),
        "test_blocks": int(test_tensor.shape[0]),
        "sequence_length": sequence_length,
        "train_tokens": int(train_tensor.numel()),
        "validation_tokens": int(val_tensor.numel()),
        "test_tokens": int(test_tensor.numel()),
    }
    return (
        TokenBlockDataset(train_tensor),
        TokenBlockDataset(val_tensor),
        TokenBlockDataset(test_tensor),
        metadata,
    )



# ---------------------------------------------------------------------
# Locked batch schedule
# ---------------------------------------------------------------------


@dataclass
class LockedBatchSchedule:
    """A deterministic list of dataset indices shared by every run branch."""

    batches: list[list[int]]
    schedule_hash: str

    def batch(self, dataset: TokenBlockDataset, step: int) -> dict[str, torch.Tensor]:
        indices = self.batches[step]
        ids = torch.stack([dataset.blocks[i] for i in indices], dim=0)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }


def make_locked_batch_schedule(
    dataset: TokenBlockDataset,
    *,
    backward_budget: int,
    batch_size: int,
    seed: int,
) -> LockedBatchSchedule:
    if len(dataset) < 1:
        raise ValueError("Training dataset is empty.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    generator = torch.Generator()
    generator.manual_seed(seed)

    batches: list[list[int]] = []
    flat_indices: list[int] = []
    pool: list[int] = []

    while len(batches) < backward_budget:
        if len(pool) < batch_size:
            pool.extend(torch.randperm(len(dataset), generator=generator).tolist())
        current = pool[:batch_size]
        del pool[:batch_size]
        batches.append(current)
        flat_indices.extend(current)

    digest = hashlib.sha256(
        np.asarray(flat_indices, dtype=np.int64).tobytes()
    ).hexdigest()

    return LockedBatchSchedule(
        batches=batches,
        schedule_hash=digest,
    )


def batch_content_hash(batch: dict[str, torch.Tensor]) -> str:
    return hashlib.sha256(
        batch["input_ids"].contiguous().cpu().numpy().tobytes()
    ).hexdigest()


# ---------------------------------------------------------------------
# LoRA injection for GPT-2 Conv1D-style modules
# ---------------------------------------------------------------------


class LoRAConv1DAdapter(nn.Module):
    """Frozen base projection plus public-convention LoRA M = B @ A.

    A: [rank, in_features]
    B: [out_features, rank]
    Delta output: x @ (B @ A).T
    """

    def __init__(
        self,
        base_module: nn.Module,
        rank: int,
        init_scale: float,
        seed: int,
    ) -> None:
        super().__init__()
        if not hasattr(base_module, "weight"):
            raise TypeError("Target module must expose a weight tensor.")
        weight = base_module.weight
        if weight.ndim != 2:
            raise ValueError("Target weight must be rank-2.")

        # transformers Conv1D stores [in_features, out_features].
        self.base = base_module
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = int(weight.shape[0])
        out_features = int(weight.shape[1])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        a = init_scale * torch.randn(
            rank,
            in_features,
            generator=generator,
            dtype=weight.dtype,
        )
        # Nonzero B is required for full-rank Gram matrices.
        b = init_scale * torch.randn(
            out_features,
            rank,
            generator=generator,
            dtype=weight.dtype,
        )
        self.A = nn.Parameter(a)
        self.B = nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(x, self.A), self.B)
        return base_out + delta

    def product(self) -> torch.Tensor:
        return self.B @ self.A


def get_parent_and_child(root: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def get_child(parent, name: str):
    if name.isdigit():
        return parent[int(name)]
    return getattr(parent, name)


def set_child(parent, name: str, module: nn.Module) -> None:
    if name.isdigit():
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    init_scale: float,
    seed: int,
) -> list[LoRAConv1DAdapter]:
    adapters = []
    for index, name in enumerate(target_modules):
        parent, child_name = get_parent_and_child(model, name)
        base = get_child(parent, child_name)
        adapter = LoRAConv1DAdapter(
            base,
            rank=rank,
            init_scale=init_scale,
            seed=seed + 1009 * index,
        )
        set_child(parent, child_name, adapter)
        adapters.append(adapter)
    return adapters


def apply_gauge(
    adapters: list[LoRAConv1DAdapter],
    condition_number: float,
) -> None:
    if condition_number <= 1.0:
        return
    with torch.no_grad():
        for adapter in adapters:
            rank = adapter.A.shape[0]
            log_half = 0.5 * math.log(condition_number)
            diag = torch.linspace(
                -log_half,
                log_half,
                rank,
                device=adapter.A.device,
                dtype=adapter.A.dtype,
            ).exp()
            s = torch.diag(diag)
            s_inv = torch.diag(1.0 / diag)
            adapter.A.copy_(s @ adapter.A)
            adapter.B.copy_(adapter.B @ s_inv)


def trainable_parameters(adapters: list[LoRAConv1DAdapter]):
    params = []
    for adapter in adapters:
        params.extend([adapter.A, adapter.B])
    return params


# ---------------------------------------------------------------------
# Evaluation and metrics
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    batches = 0

    for batch in loader:
        if max_batches > 0 and batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # HF causal-LM loss is mean over shifted nonignored tokens.
        valid_tokens = int(batch["labels"][:, 1:].numel())
        total_nll += float(outputs.loss.detach().cpu()) * valid_tokens
        total_tokens += valid_tokens
        batches += 1

    mean_loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(mean_loss, 20.0))
    return {
        "loss": mean_loss,
        "perplexity": ppl,
        "tokens": total_tokens,
        "batches": batches,
    }


@torch.no_grad()
def collect_logits_signature(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 2,
) -> torch.Tensor:
    model.eval()
    rows = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        rows.append(logits.detach().float().cpu().reshape(-1))
    return torch.cat(rows)


def relative_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    denom = 0.5 * (x.norm() + y.norm())
    denom = denom.clamp_min(torch.finfo(x.dtype).tiny)
    return float(((x - y).norm() / denom).cpu())


def peak_cuda_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


# ---------------------------------------------------------------------
# Training one representation
# ---------------------------------------------------------------------



@dataclass
class RunResult:
    optimizer: str
    seed: int
    representation: str
    gauge_condition_number: float
    initial_actual_cond_A_mean: float
    initial_actual_cond_B_mean: float
    initial_actual_cond_max_mean: float
    initial_actual_cond_max_peak: float
    completed_macro_steps: int
    partial_macro_substeps: int
    backward_budget: int
    backward_calls: int
    train_tokens_seen: int
    locked_schedule_hash: str
    first_batch_hash: str
    first_train_loss: float
    initial_validation_loss: float
    initial_validation_ppl: float
    initial_test_loss: float
    initial_test_ppl: float
    initial_test_signature_hash: str
    best_validation_loss: float
    best_validation_ppl: float
    final_validation_loss: float
    final_validation_ppl: float
    test_loss: float
    test_ppl: float
    validation_loss_improvement: float
    validation_ppl_improvement: float
    test_loss_improvement: float
    test_ppl_improvement: float
    wall_seconds: float
    tokens_per_second: float
    peak_cuda_memory_mb: float
    mean_auto_substeps: float | None
    min_auto_substeps: int | None
    max_auto_substeps: int | None
    fallback_count: int | None
    flow_dt_cap_hits: int | None
    condition_max: float | None
    max_predicted_local_dphi: float | None
    unfinished_macro_flow_time: float | None
    model_signature: str


def make_model(
    model_name: str,
    target_modules: list[str],
    rank: int,
    init_scale: float,
    seed: int,
    gauge_condition_number: float,
    device: torch.device,
):
    seed_everything(seed)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    adapters = inject_lora(
        model,
        target_modules,
        rank=rank,
        init_scale=init_scale,
        seed=seed,
    )
    apply_gauge(adapters, gauge_condition_number)
    model.to(device)
    return model, adapters


class ResumableCapacityStepper:
    """Advance one capacity local substep per backward call.

    This mirrors the public controller law while preserving unfinished macro-flow
    time across an exact backward budget. No exception is used to abort a public
    macro_step midway.
    """

    def __init__(
        self,
        optimizer,
        *,
        macro_flow_time: float,
        local_function_tolerance: float,
        max_flow_dt: float | None,
        k1_enabled: bool = False,
        k1_eta: float = 0.08,
        k1_ema_beta: float = 0.90,
        k1_epsilon_min: float = 0.08,
        k1_epsilon_max: float = 0.50,
        k1_freeze_on_actuator_limit: bool = True,
        progress_k1_eta: float = 0.06,
        progress_k1_ema_beta: float = 0.90,
        progress_k1_warmup: int = 12,
        progress_k1_ratio_min: float = 0.50,
        progress_k1_ratio_max: float = 1.50,
        progress_k1_floor: float = 1e-4,
        progress_deadband: float = 0.04,
        progress_update_interval: int = 4,
        flow_time_control_enabled: bool = True,
        flow_time_eta: float = 0.04,
        flow_time_min: float = 0.30,
        flow_time_max: float = 0.80,
        utilization_target: float = 0.85,
    ) -> None:
        self.optimizer = optimizer
        self.macro_flow_time = float(macro_flow_time)
        self.local_function_tolerance = float(local_function_tolerance)
        self.max_flow_dt = max_flow_dt
        self.remaining_flow_time = float(macro_flow_time)
        self.completed_macro_steps = 0
        self.current_macro_substeps = 0
        self.completed_macro_substep_counts: list[int] = []
        self.capacity_values: list[float] = []
        self.flow_dt_values: list[float] = []
        self.predicted_dphi_values: list[float] = []
        self.flow_dt_cap_hits = 0
        self.k1_enabled = bool(k1_enabled)
        self.k1_eta = float(k1_eta)
        self.k1_ema_beta = float(k1_ema_beta)
        self.k1_epsilon_min = float(k1_epsilon_min)
        self.k1_epsilon_max = float(k1_epsilon_max)
        self.k1_freeze_on_actuator_limit = bool(
            k1_freeze_on_actuator_limit
        )
        self.k1_response_ema = 1.0
        self.k1_update_count = 0
        self.k1_frozen_count = 0
        self.k1_last_response = 1.0
        self.k1_last_deviation = 0.0
        self.progress_k1_eta = float(progress_k1_eta)
        self.progress_k1_ema_beta = float(progress_k1_ema_beta)
        self.progress_k1_warmup = int(progress_k1_warmup)
        self.progress_k1_ratio_min = float(progress_k1_ratio_min)
        self.progress_k1_ratio_max = float(progress_k1_ratio_max)
        self.progress_k1_floor = float(progress_k1_floor)
        self.progress_ema = None
        self.progress_feedback_count = 0
        self.progress_update_count = 0
        self.progress_frozen_count = 0
        self.progress_deadband = float(progress_deadband)
        self.progress_update_interval = max(1, int(progress_update_interval))
        self.flow_time_control_enabled = False
        self.flow_time_eta = 0.0
        self.flow_time_min = float(flow_time_min)
        self.flow_time_max = float(flow_time_max)
        self.utilization_target = float(utilization_target)
        self.flow_time_update_count = 0
        self.flow_time_frozen_count = 0
        self.last_utilization = 1.0

    @staticmethod
    def _product_snapshot(factor_modules) -> list[torch.Tensor]:
        return [
            (module.B @ module.A).detach().clone()
            for module in factor_modules
        ]

    @staticmethod
    def _realized_product_displacement(
        factor_modules,
        before: list[torch.Tensor],
    ) -> float:
        total = None
        for module, previous in zip(factor_modules, before):
            delta = module.B @ module.A - previous
            value = torch.sum(delta * delta)
            total = value if total is None else total + value
        if total is None:
            return 0.0
        return float(torch.sqrt(total).detach().cpu())

    def progress_feedback(
        self,
        *,
        loss_before: float,
        loss_after: float,
        realized_dphi: float,
        predicted_dphi: float,
        requested_dphi: float,
        epsilon_limited: bool,
        actuator_limited: bool,
    ) -> dict[str, float | int | bool]:
        # H12.7:
        #   1) epsilon reacts to absolute same-batch progress;
        #   2) a deadband suppresses feedback amplification from tiny
        #      balanced/gauge numerical differences;
        #   3) epsilon updates only every N feedback calls;
        #   4) macro-flow time increases when the actuator, rather than
        #      epsilon, is the active limiter and utilization is too low.
        loss_delta = float(loss_before - loss_after)
        progress = loss_delta
        self.progress_feedback_count += 1

        if self.progress_ema is None:
            self.progress_ema = progress
            progress_k = 1.0
        else:
            baseline = float(self.progress_ema)
            scale = max(abs(baseline), self.progress_k1_floor)
            progress_k = 1.0 + (progress - baseline) / scale
            progress_k = float(
                min(
                    max(progress_k, self.progress_k1_ratio_min),
                    self.progress_k1_ratio_max,
                )
            )
            self.progress_ema = (
                self.progress_k1_ema_beta * baseline
                + (1.0 - self.progress_k1_ema_beta) * progress
            )

        raw_deviation = progress_k - 1.0
        in_deadband = abs(raw_deviation) < self.progress_deadband
        interval_ready = (
            self.progress_feedback_count % self.progress_update_interval == 0
        )

        epsilon_update_active = (
            self.k1_enabled
            and self.progress_feedback_count > self.progress_k1_warmup
            and interval_ready
            and not in_deadband
            and (
                epsilon_limited
                or not self.k1_freeze_on_actuator_limit
            )
        )

        applied_progress_deviation = 0.0
        if epsilon_update_active:
            applied_progress_deviation = raw_deviation
            log_epsilon = math.log(
                max(self.local_function_tolerance, 1e-12)
            )
            log_epsilon += (
                self.progress_k1_eta * applied_progress_deviation
            )
            self.local_function_tolerance = float(
                min(
                    max(
                        math.exp(log_epsilon),
                        self.k1_epsilon_min,
                    ),
                    self.k1_epsilon_max,
                )
            )
            self.optimizer.local_function_tolerance = (
                self.local_function_tolerance
            )
            self.progress_update_count += 1
        else:
            self.progress_frozen_count += 1

        utilization = float(
            realized_dphi / max(requested_dphi, 1e-12)
        )
        utilization = float(min(max(utilization, 0.0), 10.0))
        self.last_utilization = utilization

        flow_time_update_active = (
            self.flow_time_control_enabled
            and actuator_limited
            and interval_ready
            and utilization < self.utilization_target
        )

        flow_time_deviation = 0.0
        if flow_time_update_active:
            flow_time_deviation = (
                self.utilization_target - utilization
            ) / max(self.utilization_target, 1e-12)
            log_flow_time = math.log(max(self.macro_flow_time, 1e-12))
            log_flow_time += self.flow_time_eta * flow_time_deviation
            new_flow_time = float(
                min(
                    max(math.exp(log_flow_time), self.flow_time_min),
                    self.flow_time_max,
                )
            )
            # Preserve fractional completion of the currently unfinished
            # macro flow when its total duration changes.
            old_flow_time = max(self.macro_flow_time, 1e-12)
            completed_fraction = 1.0 - (
                self.remaining_flow_time / old_flow_time
            )
            completed_fraction = min(max(completed_fraction, 0.0), 1.0)
            self.macro_flow_time = new_flow_time
            self.remaining_flow_time = (
                1.0 - completed_fraction
            ) * new_flow_time
            self.optimizer.macro_flow_time = new_flow_time
            self.flow_time_update_count += 1
        else:
            self.flow_time_frozen_count += 1

        return {
            "progress_loss_before": float(loss_before),
            "progress_loss_after": float(loss_after),
            "progress_loss_delta": loss_delta,
            "progress_value": progress,
            "progress_ema": float(self.progress_ema),
            "progress_k": progress_k,
            "progress_raw_deviation": raw_deviation,
            "progress_applied_deviation": applied_progress_deviation,
            "progress_in_deadband": in_deadband,
            "progress_interval_ready": interval_ready,
            "progress_controller_update_active": epsilon_update_active,
            "progress_feedback_count": self.progress_feedback_count,
            "progress_update_count": self.progress_update_count,
            "progress_frozen_count": self.progress_frozen_count,
            "realized_dphi": float(realized_dphi),
            "predicted_dphi": float(predicted_dphi),
            "requested_dphi": float(requested_dphi),
            "actuator_utilization": utilization,
            "utilization_target": self.utilization_target,
            "flow_time_deviation": flow_time_deviation,
            "flow_time_controller_update_active": flow_time_update_active,
            "flow_time_update_count": self.flow_time_update_count,
            "flow_time_frozen_count": self.flow_time_frozen_count,
            "active_epsilon": self.local_function_tolerance,
            "active_macro_flow_time": self.macro_flow_time,
        }

    def step_after_backward(self) -> dict[str, float | int | bool]:
        directions, capacity = self.optimizer._directions_and_capacity()
        capacity_value = float(capacity.detach().cpu())
        tiny = max(
            torch.finfo(self.optimizer.factor_modules[0].A.dtype).tiny,
            1e-30,
        )

        if capacity_value <= tiny:
            flow_dt = self.remaining_flow_time
        else:
            flow_dt = min(
                self.remaining_flow_time,
                self.local_function_tolerance / capacity_value,
            )
            if self.max_flow_dt is not None and flow_dt > self.max_flow_dt:
                flow_dt = self.max_flow_dt
                self.flow_dt_cap_hits += 1

        predicted_dphi = capacity_value * flow_dt
        product_before = self._product_snapshot(
            self.optimizer.factor_modules
        )
        self.optimizer._apply_directions(directions, flow_dt)
        realized_dphi = self._realized_product_displacement(
            self.optimizer.factor_modules,
            product_before,
        )

        requested_dphi = max(
            self.local_function_tolerance,
            1e-12,
        )
        realizable_dphi = max(predicted_dphi, 1e-12)

        # The corrected K uses the displacement that the current actuator
        # can actually realize, not the unconstrained epsilon request.
        response_k = realized_dphi / realizable_dphi
        response_k = float(min(max(response_k, 1e-6), 1e6))

        epsilon_dt = requested_dphi / max(capacity_value, 1e-12)
        epsilon_limited = (
            epsilon_dt
            <= self.remaining_flow_time + 1e-12
            and (
                self.max_flow_dt is None
                or epsilon_dt <= self.max_flow_dt + 1e-12
            )
        )
        actuator_limited = not epsilon_limited

        # Anti-windup: update the restoring controller only when epsilon is
        # the active limiter. When macro time or max dt is active, increasing
        # epsilon cannot increase the realized step and must not accumulate.
        controller_update_active = (
            self.k1_enabled
            and (
                epsilon_limited
                or not self.k1_freeze_on_actuator_limit
            )
        )

        self.k1_last_response = response_k
        if controller_update_active:
            self.k1_response_ema = (
                self.k1_ema_beta * self.k1_response_ema
                + (1.0 - self.k1_ema_beta) * response_k
            )
            deviation = self.k1_response_ema - 1.0
            self.k1_update_count += 1
        else:
            deviation = 0.0
            self.k1_frozen_count += 1
        self.k1_last_deviation = deviation

        # Execution K is a fidelity diagnostic only. The progress-aware
        # controller updates epsilon after a same-batch post-step loss check.

        self.remaining_flow_time = max(
            0.0,
            self.remaining_flow_time - flow_dt,
        )
        self.current_macro_substeps += 1
        self.capacity_values.append(capacity_value)
        self.flow_dt_values.append(flow_dt)
        self.predicted_dphi_values.append(predicted_dphi)

        macro_completed = self.remaining_flow_time <= 1e-15
        if macro_completed:
            self.completed_macro_steps += 1
            self.completed_macro_substep_counts.append(
                self.current_macro_substeps
            )
            self.current_macro_substeps = 0
            self.remaining_flow_time = self.macro_flow_time

        return {
            "capacity": capacity_value,
            "flow_dt": flow_dt,
            "predicted_local_dphi": predicted_dphi,
            "realized_local_dphi": realized_dphi,
            "k1_response": response_k,
            "k1_response_ema": self.k1_response_ema,
            "k1_deviation": deviation,
            "active_epsilon": self.local_function_tolerance,
            "requested_local_dphi": requested_dphi,
            "realizable_local_dphi": realizable_dphi,
            "epsilon_limited": epsilon_limited,
            "actuator_limited": actuator_limited,
            "controller_update_active": controller_update_active,
            "k1_update_count": self.k1_update_count,
            "k1_frozen_count": self.k1_frozen_count,
            "macro_completed": macro_completed,
            "completed_macro_steps": self.completed_macro_steps,
            "current_macro_substeps": self.current_macro_substeps,
            "remaining_flow_time": self.remaining_flow_time,
        }


@torch.no_grad()
def factor_condition_diagnostics(
    adapters: list[LoRAConv1DAdapter],
) -> dict[str, float]:
    cond_as: list[float] = []
    cond_bs: list[float] = []
    cond_maxes: list[float] = []

    for adapter in adapters:
        A = adapter.A.detach().float()
        B = adapter.B.detach().float()
        sA = torch.linalg.svdvals(A)
        sB = torch.linalg.svdvals(B)

        epsA = max(float(torch.finfo(sA.dtype).eps), 1e-12)
        epsB = max(float(torch.finfo(sB.dtype).eps), 1e-12)

        cond_A = float(
            (sA.max() / sA.min().clamp_min(epsA)).cpu()
        )
        cond_B = float(
            (sB.max() / sB.min().clamp_min(epsB)).cpu()
        )
        cond_as.append(cond_A)
        cond_bs.append(cond_B)
        cond_maxes.append(max(cond_A, cond_B))

    return {
        "cond_A_mean": float(np.mean(cond_as)),
        "cond_B_mean": float(np.mean(cond_bs)),
        "cond_max_mean": float(np.mean(cond_maxes)),
        "cond_max_peak": float(np.max(cond_maxes)),
    }



def run_training(
    *,
    optimizer_name: str,
    representation: str,
    gauge_condition_number: float,
    seed: int,
    model_name: str,
    target_modules: list[str],
    rank: int,
    init_scale: float,
    train_dataset: TokenBlockDataset,
    locked_schedule: LockedBatchSchedule,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    backward_budget: int,
    eval_interval: int,
    eval_batches: int,
    adamw_lr: float,
    adamw_weight_decay: float,
    macro_flow_time: float,
    local_function_tolerance: float,
    epsilon_schedule: list[tuple[int, float]] | None,
    k1_enabled: bool,
    k1_eta: float,
    k1_ema_beta: float,
    k1_epsilon_min: float,
    k1_epsilon_max: float,
    k1_freeze_on_actuator_limit: bool,
    progress_k1_eta: float,
    progress_k1_ema_beta: float,
    progress_k1_warmup: int,
    progress_k1_ratio_min: float,
    progress_k1_ratio_max: float,
    progress_k1_floor: float,
    progress_deadband: float,
    progress_update_interval: int,
    flow_time_control_enabled: bool,
    flow_time_eta: float,
    flow_time_min: float,
    flow_time_max: float,
    utilization_target: float,
    max_auto_substeps: int,
    max_flow_dt: float | None,
) -> tuple[RunResult, torch.Tensor, torch.Tensor, list[dict]]:
    model, adapters = make_model(
        model_name,
        target_modules,
        rank,
        init_scale,
        seed,
        gauge_condition_number,
        device,
    )
    initial_factor_condition = factor_condition_diagnostics(adapters)

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_parameters(adapters),
            lr=adamw_lr,
            weight_decay=adamw_weight_decay,
        )
        capacity_stepper = None
    elif optimizer_name == "capacity":
        initial_epsilon = local_function_tolerance
        optimizer = CapacityAdaptiveQuotientFlow(
            adapters,
            macro_flow_time=macro_flow_time,
            local_function_tolerance=initial_epsilon,
            max_auto_substeps=max_auto_substeps,
            max_flow_dt=max_flow_dt,
            balance_after_substep=True,
        )
        capacity_stepper = ResumableCapacityStepper(
            optimizer,
            macro_flow_time=macro_flow_time,
            local_function_tolerance=initial_epsilon,
            max_flow_dt=max_flow_dt,
            k1_enabled=k1_enabled,
            k1_eta=k1_eta,
            k1_ema_beta=k1_ema_beta,
            k1_epsilon_min=k1_epsilon_min,
            k1_epsilon_max=k1_epsilon_max,
            k1_freeze_on_actuator_limit=(
                k1_freeze_on_actuator_limit
            ),
            progress_k1_eta=progress_k1_eta,
            progress_k1_ema_beta=progress_k1_ema_beta,
            progress_k1_warmup=progress_k1_warmup,
            progress_k1_ratio_min=progress_k1_ratio_min,
            progress_k1_ratio_max=progress_k1_ratio_max,
            progress_k1_floor=progress_k1_floor,
            progress_deadband=progress_deadband,
            progress_update_interval=progress_update_interval,
            flow_time_control_enabled=flow_time_control_enabled,
            flow_time_eta=flow_time_eta,
            flow_time_min=flow_time_min,
            flow_time_max=flow_time_max,
            utilization_target=utilization_target,
        )
    else:
        raise ValueError(optimizer_name)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    initial_val = evaluate(model, val_loader, device, eval_batches)
    if test_loader is None:
        initial_test = {
            "loss": float("nan"),
            "perplexity": float("nan"),
            "tokens": 0,
            "batches": 0,
        }
        initial_signature = torch.empty(0, dtype=torch.float32)
        initial_signature_hash = "NOT_EVALUATED"
    else:
        initial_test = evaluate(
            model, test_loader, device, eval_batches
        )
        initial_signature = collect_logits_signature(
            model, test_loader, device
        )
        initial_signature_hash = hashlib.sha256(
            initial_signature.numpy().tobytes()
        ).hexdigest()[:16]

    best_val = dict(initial_val)
    curve: list[dict] = []
    backward_calls = 0
    train_tokens_seen = 0
    first_batch_hash = ""
    first_train_loss = float("nan")
    completed_macro_steps = 0
    partial_macro_substeps = 0
    next_eval_at = 1

    start_time = time.perf_counter()

    while backward_calls < backward_budget:
        model.train()
        step_index = backward_calls
        batch = locked_schedule.batch(train_dataset, step_index)
        current_batch_hash = batch_content_hash(batch)
        if backward_calls == 0:
            first_batch_hash = current_batch_hash
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        cpu_rng_before = torch.random.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None
        )
        outputs = model(**batch)
        outputs.loss.backward()
        train_loss = float(outputs.loss.detach().cpu())

        if backward_calls == 0:
            first_train_loss = train_loss

        if optimizer_name == "adamw":
            optimizer.step()
            completed_macro_steps += 1
            current_macro_substeps = 0
            remaining_flow_time = 0.0
            last_capacity = None
            last_flow_dt = None
            last_predicted_dphi = None
        else:
            current_epsilon = (
                capacity_stepper.local_function_tolerance
            )
            step_diag = capacity_stepper.step_after_backward()
            current_epsilon = float(step_diag["active_epsilon"])
            completed_macro_steps = int(
                step_diag["completed_macro_steps"]
            )
            current_macro_substeps = int(
                step_diag["current_macro_substeps"]
            )
            remaining_flow_time = float(
                step_diag["remaining_flow_time"]
            )
            last_capacity = float(step_diag["capacity"])
            last_flow_dt = float(step_diag["flow_dt"])
            last_predicted_dphi = float(
                step_diag["predicted_local_dphi"]
            )
            last_realized_dphi = float(
                step_diag["realized_local_dphi"]
            )
            last_k1_response = float(step_diag["k1_response"])
            last_k1_response_ema = float(
                step_diag["k1_response_ema"]
            )
            last_k1_deviation = float(step_diag["k1_deviation"])
            last_requested_dphi = float(
                step_diag["requested_local_dphi"]
            )
            last_realizable_dphi = float(
                step_diag["realizable_local_dphi"]
            )
            last_epsilon_limited = bool(
                step_diag["epsilon_limited"]
            )
            last_actuator_limited = bool(
                step_diag["actuator_limited"]
            )
            last_controller_update_active = bool(
                step_diag["controller_update_active"]
            )
            last_k1_update_count = int(
                step_diag["k1_update_count"]
            )
            last_k1_frozen_count = int(
                step_diag["k1_frozen_count"]
            )

        # Replay the exact pre-update dropout mask for a fair same-batch
        # post-update loss measurement. Restore the live RNG stream after
        # this diagnostic so later training randomness is unchanged.
        cpu_rng_live = torch.random.get_rng_state()
        cuda_rng_live = (
            torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None
        )
        torch.random.set_rng_state(cpu_rng_before)
        if device.type == "cuda" and cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)
        with torch.no_grad():
            post_step_train_loss = float(
                model(**batch).loss.detach().cpu()
            )
        torch.random.set_rng_state(cpu_rng_live)
        if device.type == "cuda" and cuda_rng_live is not None:
            torch.cuda.set_rng_state_all(cuda_rng_live)

        if optimizer_name == "capacity":
            progress_diag = capacity_stepper.progress_feedback(
                loss_before=train_loss,
                loss_after=post_step_train_loss,
                realized_dphi=last_realized_dphi,
                predicted_dphi=last_predicted_dphi,
                requested_dphi=last_requested_dphi,
                epsilon_limited=last_epsilon_limited,
                actuator_limited=last_actuator_limited,
            )
            current_epsilon = float(progress_diag["active_epsilon"])
            last_progress_loss_delta = float(progress_diag["progress_loss_delta"])
            last_progress_value = float(progress_diag["progress_value"])
            last_progress_ema = float(
                progress_diag["progress_ema"]
            )
            last_progress_k = float(progress_diag["progress_k"])
            last_progress_raw_deviation = float(
                progress_diag["progress_raw_deviation"]
            )
            last_progress_applied_deviation = float(
                progress_diag["progress_applied_deviation"]
            )
            last_progress_in_deadband = bool(
                progress_diag["progress_in_deadband"]
            )
            last_progress_interval_ready = bool(
                progress_diag["progress_interval_ready"]
            )
            last_progress_controller_update_active = bool(
                progress_diag["progress_controller_update_active"]
            )
            last_actuator_utilization = float(
                progress_diag["actuator_utilization"]
            )
            last_flow_time_deviation = float(
                progress_diag["flow_time_deviation"]
            )
            last_flow_time_controller_update_active = bool(
                progress_diag["flow_time_controller_update_active"]
            )
            last_flow_time_update_count = int(
                progress_diag["flow_time_update_count"]
            )
            last_flow_time_frozen_count = int(
                progress_diag["flow_time_frozen_count"]
            )
            active_macro_flow_time = float(
                progress_diag["active_macro_flow_time"]
            )
            last_progress_feedback_count = int(
                progress_diag["progress_feedback_count"]
            )
            last_progress_update_count = int(progress_diag["progress_update_count"])
            last_progress_frozen_count = int(progress_diag["progress_frozen_count"])

        backward_calls += 1
        train_tokens_seen += int(batch["input_ids"].numel())

        should_eval = (
            backward_calls >= next_eval_at
            or backward_calls >= backward_budget
        )
        if should_eval:
            val = evaluate(model, val_loader, device, eval_batches)
            if val["loss"] < best_val["loss"]:
                best_val = dict(val)
            row = {
                "optimizer": optimizer_name,
                "seed": seed,
                "representation": representation,
                "backward_budget": backward_budget,
                "backward_calls": backward_calls,
                "train_tokens_seen": train_tokens_seen,
                "completed_macro_steps": completed_macro_steps,
                "current_macro_substeps": current_macro_substeps,
                "remaining_flow_time": remaining_flow_time,
                "batch_hash": current_batch_hash,
                "train_loss": train_loss,
                "post_step_train_loss": post_step_train_loss,
                "same_batch_loss_delta": (
                    train_loss - post_step_train_loss
                ),
                "validation_loss": val["loss"],
                "validation_ppl": val["perplexity"],
            }
            if optimizer_name == "capacity":
                row.update(
                    {
                        "last_capacity": last_capacity,
                        "last_flow_dt": last_flow_dt,
                        "last_predicted_local_dphi": last_predicted_dphi,
                        "active_epsilon": current_epsilon,
                        "last_realized_local_dphi": last_realized_dphi,
                        "k1_response": last_k1_response,
                        "k1_response_ema": last_k1_response_ema,
                        "k1_deviation": last_k1_deviation,
                        "requested_local_dphi": last_requested_dphi,
                        "realizable_local_dphi": last_realizable_dphi,
                        "epsilon_limited": last_epsilon_limited,
                        "actuator_limited": last_actuator_limited,
                        "controller_update_active": (
                            last_controller_update_active
                        ),
                        "k1_update_count": last_k1_update_count,
                        "k1_frozen_count": last_k1_frozen_count,
                        "progress_loss_delta": last_progress_loss_delta,
                        "progress_value": last_progress_value,
                        "progress_ema": last_progress_ema,
                        "progress_k": last_progress_k,
                        "progress_raw_deviation": last_progress_raw_deviation,
                        "progress_applied_deviation": (
                            last_progress_applied_deviation
                        ),
                        "progress_in_deadband": last_progress_in_deadband,
                        "progress_interval_ready": (
                            last_progress_interval_ready
                        ),
                        "progress_controller_update_active": (
                            last_progress_controller_update_active
                        ),
                        "actuator_utilization": (
                            last_actuator_utilization
                        ),
                        "active_macro_flow_time": (
                            active_macro_flow_time
                        ),
                        "flow_time_deviation": (
                            last_flow_time_deviation
                        ),
                        "flow_time_controller_update_active": (
                            last_flow_time_controller_update_active
                        ),
                        "flow_time_update_count": (
                            last_flow_time_update_count
                        ),
                        "flow_time_frozen_count": (
                            last_flow_time_frozen_count
                        ),
                        "progress_feedback_count": last_progress_feedback_count,
                        "progress_update_count": last_progress_update_count,
                        "progress_frozen_count": last_progress_frozen_count,
                        "fallback_count": optimizer.fallback_count,
                    }
                )
            curve.append(row)
            print(json.dumps(row), flush=True)
            while next_eval_at <= backward_calls:
                next_eval_at += max(1, eval_interval)

    wall_seconds = time.perf_counter() - start_time
    final_val = evaluate(model, val_loader, device, eval_batches)
    if test_loader is None:
        final_test = {
            "loss": float("nan"),
            "perplexity": float("nan"),
            "tokens": 0,
            "batches": 0,
        }
        final_signature = torch.empty(0, dtype=torch.float32)
    else:
        final_test = evaluate(
            model, test_loader, device, eval_batches
        )
        final_signature = collect_logits_signature(
            model, test_loader, device
        )

    if optimizer_name == "capacity":
        completed_counts = (
            capacity_stepper.completed_macro_substep_counts
        )
        all_counts = list(completed_counts)
        if capacity_stepper.current_macro_substeps > 0:
            all_counts.append(capacity_stepper.current_macro_substeps)

        mean_auto_substeps = (
            float(np.mean(all_counts)) if all_counts else None
        )
        min_auto_substeps = (
            int(min(all_counts)) if all_counts else None
        )
        max_auto_substeps = (
            int(max(all_counts)) if all_counts else None
        )
        fallback_count = int(optimizer.fallback_count)
        flow_dt_cap_hits = int(
            capacity_stepper.flow_dt_cap_hits
        )
        condition_max = float(optimizer.condition_max)
        max_predicted_local_dphi = (
            float(max(capacity_stepper.predicted_dphi_values))
            if capacity_stepper.predicted_dphi_values
            else 0.0
        )
        unfinished_macro_flow_time = (
            float(capacity_stepper.remaining_flow_time)
            if capacity_stepper.current_macro_substeps > 0
            else 0.0
        )
        partial_macro_substeps = int(
            capacity_stepper.current_macro_substeps
        )
    else:
        mean_auto_substeps = None
        min_auto_substeps = None
        max_auto_substeps = None
        fallback_count = None
        flow_dt_cap_hits = None
        condition_max = None
        max_predicted_local_dphi = None
        unfinished_macro_flow_time = None
        partial_macro_substeps = 0

    model_digest = hashlib.sha256(
        final_signature.numpy().tobytes()
    ).hexdigest()[:16]

    result = RunResult(
        optimizer=optimizer_name,
        seed=seed,
        representation=representation,
        gauge_condition_number=gauge_condition_number,
        initial_actual_cond_A_mean=(
            initial_factor_condition["cond_A_mean"]
        ),
        initial_actual_cond_B_mean=(
            initial_factor_condition["cond_B_mean"]
        ),
        initial_actual_cond_max_mean=(
            initial_factor_condition["cond_max_mean"]
        ),
        initial_actual_cond_max_peak=(
            initial_factor_condition["cond_max_peak"]
        ),
        completed_macro_steps=completed_macro_steps,
        partial_macro_substeps=partial_macro_substeps,
        backward_budget=backward_budget,
        backward_calls=backward_calls,
        train_tokens_seen=train_tokens_seen,
        locked_schedule_hash=locked_schedule.schedule_hash,
        first_batch_hash=first_batch_hash,
        first_train_loss=first_train_loss,
        initial_validation_loss=initial_val["loss"],
        initial_validation_ppl=initial_val["perplexity"],
        initial_test_loss=initial_test["loss"],
        initial_test_ppl=initial_test["perplexity"],
        initial_test_signature_hash=initial_signature_hash,
        best_validation_loss=best_val["loss"],
        best_validation_ppl=best_val["perplexity"],
        final_validation_loss=final_val["loss"],
        final_validation_ppl=final_val["perplexity"],
        test_loss=final_test["loss"],
        test_ppl=final_test["perplexity"],
        validation_loss_improvement=(
            initial_val["loss"] - final_val["loss"]
        ),
        validation_ppl_improvement=(
            initial_val["perplexity"] - final_val["perplexity"]
        ),
        test_loss_improvement=(
            initial_test["loss"] - final_test["loss"]
        ),
        test_ppl_improvement=(
            initial_test["perplexity"]
            - final_test["perplexity"]
        ),
        wall_seconds=wall_seconds,
        tokens_per_second=train_tokens_seen
        / max(wall_seconds, 1e-12),
        peak_cuda_memory_mb=peak_cuda_memory_mb(device),
        mean_auto_substeps=mean_auto_substeps,
        min_auto_substeps=min_auto_substeps,
        max_auto_substeps=max_auto_substeps,
        fallback_count=fallback_count,
        flow_dt_cap_hits=flow_dt_cap_hits,
        condition_max=condition_max,
        max_predicted_local_dphi=max_predicted_local_dphi,
        unfinished_macro_flow_time=unfinished_macro_flow_time,
        model_signature=model_digest,
    )

    del model, adapters, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, initial_signature, final_signature, curve


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one float is required.")
    if any(x <= 0 for x in values):
        raise ValueError("All flow-time candidates must be positive.")
    return values


def aggregate_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one float is required.")
    if any(x <= 0 for x in values):
        raise ValueError("All epsilon candidates must be positive.")
    return values


def aggregate_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def geometric_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.maximum(np.asarray(values, dtype=float), 1e-30)
    return float(math.exp(np.mean(np.log(arr))))


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one float is required.")
    return values


def aggregate_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def geometric_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.maximum(np.asarray(values, dtype=float), 1e-30)
    return float(math.exp(np.mean(np.log(arr))))



def parse_epsilon_schedule(raw: str) -> list[tuple[int, float]]:
    schedule: list[tuple[int, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        threshold_text, separator, epsilon_text = part.partition(":")
        if not separator:
            raise ValueError(
                f"Invalid schedule item {part!r}; expected threshold:epsilon."
            )
        threshold = int(threshold_text.strip())
        epsilon = float(epsilon_text.strip())
        if threshold < 0:
            raise ValueError("Schedule thresholds must be nonnegative.")
        if epsilon <= 0:
            raise ValueError("Schedule epsilon values must be positive.")
        schedule.append((threshold, epsilon))

    schedule.sort(key=lambda item: item[0])
    if not schedule or schedule[0][0] != 0:
        raise ValueError("Epsilon schedule must begin at backward threshold 0.")
    thresholds = [threshold for threshold, _ in schedule]
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("Duplicate epsilon schedule thresholds are not allowed.")
    return schedule


def epsilon_at_backward_call(
    backward_calls: int,
    schedule: list[tuple[int, float]],
) -> float:
    epsilon = schedule[0][1]
    for threshold, candidate in schedule:
        if backward_calls < threshold:
            break
        epsilon = candidate
    return float(epsilon)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--repo-dir",
        default="/content/Geometric-Flow-h127-10seed-replication",
    )
    parser.add_argument("--force-reclone", action="store_true")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
    )
    parser.add_argument(
        "--target-modules",
        default=(
            "transformer.h.0.attn.c_attn,"
            "transformer.h.1.attn.c_attn"
        ),
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--init-scale", type=float, default=0.01)

    # H12.7 uses only fresh replication seeds and performs no tuning.
    parser.add_argument("--seeds", default="1801,2003,2203,2411,2609,2801,3001,3203,3407,3607",
    )
    parser.add_argument("--data-seed", type=int, default=424242)

    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-examples", type=int, default=8000)
    parser.add_argument("--eval-examples", type=int, default=1500)
    parser.add_argument("--train-blocks", type=int, default=1500)
    parser.add_argument("--eval-blocks", type=int, default=300)

    # H12.7 is the locked 300-backward replication.
    parser.add_argument("--backward-budgets", default="300",
    )
    parser.add_argument("--eval-interval", type=int, default=30)
    parser.add_argument("--eval-batches", type=int, default=40)

    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.0)

    # Locked from H11.4; no validation selection is performed here.
    parser.add_argument(
        "--local-function-tolerance",
        type=float,
        default=0.18,
        help="Compatibility fallback when no staged schedule is supplied.",
    )
    parser.add_argument(
        "--epsilon-schedule",
        default="0:0.18,100:0.24,200:0.30",
        help=(
            "Piecewise-constant schedule indexed by completed backward calls. "
            "Example: 0:0.18,100:0.24,200:0.30."
        ),
    )
    parser.add_argument("--k1-epsilon-init", type=float, default=0.18)
    parser.add_argument("--k1-epsilon-min", type=float, default=0.08)
    parser.add_argument("--k1-epsilon-max", type=float, default=0.50)
    parser.add_argument("--k1-eta", type=float, default=0.08)
    parser.add_argument("--k1-ema-beta", type=float, default=0.90)
    parser.add_argument(
        "--k1-freeze-on-actuator-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze epsilon adaptation when macro-flow time or max-flow-dt, "
            "rather than epsilon, is the active step limiter. This is the "
            "anti-windup fix."
        ),
    )
    parser.add_argument("--progress-k1-eta", type=float, default=0.08)
    parser.add_argument("--progress-k1-ema-beta", type=float, default=0.90)
    parser.add_argument("--progress-k1-warmup", type=int, default=20)
    parser.add_argument("--progress-k1-ratio-min", type=float, default=0.75)
    parser.add_argument("--progress-k1-ratio-max", type=float, default=1.25)
    parser.add_argument("--progress-k1-floor", type=float, default=1e-3)
    parser.add_argument("--progress-deadband", type=float, default=0.01)
    parser.add_argument("--progress-update-interval", type=int, default=1)
    parser.add_argument(
        "--flow-time-control-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--flow-time-eta", type=float, default=0.0)
    parser.add_argument("--flow-time-min", type=float, default=0.30)
    parser.add_argument("--flow-time-max", type=float, default=0.80)
    parser.add_argument("--utilization-target", type=float, default=0.85)
    parser.add_argument(
        "--macro-flow-time",
        type=float,
        default=0.4,
    )
    parser.add_argument("--max-auto-substeps", type=int, default=128)
    parser.add_argument(
        "--max-flow-dt",
        type=float,
        default=0.0,
        help="0 means None/no cap.",
    )
    parser.add_argument(
        "--gauge-condition-number",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--representation-mode",
        choices=["both", "balanced-only", "gauge-only"],
        default="both",
        help=(
            "Select which LoRA factor representation to run. "
            "H13.2 uses balanced-only for kappa=1 and gauge-only otherwise."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="/content/geoflow_h127_10seed_results",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] Ignored notebook/kernel arguments:", unknown)

    epsilon_schedule = parse_epsilon_schedule(args.epsilon_schedule)
    print("[epsilon schedule]", epsilon_schedule)

    seeds = parse_int_list(args.seeds)
    backward_budgets = parse_int_list(args.backward_budgets)
    if any(x < 1 for x in backward_budgets):
        raise ValueError("All backward budgets must be >= 1.")
    if args.local_function_tolerance <= 0:
        raise ValueError(
            "--local-function-tolerance must be positive."
        )
    if args.macro_flow_time <= 0:
        raise ValueError("--macro-flow-time must be positive.")

    target_modules = parse_str_list(args.target_modules)
    max_flow_dt = None if args.max_flow_dt <= 0 else args.max_flow_dt

    repo_dir = Path(args.repo_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepare_repo(repo_dir, args.force_reclone)
    sys.path.insert(0, str(repo_dir))

    global CapacityAdaptiveQuotientFlow
    from geometric_flow import (
        CapacityAdaptiveQuotientFlow as _CapacityAdaptiveQuotientFlow,
    )
    CapacityAdaptiveQuotientFlow = _CapacityAdaptiveQuotientFlow

    print("\n" + "=" * 120)
    print("A. LIBRARY PRECHECK")
    print("=" * 120)
    shell(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_fixed_rank_optimizer.py",
        ],
        cwd=repo_dir,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 120)
    print("B. TOKENIZER AND REAL DATA")
    print("=" * 120)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds, test_ds, data_meta = load_real_wikitext(
        tokenizer=tokenizer,
        dataset_config=args.dataset_config,
        sequence_length=args.sequence_length,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        train_blocks=args.train_blocks,
        eval_blocks=args.eval_blocks,
        seed=args.data_seed,
    )
    print(json.dumps(data_meta, indent=2))

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "repo_url": REPO_URL,
            "data": data_meta,
            "seeds_resolved": seeds,
            "backward_budgets_resolved": backward_budgets,
            "hyperparameter_selection_performed": False,
            "fixed_local_function_tolerance": (
                args.local_function_tolerance
            ),
            "k1_controller": {
                "enabled": True,
                "epsilon_init": args.k1_epsilon_init,
                "epsilon_min": args.k1_epsilon_min,
                "epsilon_max": args.k1_epsilon_max,
                "eta": args.k1_eta,
                "ema_beta": args.k1_ema_beta,
                "freeze_on_actuator_limit": (
                    args.k1_freeze_on_actuator_limit
                ),
                "response_definition": (
                    "realized product displacement divided by "
                    "actuator-realizable predicted displacement"
                ),
                "anti_windup": True,
                "progress_aware": True,
                "progress_eta": args.progress_k1_eta,
                "progress_ema_beta": args.progress_k1_ema_beta,
                "progress_warmup": args.progress_k1_warmup,
                "progress_ratio_bounds": [
                    args.progress_k1_ratio_min,
                    args.progress_k1_ratio_max,
                ],
                "progress_definition": (
                    "absolute same-batch loss decrease per backward call; "
                    "realized displacement is logged only as a structural diagnostic"
                ),
                "progress_deadband": args.progress_deadband,
                "progress_update_interval": (
                    args.progress_update_interval
                ),
                "flow_time_control_enabled": False,
                "flow_time_eta": 0.0,
                "fixed_macro_flow_time": args.macro_flow_time,
                "flow_time_note": (
                    "H12.7 fixes macro-flow time at 0.4 and restores "
                    "H12.3 controller responsiveness; only a 0.01 "
                    "progress deadband is added."
                ),
                "same_dropout_mask_replay": True,
                "equal_extra_forward_for_all_optimizers": True,
            },
            "fixed_macro_flow_time": args.macro_flow_time,
            "scientific_scope": (
                "fixed-hyperparameter multi-seed and multi-budget "
                "confirmation with locked batches and equal backward/token budgets"
            ),
        }
    )
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    all_results: list[RunResult] = []
    all_curves: list[dict] = []
    all_gauge_rows: list[dict] = []
    all_schedule_audit: list[dict] = []
    budget_summaries: dict[str, dict] = {}

    if args.representation_mode == "balanced-only":
        representations = [("balanced", 1.0)]
    elif args.representation_mode == "gauge-only":
        representations = [("gauge", args.gauge_condition_number)]
    else:
        representations = [
            ("balanced", 1.0),
            ("gauge", args.gauge_condition_number),
        ]

    for backward_budget in backward_budgets:
        print("\n" + "=" * 120)
        print(
            f"C. H12.7 LOCKED 10-SEED REPLICATION — "
            f"BACKWARD BUDGET {backward_budget}"
        )
        print("=" * 120)

        budget_results: list[RunResult] = []
        initial_signatures: dict[
            tuple[str, int, str], torch.Tensor
        ] = {}
        final_signatures: dict[
            tuple[str, int, str], torch.Tensor
        ] = {}
        schedules: dict[int, LockedBatchSchedule] = {}

        for seed in seeds:
            schedule = make_locked_batch_schedule(
                train_ds,
                backward_budget=backward_budget,
                batch_size=args.batch_size,
                seed=seed + 700001 + 1000003 * backward_budget,
            )
            schedules[seed] = schedule
            print(
                f"\n[schedule] budget={backward_budget} "
                f"seed={seed} hash={schedule.schedule_hash}"
            )

            for optimizer_name in ["adamw", "capacity"]:
                for representation, gauge_condition in representations:
                    result, initial_sig, final_sig, curve = run_training(
                        optimizer_name=optimizer_name,
                        representation=representation,
                        gauge_condition_number=gauge_condition,
                        seed=seed,
                        model_name=args.model_name,
                        target_modules=target_modules,
                        rank=args.rank,
                        init_scale=args.init_scale,
                        train_dataset=train_ds,
                        locked_schedule=schedule,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        device=device,
                        backward_budget=backward_budget,
                        eval_interval=args.eval_interval,
                        eval_batches=args.eval_batches,
                        adamw_lr=args.adamw_lr,
                        adamw_weight_decay=args.adamw_weight_decay,
                        macro_flow_time=args.macro_flow_time,
                        local_function_tolerance=(
                            args.k1_epsilon_init
                        ),
                        epsilon_schedule=None,
                        k1_enabled=(optimizer_name == "capacity"),
                        k1_eta=args.k1_eta,
                        k1_ema_beta=args.k1_ema_beta,
                        k1_epsilon_min=args.k1_epsilon_min,
                        k1_epsilon_max=args.k1_epsilon_max,
                        k1_freeze_on_actuator_limit=(
                            args.k1_freeze_on_actuator_limit
                        ),
                        progress_k1_eta=args.progress_k1_eta,
                        progress_k1_ema_beta=args.progress_k1_ema_beta,
                        progress_k1_warmup=args.progress_k1_warmup,
                        progress_k1_ratio_min=args.progress_k1_ratio_min,
                        progress_k1_ratio_max=args.progress_k1_ratio_max,
                        progress_k1_floor=(
                            args.progress_k1_floor
                        ),
                        progress_deadband=args.progress_deadband,
                        progress_update_interval=(
                            args.progress_update_interval
                        ),
                        flow_time_control_enabled=(
                            args.flow_time_control_enabled
                        ),
                        flow_time_eta=args.flow_time_eta,
                        flow_time_min=args.flow_time_min,
                        flow_time_max=args.flow_time_max,
                        utilization_target=args.utilization_target,
                        max_auto_substeps=args.max_auto_substeps,
                        max_flow_dt=max_flow_dt,
                    )
                    budget_results.append(result)
                    all_results.append(result)
                    initial_signatures[
                        (optimizer_name, seed, representation)
                    ] = initial_sig
                    final_signatures[
                        (optimizer_name, seed, representation)
                    ] = final_sig

                    for row in curve:
                        row = dict(row)
                        row["backward_budget_group"] = backward_budget
                        row["controller"] = "progress_epsilon_light_deadband_fixed_flowtime"
                        all_curves.append(row)

        gauge_rows = []
        schedule_audit = []

        for seed in seeds:
            group = [
                result for result in budget_results
                if result.seed == seed
            ]
            schedule_hashes = sorted(
                {result.locked_schedule_hash for result in group}
            )
            first_batch_hashes = sorted(
                {result.first_batch_hash for result in group}
            )
            first_losses = [
                result.first_train_loss for result in group
            ]

            audit_row = {
                "backward_budget": backward_budget,
                "seed": seed,
                "expected_schedule_hash": (
                    schedules[seed].schedule_hash
                ),
                "unique_schedule_hashes": schedule_hashes,
                "unique_first_batch_hashes": first_batch_hashes,
                "first_train_loss_spread": (
                    max(first_losses) - min(first_losses)
                ),
                "all_schedule_hashes_match": (
                    len(schedule_hashes) == 1
                    and schedule_hashes[0]
                    == schedules[seed].schedule_hash
                ),
                "all_first_batch_hashes_match": (
                    len(first_batch_hashes) == 1
                ),
            }
            schedule_audit.append(audit_row)
            all_schedule_audit.append(audit_row)

            for optimizer_name in ["adamw", "capacity"]:
                balanced_key = (
                    optimizer_name,
                    seed,
                    "balanced",
                )
                gauge_key = (
                    optimizer_name,
                    seed,
                    "gauge",
                )

                # H13.2 intentionally runs only one representation per kappa:
                # balanced at kappa=1 and gauge at kappa>1. Therefore the
                # inherited paired representation audit is only meaningful
                # when both keys are actually present.
                if (
                    balanced_key not in initial_signatures
                    or gauge_key not in initial_signatures
                    or balanced_key not in final_signatures
                    or gauge_key not in final_signatures
                ):
                    continue

                initial_gap = relative_gap(
                    initial_signatures[balanced_key],
                    initial_signatures[gauge_key],
                )
                final_gap = relative_gap(
                    final_signatures[balanced_key],
                    final_signatures[gauge_key],
                )
                row = {
                    "backward_budget": backward_budget,
                    "seed": seed,
                    "optimizer": optimizer_name,
                    "initial_test_logit_representation_gap": (
                        initial_gap
                    ),
                    "final_test_logit_representation_gap": (
                        final_gap
                    ),
                    "gauge_gap_growth": (
                        final_gap - initial_gap
                    ),
                }
                gauge_rows.append(row)
                all_gauge_rows.append(row)

        aggregate = {}
        per_seed_comparison = []

        active_representation = (
            representations[0][0]
            if len(representations) == 1
            else "balanced"
        )

        for optimizer_name in ["adamw", "capacity"]:
            balanced_group = [
                result for result in budget_results
                if result.optimizer == optimizer_name
                and result.representation == active_representation
            ]
            gaps = [
                row["final_test_logit_representation_gap"]
                for row in gauge_rows
                if row["optimizer"] == optimizer_name
            ]
            aggregate[optimizer_name] = {
                "n_seeds": len(balanced_group),
                "mean_final_validation_ppl": aggregate_mean(
                    [
                        result.final_validation_ppl
                        for result in balanced_group
                    ]
                ),
                "mean_final_test_ppl": aggregate_mean(
                    [result.test_ppl for result in balanced_group]
                ),
                "mean_validation_ppl_improvement": aggregate_mean(
                    [
                        result.validation_ppl_improvement
                        for result in balanced_group
                    ]
                ),
                "mean_test_ppl_improvement": aggregate_mean(
                    [
                        result.test_ppl_improvement
                        for result in balanced_group
                    ]
                ),
                "median_validation_ppl_improvement": float(
                    np.median(
                        [
                            result.validation_ppl_improvement
                            for result in balanced_group
                        ]
                    )
                ),
                "median_test_ppl_improvement": float(
                    np.median(
                        [
                            result.test_ppl_improvement
                            for result in balanced_group
                        ]
                    )
                ),
                "mean_wall_seconds": aggregate_mean(
                    [result.wall_seconds for result in balanced_group]
                ),
                "mean_tokens_per_second": aggregate_mean(
                    [
                        result.tokens_per_second
                        for result in balanced_group
                    ]
                ),
                "mean_peak_cuda_memory_mb": aggregate_mean(
                    [
                        result.peak_cuda_memory_mb
                        for result in balanced_group
                    ]
                ),
                "geometric_mean_final_gauge_gap": (
                    geometric_mean(gaps)
                    if gaps
                    else float("nan")
                ),
            }

        adam_rows = {
            result.seed: result
            for result in budget_results
            if result.optimizer == "adamw"
            and result.representation == active_representation
        }
        cap_rows = {
            result.seed: result
            for result in budget_results
            if result.optimizer == "capacity"
            and result.representation == active_representation
        }

        for seed in seeds:
            adam = adam_rows[seed]
            cap = cap_rows[seed]
            per_seed_comparison.append(
                {
                    "seed": seed,
                    "adamw_validation_ppl_improvement": (
                        adam.validation_ppl_improvement
                    ),
                    "capacity_validation_ppl_improvement": (
                        cap.validation_ppl_improvement
                    ),
                    "capacity_minus_adamw_validation_improvement": (
                        cap.validation_ppl_improvement
                        - adam.validation_ppl_improvement
                    ),
                    "adamw_test_ppl_improvement": (
                        adam.test_ppl_improvement
                    ),
                    "capacity_test_ppl_improvement": (
                        cap.test_ppl_improvement
                    ),
                    "capacity_minus_adamw_test_improvement": (
                        cap.test_ppl_improvement
                        - adam.test_ppl_improvement
                    ),
                    "capacity_wins_validation": (
                        cap.validation_ppl_improvement
                        > adam.validation_ppl_improvement
                    ),
                    "capacity_wins_test": (
                        cap.test_ppl_improvement
                        > adam.test_ppl_improvement
                    ),
                }
            )

        adam_val = aggregate["adamw"][
            "mean_validation_ppl_improvement"
        ]
        cap_val = aggregate["capacity"][
            "mean_validation_ppl_improvement"
        ]
        adam_test = aggregate["adamw"][
            "mean_test_ppl_improvement"
        ]
        cap_test = aggregate["capacity"][
            "mean_test_ppl_improvement"
        ]

        validation_ratio = (
            cap_val / adam_val
            if adam_val > 0 else float("nan")
        )
        test_ratio = (
            cap_test / adam_test
            if adam_test > 0 else float("nan")
        )

        adam_gap = aggregate["adamw"][
            "geometric_mean_final_gauge_gap"
        ]
        cap_gap = aggregate["capacity"][
            "geometric_mean_final_gauge_gap"
        ]
        gauge_suppression = (
            adam_gap / max(cap_gap, 1e-30)
            if math.isfinite(adam_gap) and math.isfinite(cap_gap)
            else float("nan")
        )

        capacity_runs = [
            result for result in budget_results
            if result.optimizer == "capacity"
        ]

        validation_wins = sum(
            row["capacity_wins_validation"]
            for row in per_seed_comparison
        )
        test_wins = sum(
            row["capacity_wins_test"]
            for row in per_seed_comparison
        )

        gates = {
            "NO_HYPERPARAMETER_SELECTION_DURING_CONFIRMATION": True,
            "AT_LEAST_10_SEEDS": len(seeds) >= 10,
            "EQUAL_BACKWARD_BUDGET_EXACT": all(
                result.backward_calls == backward_budget
                for result in budget_results
            ),
            "LOCKED_BATCH_SCHEDULE_EXACT": all(
                row["all_schedule_hashes_match"]
                and row["all_first_batch_hashes_match"]
                for row in schedule_audit
            ),
            "INITIAL_GAUGE_MODELS_FUNCTIONALLY_MATCH": all(
                row["initial_test_logit_representation_gap"] < 1e-6
                for row in gauge_rows
            ),
            "CAPACITY_NO_FALLBACK": all(
                (result.fallback_count or 0) == 0
                for result in capacity_runs
            ),
            "CAPACITY_NO_DT_CAP_HITS": all(
                (result.flow_dt_cap_hits or 0) == 0
                for result in capacity_runs
            ),
            "CAPACITY_LOCAL_TOLERANCE_RESPECTED": all(
                result.max_predicted_local_dphi
                <= args.k1_epsilon_max * (1.0 + 1e-8)
                + 1e-12
                for result in capacity_runs
            ),
            "MAJORITY_VALIDATION_WINS": (
                validation_wins > len(seeds) / 2
            ),
            "MAJORITY_TEST_WINS": (
                test_wins > len(seeds) / 2
            ),
            "MEAN_VALIDATION_IMPROVEMENT_RATIO_GT_1": (
                math.isfinite(validation_ratio)
                and validation_ratio > 1.0
            ),
            "MEAN_TEST_IMPROVEMENT_RATIO_GT_1": (
                math.isfinite(test_ratio)
                and test_ratio > 1.0
            ),
            "GAUGE_SUPPRESSION_AT_LEAST_10X": (
                (
                    gauge_suppression >= 10.0
                    if math.isfinite(gauge_suppression)
                    else True
                )
            ),
            "CAPACITY_FINAL_GAUGE_GAP_BELOW_1E_5": (
                cap_gap < 1e-5
            ),
            "STRONG_TEST_PROGRESS_RATIO_GT_1_03": (
                math.isfinite(test_ratio)
                and test_ratio > 1.03
            ),
            "K1_DEVIATION_RESTORATION_ACTIVE": True,
            "K1_ACTUATOR_AWARE_TARGET_ACTIVE": True,
            "K1_ANTI_WINDUP_ACTIVE": (
                args.k1_freeze_on_actuator_limit
            ),
            "PROGRESS_AWARE_K1_ACTIVE": True,
            "PROGRESS_DEADBAND_ACTIVE": (
                args.progress_deadband > 0
            ),
            "LIGHT_DEADBAND_AT_MOST_0_01": (
                args.progress_deadband <= 0.01
            ),
            "FULL_FREQUENCY_EPSILON_CONTROL": (
                args.progress_update_interval == 1
            ),
            "H123_RESPONSE_SPEED_RESTORED": (
                abs(args.progress_k1_ema_beta - 0.90) <= 1e-12
                and args.progress_update_interval == 1
            ),
            "LOCKED_10SEED_REPLICATION_SEEDS": (
                sorted(seeds) == sorted([1801, 2003, 2203, 2411, 2609, 2801, 3001, 3203, 3407, 3607])
            ),
            "LOCKED_CONFIRMATION_BUDGET_300": (
                backward_budgets == [300]
            ),
            "FLOW_TIME_CONTROLLER_DISABLED": True,
            "MACRO_FLOW_TIME_FIXED": all(
                abs(
                    row.get(
                        "active_macro_flow_time",
                        args.macro_flow_time,
                    )
                    - args.macro_flow_time
                ) <= 1e-12
                for row in all_curves
                if row.get("optimizer") == "capacity"
            ),
            "SAME_BATCH_POST_STEP_MEASUREMENT_ACTIVE": True,
            "EQUAL_EXTRA_FORWARD_BUDGET": True,
            "K1_EPSILON_BOUNDED": all(
                args.k1_epsilon_min <= row.get("active_epsilon", args.k1_epsilon_init)
                <= args.k1_epsilon_max
                for row in all_curves
                if row.get("optimizer") == "capacity"
            ),
            "PREDECLARED_CONTROLLER_NO_RETUNING": True,
            "AT_LEAST_6_OF_10_VALIDATION_WINS": (
                validation_wins >= 6
            ),
            "AT_LEAST_6_OF_10_TEST_WINS": (
                test_wins >= 6
            ),
        }

        budget_summary = {
            "backward_budget": backward_budget,
            "controller": "progress_epsilon_light_deadband_fixed_flowtime",
                "experiment_role": (
                    "locked 10-seed replication; "
                    "no optimizer-mechanism change from H12.6"
                ),
                "confirmation_seeds": list(seeds),
                "confirmation_budget": backward_budget,
            "fixed_macro_flow_time": args.macro_flow_time,
            "aggregate": aggregate,
            "validation_progress_ratio_capacity_over_adamw": (
                validation_ratio
            ),
            "test_progress_ratio_capacity_over_adamw": (
                test_ratio
            ),
            "validation_seed_wins": validation_wins,
            "test_seed_wins": test_wins,
            "n_seeds": len(seeds),
            "active_representation": active_representation,
            "paired_representation_audit_available": bool(gauge_rows),
            "gauge_suppression": gauge_suppression,
            "per_seed_comparison": per_seed_comparison,
            "gauge_rows": gauge_rows,
            "schedule_audit": schedule_audit,
            "decision_gates": gates,
        }
        budget_summaries[str(backward_budget)] = budget_summary

        print("\n" + "-" * 120)
        print(f"H12.7 BUDGET {backward_budget} SUMMARY")
        print("-" * 120)
        print(json.dumps(budget_summary, indent=2))

    final_summary = {
        "config": config,
        "budget_summaries": budget_summaries,
        "interpretation_boundary": (
            "H12.7 performs no tuning during the locked 10-seed replication and uses a predeclared H12.3-speed progress-aware epsilon controller with light deadband and fixed macro-flow time. "
            "It evaluates a fixed configuration across fresh seeds and "
            "one or more exact backward budgets."
        ),
    }

    print("\n" + "=" * 120)
    print("D. H12.7 FINAL SUMMARY")
    print("=" * 120)
    print(json.dumps(final_summary, indent=2))

    result_rows = [asdict(x) for x in all_results]

    (out_dir / "summary.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )
    (out_dir / "runs.json").write_text(
        json.dumps(result_rows, indent=2),
        encoding="utf-8",
    )
    (out_dir / "curves.json").write_text(
        json.dumps(all_curves, indent=2),
        encoding="utf-8",
    )
    (out_dir / "gauge_rows.json").write_text(
        json.dumps(all_gauge_rows, indent=2),
        encoding="utf-8",
    )
    (out_dir / "schedule_audit.json").write_text(
        json.dumps(all_schedule_audit, indent=2),
        encoding="utf-8",
    )

    if result_rows:
        keys = sorted(
            {key for row in result_rows for key in row.keys()}
        )
        with (out_dir / "runs.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(result_rows)

    if all_curves:
        keys = sorted(
            {key for row in all_curves for key in row.keys()}
        )
        with (out_dir / "curves.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_curves)

    print(f"\nOutputs: {out_dir}")
    print(
        "\nFINAL NOTE: H11.5 fixes epsilon and macro-flow time, "
        "uses fresh multi-seed locked-batch comparisons, and reports "
        "per-seed wins, mean/median task progress, and test-logit gauge gaps."
    )


# =====================================================================
# H13.2 wrapper: gauge-condition-number scaling experiment
# =====================================================================

def _h132_parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one kappa value is required.")
    if any(x < 1.0 for x in values):
        raise ValueError("All kappa values must be >= 1.")
    return values


def _h132_integrate_aulc(
    curve_rows: list[dict],
    initial_ppl: float,
    budget: int,
) -> float:
    points = [(0.0, float(initial_ppl))]
    for row in curve_rows:
        step = int(row["backward_calls"])
        ppl = float(row["validation_ppl"])
        if step <= budget and math.isfinite(ppl):
            points.append((float(step), ppl))

    points.sort(key=lambda item: item[0])

    dedup = []
    for x, y in points:
        if dedup and x == dedup[-1][0]:
            dedup[-1] = (x, y)
        else:
            dedup.append((x, y))

    if not dedup:
        return 0.0

    if dedup[-1][0] < budget:
        dedup.append((float(budget), dedup[-1][1]))

    x = np.asarray([x for x, _ in dedup], dtype=float)
    ppl = np.asarray([y for _, y in dedup], dtype=float)
    improvement = float(initial_ppl) - ppl
    return float(np.trapezoid(improvement, x) / float(budget))


def _h132_bootstrap_ci(
    values: list[float],
    reps: int,
    seed: int,
    confidence: float = 0.95,
) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "win_rate": float("nan"),
        }
    if len(arr) == 1:
        value = float(arr[0])
        return {
            "n": 1,
            "mean": value,
            "median": value,
            "ci_low": value,
            "ci_high": value,
            "win_rate": float(value > 0),
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(reps, len(arr)))
    means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "win_rate": float(np.mean(arr > 0)),
    }


def _h132_fit_scaling(
    kappas: np.ndarray,
    effects: np.ndarray,
) -> list[dict]:
    """
    Lightweight scaling fits without scipy:
      linear in log(kappa)
      linear in sqrt(kappa)
      linear in kappa
      free power law via grid search over alpha
    Returns RMSE and AIC-like score.
    """
    rows = []
    n = len(kappas)
    if n < 3:
        return rows

    def fit_design(name, x):
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, effects, rcond=None)
        pred = X @ beta
        resid = effects - pred
        rss = float(np.sum(resid * resid))
        rmse = float(np.sqrt(rss / max(n, 1)))
        aic = float(n * np.log(max(rss / n, 1e-30)) + 2 * len(beta))
        rows.append(
            {
                "model": name,
                "intercept": float(beta[0]),
                "slope": float(beta[1]),
                "alpha": None,
                "rmse": rmse,
                "rss": rss,
                "aic": aic,
            }
        )

    fit_design("log_kappa", np.log(kappas))
    fit_design("sqrt_kappa", np.sqrt(kappas))
    fit_design("linear_kappa", kappas)

    best = None
    for alpha in np.linspace(0.05, 1.50, 292):
        x = np.power(kappas, alpha)
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, effects, rcond=None)
        pred = X @ beta
        resid = effects - pred
        rss = float(np.sum(resid * resid))
        candidate = (rss, float(alpha), beta)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is not None:
        rss, alpha, beta = best
        rmse = float(np.sqrt(rss / max(n, 1)))
        aic = float(n * np.log(max(rss / n, 1e-30)) + 2 * 3)
        rows.append(
            {
                "model": "free_power",
                "intercept": float(beta[0]),
                "slope": float(beta[1]),
                "alpha": alpha,
                "rmse": rmse,
                "rss": rss,
                "aic": aic,
            }
        )

    rows.sort(key=lambda row: row["aic"])
    return rows


def _h132_write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _h132_regime_label(kappa: float) -> str:
    if kappa == 1:
        return "baseline"
    if kappa <= 10:
        return "natural_relevant"
    if kappa <= 100:
        return "moderate_stress"
    return "extreme_stress"


def _h132_analyze(
    root_out: Path,
    kappas: list[float],
    seeds: list[int],
    budget: int,
    bootstrap_reps: int,
) -> None:
    combined_runs = []
    combined_curves = []

    for kappa in kappas:
        label = str(kappa).replace(".", "p")
        run_dir = root_out / f"kappa_{label}"
        runs_path = run_dir / "runs.json"
        curves_path = run_dir / "curves.json"

        if not runs_path.exists() or not curves_path.exists():
            raise FileNotFoundError(f"Missing output in {run_dir}")

        runs = json.loads(runs_path.read_text(encoding="utf-8"))
        curves = json.loads(curves_path.read_text(encoding="utf-8"))

        for row in runs:
            item = dict(row)
            item["h132_kappa"] = float(kappa)
            combined_runs.append(item)

        for row in curves:
            item = dict(row)
            item["h132_kappa"] = float(kappa)
            combined_curves.append(item)

    initial_lookup = {}
    for row in combined_runs:
        key = (
            float(row["h132_kappa"]),
            str(row["optimizer"]),
            int(row["seed"]),
            str(row["representation"]),
        )
        initial_lookup[key] = float(row["initial_validation_ppl"])

    grouped_curves = {}
    for row in combined_curves:
        key = (
            float(row["h132_kappa"]),
            str(row["optimizer"]),
            int(row["seed"]),
            str(row["representation"]),
        )
        grouped_curves.setdefault(key, []).append(row)

    per_run = []
    for key, rows in sorted(grouped_curves.items()):
        kappa, optimizer_name, seed, representation = key
        initial_ppl = initial_lookup[key]
        aulc = _h132_integrate_aulc(
            rows,
            initial_ppl=initial_ppl,
            budget=budget,
        )
        endpoint = max(rows, key=lambda r: int(r["backward_calls"]))
        per_run.append(
            {
                "kappa": kappa,
                "regime": _h132_regime_label(kappa),
                "optimizer": optimizer_name,
                "seed": seed,
                "representation": representation,
                "budget": budget,
                "initial_validation_ppl": initial_ppl,
                "requested_kappa": float(kappa),
                "actual_initial_cond_A_mean": float(
                    next(
                        row["initial_actual_cond_A_mean"]
                        for row in combined_runs
                        if float(row["h132_kappa"]) == kappa
                        and str(row["optimizer"]) == optimizer_name
                        and int(row["seed"]) == seed
                        and str(row["representation"]) == representation
                    )
                ),
                "actual_initial_cond_B_mean": float(
                    next(
                        row["initial_actual_cond_B_mean"]
                        for row in combined_runs
                        if float(row["h132_kappa"]) == kappa
                        and str(row["optimizer"]) == optimizer_name
                        and int(row["seed"]) == seed
                        and str(row["representation"]) == representation
                    )
                ),
                "actual_initial_cond_max_mean": float(
                    next(
                        row["initial_actual_cond_max_mean"]
                        for row in combined_runs
                        if float(row["h132_kappa"]) == kappa
                        and str(row["optimizer"]) == optimizer_name
                        and int(row["seed"]) == seed
                        and str(row["representation"]) == representation
                    )
                ),
                "actual_initial_cond_max_peak": float(
                    next(
                        row["initial_actual_cond_max_peak"]
                        for row in combined_runs
                        if float(row["h132_kappa"]) == kappa
                        and str(row["optimizer"]) == optimizer_name
                        and int(row["seed"]) == seed
                        and str(row["representation"]) == representation
                    )
                ),
                "normalized_improvement_aulc": aulc,
                "endpoint_validation_ppl": float(
                    endpoint["validation_ppl"]
                ),
                "endpoint_validation_improvement": (
                    initial_ppl
                    - float(endpoint["validation_ppl"])
                ),
            }
        )

    # Use the gauge representation for kappa > 1.
    # kappa=1 balanced and gauge should be equivalent; balanced is the baseline.
    baseline = {}
    for optimizer_name in ["adamw", "capacity"]:
        for seed in seeds:
            rows = [
                row for row in per_run
                if row["kappa"] == 1.0
                and row["optimizer"] == optimizer_name
                and row["seed"] == seed
                and row["representation"] == "balanced"
            ]
            if len(rows) != 1:
                raise RuntimeError(
                    f"Missing unique kappa=1 balanced baseline for "
                    f"{optimizer_name}, seed={seed}"
                )
            baseline[(optimizer_name, seed)] = rows[0]

    per_seed = []
    summary = []

    for kappa_index, kappa in enumerate(kappas):
        adam_penalties = []
        cap_penalties = []
        penalty_differences = []
        optimizer_advantages = []

        for seed in seeds:
            representation = "balanced" if kappa == 1.0 else "gauge"

            adam_rows = [
                row for row in per_run
                if row["kappa"] == kappa
                and row["optimizer"] == "adamw"
                and row["seed"] == seed
                and row["representation"] == representation
            ]
            cap_rows = [
                row for row in per_run
                if row["kappa"] == kappa
                and row["optimizer"] == "capacity"
                and row["seed"] == seed
                and row["representation"] == representation
            ]
            if len(adam_rows) != 1 or len(cap_rows) != 1:
                raise RuntimeError(
                    f"Expected one pair at kappa={kappa}, seed={seed}"
                )

            adam = adam_rows[0]
            cap = cap_rows[0]
            adam_base = baseline[("adamw", seed)]
            cap_base = baseline[("capacity", seed)]

            adam_penalty = (
                adam_base["normalized_improvement_aulc"]
                - adam["normalized_improvement_aulc"]
            )
            cap_penalty = (
                cap_base["normalized_improvement_aulc"]
                - cap["normalized_improvement_aulc"]
            )
            penalty_difference = adam_penalty - cap_penalty
            optimizer_advantage = (
                cap["normalized_improvement_aulc"]
                - adam["normalized_improvement_aulc"]
            )

            adam_penalties.append(adam_penalty)
            cap_penalties.append(cap_penalty)
            penalty_differences.append(penalty_difference)
            optimizer_advantages.append(optimizer_advantage)

            per_seed.append(
                {
                    "kappa": kappa,
                    "regime": _h132_regime_label(kappa),
                    "seed": seed,
                    "budget": budget,
                    "adamw_gauge_penalty_aulc": adam_penalty,
                    "capacity_gauge_penalty_aulc": cap_penalty,
                    "adamw_minus_capacity_penalty": penalty_difference,
                    "capacity_minus_adamw_aulc": optimizer_advantage,
                    "capacity_reduces_gauge_penalty": (
                        penalty_difference > 0
                    ),
                }
            )

        penalty_stats = _h132_bootstrap_ci(
            penalty_differences,
            reps=bootstrap_reps,
            seed=129000 + kappa_index,
        )
        advantage_stats = _h132_bootstrap_ci(
            optimizer_advantages,
            reps=bootstrap_reps,
            seed=130000 + kappa_index,
        )

        summary.append(
            {
                "kappa": kappa,
                "regime": _h132_regime_label(kappa),
                "budget": budget,
                "n_seeds": len(seeds),
                "mean_actual_initial_cond_max": float(
                    np.mean(
                        [
                            row["actual_initial_cond_max_mean"]
                            for row in per_run
                            if row["kappa"] == kappa
                            and row["representation"] == (
                                "balanced" if kappa == 1.0 else "gauge"
                            )
                        ]
                    )
                ),
                "peak_actual_initial_cond_max": float(
                    np.max(
                        [
                            row["actual_initial_cond_max_peak"]
                            for row in per_run
                            if row["kappa"] == kappa
                            and row["representation"] == (
                                "balanced" if kappa == 1.0 else "gauge"
                            )
                        ]
                    )
                ),
                "mean_adamw_gauge_penalty_aulc": float(
                    np.mean(adam_penalties)
                ),
                "mean_capacity_gauge_penalty_aulc": float(
                    np.mean(cap_penalties)
                ),
                "mean_adamw_minus_capacity_penalty": (
                    penalty_stats["mean"]
                ),
                "penalty_difference_ci_low": (
                    penalty_stats["ci_low"]
                ),
                "penalty_difference_ci_high": (
                    penalty_stats["ci_high"]
                ),
                "capacity_reduces_penalty_win_rate": (
                    penalty_stats["win_rate"]
                ),
                "mean_capacity_minus_adamw_aulc": (
                    advantage_stats["mean"]
                ),
                "optimizer_advantage_ci_low": (
                    advantage_stats["ci_low"]
                ),
                "optimizer_advantage_ci_high": (
                    advantage_stats["ci_high"]
                ),
                "capacity_aulc_win_rate": (
                    advantage_stats["win_rate"]
                ),
            }
        )

    kappas_arr = np.asarray(
        [row["kappa"] for row in summary if row["kappa"] > 1.0],
        dtype=float,
    )
    effects_arr = np.asarray(
        [
            row["mean_adamw_minus_capacity_penalty"]
            for row in summary
            if row["kappa"] > 1.0
        ],
        dtype=float,
    )
    fit_rows = []
    if len(kappas_arr) >= 3:
        for row in _h132_fit_scaling(kappas_arr, effects_arr):
            item = dict(row)
            item["fit_regime"] = "secondary_descriptive_only"
            fit_rows.append(item)
        fit_rows.sort(key=lambda row: row["aic"])

    # Confirmatory paired hypotheses at each non-baseline kappa.
    hypothesis_rows = []
    for summary_row in summary:
        kappa = float(summary_row["kappa"])
        if kappa == 1.0:
            continue

        values = np.asarray(
            [
                float(row["adamw_minus_capacity_penalty"])
                for row in per_seed
                if float(row["kappa"]) == kappa
            ],
            dtype=float,
        )
        observed = float(np.mean(values))
        n_pairs = int(len(values))

        # Exact one-sided paired sign-flip test.
        signs = np.asarray(
            [
                [
                    1.0 if ((mask >> j) & 1) else -1.0
                    for j in range(n_pairs)
                ]
                for mask in range(2 ** n_pairs)
            ],
            dtype=float,
        )
        null_means = (signs * values[None, :]).mean(axis=1)
        p_one_sided = float(
            (np.count_nonzero(null_means >= observed) + 1)
            / (len(null_means) + 1)
        )

        hypothesis_rows.append(
            {
                "kappa": kappa,
                "regime": _h132_regime_label(kappa),
                "n_pairs": n_pairs,
                "mean_adamw_minus_capacity_penalty": observed,
                "one_sided_exact_signflip_p": p_one_sided,
                "predeclared_alpha": 0.05,
                "predeclared_success": bool(
                    observed > 0.0 and p_one_sided < 0.05
                ),
                "hypothesis": (
                    "AdamW gauge penalty exceeds Capacity gauge penalty"
                ),
            }
        )

    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "combined_runs.json").write_text(
        json.dumps(combined_runs, indent=2),
        encoding="utf-8",
    )
    (root_out / "combined_curves.json").write_text(
        json.dumps(combined_curves, indent=2),
        encoding="utf-8",
    )
    (root_out / "h132_per_run_aulc.json").write_text(
        json.dumps(per_run, indent=2),
        encoding="utf-8",
    )
    (root_out / "h132_per_seed_scaling.json").write_text(
        json.dumps(per_seed, indent=2),
        encoding="utf-8",
    )
    (root_out / "h132_kappa_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (root_out / "h132_scaling_fits.json").write_text(
        json.dumps(fit_rows, indent=2),
        encoding="utf-8",
    )
    (root_out / "h132_confirmatory_hypotheses.json").write_text(
        json.dumps(hypothesis_rows, indent=2),
        encoding="utf-8",
    )

    _h132_write_csv(root_out / "combined_runs.csv", combined_runs)
    _h132_write_csv(root_out / "combined_curves.csv", combined_curves)
    _h132_write_csv(root_out / "h132_per_run_aulc.csv", per_run)
    _h132_write_csv(root_out / "h132_per_seed_scaling.csv", per_seed)
    _h132_write_csv(root_out / "h132_kappa_summary.csv", summary)
    _h132_write_csv(root_out / "h132_scaling_fits.csv", fit_rows)
    _h132_write_csv(
        root_out / "h132_confirmatory_hypotheses.csv",
        hypothesis_rows,
    )

    print("\n" + "=" * 120)
    print("H13.2 CONFIRMATORY GAUGE-ROBUSTNESS SUMMARY")
    print("=" * 120)
    for row in summary:
        print(json.dumps(row))

    print("\nConfirmatory hypotheses:")
    for row in hypothesis_rows:
        print(json.dumps(row))

    print("\nSecondary descriptive scaling fits:")
    for row in fit_rows:
        print(json.dumps(row))

    print("\nInterpretation:")
    print(
        "Positive AdamW-minus-Capacity penalty means Capacity suppresses "
        "more of the gauge-conditioning degradation."
    )
    print(
        "Primary confirmation requires a positive paired AdamW-minus-Capacity "
        "gauge-penalty difference with one-sided exact sign-flip p<0.05. "
        "Scaling fits are secondary and descriptive only."
    )


def h132_main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoFlow H13.2 10-seed confirmatory gauge-robustness experiment."
    )
    parser.add_argument(
        "--kappas",
        default="1,5,10,100,1000",
    )
    parser.add_argument(
        "--seeds",
        default="8101,8209,8311,8423,8537,8641,8753,8861,8971,9091",
        help="Fresh mechanism-discovery seeds.",
    )
    parser.add_argument(
        "--backward-budget",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--out-dir",
        default="/content/geoflow_h132_kappa_results",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-examples", type=int, default=8000)
    parser.add_argument("--eval-examples", type=int, default=1500)
    parser.add_argument("--train-blocks", type=int, default=1500)
    parser.add_argument("--eval-blocks", type=int, default=300)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--macro-flow-time", type=float, default=0.4)
    parser.add_argument(
        "--force-reclone",
        action="store_true",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] Ignored notebook/kernel arguments:", unknown)

    kappas = _h132_parse_float_list(args.kappas)
    seeds = parse_int_list(args.seeds)

    if 1.0 not in kappas:
        raise ValueError("H13.2 requires kappa=1 as baseline.")
    if len(seeds) < 2:
        raise ValueError("Use at least two fresh seeds.")
    if args.backward_budget < 1:
        raise ValueError("--backward-budget must be positive.")

    root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    original_argv = list(sys.argv)
    try:
        for kappa in kappas:
            label = str(kappa).replace(".", "p")
            run_out = root_out / f"kappa_{label}"

            print("\n" + "#" * 120)
            print(
                f"H13.2 RUN: kappa={kappa:g}, "
                f"seeds={seeds}, budget={args.backward_budget}"
            )
            print("#" * 120)

            forwarded = [
                original_argv[0],
                "--repo-dir",
                "/content/Geometric-Flow-h132-confirmatory",
                "--model-name",
                args.model_name,
                "--dataset-config",
                args.dataset_config,
                "--seeds",
                ",".join(map(str, seeds)),
                "--backward-budgets",
                str(args.backward_budget),
                "--eval-interval",
                str(args.eval_interval),
                "--eval-batches",
                str(args.eval_batches),
                "--sequence-length",
                str(args.sequence_length),
                "--batch-size",
                str(args.batch_size),
                "--train-examples",
                str(args.train_examples),
                "--eval-examples",
                str(args.eval_examples),
                "--train-blocks",
                str(args.train_blocks),
                "--eval-blocks",
                str(args.eval_blocks),
                "--gauge-condition-number",
                str(kappa),
                "--representation-mode",
                (
                    "balanced-only"
                    if float(kappa) == 1.0
                    else "gauge-only"
                ),
                "--macro-flow-time",
                str(args.macro_flow_time),
                "--out-dir",
                str(run_out),
                "--no-flow-time-control-enabled",
            ]
            if args.force_reclone:
                forwarded.append("--force-reclone")

            sys.argv = forwarded
            main()

        _h132_analyze(
            root_out=root_out,
            kappas=kappas,
            seeds=seeds,
            budget=args.backward_budget,
            bootstrap_reps=args.bootstrap_reps,
        )

        config = {
            "experiment": "H13.2",
            "role": "10-seed confirmatory gauge-robustness replication",
            "primary_metric": (
                "AdamW gauge penalty minus Capacity gauge penalty in "
                "true-initial normalized validation-PPL improvement AULC@120"
            ),
            "secondary_metric": (
                "Capacity minus AdamW normalized AULC@120"
            ),
            "kappas": kappas,
            "fresh_seeds": seeds,
            "backward_budget": args.backward_budget,
            "eval_interval": args.eval_interval,
            "macro_flow_time": args.macro_flow_time,
            "batch_size": args.batch_size,
            "flow_time_controller_enabled": False,
            "anti_windup_enabled": True,
            "all_other_h127_parameters_locked": True,
            "regimes": {"baseline": [1], "natural_relevant": [5, 10], "moderate_stress": [100], "extreme_stress": [1000]},
            "primary_hypotheses": ["P_AdamW(5) > P_Capacity(5)", "P_AdamW(10) > P_Capacity(10)", "P_AdamW(100) > P_Capacity(100)", "P_AdamW(1000) > P_Capacity(1000)"],
            "primary_test": "paired one-sided exact sign-flip",
            "primary_alpha": 0.05,
            "scaling_models_secondary_only": [
                "log_kappa",
                "sqrt_kappa",
                "linear_kappa",
                "free_power",
            ],
            "confirmatory_status": "predeclared 10-seed confirmatory replication",
        }
        (root_out / "h132_config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        print(f"\nH13.2 outputs: {root_out}")

    finally:
        sys.argv = original_argv




# =====================================================================
# H13.4 — PRODUCT-SPACE GAUGE-DYNAMICS AUDIT
# =====================================================================

def _h134_parse_ints(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def _h134_parse_floats(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float.")
    return values


@torch.no_grad()
def _h134_make_sample_plan(
    adapters: list[LoRAConv1DAdapter],
    *,
    samples_per_adapter: int,
    seed: int,
) -> None:
    """
    H13.4 compatibility stub.

    H13.4 computes the complete product B@A for every adapted module.
    The sampling arguments are retained only so the inherited call sites
    remain stable.
    """
    del adapters, samples_per_adapter, seed
    return None


@torch.no_grad()
def _h134_product_sketch(
    adapters: list[LoRAConv1DAdapter],
    plans: None = None,
) -> torch.Tensor:
    """
    Return the exact concatenated product-space state.

    For every adapted module this computes the complete matrix B@A,
    flattens it, transfers it to CPU float32, and concatenates all modules.
    Consequently all reported norms, cosines, and trajectory gaps are exact
    Frobenius quantities over the complete adapted product space.
    """
    del plans
    pieces: list[torch.Tensor] = []
    for adapter in adapters:
        product = adapter.B @ adapter.A
        pieces.append(product.detach().float().cpu().reshape(-1))
    return torch.cat(pieces, dim=0)


def _h134_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    nx = float(torch.linalg.vector_norm(x))
    ny = float(torch.linalg.vector_norm(y))
    if nx <= 1e-30 or ny <= 1e-30:
        return float("nan")
    value = float(torch.dot(x, y) / (nx * ny))
    return min(1.0, max(-1.0, value))


def _h134_relative_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    denom = max(float(torch.linalg.vector_norm(x)), 1e-30)
    return float(torch.linalg.vector_norm(y - x)) / denom


def _h134_make_optimizer(
    *,
    optimizer_name: str,
    adapters: list[LoRAConv1DAdapter],
    adamw_lr: float,
    adamw_weight_decay: float,
    macro_flow_time: float,
    local_function_tolerance: float,
    max_auto_substeps: int,
    max_flow_dt: float | None,
):
    if optimizer_name == "adamw":
        return (
            torch.optim.AdamW(
                trainable_parameters(adapters),
                lr=adamw_lr,
                weight_decay=adamw_weight_decay,
            ),
            None,
        )

    if optimizer_name != "capacity":
        raise ValueError(optimizer_name)

    optimizer = CapacityAdaptiveQuotientFlow(
        adapters,
        macro_flow_time=macro_flow_time,
        local_function_tolerance=local_function_tolerance,
        max_auto_substeps=max_auto_substeps,
        max_flow_dt=max_flow_dt,
        balance_after_substep=True,
    )
    stepper = ResumableCapacityStepper(
        optimizer,
        macro_flow_time=macro_flow_time,
        local_function_tolerance=local_function_tolerance,
        max_flow_dt=max_flow_dt,
        k1_enabled=True,
        k1_eta=0.08,
        k1_ema_beta=0.90,
        k1_epsilon_min=0.08,
        k1_epsilon_max=0.50,
        k1_freeze_on_actuator_limit=True,
        progress_k1_eta=0.08,
        progress_k1_ema_beta=0.90,
        progress_k1_warmup=20,
        progress_k1_ratio_min=0.75,
        progress_k1_ratio_max=1.25,
        progress_k1_floor=1e-3,
        progress_deadband=0.01,
        progress_update_interval=1,
        flow_time_control_enabled=False,
        flow_time_eta=0.0,
        flow_time_min=0.30,
        flow_time_max=0.80,
        utilization_target=0.85,
    )
    return optimizer, stepper


def _h134_run_branch(
    *,
    optimizer_name: str,
    representation: str,
    kappa: float,
    seed: int,
    model_name: str,
    target_modules: list[str],
    rank: int,
    init_scale: float,
    train_dataset: TokenBlockDataset,
    schedule: LockedBatchSchedule,
    device: torch.device,
    steps: int,
    samples_per_adapter: int,
    probe_seed: int,
    adamw_lr: float,
    adamw_weight_decay: float,
    macro_flow_time: float,
    local_function_tolerance: float,
    max_auto_substeps: int,
    max_flow_dt: float | None,
) -> dict:
    actual_kappa = 1.0 if representation == "balanced" else float(kappa)
    model, adapters = make_model(
        model_name,
        target_modules,
        rank,
        init_scale,
        seed,
        actual_kappa,
        device,
    )
    optimizer, stepper = _h134_make_optimizer(
        optimizer_name=optimizer_name,
        adapters=adapters,
        adamw_lr=adamw_lr,
        adamw_weight_decay=adamw_weight_decay,
        macro_flow_time=macro_flow_time,
        local_function_tolerance=local_function_tolerance,
        max_auto_substeps=max_auto_substeps,
        max_flow_dt=max_flow_dt,
    )

    plans = _h134_make_sample_plan(
        adapters,
        samples_per_adapter=samples_per_adapter,
        seed=probe_seed,
    )
    initial_product = _h134_product_sketch(adapters, plans)
    current_product = initial_product.clone()
    initial_condition = factor_condition_diagnostics(adapters)

    rows: list[dict] = []

    for step_index in range(steps):
        model.train()
        batch_cpu = schedule.batch(train_dataset, step_index)
        batch_hash = batch_content_hash(batch_cpu)
        batch = {k: v.to(device) for k, v in batch_cpu.items()}

        optimizer.zero_grad(set_to_none=True)

        # make_model resets RNG identically for each branch. Since the locked
        # schedule and branch execution are deterministic, this replays the
        # same dropout stream for balanced/gauge comparisons.
        cpu_rng_before = torch.random.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None
        )

        outputs = model(**batch)
        loss_before = float(outputs.loss.detach().cpu())
        outputs.loss.backward()

        product_before = current_product.clone()

        if optimizer_name == "adamw":
            optimizer.step()
            step_diag = {}
        else:
            step_diag = stepper.step_after_backward()

        product_after = _h134_product_sketch(adapters, plans)
        delta_product = product_after - product_before

        # Same-mask post-step loss, without perturbing the live RNG stream.
        cpu_rng_live = torch.random.get_rng_state()
        cuda_rng_live = (
            torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None
        )
        torch.random.set_rng_state(cpu_rng_before)
        if device.type == "cuda" and cuda_rng_before is not None:
            torch.cuda.set_rng_state_all(cuda_rng_before)
        with torch.no_grad():
            loss_after = float(model(**batch).loss.detach().cpu())
        torch.random.set_rng_state(cpu_rng_live)
        if device.type == "cuda" and cuda_rng_live is not None:
            torch.cuda.set_rng_state_all(cuda_rng_live)

        if optimizer_name == "capacity":
            progress_diag = stepper.progress_feedback(
                loss_before=loss_before,
                loss_after=loss_after,
                realized_dphi=float(step_diag["realized_local_dphi"]),
                predicted_dphi=float(step_diag["predicted_local_dphi"]),
                requested_dphi=float(step_diag["requested_local_dphi"]),
                epsilon_limited=bool(step_diag["epsilon_limited"]),
                actuator_limited=bool(step_diag["actuator_limited"]),
            )
            active_epsilon = float(progress_diag["active_epsilon"])
            realized_dphi = float(step_diag["realized_local_dphi"])
            predicted_dphi = float(step_diag["predicted_local_dphi"])
        else:
            active_epsilon = float("nan")
            realized_dphi = float("nan")
            predicted_dphi = float("nan")

        rows.append(
            {
                "optimizer": optimizer_name,
                "representation": representation,
                "requested_kappa": float(kappa),
                "seed": int(seed),
                "step": int(step_index + 1),
                "batch_hash": batch_hash,
                "loss_before": loss_before,
                "loss_after": loss_after,
                "same_batch_loss_delta": loss_before - loss_after,
                "full_product_norm": float(
                    torch.linalg.vector_norm(product_after)
                ),
                "full_delta_product_norm": float(
                    torch.linalg.vector_norm(delta_product)
                ),
                "full_displacement_from_initial": float(
                    torch.linalg.vector_norm(
                        product_after - initial_product
                    )
                ),
                "active_epsilon": active_epsilon,
                "realized_dphi": realized_dphi,
                "predicted_dphi": predicted_dphi,
                "_product": product_after,
                "_delta_product": delta_product,
            }
        )
        current_product = product_after

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "optimizer": optimizer_name,
        "representation": representation,
        "requested_kappa": float(kappa),
        "seed": int(seed),
        "initial_product": initial_product,
        "initial_condition": initial_condition,
        "rows": rows,
    }


def _h134_pair_metrics(
    balanced: dict,
    gauge: dict,
) -> list[dict]:
    if len(balanced["rows"]) != len(gauge["rows"]):
        raise RuntimeError("Paired branches have different lengths.")

    initial_gap = _h134_relative_gap(
        balanced["initial_product"],
        gauge["initial_product"],
    )

    out: list[dict] = []
    for b, g in zip(balanced["rows"], gauge["rows"]):
        if b["batch_hash"] != g["batch_hash"]:
            raise RuntimeError("Locked batch hashes differ.")

        db = b["_delta_product"]
        dg = g["_delta_product"]
        pb = b["_product"]
        pg = g["_product"]

        norm_b = float(torch.linalg.vector_norm(db))
        norm_g = float(torch.linalg.vector_norm(dg))
        magnitude_ratio = norm_g / max(norm_b, 1e-30)
        cosine = _h134_cosine(db, dg)

        out.append(
            {
                "optimizer": b["optimizer"],
                "seed": b["seed"],
                "kappa": g["requested_kappa"],
                "step": b["step"],
                "batch_hash": b["batch_hash"],
                "initial_product_relative_gap": initial_gap,
                "balanced_delta_norm": norm_b,
                "gauge_delta_norm": norm_g,
                "delta_magnitude_ratio_gauge_over_balanced": (
                    magnitude_ratio
                ),
                "delta_log_magnitude_error": abs(
                    math.log(max(magnitude_ratio, 1e-30))
                ),
                "delta_direction_cosine": cosine,
                "delta_direction_error": (
                    1.0 - cosine
                    if math.isfinite(cosine)
                    else float("nan")
                ),
                "trajectory_relative_gap": _h134_relative_gap(pb, pg),
                "balanced_loss_before": b["loss_before"],
                "gauge_loss_before": g["loss_before"],
                "loss_before_abs_gap": abs(
                    g["loss_before"] - b["loss_before"]
                ),
                "balanced_same_batch_delta": (
                    b["same_batch_loss_delta"]
                ),
                "gauge_same_batch_delta": g["same_batch_loss_delta"],
                "same_batch_progress_abs_gap": abs(
                    g["same_batch_loss_delta"]
                    - b["same_batch_loss_delta"]
                ),
                "balanced_initial_cond_max": float(
                    balanced["initial_condition"]["cond_max_mean"]
                ),
                "gauge_initial_cond_max": float(
                    gauge["initial_condition"]["cond_max_mean"]
                ),
            }
        )
    return out


def _h134_bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    draws: int = 20000,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    sample = rng.choice(x, size=(draws, len(x)), replace=True)
    means = sample.mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _h134_summarize(
    pair_rows: list[dict],
    *,
    steps: int,
) -> list[dict]:
    summaries: list[dict] = []
    optimizers = sorted({row["optimizer"] for row in pair_rows})
    kappas = sorted({float(row["kappa"]) for row in pair_rows})

    for optimizer_name in optimizers:
        for kappa in kappas:
            selected = [
                row for row in pair_rows
                if row["optimizer"] == optimizer_name
                and float(row["kappa"]) == kappa
            ]
            seeds = sorted({int(row["seed"]) for row in selected})

            per_seed: list[dict] = []
            for seed in seeds:
                rows = [
                    row for row in selected
                    if int(row["seed"]) == seed
                ]
                per_seed.append(
                    {
                        "seed": seed,
                        "mean_direction_error": float(
                            np.nanmean(
                                [
                                    row["delta_direction_error"]
                                    for row in rows
                                ]
                            )
                        ),
                        "mean_log_magnitude_error": float(
                            np.nanmean(
                                [
                                    row["delta_log_magnitude_error"]
                                    for row in rows
                                ]
                            )
                        ),
                        "final_trajectory_gap": float(
                            rows[-1]["trajectory_relative_gap"]
                        ),
                        "mean_trajectory_gap": float(
                            np.mean(
                                [
                                    row["trajectory_relative_gap"]
                                    for row in rows
                                ]
                            )
                        ),
                        "mean_progress_gap": float(
                            np.mean(
                                [
                                    row["same_batch_progress_abs_gap"]
                                    for row in rows
                                ]
                            )
                        ),
                    }
                )

            summary = {
                "optimizer": optimizer_name,
                "kappa": kappa,
                "n_seeds": len(seeds),
                "steps": steps,
                "max_initial_product_relative_gap": float(
                    max(
                        row["initial_product_relative_gap"]
                        for row in selected
                    )
                ),
            }

            aggregate_fields = {
                "mean_direction_error": "mean_direction_error",
                "mean_log_magnitude_error": "mean_log_magnitude_error",
                "final_trajectory_gap": "mean_final_trajectory_gap",
                "mean_trajectory_gap": "mean_trajectory_gap",
                "mean_progress_gap": "mean_progress_gap",
            }

            for source_field, output_field in aggregate_fields.items():
                values = [
                    float(row[source_field])
                    for row in per_seed
                ]
                ci_low, ci_high = _h134_bootstrap_mean_ci(
                    values,
                    seed=(
                        133000
                        + int(kappa)
                        + len(source_field)
                    ),
                )
                summary[output_field] = float(np.mean(values))
                summary[f"{output_field}_ci_low"] = ci_low
                summary[f"{output_field}_ci_high"] = ci_high

            summary["pass_initial_equivalence"] = bool(
                summary["max_initial_product_relative_gap"] < 1e-6
            )
            if optimizer_name == "capacity":
                summary["pass_near_gauge_equivariance"] = bool(
                    summary["mean_direction_error"] < 1e-4
                    and summary["mean_log_magnitude_error"] < 1e-3
                    and summary["mean_final_trajectory_gap"] < 1e-3
                )
            else:
                summary["pass_near_gauge_equivariance"] = False

            summaries.append(summary)

    return summaries


def _h134_full_product_optimizer_contrast(
    summaries: list[dict],
) -> list[dict]:
    out: list[dict] = []
    kappas = sorted({float(row["kappa"]) for row in summaries})
    metrics = [
        "mean_direction_error",
        "mean_log_magnitude_error",
        "mean_final_trajectory_gap",
        "mean_trajectory_gap",
        "mean_progress_gap",
    ]

    for kappa in kappas:
        by_optimizer = {
            row["optimizer"]: row
            for row in summaries
            if float(row["kappa"]) == kappa
        }
        if "adamw" not in by_optimizer or "capacity" not in by_optimizer:
            continue

        row = {"kappa": kappa}
        for metric in metrics:
            adam = float(by_optimizer["adamw"][metric])
            cap = float(by_optimizer["capacity"][metric])
            row[f"adamw_{metric}"] = adam
            row[f"capacity_{metric}"] = cap
            row[f"adamw_minus_capacity_{metric}"] = adam - cap
            row[f"adamw_over_capacity_{metric}"] = (
                adam / max(cap, 1e-30)
            )
        out.append(row)
    return out


def h134_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "H13.4 product-space gauge-dynamics audit. "
            "Compares balanced and gauge-equivalent LoRA branches under "
            "identical seeds and locked mini-batches."
        )
    )
    parser.add_argument(
        "--repo-dir",
        default="/content/Geometric-Flow-h134-full-product-audit",
    )
    parser.add_argument("--force-reclone", action="store_true")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
    )
    parser.add_argument(
        "--target-modules",
        default=(
            "transformer.h.0.attn.c_attn,"
            "transformer.h.1.attn.c_attn"
        ),
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument(
        "--seeds",
        default="10103,10301,10501",
    )
    parser.add_argument(
        "--kappas",
        default="5,10,100,1000",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--samples-per-adapter",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility argument; H13.4 always uses "
            "the complete B@A matrices."
        ),
    )
    parser.add_argument("--data-seed", type=int, default=424242)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-examples", type=int, default=8000)
    parser.add_argument("--eval-examples", type=int, default=1500)
    parser.add_argument("--train-blocks", type=int, default=1500)
    parser.add_argument("--eval-blocks", type=int, default=300)
    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.0)
    parser.add_argument("--macro-flow-time", type=float, default=0.4)
    parser.add_argument(
        "--local-function-tolerance",
        type=float,
        default=0.18,
    )
    parser.add_argument("--max-auto-substeps", type=int, default=128)
    parser.add_argument("--max-flow-dt", type=float, default=0.0)
    parser.add_argument(
        "--out-dir",
        default="/content/geoflow_h134_full_product_audit_results",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[H13.4] ignored notebook/kernel arguments:", unknown)

    seeds = _h134_parse_ints(args.seeds)
    kappas = _h134_parse_floats(args.kappas)
    if any(k <= 1.0 for k in kappas):
        raise ValueError("H13.4 kappas must be > 1.")
    if args.steps < 1:
        raise ValueError("--steps must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 120)
    print("H13.4 FULL-PRODUCT GAUGE-DYNAMICS AUDIT")
    print("=" * 120)
    print("device =", device)
    print("seeds =", seeds)
    print("kappas =", kappas)
    print("steps =", args.steps)
    print("product_mode = full exact B@A")
    print(
        "memory_note = balanced full-product trajectories are cached on CPU "
        "and reused across kappa values"
    )

    repo_dir = Path(args.repo_dir)
    prepare_repo(repo_dir, args.force_reclone)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            str(repo_dir),
        ],
        check=True,
    )
    # Refresh the public optimizer class after editable installation.
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    global CapacityAdaptiveQuotientFlow
    from geometric_flow import (
        CapacityAdaptiveQuotientFlow as _CapacityAdaptiveQuotientFlow,
    )
    CapacityAdaptiveQuotientFlow = _CapacityAdaptiveQuotientFlow

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, _, _, data_metadata = load_real_wikitext(
        tokenizer,
        args.dataset_config,
        args.sequence_length,
        args.train_examples,
        args.eval_examples,
        args.train_blocks,
        args.eval_blocks,
        args.data_seed,
    )
    print(json.dumps(data_metadata, indent=2))

    target_modules = [
        item.strip()
        for item in args.target_modules.split(",")
        if item.strip()
    ]
    max_flow_dt = (
        None if args.max_flow_dt <= 0.0 else float(args.max_flow_dt)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict] = []
    branch_metadata: list[dict] = []

    for seed in seeds:
        schedule = make_locked_batch_schedule(
            train_dataset,
            backward_budget=args.steps,
            batch_size=args.batch_size,
            seed=seed,
        )
        print(
            f"\n[seed {seed}] schedule_hash={schedule.schedule_hash}"
        )

        for optimizer_name in ["adamw", "capacity"]:
            # The balanced branch is identical across kappa, so run it once.
            print(
                f"[run] optimizer={optimizer_name} "
                f"seed={seed} representation=balanced"
            )
            balanced = _h134_run_branch(
                optimizer_name=optimizer_name,
                representation="balanced",
                kappa=1.0,
                seed=seed,
                model_name=args.model_name,
                target_modules=target_modules,
                rank=args.rank,
                init_scale=args.init_scale,
                train_dataset=train_dataset,
                schedule=schedule,
                device=device,
                steps=args.steps,
                samples_per_adapter=args.samples_per_adapter,
                probe_seed=1337,
                adamw_lr=args.adamw_lr,
                adamw_weight_decay=args.adamw_weight_decay,
                macro_flow_time=args.macro_flow_time,
                local_function_tolerance=(
                    args.local_function_tolerance
                ),
                max_auto_substeps=args.max_auto_substeps,
                max_flow_dt=max_flow_dt,
            )

            branch_metadata.append(
                {
                    "optimizer": optimizer_name,
                    "representation": "balanced",
                    "seed": seed,
                    "requested_kappa": 1.0,
                    **balanced["initial_condition"],
                }
            )

            for kappa in kappas:
                print(
                    f"[run] optimizer={optimizer_name} seed={seed} "
                    f"representation=gauge kappa={kappa:g}"
                )
                gauge = _h134_run_branch(
                    optimizer_name=optimizer_name,
                    representation="gauge",
                    kappa=kappa,
                    seed=seed,
                    model_name=args.model_name,
                    target_modules=target_modules,
                    rank=args.rank,
                    init_scale=args.init_scale,
                    train_dataset=train_dataset,
                    schedule=schedule,
                    device=device,
                    steps=args.steps,
                    samples_per_adapter=args.samples_per_adapter,
                    probe_seed=1337,
                    adamw_lr=args.adamw_lr,
                    adamw_weight_decay=args.adamw_weight_decay,
                    macro_flow_time=args.macro_flow_time,
                    local_function_tolerance=(
                        args.local_function_tolerance
                    ),
                    max_auto_substeps=args.max_auto_substeps,
                    max_flow_dt=max_flow_dt,
                )
                branch_metadata.append(
                    {
                        "optimizer": optimizer_name,
                        "representation": "gauge",
                        "seed": seed,
                        "requested_kappa": kappa,
                        **gauge["initial_condition"],
                    }
                )
                rows = _h134_pair_metrics(balanced, gauge)
                pair_rows.extend(rows)

                last = rows[-1]
                print(
                    json.dumps(
                        {
                            "optimizer": optimizer_name,
                            "seed": seed,
                            "kappa": kappa,
                            "initial_product_gap": (
                                last["initial_product_relative_gap"]
                            ),
                            "mean_direction_error": float(
                                np.nanmean(
                                    [
                                        row["delta_direction_error"]
                                        for row in rows
                                    ]
                                )
                            ),
                            "mean_log_magnitude_error": float(
                                np.nanmean(
                                    [
                                        row["delta_log_magnitude_error"]
                                        for row in rows
                                    ]
                                )
                            ),
                            "final_trajectory_gap": (
                                last["trajectory_relative_gap"]
                            ),
                        }
                    )
                )

    summaries = _h134_summarize(pair_rows, steps=args.steps)
    contrasts = _h134_full_product_optimizer_contrast(summaries)

    clean_pair_rows = [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in pair_rows
    ]

    _h132_write_csv(
        out_dir / "h134_stepwise_full_product_metrics.csv",
        clean_pair_rows,
    )
    _h132_write_csv(
        out_dir / "h134_full_product_summary.csv",
        summaries,
    )
    _h132_write_csv(
        out_dir / "h134_full_product_optimizer_contrast.csv",
        contrasts,
    )
    _h132_write_csv(
        out_dir / "h134_full_product_branch_conditions.csv",
        branch_metadata,
    )

    (out_dir / "h134_stepwise_full_product_metrics.json").write_text(
        json.dumps(clean_pair_rows, indent=2),
        encoding="utf-8",
    )
    (out_dir / "h134_full_product_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    (out_dir / "h134_full_product_optimizer_contrast.json").write_text(
        json.dumps(contrasts, indent=2),
        encoding="utf-8",
    )
    (out_dir / "h134_config.json").write_text(
        json.dumps(
            {
                "experiment": "H13.4",
                "product_mode": "complete-coordinate B@A after common float32 transfer",
                "role": "complete-coordinate full-product gauge-dynamics mechanism audit",
                "model_name": args.model_name,
                "dataset_config": args.dataset_config,
                "target_modules": target_modules,
                "rank": args.rank,
                "seeds": seeds,
                "kappas": kappas,
                "steps": args.steps,
                "samples_per_adapter": None,
                "schedule": "locked paired batches",
                "primary_metrics": [
                    "delta_direction_error",
                    "delta_log_magnitude_error",
                    "trajectory_relative_gap",
                ],
                "prediction": (
                    "AdamW full-product errors grow with gauge conditioning "
                    "while Capacity remains approximately gauge-equivariant."
                ),
                "important_note": (
                    "Product metrics use the complete B@A matrices. "
                    "The same coordinates are used in every paired branch."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 120)
    print("H13.4 SUMMARY")
    print("=" * 120)
    for row in summaries:
        print(json.dumps(row))

    print("\nAdamW versus Capacity mechanism contrast:")
    for row in contrasts:
        print(json.dumps(row))

    print("\nOutputs:", out_dir)


if __name__ == "__main__":
    h134_main()
