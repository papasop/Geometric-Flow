"""Tiny capacity-adaptive quotient-flow smoke benchmark.

This script is intentionally small: it uses a synthetic LoRA-style classifier
and does not download GPT-2. It validates the public controller wiring,
dynamic local substep counts, and gauge-divergence reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometric_flow import CapacityAdaptiveQuotientFlow


class LoRAFactor(torch.nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.A = torch.nn.Parameter(a.clone())
        self.B = torch.nn.Parameter(b.clone())

    def product(self) -> torch.Tensor:
        return self.B @ self.A


class TinyLoRAClassifier(torch.nn.Module):
    def __init__(
        self,
        *,
        w0: torch.Tensor,
        head: torch.Tensor,
        bias: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("w0", w0.clone())
        self.register_buffer("head", head.clone())
        self.register_buffer("bias", bias.clone())
        self.lora = LoRAFactor(a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(x @ (self.w0 + self.lora.product()).T)
        return hidden @ self.head.T + self.bias

    def product(self) -> torch.Tensor:
        return self.lora.product()


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def pairwise_divergence(tensors: list[torch.Tensor]) -> float:
    if len(tensors) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            denom = 0.5 * (tensors[i].norm() + tensors[j].norm()).clamp_min(torch.finfo(tensors[i].dtype).tiny)
            total += float(((tensors[i] - tensors[j]).norm() / denom).detach().cpu())
            count += 1
    return total / count


def make_problem(seed: int, *, dtype: torch.dtype):
    torch.manual_seed(seed)
    n_samples = 96
    in_features = 10
    hidden = 12
    classes = 4
    rank = 3
    x = torch.randn(n_samples, in_features, dtype=dtype)
    w0 = 0.35 * torch.randn(hidden, in_features, dtype=dtype)
    head = 0.5 * torch.randn(classes, hidden, dtype=dtype)
    bias = 0.05 * torch.randn(classes, dtype=dtype)
    teacher_delta = 0.8 * torch.randn(hidden, in_features, dtype=dtype)
    teacher_hidden = torch.tanh(x @ (w0 + teacher_delta).T)
    labels = (teacher_hidden @ head.T + bias).argmax(dim=-1)
    a = 0.2 * torch.randn(rank, in_features, dtype=dtype)
    b = 0.2 * torch.randn(hidden, rank, dtype=dtype)
    gauges = []
    for idx in range(3):
        scale = torch.linspace(-0.35, 0.35, rank, dtype=dtype) * (idx + 1)
        gauges.append(torch.diag(torch.exp(scale)))
    return x, labels, w0, head, bias, a, b, gauges


def train_factor_adam(model: TinyLoRAClassifier, x: torch.Tensor, labels: torch.Tensor, *, steps: int, lr: float):
    optimizer = torch.optim.Adam([model.lora.A, model.lora.B], lr=lr)
    initial_loss = float(F.cross_entropy(model(x), labels).detach().cpu())
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(x)
        final_loss = float(F.cross_entropy(logits, labels).cpu())
        accuracy = float((logits.argmax(dim=-1) == labels).float().mean().cpu())
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_progress": initial_loss - final_loss,
        "accuracy": accuracy,
        "product": model.product().detach().clone(),
        "logits": logits.detach().clone(),
    }


def train_capacity(
    model: TinyLoRAClassifier,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int,
    macro_flow_time: float,
    local_function_tolerance: float,
):
    optimizer = CapacityAdaptiveQuotientFlow(
        [model.lora],
        macro_flow_time=macro_flow_time,
        local_function_tolerance=local_function_tolerance,
        balance_after_substep=True,
    )
    initial_loss = float(F.cross_entropy(model(x), labels).detach().cpu())
    substeps = []
    for _ in range(steps):
        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), labels)
            loss.backward()
            return loss

        optimizer.macro_step(closure)
        substeps.append(optimizer.last_auto_substeps)
    with torch.no_grad():
        logits = model(x)
        final_loss = float(F.cross_entropy(logits, labels).cpu())
        accuracy = float((logits.argmax(dim=-1) == labels).float().mean().cpu())
    diagnostics = dict(optimizer.last_diagnostics)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_progress": initial_loss - final_loss,
        "accuracy": accuracy,
        "product": model.product().detach().clone(),
        "logits": logits.detach().clone(),
        "substeps": substeps,
        "diagnostics": diagnostics,
    }


def run_seed(args, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    dtype = torch.float64
    x, labels, w0, head, bias, a, b, gauges = make_problem(seed, dtype=dtype)
    gauges = gauges[: args.representations]
    rows = []
    products = {"factor_adam": [], "capacity_adaptive": []}
    logits = {"factor_adam": [], "capacity_adaptive": []}
    progress = {"factor_adam": [], "capacity_adaptive": []}
    substep_values = []
    for rep, gauge in enumerate(gauges):
        a_g = gauge @ a
        b_g = b @ torch.linalg.inv(gauge)

        adam_model = TinyLoRAClassifier(w0=w0, head=head, bias=bias, a=a_g, b=b_g)
        adam = train_factor_adam(adam_model, x, labels, steps=args.steps, lr=args.factor_adam_lr)
        products["factor_adam"].append(adam["product"])
        logits["factor_adam"].append(adam["logits"])
        progress["factor_adam"].append(adam["loss_progress"])
        rows.append(
            {
                "seed": seed,
                "representation": rep,
                "optimizer": "factor_adam",
                "initial_loss": adam["initial_loss"],
                "final_loss": adam["final_loss"],
                "loss_progress": adam["loss_progress"],
                "accuracy": adam["accuracy"],
                "last_auto_substeps": "",
                "mean_auto_substeps": "",
                "fallback_count": "",
            }
        )

        capacity_model = TinyLoRAClassifier(w0=w0, head=head, bias=bias, a=a_g, b=b_g)
        capacity = train_capacity(
            capacity_model,
            x,
            labels,
            steps=args.steps,
            macro_flow_time=args.macro_flow_time,
            local_function_tolerance=args.local_function_tolerance,
        )
        products["capacity_adaptive"].append(capacity["product"])
        logits["capacity_adaptive"].append(capacity["logits"])
        progress["capacity_adaptive"].append(capacity["loss_progress"])
        substep_values.extend(capacity["substeps"])
        rows.append(
            {
                "seed": seed,
                "representation": rep,
                "optimizer": "capacity_adaptive",
                "initial_loss": capacity["initial_loss"],
                "final_loss": capacity["final_loss"],
                "loss_progress": capacity["loss_progress"],
                "accuracy": capacity["accuracy"],
                "last_auto_substeps": capacity["diagnostics"]["last_auto_substeps"],
                "mean_auto_substeps": capacity["diagnostics"]["mean_auto_substeps"],
                "fallback_count": capacity["diagnostics"]["fallback_count"],
            }
        )

    adam_product_div = pairwise_divergence(products["factor_adam"])
    capacity_product_div = pairwise_divergence(products["capacity_adaptive"])
    adam_logit_div = pairwise_divergence(logits["factor_adam"])
    capacity_logit_div = pairwise_divergence(logits["capacity_adaptive"])
    summary = {
        "seed": seed,
        "factor_adam_product_divergence": adam_product_div,
        "capacity_product_divergence": capacity_product_div,
        "product_gauge_suppression": adam_product_div / max(capacity_product_div, 1e-30),
        "factor_adam_logit_divergence": adam_logit_div,
        "capacity_logit_divergence": capacity_logit_div,
        "logit_gauge_suppression": adam_logit_div / max(capacity_logit_div, 1e-30),
        "factor_adam_mean_progress": sum(progress["factor_adam"]) / len(progress["factor_adam"]),
        "capacity_mean_progress": sum(progress["capacity_adaptive"]) / len(progress["capacity_adaptive"]),
        "min_auto_substeps": min(substep_values),
        "max_auto_substeps": max(substep_values),
        "mean_auto_substeps": sum(substep_values) / len(substep_values),
    }
    summary["matched_progress"] = summary["capacity_mean_progress"] >= 0.8 * summary["factor_adam_mean_progress"]
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="101,211,307")
    parser.add_argument("--representations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--factor-adam-lr", type=float, default=0.03)
    parser.add_argument("--macro-flow-time", type=float, default=2.6)
    parser.add_argument("--local-function-tolerance", type=float, default=0.05)
    parser.add_argument("--out-dir", default="artifacts/capacity_adaptive_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    for seed in parse_seeds(args.seeds):
        rows, summary = run_seed(args, seed)
        all_rows.extend(rows)
        summaries.append(summary)

    rows_path = out_dir / "runs.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump({"summaries": summaries}, handle, indent=2)

    suppressions = [item["product_gauge_suppression"] for item in summaries]
    geo_suppression = math.exp(sum(math.log(max(value, 1e-30)) for value in suppressions) / len(suppressions))
    matched = sum(1 for item in summaries if item["matched_progress"])
    min_k = min(item["min_auto_substeps"] for item in summaries)
    max_k = max(item["max_auto_substeps"] for item in summaries)
    print(
        "capacity_adaptive_smoke "
        f"matched={matched}/{len(summaries)} "
        f"product_gauge_suppression={geo_suppression:.3f}x "
        f"auto_substeps=[{min_k},{max_k}]"
    )
    print(f"wrote {rows_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
