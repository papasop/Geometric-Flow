"""Recommended CIFAR-10 experiment configurations for GeoFlow."""

from __future__ import annotations

RECOMMENDED_CIFAR10_CONFIGS = {
    "hybrid_diagonal_500": {
        "description": "Default real CIFAR-10 benchmark: Adam warm-up plus Fisher diagonal GeoFlow.",
        "steps": 500,
        "trials": 3,
        "channels": 32,
        "conv_layers": 6,
        "lr": 3e-3,
        "damping": 1e-3,
        "lr_scale": 3.0,
        "curvature_reuse": 5,
        "preconditioner": "diagonal",
        "use_fisher": True,
        "precond_scale": 0.5,
        "grad_smoothing": 0.0,
        "max_grad_norm": 2.0,
        "hybrid_warmup_steps": [10, 30, 50, 80],
    },
    "hybrid_deep_scan": {
        "description": "Deeper GeoCNN scan intended to test whether geometry helps more as scale grows.",
        "steps": 1000,
        "trials": 5,
        "channels": 48,
        "conv_layers": 6,
        "lr": 3e-3,
        "damping": 1e-3,
        "lr_scale": 3.0,
        "curvature_reuse": 5,
        "preconditioner": "diagonal",
        "use_fisher": True,
        "precond_scale": 0.5,
        "grad_smoothing": 0.0,
        "max_grad_norm": 2.0,
        "hybrid_warmup_steps": [30, 80, 150],
    },
}


def config_names():
    return sorted(RECOMMENDED_CIFAR10_CONFIGS)


def get_config(name: str):
    try:
        return dict(RECOMMENDED_CIFAR10_CONFIGS[name])
    except KeyError as exc:
        available = ", ".join(config_names())
        raise ValueError(f"unknown CIFAR-10 config {name!r}; available: {available}") from exc
