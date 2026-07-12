"""Flat parameter helpers used by the geometric flow modules."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def trainable_params(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [p for p in params if p.requires_grad]


def flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.empty(0)
    return torch.cat([t.reshape(-1) for t in tensors])


def flatten_grads(
    grads: Sequence[torch.Tensor | None],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    flat = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
        else:
            flat.append(grad.reshape(-1))
    return flatten_tensors(flat)


def zeros_like_params(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    if not params:
        return torch.empty(0)
    return flatten_tensors([torch.zeros_like(p) for p in params])


def assign_flat_update(
    params: Sequence[torch.nn.Parameter],
    update: torch.Tensor,
    scale: float = 1.0,
) -> None:
    offset = 0
    with torch.no_grad():
        for param in params:
            n = param.numel()
            param.add_(update[offset : offset + n].view_as(param), alpha=scale)
            offset += n


def set_flat_params(params: Sequence[torch.nn.Parameter], values: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for param in params:
            n = param.numel()
            param.copy_(values[offset : offset + n].view_as(param))
            offset += n


def get_flat_params(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    with torch.no_grad():
        return flatten_tensors([p.detach().clone() for p in params])
