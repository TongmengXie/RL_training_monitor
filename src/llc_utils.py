from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch


EPS = 1e-8


def flatten_params(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in params])


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    grads = [p.grad.detach().view(-1) for p in params if p.grad is not None]
    if not grads:
        return 0.0
    stacked = torch.cat(grads)
    return torch.norm(stacked, p=2).item()


def delta_norm(before: torch.Tensor, after: torch.Tensor) -> float:
    return torch.norm(after - before, p=2).item()


def compute_llc(delta: float, grad: float, eps: float = EPS) -> float:
    return delta / (grad + eps)


@dataclass
class LLCMetrics:
    grad_norm: float
    delta_norm: float
    llc: float


def aggregate_llc(records: List[LLCMetrics]) -> Tuple[float, float, float]:
    if not records:
        return 0.0, 0.0, 0.0
    grad = float(np.mean([r.grad_norm for r in records]))
    delta = float(np.mean([r.delta_norm for r in records]))
    llc = float(np.mean([r.llc for r in records]))
    return grad, delta, llc