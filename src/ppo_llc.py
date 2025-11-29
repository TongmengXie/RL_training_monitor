from __future__ import annotations

from typing import List

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import RolloutReturn

from .llc_utils import LLCMetrics, aggregate_llc, compute_llc, delta_norm, flatten_params, grad_norm


class LLCPPO(PPO):
    def __init__(self, *args, max_interventions: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_interventions = max_interventions
        self.llc_history: List[float] = []
        self.num_interventions = 0
        self.last_llc = np.nan
        self.last_grad_norm = np.nan
        self.last_delta_norm = np.nan
        self.last_intervention = False
        self.last_lr = None
        self._patch_optimizer()

    def _patch_optimizer(self) -> None:
        original_step = self.policy.optimizer.step
        model = self

        def patched_step(closure=None):
            params = list(model.policy.parameters())
            before = flatten_params(params)
            grad = grad_norm(params)
            result = original_step(closure)
            after = flatten_params(params)
            delta = delta_norm(before, after)
            llc_value = compute_llc(delta, grad)
            model._llc_records.append(LLCMetrics(grad_norm=grad, delta_norm=delta, llc=llc_value))
            return result

        self.policy.optimizer.step = patched_step  # type: ignore[method-assign]

    def train(self) -> None:
        self._llc_records: List[LLCMetrics] = []
        self.last_intervention = False
        super().train()
        grad, delta, llc_value = aggregate_llc(self._llc_records)
        self.last_grad_norm = grad
        self.last_delta_norm = delta
        self.last_llc = llc_value
        if llc_value > 0:
            self.llc_history.append(llc_value)
        self.last_lr = self.policy.optimizer.param_groups[0]["lr"]
        self._maybe_intervene()

    def _maybe_intervene(self) -> None:
        if len(self.llc_history) < 20:
            return
        mu = np.mean(self.llc_history)
        sigma = np.std(self.llc_history)
        if self.last_llc > mu + sigma and self.num_interventions < self.max_interventions:
            for group in self.policy.optimizer.param_groups:
                group["lr"] *= 0.5
            self.num_interventions += 1
            self.last_intervention = True
            self.last_lr = self.policy.optimizer.param_groups[0]["lr"]

    def collect_rollouts(self, *args, **kwargs) -> RolloutReturn:
        # reset per-rollout LLC stats
        self._llc_records = []
        return super().collect_rollouts(*args, **kwargs)