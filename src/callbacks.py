import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback


class StatsRecorder(BaseCallback):
    def __init__(self, csv_path: str, run_type: str, seed: int):
        super().__init__()
        self.csv_path = csv_path
        self.run_type = run_type
        self.seed = seed
        self.rows: List[dict] = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        mean_reward = float(np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])) if len(self.model.ep_info_buffer) > 0 else np.nan
        success_entries = [ep_info.get("is_success", np.nan) for ep_info in self.model.ep_info_buffer]
        success_rate = float(np.mean(success_entries)) if len(success_entries) > 0 else np.nan
        entropy = self._compute_entropy()
        grad_norm = getattr(self.model, "last_grad_norm", np.nan)
        delta_norm = getattr(self.model, "last_delta_norm", np.nan)
        llc = getattr(self.model, "last_llc", np.nan)
        intervention = 1 if getattr(self.model, "last_intervention", False) else 0
        current_lr = self._current_lr()

        self.rows.append(
            {
                "timestep": self.num_timesteps,
                "seed": self.seed,
                "run_type": self.run_type,
                "mean_episode_reward": mean_reward,
                "success_rate": success_rate,
                "policy_entropy": entropy,
                "grad_norm": grad_norm,
                "delta_norm": delta_norm,
                "llc": llc,
                "intervention_flag": intervention,
                "current_lr": current_lr,
                "elapsed_sec": time.time() - self.start_time,
            }
        )

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.rows)
        df.to_csv(self.csv_path, index=False)

    def _compute_entropy(self) -> float:
        buffer = getattr(self.model, "rollout_buffer", None)
        if buffer is None or buffer.size() == 0:
            return np.nan
        obs = torch.as_tensor(buffer.observations, device=self.model.device)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs)
            entropy = dist.entropy()
            if entropy.dim() > 1:
                entropy = entropy.mean(dim=1)
        return float(entropy.mean().item())

    def _current_lr(self) -> Optional[float]:
        if hasattr(self.model.policy, "optimizer"):
            if len(self.model.policy.optimizer.param_groups) > 0:
                return float(self.model.policy.optimizer.param_groups[0]["lr"])
        return np.nan