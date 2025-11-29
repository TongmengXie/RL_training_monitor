import os
from typing import List

from stable_baselines3.common.vec_env import DummyVecEnv

from .callbacks import StatsRecorder
from .env_utils import make_env
from .ppo_llc import LLCPPO

HYPERPARAMS = {
    "learning_rate": 2.5e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

SEEDS: List[int] = [0, 1, 2]
TOTAL_TIMESTEPS = 20_000


def run_llc():
    os.makedirs("logs/llc", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    for seed in SEEDS:
        env = DummyVecEnv([make_env(seed)])
        model = LLCPPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            **HYPERPARAMS,
        )
        callback = StatsRecorder(
            csv_path=f"logs/llc/llc_seed{seed}.csv",
            run_type="llc",
            seed=seed,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        model.save(f"models/llc_seed{seed}.zip")
        env.close()


if __name__ == "__main__":
    run_llc()