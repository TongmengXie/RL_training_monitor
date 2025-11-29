from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

BASELINE_LOGS = [f"logs/baseline/baseline_seed{s}.csv" for s in [0, 1, 2]]
LLC_LOGS = [f"logs/llc/llc_seed{s}.csv" for s in [0, 1, 2]]
OUTPUT_DIR = "plots"


def load_runs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = df.groupby("timestep")[metric]
    return pd.DataFrame({"timestep": grouped.mean().index, "mean": grouped.mean().values, "std": grouped.std().values})


def plot_reward(baseline_df: pd.DataFrame, llc_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    base_stats = aggregate(baseline_df, "mean_episode_reward")
    llc_stats = aggregate(llc_df, "mean_episode_reward")
    plt.plot(base_stats["timestep"], base_stats["mean"], label="Baseline")
    plt.fill_between(base_stats["timestep"], base_stats["mean"] - base_stats["std"], base_stats["mean"] + base_stats["std"], alpha=0.2)
    plt.plot(llc_stats["timestep"], llc_stats["mean"], label="LLC-PPO")
    plt.fill_between(llc_stats["timestep"], llc_stats["mean"] - llc_stats["std"], llc_stats["mean"] + llc_stats["std"], alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "reward_baseline_vs_llc.png"))
    plt.close()


def plot_llc(llc_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    llc_stats = aggregate(llc_df, "llc")
    plt.plot(llc_stats["timestep"], llc_stats["mean"], label="LLC")
    plt.fill_between(llc_stats["timestep"], llc_stats["mean"] - llc_stats["std"], llc_stats["mean"] + llc_stats["std"], alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("LLC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "llc_over_time.png"))
    plt.close()


def plot_entropy(baseline_df: pd.DataFrame, llc_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    base_stats = aggregate(baseline_df, "policy_entropy")
    llc_stats = aggregate(llc_df, "policy_entropy")
    plt.plot(base_stats["timestep"], base_stats["mean"], label="Baseline")
    plt.fill_between(base_stats["timestep"], base_stats["mean"] - base_stats["std"], base_stats["mean"] + base_stats["std"], alpha=0.2)
    plt.plot(llc_stats["timestep"], llc_stats["mean"], label="LLC-PPO")
    plt.fill_between(llc_stats["timestep"], llc_stats["mean"] - llc_stats["std"], llc_stats["mean"] + llc_stats["std"], alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Policy Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "entropy_baseline_vs_llc.png"))
    plt.close()


def plot_interventions(llc_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    reward_stats = aggregate(llc_df, "mean_episode_reward")
    plt.plot(reward_stats["timestep"], reward_stats["mean"], label="LLC-PPO reward")
    for seed in llc_df["seed"].unique():
        seed_df = llc_df[llc_df["seed"] == seed]
        intervention_steps = seed_df.loc[seed_df["intervention_flag"] == 1, "timestep"].values
        for step in intervention_steps:
            plt.axvline(step, color="red", linestyle="--", alpha=0.3)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "intervention_events.png"))
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    baseline_df = load_runs(BASELINE_LOGS)
    llc_df = load_runs(LLC_LOGS)
    if baseline_df.empty or llc_df.empty:
        raise FileNotFoundError("Missing logs for plotting. Please run training scripts first.")
    plot_reward(baseline_df, llc_df)
    plot_llc(llc_df)
    plot_entropy(baseline_df, llc_df)
    plot_interventions(llc_df)


if __name__ == "__main__":
    main()