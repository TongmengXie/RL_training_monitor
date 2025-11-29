# LLC-RL Benchmarked PPO on MiniGrid-FourRooms

This project trains Stable-Baselines3 PPO agents on `MiniGrid-FourRooms-v0` with and without Local Learning Coefficient (LLC) monitoring.

### Environment and reward design
- **Env**: `MiniGrid-FourRooms-v0` from `gymnasium` + `minigrid`, wrapped with `FullyObsWrapper` and `FlatObsWrapper` so the agent receives a flattened, fully observable grid.
- **Reward**: sparse MiniGrid reward; the agent receives `1 - 0.9 * (steps / max_steps)` only upon reaching the goal and `0` otherwise, incentivizing shorter paths.
- **Why this env?** FourRooms is a classic sparse-reward navigation task with partial observability in its base form; making it fully observable isolates planning and credit-assignment under sparsity. It is small enough for quick PPO iteration while still exhibiting exploration and stability challenges, making LLC-triggered adaptations interesting to study.
- **Reward threshold realism**: the reporting threshold of mean reward ≥ 0.4 aligns with typical PPO performance ranges on FourRooms (HuggingFace baseline ≈0.5). It is achievable within the 200k-step budget yet non-trivial, providing a meaningful sample-efficiency marker.

### Model and library choices
- **Algorithm**: Stable-Baselines3 PPO (`MlpPolicy`) with 2×64 ReLU MLP heads; chosen for its strong baseline performance and ease of instrumentation.
- **Packages**: `stable-baselines3`, `gymnasium`, and `minigrid` supply the PPO implementation and environment; `torch` backs the neural networks. These choices mirror widely used baselines and keep comparisons to the HuggingFace reference fair.

## Structure
- `llc_rl/src/env_utils.py`: environment factory using `FullyObsWrapper` and `FlatObsWrapper`.
- `llc_rl/src/train_baseline.py`: baseline PPO training across seeds.
- `llc_rl/src/train_llc.py`: LLC-instrumented PPO training with learning rate halving interventions.
- `llc_rl/src/ppo_llc.py`: PPO subclass that tracks LLC and applies interventions.
- `llc_rl/src/callbacks.py`: CSV logger for rewards, entropy, LLC metrics, and interventions.
- `llc_rl/src/plotting.py`: aggregates logs and produces required plots.

## Running
Install dependencies from `llc_rl/requirements.txt`, then run:

```bash
python -m llc_rl.src.train_baseline
python -m llc_rl.src.train_llc
python -m llc_rl.src.plotting
```

Artifacts are written under `llc_rl/logs`, `llc_rl/models`, and `llc_rl/plots`. To keep the repo lightweight, logs (`*.csv`), models (`*.zip`), and plots (`*.png`) are not checked in; regenerate them locally by rerunning the training and plotting scripts.

### How to regenerate and view artifacts locally
1) Install dependencies (`pip install -r llc_rl/requirements.txt`).
2) Run baseline training: `python -m llc_rl.src.train_baseline`.
3) Run LLC-PPO training: `python -m llc_rl.src.train_llc`.
4) Generate plots: `python -m llc_rl.src.plotting`.

After these commands finish, you will have fresh artifacts on your machine:
- CSV logs: `llc_rl/logs/baseline/baseline_seed{0,1,2}.csv` and `llc_rl/logs/llc/llc_seed{0,1,2}.csv`.
- Models: `llc_rl/models/baseline_seed*.zip` and `llc_rl/models/llc_seed*.zip`.
- Plots: `llc_rl/plots/reward_baseline_vs_llc.png`, `llc_rl/plots/llc_over_time.png`, `llc_rl/plots/entropy_baseline_vs_llc.png`, and `llc_rl/plots/intervention_events.png`.

Because these files are ignored by Git, they remain only on your machine after generation; if you need to share them, zip the directories or upload the images/models separately.