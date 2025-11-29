import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor


def make_env(seed: int):
    def _init():
        env = gym.make("MiniGrid-FourRooms-v0")
        env = FullyObsWrapper(env)
        env = FlatObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init