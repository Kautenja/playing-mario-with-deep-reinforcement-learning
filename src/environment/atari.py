"""Methods for setting up an Atari environment."""
import gym
from src.environment.wrappers import (
    NoopResetEnv,
    PenalizeDeathEnv,
    ClipRewardEnv,
    FireResetEnv,
    DownsampleEnv
)


def build_atari_environment(game_name: str,
    env_spec: str='-v4',
    image_size: tuple=(84, 84),
    noop_max: int=30,
    death_penalty: int=-1,
    clip_rewards: bool=True,
):
    """
    Build and return a configured Atari environment.

    Args:
        TODO

    Returns:
        a gym environment configured for this experiment

    """
    # make the initial environment
    env = gym.make(game_name + env_spec)
    # apply the no op max feature if enabled
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    # apply the wrapper for firing at the beginning of games that require
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # apply a down-sampler for the given game
    env = DownsampleEnv(env, image_size, **DownsampleEnv.metadata[game_name])
    # apply the death penalty feature if enabled
    if death_penalty is not None:
        env = PenalizeDeathEnv(env, penalty=death_penalty)
    # clip the rewards in {-1, 0, +1} if the feature is enabled
    if clip_rewards:
        env = ClipRewardEnv(env)

    return env
