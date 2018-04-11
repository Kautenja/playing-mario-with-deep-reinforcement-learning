"""Methods for setting up an Atari environment."""
import gym
from src.environment.wrappers import (
    NoopResetEnv,
    FireResetEnv,
    DownsampleEnv,
    PenalizeDeathEnv,
    ClipRewardEnv,
    FrameStackEnv
)


def build_atari_environment(game_name: str,
    env_spec: str='-v4',
    image_size: tuple=(84, 84),
    noop_max: int=30,
    death_penalty: int=-1,
    clip_rewards: bool=True,
    agent_history_length: int=4
):
    """
    Build and return a configured Atari environment.

    Args:
        game_name: the name of the Atari game to make
        env_spec: the specification for the environment
        image_size: the size to down-sample images to
        noop_max: the max number of random no-ops at the beginning of a game
        death_penatly: the penalty for losing a life in a game
        clip_rewards: whether to clip rewards in {-1, 0, +1}
        agent_history_length: the size of the frame buffer for the agent

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
    # apply the back history of frames if the feature is enabled
    if agent_history_length is not None:
        env = FrameStackEnv(env, agent_history_length)

    return env


# explicitly specify the outward facing API of this module
__all__ = [build_atari_environment.__name__]
