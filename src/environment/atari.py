"""Methods for setting up an Atari environment."""
import gym
from src.environment.wrappers import (
    ClipRewardEnv,
    DownsampleEnv,
    FireResetEnv,
    FrameStackEnv,
    MaxFrameskipEnv,
    NoopResetEnv,
    PenalizeDeathEnv,
    RewardCacheEnv,
)


def build_atari_environment(game_name: str,
    is_validation: bool=False,
    image_size: tuple=(84, 84),
    noop_max: int=30,
    skip_frames: int=4,
    death_penalty: int=-1,
    clip_rewards: bool=True,
    agent_history_length: int=4
):
    """
    Build and return a configured Atari environment.

    Args:
        game_name: the name of the Atari game to make
        is_validation: whether to use the validation or training environment
        image_size: the size to down-sample images to
        noop_max: the max number of random no-ops at the beginning of a game
        skip_frames: the number of frames to hold each action for
        death_penatly: the penalty for losing a life in a game
        clip_rewards: whether to clip rewards in {-1, 0, +1}
        agent_history_length: the size of the frame buffer for the agent

    Returns:
        a gym environment configured for this experiment

    """
    # make the initial environment
    if is_validation:
        env = gym.make('{}NoFrameskip-v10'.format(game_name))
    else:
        env = gym.make('{}NoFrameskip-v4'.format(game_name))
    # wrap the environment with a reward cacher
    env = RewardCacheEnv(env)
    # apply the no op max feature if enabled
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    # apply the frame skip feature if enabled
    if skip_frames is not None:
        env = MaxFrameskipEnv(env, skip=skip_frames)
    # apply the wrapper for firing at the beginning of games that require
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # apply a down-sampler for the given game
    env = DownsampleEnv(env, image_size)
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
