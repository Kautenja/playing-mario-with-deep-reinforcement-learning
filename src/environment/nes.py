"""Methods for setting up an NES environment."""
import gym
from src.environment.wrappers import (
    ClipRewardEnv,
    DownsampleEnv,
    FrameStackEnv,
    MaxFrameskipEnv,
    NoopResetEnv,
)
from super_mario.wrappers import (
    ToDiscreteWrapper,
    PenalizeDeathEnv,
    RewardCacheEnv
)


def build_nes_environment(game_name: str,
    image_size: tuple=(84, 84),
    death_penalty: int=-1,
    clip_rewards: bool=True,
    agent_history_length: int=4
):
    """
    Build and return a configured NES environment.

    Args:
        game_name: the name of the NES game to make
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
    env = gym.make('{}-v0'.format(game_name))
    env = ToDiscreteWrapper(env)
    r_cache = RewardCacheEnv(env)
    env = r_cache
    # apply the frame skip feature if enabled
    # if skip_frames is not None:
    #     env = MaxFrameskipEnv(env, skip=skip_frames)
    # apply a down-sampler for the given game
    downsampler = DownsampleEnv.metadata[game_name.split('-')[0]]
    env = DownsampleEnv(env, image_size, **downsampler)
    # apply the death penalty feature if enabled
    if death_penalty is not None:
        env = PenalizeDeathEnv(env, penalty=death_penalty)
    # clip the rewards in {-1, 0, +1} if the feature is enabled
    if clip_rewards:
        env = ClipRewardEnv(env)
    # apply the back history of frames if the feature is enabled
    if agent_history_length is not None:
        env = FrameStackEnv(env, agent_history_length)

    return env, r_cache


# explicitly specify the outward facing API of this module
__all__ = [build_nes_environment.__name__]
