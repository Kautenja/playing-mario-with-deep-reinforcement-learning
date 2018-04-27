"""Methods for setting up an NES environment."""
from src.environment.wrappers import (
    ClipRewardEnv,
    DownsampleEnv,
    FrameStackEnv,
    RewardCacheEnv
)
import gym_super_mario_bros


def build_nes_environment(game_name: str,
    image_size: tuple=(84, 84),
    clip_rewards: bool=True,
    agent_history_length: int=4,
):
    """
    Build and return a configured NES environment.

    Args:
        game_name: the name of the NES game to make
        image_size: the size to down-sample images to
        clip_rewards: whether to clip rewards in {-1, 0, +1}
        agent_history_length: the size of the frame buffer for the agent

    Returns:
        a gym environment configured for this experiment

    """
    # make the initial environment
    env = gym_super_mario_bros.make('{}-v0'.format(game_name))
    # add a reward cache for scoring episodes
    env = RewardCacheEnv(env)
    # apply a down-sampler for the given game
    downsampler = DownsampleEnv.metadata[game_name.split('-')[0]]
    env = DownsampleEnv(env, image_size, **downsampler)
    # clip the rewards in {-1, 0, +1} if the feature is enabled
    if clip_rewards:
        env = ClipRewardEnv(env)
    # apply the back history of frames if the feature is enabled
    if agent_history_length is not None:
        env = FrameStackEnv(env, agent_history_length)

    return env


# explicitly specify the outward facing API of this module
__all__ = [build_nes_environment.__name__]
