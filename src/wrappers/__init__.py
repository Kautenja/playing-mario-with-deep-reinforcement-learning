"""
Wrappers used in this experiment.

All of the wrappers in this package derive from, or are heavily inspired by
the wrappers found here:
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

These are provided by OpenAI baselines as a means of recreating some of the
DeepMind functionality.
"""
from .clip_reward_env import ClipRewardEnv
from .downsample_env import DownsampleEnv
from .fire_reset_env import FireResetEnv
from .frame_skip_env import FrameSkipEnv
from .frame_stack_env import FrameStackEnv
from .max_frameskip_env import MaxFrameskipEnv
from .noop_reset_env import NoopResetEnv
from .penalize_death_env import PenalizeDeathEnv
from .reward_cache_env import RewardCacheEnv


def wrap(env,
    cache_rewards: bool=True,
    noop_max: int=30,
    frame_skip: int=4,
    max_frame_skip: bool=False,
    image_size: tuple=(84, 84),
    death_penalty: float=None,
    clip_rewards: bool=False,
    agent_history_length: int=4,
):
    """
    Wrap an environment with standard wrappers.
    Args:
        env (gym.Env): the environment to wrap
        cache_rewards (bool): True to use a reward cache for raw rewards
        noop_max: the max number of random no-ops at the beginning of a game
        frame_skip (int): the number of frames to skip between observations
        image_size (tuple): the size to down-sample images to
        death_penatly (float): the penalty for losing a life in a game
        clip_rewards (bool): whether to clip rewards in {-1, 0, +1}
        agent_history_length (int): the size of the frame buffer for the agent
    Returns:
        a gym environment configured for this experiment
    """
    # wrap the environment with a reward cacher
    if cache_rewards:
        env = RewardCacheEnv(env)
    # apply the no op max feature if enabled
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    # apply the wrapper for firing at the beginning of games that require
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # wrap the environment with a frame skipper
    if frame_skip:
        if max_frame_skip:
            env = MaxFrameskipEnv(env, skip=frame_skip)
        else:
            env = FrameSkipEnv(env, frame_skip)
    # apply a down-sampler for the given game
    if image_size is not None:
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


# explicitly specify the outward facing API of this package
__all__ = [
    ClipRewardEnv.__name__,
    DownsampleEnv.__name__,
    FrameSkipEnv.__name__,
    FireResetEnv.__name__,
    FrameStackEnv.__name__,
    MaxFrameskipEnv.__name__,
    NoopResetEnv.__name__,
    PenalizeDeathEnv.__name__,
    RewardCacheEnv.__name__,
]
