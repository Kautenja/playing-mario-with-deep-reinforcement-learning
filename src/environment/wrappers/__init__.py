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
from .frame_stack_env import FrameStackEnv
from .max_frameskip_env import MaxFrameskipEnv
from .noop_reset_env import NoopResetEnv
from .penalize_death_env import PenalizeDeathEnv
from .reward_cache_env import RewardCacheEnv


# explicitly specify the outward facing API of this package
__all__ = [
    ClipRewardEnv.__name__,
    DownsampleEnv.__name__,
    FireResetEnv.__name__,
    FrameStackEnv.__name__,
    MaxFrameskipEnv.__name__,
    NoopResetEnv.__name__,
    PenalizeDeathEnv.__name__,
    RewardCacheEnv.__name__,
]
