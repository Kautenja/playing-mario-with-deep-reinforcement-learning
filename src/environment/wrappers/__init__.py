"""
Wrappers used in this experiment.

All of the wrappers in this package derive from, or are heavily inspired by
the wrappers found here:
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

These are provided by OpenAI baselines as a means of recreating some of the
DeepMind functionality.
"""
from .clip_reward_env import ClipRewardEnv
from .fire_reset_env import FireResetEnv
from .noop_reset_env import NoopResetEnv
from .penalize_death_env import PenalizeDeathEnv
from .downsample_env import DownsampleEnv
from .frame_stack_env import FrameStackEnv


# explicitly specify the outward facing API of this package
__all__ = [
    ClipRewardEnv.__name__,
    FireResetEnv.__name__,
    NoopResetEnv.__name__,
    PenalizeDeathEnv.__name__,
    DownsampleEnv.__name__,
    FrameStackEnv.__name__,
]
