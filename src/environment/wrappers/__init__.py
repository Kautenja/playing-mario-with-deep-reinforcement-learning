"""Wrappers used in this experiment."""
from .clip_reward_env import ClipRewardEnv
from .fire_reset_env import FireResetEnv
from .noop_reset_env import NoopResetEnv
from .penalized_death_env import PenalizeDeathEnv
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
