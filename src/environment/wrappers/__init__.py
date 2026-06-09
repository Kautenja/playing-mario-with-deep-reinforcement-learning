"""Legacy Atari wrappers used in the original experiment."""
from importlib import import_module


_WRAPPER_MODULES = {
    "ClipRewardEnv": ".clip_reward_env",
    "DownsampleEnv": ".downsample_env",
    "FireResetEnv": ".fire_reset_env",
    "FrameStackEnv": ".frame_stack_env",
    "MaxFrameskipEnv": ".max_frameskip_env",
    "NoopResetEnv": ".noop_reset_env",
    "PenalizeDeathEnv": ".penalize_death_env",
    "RewardCacheEnv": ".reward_cache_env",
}

__all__ = list(_WRAPPER_MODULES)


def __getattr__(name):
    """Lazily import wrappers and their legacy Gym dependency."""
    if name in _WRAPPER_MODULES:
        module = import_module(_WRAPPER_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
