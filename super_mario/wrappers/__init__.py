"""Wrappers specific to the Super Mario Bros. game."""
from .penalize_death_env import PenalizeDeathEnv
from .reward_cache_env import RewardCacheEnv
from .to_discrete_action_space_env import ToDiscreteActionSpaceEnv


__all__ = [
    PenalizeDeathEnv.__name__,
    RewardCacheEnv.__name__,
    ToDiscreteActionSpaceEnv.__name__,
]
