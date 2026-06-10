"""Configuration object for modern Mario environment creation."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MarioEnvConfig:
    """Configuration for :func:`mario_rl.envs.make_env`."""

    env_id: str = "SuperMarioBros-1-1-v0"
    render_mode: str | None = None
    seed: int | None = None
    action_set: str | tuple[tuple[str, ...], ...] = "simple"
    preprocess: bool = True
    frame_skip: int | None = 4
    image_size: tuple[int, int] = (84, 84)
    interpolation: str = "area"
    grayscale: bool = True
    dtype: type[np.uint8] = np.uint8
    channel_first: bool = True
    frame_stack: int | None = 4
    clip_rewards: bool = False
    record_statistics: bool = True
    video_dir: str | Path | None = None
    video_episode_trigger: Callable[[int], bool] | None = None
    video_length: int = 0
    video_name_prefix: str = "mario-rl"


_FIELD_NAMES = {field.name for field in fields(MarioEnvConfig)}
UNSET = object()


def coerce_config(config: Any | None = None, **overrides: Any) -> MarioEnvConfig:
    """Build a config from a dataclass, mapping, light object, and overrides."""
    if config is None:
        result = MarioEnvConfig()
    elif isinstance(config, MarioEnvConfig):
        result = config
    elif isinstance(config, Mapping):
        result = MarioEnvConfig(**_known_fields(dict(config)))
    else:
        values = {
            name: getattr(config, name)
            for name in _FIELD_NAMES
            if hasattr(config, name)
        }
        result = MarioEnvConfig(**values)

    explicit = {
        name: value
        for name, value in overrides.items()
        if value is not UNSET and name in _FIELD_NAMES
    }
    if explicit:
        result = replace(result, **explicit)
    return result


def _known_fields(values: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown mapping keys before creating a config object."""
    return {name: value for name, value in values.items() if name in _FIELD_NAMES}


__all__ = ["MarioEnvConfig", "UNSET", "coerce_config"]
