"""Mario Gymnasium environment factory."""
from pathlib import Path

import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from .actions import get_action_set
from .config import UNSET, MarioEnvConfig, coerce_config
from .wrappers import (
    ClipRewardEnv,
    DefaultSeedEnv,
    DownsampleObservationEnv,
    FrameStackEnv,
    MaxFrameskipEnv,
    OpenCVRecordVideoEnv,
)


def make_env(
    env_id: str | None = None,
    *,
    config=None,
    render_mode=UNSET,
    seed=UNSET,
    action_set=UNSET,
    preprocess=UNSET,
    frame_skip=UNSET,
    image_size=UNSET,
    interpolation=UNSET,
    grayscale=UNSET,
    dtype=UNSET,
    channel_first=UNSET,
    frame_stack=UNSET,
    clip_rewards=UNSET,
    record_statistics=UNSET,
    video_dir=UNSET,
    video_episode_trigger=UNSET,
    video_length=UNSET,
    video_name_prefix=UNSET,
) -> gym.Env:
    """
    Create a Gymnasium-compatible Super Mario Bros. environment.

    The default preprocessing emits channel-first grayscale stacks with shape
    ``(frame_stack, height, width)`` for direct PyTorch tensor conversion.
    """
    overrides = {
        "env_id": env_id if env_id is not None else UNSET,
        "render_mode": render_mode,
        "seed": seed,
        "action_set": action_set,
        "preprocess": preprocess,
        "frame_skip": frame_skip,
        "image_size": image_size,
        "interpolation": interpolation,
        "grayscale": grayscale,
        "dtype": dtype,
        "channel_first": channel_first,
        "frame_stack": frame_stack,
        "clip_rewards": clip_rewards,
        "record_statistics": record_statistics,
        "video_dir": video_dir,
        "video_episode_trigger": video_episode_trigger,
        "video_length": video_length,
        "video_name_prefix": video_name_prefix,
    }
    cfg: MarioEnvConfig = coerce_config(config, **overrides)

    if cfg.video_dir is not None and cfg.render_mode != "rgb_array":
        raise ValueError("video recording requires render_mode='rgb_array'")

    env = gym_super_mario_bros.make(cfg.env_id, render_mode=cfg.render_mode)
    env = JoypadSpace(env, get_action_set(cfg.action_set))

    if cfg.seed is not None:
        env = DefaultSeedEnv(env, cfg.seed)

    if cfg.preprocess:
        if cfg.frame_skip is not None and cfg.frame_skip > 1:
            env = MaxFrameskipEnv(env, skip=cfg.frame_skip)

        env = DownsampleObservationEnv(
            env,
            image_size=cfg.image_size,
            interpolation=cfg.interpolation,
            grayscale=cfg.grayscale,
            dtype=cfg.dtype,
            channel_first=cfg.channel_first,
        )

        if cfg.frame_stack is not None and cfg.frame_stack > 1:
            env = FrameStackEnv(
                env,
                num_stack=cfg.frame_stack,
                channel_first=cfg.channel_first,
            )

    if cfg.clip_rewards:
        env = ClipRewardEnv(env)

    if cfg.record_statistics:
        env = gym.wrappers.RecordEpisodeStatistics(env)

    if cfg.video_dir is not None:
        env = OpenCVRecordVideoEnv(
            env,
            video_dir=Path(cfg.video_dir),
            episode_trigger=cfg.video_episode_trigger,
            video_length=cfg.video_length,
            name_prefix=cfg.video_name_prefix,
        )

    return env


__all__ = ["make_env"]
