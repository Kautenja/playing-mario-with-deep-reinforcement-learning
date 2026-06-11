"""Mario Gymnasium environment factory."""
from pathlib import Path

import gymnasium as gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from .actions import resolve_action_set
from .config import UNSET, MarioEnvConfig, coerce_config
from .wrappers import (
    ClipRewardEnv,
    DefaultSeedEnv,
    DownsampleObservationEnv,
    FrameStackEnv,
    MaxFrameskipEnv,
    OpenCVLiveRenderEnv,
    OpenCVRecordVideoEnv,
    TrainingTimeoutEnv,
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
    max_episode_steps=UNSET,
    no_progress_timeout_steps=UNSET,
    stuck_penalty=UNSET,
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
        "max_episode_steps": max_episode_steps,
        "no_progress_timeout_steps": no_progress_timeout_steps,
        "stuck_penalty": stuck_penalty,
        "video_dir": video_dir,
        "video_episode_trigger": video_episode_trigger,
        "video_length": video_length,
        "video_name_prefix": video_name_prefix,
    }
    cfg: MarioEnvConfig = coerce_config(config, **overrides)

    if cfg.video_dir is not None and cfg.render_mode not in ("rgb_array", "human"):
        raise ValueError("video recording requires render_mode='rgb_array' or 'human'")

    base_render_mode = "rgb_array" if cfg.render_mode == "human" else cfg.render_mode
    env = gym_super_mario_bros.make(cfg.env_id, render_mode=base_render_mode)
    resolved_action_set = resolve_action_set(cfg.action_set, env=env)
    if not resolved_action_set.native:
        env = JoypadSpace(env, resolved_action_set.actions)

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

    if cfg.max_episode_steps is not None or cfg.no_progress_timeout_steps is not None:
        env = TrainingTimeoutEnv(
            env,
            max_episode_steps=cfg.max_episode_steps,
            no_progress_timeout_steps=cfg.no_progress_timeout_steps,
            stuck_penalty=cfg.stuck_penalty,
        )

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

    if cfg.render_mode == "human":
        env = OpenCVLiveRenderEnv(env, window_name=cfg.env_id)

    env.mario_rl_action_set = resolved_action_set.name
    env.mario_rl_action_count = resolved_action_set.num_actions
    env.mario_rl_native_action_space = resolved_action_set.native
    return env


__all__ = ["make_env"]
