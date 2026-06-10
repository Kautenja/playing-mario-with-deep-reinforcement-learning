"""Gymnasium preprocessing wrappers for Mario training observations."""
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np


_INTERPOLATION = {
    "area": cv2.INTER_AREA,
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


class DefaultSeedEnv(gym.Wrapper):
    """Use a configured seed when callers reset without one."""

    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self.seed = seed
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.seed
        return self.env.reset(seed=seed, options=options)


class MaxFrameskipEnv(gym.Wrapper):
    """Repeat actions and max-pool over the last two raw observations."""

    def __init__(self, env: gym.Env, skip: int = 4):
        if skip < 1:
            raise ValueError("skip must be at least 1")
        super().__init__(env)
        self.skip = skip
        self._obs_buffer = deque(maxlen=2)

    def reset(self, *, seed=None, options=None):
        self._obs_buffer.clear()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs = None
        frames_skipped = 0
        raw_reward_sum = 0.0
        has_raw_reward = False
        unclipped_reward_sum = 0.0
        has_unclipped_reward = False
        clipped_reward_sum = 0.0
        has_clipped_reward = False
        component_sums: dict[str, float] = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            frames_skipped += 1
            total_reward += float(reward)
            if isinstance(info, dict):
                raw_reward = info.get("raw_reward")
                if raw_reward is not None:
                    raw_reward_sum += float(raw_reward)
                    has_raw_reward = True
                unclipped_reward = info.get("reward_total_unclipped")
                if unclipped_reward is not None:
                    unclipped_reward_sum += float(unclipped_reward)
                    has_unclipped_reward = True
                clipped_reward = info.get("reward_total_clipped")
                if clipped_reward is not None:
                    clipped_reward_sum += float(clipped_reward)
                    has_clipped_reward = True
                components = info.get("reward_components")
                if isinstance(components, dict):
                    for name, value in components.items():
                        component_sums[str(name)] = (
                            component_sums.get(str(name), 0.0) + float(value)
                        )
            self._obs_buffer.append(np.array(obs, copy=True))
            if terminated or truncated:
                break

        if obs is None:
            raise RuntimeError("frame skip wrapper did not step the environment")
        if len(self._obs_buffer) == 2:
            obs = np.maximum(self._obs_buffer[0], self._obs_buffer[1])

        info = dict(info)
        if has_raw_reward:
            info["raw_reward"] = raw_reward_sum
        if has_unclipped_reward:
            info["reward_total_unclipped"] = unclipped_reward_sum
        if has_clipped_reward:
            info["reward_total_clipped"] = clipped_reward_sum
        if component_sums:
            info["reward_components"] = component_sums
        info["frames_skipped"] = frames_skipped
        return obs, total_reward, terminated, truncated, info


class DownsampleObservationEnv(gym.ObservationWrapper):
    """Resize RGB observations with explicit color, dtype, and axis behavior."""

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] = (84, 84),
        interpolation: str = "area",
        grayscale: bool = True,
        dtype=np.uint8,
        channel_first: bool = True,
    ):
        super().__init__(env)
        if interpolation not in _INTERPOLATION:
            choices = ", ".join(sorted(_INTERPOLATION))
            raise ValueError(f"unknown interpolation {interpolation!r}; choose {choices}")

        self.image_size = image_size
        self.interpolation = interpolation
        self.grayscale = grayscale
        self.dtype = np.dtype(dtype)
        self.channel_first = channel_first

        height, width = image_size
        channels = 1 if grayscale else 3
        if channel_first:
            shape = (channels, height, width)
        else:
            shape = (height, width, channels)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=self.dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        height, width = self.image_size
        resized = cv2.resize(
            frame,
            (width, height),
            interpolation=_INTERPOLATION[self.interpolation],
        )
        if self.grayscale:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            resized = resized[:, :, np.newaxis]
        if self.channel_first:
            resized = np.moveaxis(resized, -1, 0)
        return resized.astype(self.dtype, copy=False)


class FrameStackEnv(gym.Wrapper):
    """Concatenate the last ``num_stack`` observations along the channel axis."""

    def __init__(self, env: gym.Env, num_stack: int = 4, channel_first: bool = True):
        if num_stack < 1:
            raise ValueError("num_stack must be at least 1")
        super().__init__(env)
        self.num_stack = num_stack
        self.channel_first = channel_first
        self.frames = deque(maxlen=num_stack)

        axis = 0 if channel_first else -1
        low = np.concatenate([env.observation_space.low] * num_stack, axis=axis)
        high = np.concatenate([env.observation_space.high] * num_stack, axis=axis)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(np.array(obs, copy=True))
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.array(obs, copy=True))
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        if len(self.frames) != self.num_stack:
            raise RuntimeError("frame stack is not initialized")
        axis = 0 if self.channel_first else -1
        return np.concatenate(tuple(self.frames), axis=axis)


class ClipRewardEnv(gym.Wrapper):
    """Clip rewards to their sign while preserving raw episode totals."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.raw_episode_reward = 0.0
        self.clipped_episode_reward = 0.0

    def reset(self, *, seed=None, options=None):
        self.raw_episode_reward = 0.0
        self.clipped_episode_reward = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(reward)
        clipped_reward = float(np.sign(raw_reward))
        self.raw_episode_reward += raw_reward
        self.clipped_episode_reward += clipped_reward

        info = dict(info)
        info["raw_reward"] = raw_reward
        if terminated or truncated:
            info["episode_raw_reward"] = self.raw_episode_reward
            info["episode_clipped_reward"] = self.clipped_episode_reward

        return obs, clipped_reward, terminated, truncated, info


class OpenCVRecordVideoEnv(gym.Wrapper):
    """Record rendered RGB frames to MP4 files with Gymnasium wrapper semantics."""

    def __init__(
        self,
        env: gym.Env,
        video_dir: str | Path,
        episode_trigger=None,
        video_length: int = 0,
        name_prefix: str = "mario-rl",
    ):
        super().__init__(env)
        self.video_dir = Path(video_dir)
        self.episode_trigger = episode_trigger or (lambda episode_id: True)
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.episode_id = -1
        self._writer = None
        self._frames_recorded = 0
        self._video_path = None
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, *, seed=None, options=None):
        self._close_writer()
        obs, info = self.env.reset(seed=seed, options=options)
        self.episode_id += 1
        if self.episode_trigger(self.episode_id):
            self._start_writer()
            self._record_rendered_frame()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._record_rendered_frame()
        if terminated or truncated:
            self._close_writer()
        return obs, reward, terminated, truncated, info

    def close(self):
        self._close_writer()
        return self.env.close()

    def _start_writer(self):
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("video recording requires render_mode='rgb_array'")

        frame = np.asarray(frame)
        height, width = frame.shape[:2]
        fps = int(self.metadata.get("render_fps", 60))
        self._video_path = self.video_dir / (
            f"{self.name_prefix}-episode-{self.episode_id}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self._video_path),
            fourcc,
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self._video_path}")
        self._frames_recorded = 0

    def _record_rendered_frame(self):
        if self._writer is None:
            return
        if self.video_length and self._frames_recorded >= self.video_length:
            self._close_writer()
            return

        frame = self.env.render()
        if frame is None:
            return
        frame = np.asarray(frame)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)
        self._frames_recorded += 1
        if self.video_length and self._frames_recorded >= self.video_length:
            self._close_writer()

    def _close_writer(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


__all__ = [
    "ClipRewardEnv",
    "DefaultSeedEnv",
    "DownsampleObservationEnv",
    "FrameStackEnv",
    "MaxFrameskipEnv",
    "OpenCVRecordVideoEnv",
]
