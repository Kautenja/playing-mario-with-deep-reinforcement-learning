"""Gymnasium preprocessing wrappers for Mario training observations."""
from collections import deque
from collections.abc import Sequence
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


class _StepInfoAccumulator:
    """Accumulate reward diagnostics across repeated low-level env steps."""

    def __init__(self) -> None:
        self.total_reward = 0.0
        self.frames_skipped = 0
        self.raw_reward_sum = 0.0
        self.has_raw_reward = False
        self.unclipped_reward_sum = 0.0
        self.has_unclipped_reward = False
        self.clipped_reward_sum = 0.0
        self.has_clipped_reward = False
        self.component_sums: dict[str, float] = {}

    def add(self, reward: float, info: dict[str, Any] | None, *, frame_count: int = 1) -> None:
        self.total_reward += float(reward)
        self.frames_skipped += max(int(frame_count), 1)
        if not isinstance(info, dict):
            return
        raw_reward = info.get("raw_reward")
        if raw_reward is not None:
            self.raw_reward_sum += float(raw_reward)
            self.has_raw_reward = True
        unclipped_reward = info.get("reward_total_unclipped")
        if unclipped_reward is not None:
            self.unclipped_reward_sum += float(unclipped_reward)
            self.has_unclipped_reward = True
        clipped_reward = info.get("reward_total_clipped")
        if clipped_reward is not None:
            self.clipped_reward_sum += float(clipped_reward)
            self.has_clipped_reward = True
        components = info.get("reward_components")
        if isinstance(components, dict):
            for name, value in components.items():
                key = str(name)
                self.component_sums[key] = self.component_sums.get(key, 0.0) + float(value)

    def apply(self, info: dict[str, Any] | None) -> dict[str, Any]:
        result = dict(info or {})
        if self.has_raw_reward:
            result["raw_reward"] = self.raw_reward_sum
        if self.has_unclipped_reward:
            result["reward_total_unclipped"] = self.unclipped_reward_sum
        if self.has_clipped_reward:
            result["reward_total_clipped"] = self.clipped_reward_sum
        if self.component_sums:
            result["reward_components"] = dict(self.component_sums)
        result["frames_skipped"] = self.frames_skipped
        return result


def _info_frame_count(info: dict[str, Any] | None) -> int:
    if not isinstance(info, dict):
        return 1
    value = info.get("frames_skipped", 1)
    return max(int(value), 1)


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
        accumulator = _StepInfoAccumulator()
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs = None

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            accumulator.add(float(reward), info, frame_count=1)
            self._obs_buffer.append(np.array(obs, copy=True))
            if terminated or truncated:
                break

        if obs is None:
            raise RuntimeError("frame skip wrapper did not step the environment")
        if len(self._obs_buffer) == 2:
            obs = np.maximum(self._obs_buffer[0], self._obs_buffer[1])

        info = accumulator.apply(info)
        return obs, accumulator.total_reward, terminated, truncated, info


class MacroActionEnv(gym.Wrapper):
    """Expose named deterministic sequences of existing Joypad actions."""

    def __init__(
        self,
        env: gym.Env,
        macro_actions: Sequence[Any],
        *,
        macro_action_set: str,
    ):
        actions = tuple(macro_actions)
        if not actions:
            raise ValueError("macro_actions must contain at least one action")
        super().__init__(env)
        self.macro_actions = actions
        self.macro_action_set = str(macro_action_set)
        self.action_space = gym.spaces.Discrete(len(actions))

    def step(self, action):
        macro_index = int(action)
        try:
            macro = self.macro_actions[macro_index]
        except IndexError as exc:
            raise ValueError(f"macro action index out of range: {macro_index}") from exc

        accumulator = _StepInfoAccumulator()
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs = None
        macro_steps = 0

        for joypad_action in tuple(macro.action_indices):
            obs, reward, terminated, truncated, info = self.env.step(int(joypad_action))
            macro_steps += 1
            accumulator.add(
                float(reward),
                info if isinstance(info, dict) else None,
                frame_count=_info_frame_count(info),
            )
            if terminated or truncated:
                break

        if obs is None:
            raise RuntimeError("macro action wrapper did not step the environment")

        info = accumulator.apply(info)
        info["macro_action"] = macro_index
        info["macro_action_name"] = str(macro.name)
        info["macro_action_sequence"] = [int(index) for index in macro.action_indices]
        info["macro_action_length"] = int(len(macro.action_indices))
        info["macro_steps"] = int(macro_steps)
        info["macro_actions_enabled"] = True
        info["macro_action_set"] = self.macro_action_set
        return obs, accumulator.total_reward, terminated, truncated, info


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


class TrainingTimeoutEnv(gym.Wrapper):
    """Truncate training episodes that run too long or stop making progress."""

    _PROGRESS_KEYS = (
        "progress_max",
        "position_progress_max",
        "progress",
        "position_progress",
        "x_pos",
        "x_position",
    )

    def __init__(
        self,
        env: gym.Env,
        *,
        max_episode_steps: int | None = 4000,
        no_progress_timeout_steps: int | None = 600,
        stuck_penalty: float = 0.01,
    ):
        super().__init__(env)
        self.max_episode_steps = _positive_optional_int(
            max_episode_steps,
            "max_episode_steps",
        )
        self.no_progress_timeout_steps = _positive_optional_int(
            no_progress_timeout_steps,
            "no_progress_timeout_steps",
        )
        self.stuck_penalty = float(stuck_penalty)
        self._episode_steps = 0
        self._steps_since_progress = 0
        self._best_progress = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_steps = 0
        self._steps_since_progress = 0
        self._best_progress = self._progress_value(info if isinstance(info, dict) else {})
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        self._episode_steps += 1
        self._update_progress(info)
        timeout = False
        timeout_reason = None

        if (
            self.max_episode_steps is not None
            and self._episode_steps >= self.max_episode_steps
            and not terminated
            and not truncated
        ):
            timeout = True
            timeout_reason = "max_episode_steps"
        elif (
            self.no_progress_timeout_steps is not None
            and self._steps_since_progress >= self.no_progress_timeout_steps
            and not terminated
            and not truncated
        ):
            timeout = True
            timeout_reason = "no_progress"

        if self.stuck_penalty and self._steps_since_progress > 0 and not terminated:
            reward = float(reward) - self.stuck_penalty

        if timeout:
            truncated = True
            info["timeout"] = True
            info["training_timeout"] = True
            info["training_timeout_reason"] = timeout_reason
        info["episode_steps"] = self._episode_steps
        info["no_progress_steps"] = self._steps_since_progress
        return obs, float(reward), terminated, truncated, info

    def _update_progress(self, info: dict[str, Any]) -> None:
        progress = self._progress_value(info)
        if progress is None:
            return
        if self._best_progress is None or progress > self._best_progress:
            self._best_progress = progress
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

    @classmethod
    def _progress_value(cls, info: dict[str, Any]) -> float | None:
        for key in cls._PROGRESS_KEYS:
            value = info.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None


class OpenCVLiveRenderEnv(gym.Wrapper):
    """Display rendered RGB frames with OpenCV after reset and every step."""

    def __init__(
        self,
        env: gym.Env,
        *,
        window_name: str = "mario-rl",
        wait_ms: int = 1,
    ):
        super().__init__(env)
        self.window_name = str(window_name)
        self.wait_ms = int(wait_ms)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._show_frame()
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self._show_frame()
        return result

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        return self.env.close()

    def _show_frame(self):
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("live rendering requires render_mode='rgb_array'")
        frame = np.asarray(frame)
        cv2.imshow(self.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(max(self.wait_ms, 1))


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
    "OpenCVLiveRenderEnv",
    "OpenCVRecordVideoEnv",
    "TrainingTimeoutEnv",
]


def _positive_optional_int(value, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value
