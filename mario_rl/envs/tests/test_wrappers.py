"""Gymnasium preprocessing wrapper contract tests."""
from unittest import TestCase

import gymnasium as gym
import numpy as np

from mario_rl.envs.wrappers import (
    ClipRewardEnv,
    DownsampleObservationEnv,
    FrameStackEnv,
    MaxFrameskipEnv,
)


class TinyImageEnv(gym.Env):
    """Small deterministic image env for wrapper tests."""

    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(8, 10, 3),
        dtype=np.uint8,
    )

    def __init__(self, steps):
        self.steps = list(steps)
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self._obs(0), {"seed": seed, "reset_options": options}

    def step(self, action):
        reward, terminated, truncated, info = self.steps[self.index]
        self.index += 1
        return self._obs(self.index), reward, terminated, truncated, dict(info)

    def _obs(self, value):
        return np.full(self.observation_space.shape, value, dtype=np.uint8)


class PreprocessingWrappersTest(TestCase):
    """Validate the modern wrappers against Gymnasium semantics."""

    def test_downsample_and_frame_stack_use_channel_first_training_shape(self):
        env = TinyImageEnv([(0.0, False, False, {})])
        env = DownsampleObservationEnv(
            env,
            image_size=(4, 5),
            interpolation="nearest",
            grayscale=True,
            dtype=np.uint8,
            channel_first=True,
        )
        env = FrameStackEnv(env, num_stack=4, channel_first=True)

        try:
            obs, info = env.reset(seed=123, options={"mode": "test"})
            self.assertEqual((4, 4, 5), obs.shape)
            self.assertEqual(np.uint8, obs.dtype)
            self.assertEqual(123, info["seed"])
            self.assertEqual((4, 4, 5), env.observation_space.shape)
        finally:
            env.close()

    def test_frame_skip_stops_on_terminated_and_preserves_reward_sum(self):
        env = TinyImageEnv([
            (1.5, False, False, {"score": 10}),
            (2.5, True, False, {"score": 20}),
        ])
        env = MaxFrameskipEnv(env, skip=4)

        try:
            obs, reward, terminated, truncated, info = env.step(0)
            self.assertEqual((8, 10, 3), obs.shape)
            self.assertEqual(4.0, reward)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(2, info["frames_skipped"])
            self.assertEqual(20, info["score"])
        finally:
            env.close()

    def test_frame_skip_stops_on_truncated_and_reward_clipper_keeps_raw_reward(self):
        env = TinyImageEnv([
            (-0.25, False, False, {"score": 5}),
            (-3.5, False, True, {"score": 6}),
        ])
        env = MaxFrameskipEnv(env, skip=4)
        env = ClipRewardEnv(env)

        try:
            _, reward, terminated, truncated, info = env.step(0)
            self.assertEqual(-1.0, reward)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(-3.75, info["raw_reward"])
            self.assertEqual(-3.75, info["episode_raw_reward"])
            self.assertEqual(-1.0, info["episode_clipped_reward"])
        finally:
            env.close()
