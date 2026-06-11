"""Gymnasium preprocessing wrapper contract tests."""
from unittest import TestCase
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from mario_rl.envs.wrappers import (
    ClipRewardEnv,
    DownsampleObservationEnv,
    FrameStackEnv,
    MacroActionEnv,
    MaxFrameskipEnv,
    OpenCVLiveRenderEnv,
    TrainingTimeoutEnv,
)
from mario_rl.envs.actions import MacroAction
from mario_rl.rewards import RewardTransformConfig, RewardTransformer


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
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self._obs(0), {"seed": seed, "reset_options": options}

    def step(self, action):
        self.actions.append(int(action))
        reward, terminated, truncated, info = self.steps[self.index]
        self.index += 1
        return self._obs(self.index), reward, terminated, truncated, dict(info)

    def _obs(self, value):
        return np.full(self.observation_space.shape, value, dtype=np.uint8)


class RenderCountingEnv(TinyImageEnv):
    """Tiny image env that counts explicit render calls."""

    def __init__(self, steps):
        super().__init__(steps)
        self.render_calls = 0

    def render(self):
        self.render_calls += 1
        return self._obs(self.index)


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

    def test_opencv_live_render_env_shows_frames_after_reset_and_step(self):
        base_env = RenderCountingEnv([(0.0, False, False, {})])
        env = OpenCVLiveRenderEnv(base_env, window_name="test-window")

        with (
            patch("mario_rl.envs.wrappers.cv2.imshow") as imshow,
            patch("mario_rl.envs.wrappers.cv2.waitKey") as wait_key,
            patch("mario_rl.envs.wrappers.cv2.destroyWindow"),
        ):
            try:
                env.reset(seed=123)
                self.assertEqual(1, base_env.render_calls)
                env.step(0)
                self.assertEqual(2, base_env.render_calls)
                self.assertEqual(2, imshow.call_count)
                self.assertEqual(2, wait_key.call_count)
            finally:
                env.close()

    def test_training_timeout_env_truncates_at_hard_step_limit(self):
        env = TrainingTimeoutEnv(
            TinyImageEnv([
                (1.0, False, False, {"progress": 0}),
                (1.0, False, False, {"progress": 1}),
            ]),
            max_episode_steps=2,
            no_progress_timeout_steps=None,
            stuck_penalty=0.0,
        )

        try:
            env.reset(seed=123)
            _, _, _, truncated, _ = env.step(0)
            self.assertFalse(truncated)
            _, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(info["training_timeout"])
            self.assertEqual("max_episode_steps", info["training_timeout_reason"])
        finally:
            env.close()

    def test_training_timeout_env_truncates_when_progress_stalls(self):
        env = TrainingTimeoutEnv(
            TinyImageEnv([
                (1.0, False, False, {"progress": 0}),
                (1.0, False, False, {"progress": 0}),
            ]),
            max_episode_steps=None,
            no_progress_timeout_steps=1,
            stuck_penalty=0.25,
        )

        try:
            env.reset(seed=123)
            _, reward, _, truncated, info = env.step(0)
            self.assertFalse(truncated)
            self.assertEqual(1.0, reward)
            self.assertEqual(0, info["no_progress_steps"])
            _, reward, _, truncated, info = env.step(0)
            self.assertTrue(truncated)
            self.assertEqual(0.75, reward)
            self.assertTrue(info["training_timeout"])
            self.assertEqual("no_progress", info["training_timeout_reason"])
        finally:
            env.close()


class RewardTransformTest(TestCase):
    """Validate training-reward transforms against fake step info."""

    def test_env_sign_unclipped_clipped_and_component_modes(self):
        info = {
            "raw_reward": 4.0,
            "reward_total_unclipped": 4.0,
            "reward_total_clipped": 2.0,
            "reward_components": {
                "progress": 5.0,
                "death": -1.0,
            },
        }

        self.assertEqual(
            -3.0,
            RewardTransformer(RewardTransformConfig(mode="env")).transform(-3.0, info).training_reward,
        )
        self.assertEqual(
            -1.0,
            RewardTransformer(RewardTransformConfig(mode="sign")).transform(-3.0, info).training_reward,
        )
        self.assertEqual(
            4.0,
            RewardTransformer(RewardTransformConfig(mode="unclipped")).transform(-3.0, info).training_reward,
        )
        self.assertEqual(
            2.0,
            RewardTransformer(RewardTransformConfig(mode="clipped")).transform(-3.0, info).training_reward,
        )
        self.assertEqual(
            3.0,
            RewardTransformer(
                RewardTransformConfig(
                    mode="component_weights",
                    component_weights={"progress": 0.5, "death": -0.5},
                )
            ).transform(-3.0, info).training_reward,
        )

    def test_missing_total_fields_use_explicit_error_or_env_fallback(self):
        with self.assertRaises(KeyError):
            RewardTransformer(RewardTransformConfig(mode="unclipped")).transform(3.0, {})

        fallback = RewardTransformer(
            RewardTransformConfig(
                mode="clipped",
                missing_total_policy="env",
            )
        ).transform(3.0, {})

        self.assertEqual(3.0, fallback.training_reward)

    def test_missing_reward_components_are_zero_or_errors(self):
        zero = RewardTransformer(
            RewardTransformConfig(
                mode="component_weights",
                component_weights={"progress": 1.0},
                missing_component_policy="zero",
            )
        ).transform(3.0, {})

        self.assertEqual(0.0, zero.training_reward)

        with self.assertRaises(KeyError):
            RewardTransformer(
                RewardTransformConfig(
                    mode="component_weights",
                    component_weights={"progress": 1.0},
                    missing_component_policy="error",
                )
            ).transform(3.0, {})

    def test_sign_mode_matches_legacy_reward_clipper(self):
        env = ClipRewardEnv(TinyImageEnv([(-3.75, False, True, {})]))
        try:
            _, clipped_reward, _, _, info = env.step(0)
        finally:
            env.close()

        transformed = RewardTransformer(
            RewardTransformConfig(mode="sign")
        ).transform(info["raw_reward"], info)

        self.assertEqual(clipped_reward, transformed.training_reward)

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

    def test_frame_skip_aggregates_reward_diagnostics(self):
        env = TinyImageEnv([
            (
                1.5,
                False,
                False,
                {
                    "raw_reward": 1.5,
                    "reward_total_unclipped": 1.5,
                    "reward_total_clipped": 1.0,
                    "reward_components": {"progress": 2.0, "death": 0.0},
                },
            ),
            (
                2.5,
                True,
                False,
                {
                    "raw_reward": 2.5,
                    "reward_total_unclipped": 2.5,
                    "reward_total_clipped": 2.0,
                    "reward_components": {"progress": 3.0, "death": -1.0},
                },
            ),
        ])
        env = MaxFrameskipEnv(env, skip=4)

        try:
            _, reward, _, _, info = env.step(0)
            self.assertEqual(4.0, reward)
            self.assertEqual(4.0, info["raw_reward"])
            self.assertEqual(4.0, info["reward_total_unclipped"])
            self.assertEqual(3.0, info["reward_total_clipped"])
            self.assertEqual(5.0, info["reward_components"]["progress"])
            self.assertEqual(-1.0, info["reward_components"]["death"])
        finally:
            env.close()

    def test_macro_action_sequences_stop_early_and_aggregate_diagnostics(self):
        base_env = TinyImageEnv([
            (
                1.0,
                False,
                False,
                {
                    "frames_skipped": 2,
                    "raw_reward": 1.0,
                    "reward_total_unclipped": 1.5,
                    "reward_total_clipped": 1.0,
                    "reward_components": {"progress": 2.0},
                },
            ),
            (
                2.0,
                True,
                False,
                {
                    "frames_skipped": 3,
                    "raw_reward": 2.0,
                    "reward_total_unclipped": 2.5,
                    "reward_total_clipped": 1.0,
                    "reward_components": {"progress": 3.0, "death": -1.0},
                },
            ),
            (99.0, False, False, {"frames_skipped": 1}),
        ])
        env = MacroActionEnv(
            base_env,
            (
                MacroAction(
                    name="test_macro",
                    action_indices=(3, 4, 5),
                    button_sequence=(("right",), ("right", "A"), ("right", "B")),
                    description="Test sequence.",
                ),
            ),
            macro_action_set="unit",
        )

        try:
            _, reward, terminated, truncated, info = env.step(0)
            self.assertEqual([3, 4], base_env.actions)
            self.assertEqual(3.0, reward)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(5, info["frames_skipped"])
            self.assertEqual(2, info["macro_steps"])
            self.assertEqual("test_macro", info["macro_action_name"])
            self.assertEqual([3, 4, 5], info["macro_action_sequence"])
            self.assertEqual(3.0, info["raw_reward"])
            self.assertEqual(4.0, info["reward_total_unclipped"])
            self.assertEqual(2.0, info["reward_total_clipped"])
            self.assertEqual(5.0, info["reward_components"]["progress"])
            self.assertEqual(-1.0, info["reward_components"]["death"])
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
