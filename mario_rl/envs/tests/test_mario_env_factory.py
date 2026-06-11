"""Mario Gymnasium environment factory contract tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from mario_rl.envs import (
    ACTION_SETS,
    MarioEnvConfig,
    NATIVE_ACTION_COUNT,
    available_env_ids,
    choose_stage_env_id,
    get_action_set,
    make_env,
    resolve_action_set,
    task_for_env_id,
)


def _first_rollout(seed=123, env_id="SuperMarioBros-1-1-v0", actions=(0, 1, 0)):
    """Run a short deterministic rollout and return comparable metadata."""
    env = make_env(
        env_id,
        render_mode="rgb_array",
        seed=seed,
        action_set="simple",
        frame_skip=1,
        frame_stack=2,
        image_size=(16, 16),
        clip_rewards=False,
    )
    try:
        obs, info = env.reset(seed=seed)
        steps = []
        for action in actions:
            _, reward, terminated, truncated, step_info = env.step(action)
            steps.append((
                float(reward),
                bool(terminated),
                bool(truncated),
                step_info.get("world"),
                step_info.get("stage"),
                step_info.get("score"),
            ))
        return obs.shape, info.get("world"), info.get("stage"), tuple(steps)
    finally:
        env.close()


class MarioEnvFactoryTest(TestCase):
    """Validate the public modern Mario factory."""

    def test_action_set_resolution_covers_supported_names_and_aliases(self):
        self.assertEqual(
            {"nes", "right", "right_only", "simple", "complex"},
            set(ACTION_SETS),
        )
        self.assertIsNone(get_action_set("nes"))

        expected_counts = {
            "nes": NATIVE_ACTION_COUNT,
            "right": 5,
            "right_only": 5,
            "simple": 7,
            "complex": 12,
        }
        for name, expected_count in expected_counts.items():
            with self.subTest(name=name):
                resolved = resolve_action_set(name)
                self.assertEqual(expected_count, resolved.num_actions)
                self.assertEqual(name == "nes", resolved.native)

        right = resolve_action_set("right")
        right_only = resolve_action_set("right_only")
        self.assertEqual("right_only", right.name)
        self.assertEqual(right_only.actions, right.actions)

    def test_factory_creates_preprocessed_gymnasium_env(self):
        env = make_env(
            "SuperMarioBros-1-1-v0",
            render_mode="rgb_array",
            seed=123,
            action_set="simple",
            frame_skip=1,
            frame_stack=4,
            image_size=(20, 24),
        )

        try:
            obs, info = env.reset(seed=123)
            result = env.step(env.action_space.sample())

            self.assertEqual((4, 20, 24), obs.shape)
            self.assertIsInstance(info, dict)
            self.assertEqual(5, len(result))
            self.assertIsInstance(result[1], float)
            self.assertIsInstance(result[4], dict)
            self.assertEqual(7, env.action_space.n)
        finally:
            env.close()

    def test_factory_creates_rgb_frame_stack(self):
        env = make_env(
            "SuperMarioBros-1-1-v0",
            render_mode="rgb_array",
            seed=123,
            action_set="simple",
            frame_skip=1,
            frame_stack=2,
            image_size=(30, 32),
            grayscale=False,
        )

        try:
            obs, _ = env.reset(seed=123)

            self.assertEqual((6, 30, 32), obs.shape)
            self.assertEqual(np.uint8, obs.dtype)
            self.assertEqual((6, 30, 32), env.observation_space.shape)
        finally:
            env.close()

    def test_factory_can_keep_native_nes_action_space(self):
        env = make_env(
            "SuperMarioBros-1-1-v0",
            render_mode="rgb_array",
            seed=123,
            action_set="nes",
            preprocess=False,
            record_statistics=False,
        )

        try:
            obs, info = env.reset(seed=123)
            self.assertEqual((240, 256, 3), obs.shape)
            self.assertIsInstance(info, dict)
            self.assertEqual(NATIVE_ACTION_COUNT, env.action_space.n)
            self.assertEqual("nes", env.mario_rl_action_set)
            self.assertEqual(NATIVE_ACTION_COUNT, env.mario_rl_action_count)
        finally:
            env.close()

    def test_factory_creates_constrained_action_space_counts(self):
        for action_set, expected_count in (
            ("right", 5),
            ("right_only", 5),
            ("simple", 7),
            ("complex", 12),
        ):
            with self.subTest(action_set=action_set):
                env = make_env(
                    "SuperMarioBros-1-1-v0",
                    render_mode="rgb_array",
                    seed=123,
                    action_set=action_set,
                    preprocess=False,
                    record_statistics=False,
                )
                try:
                    self.assertEqual(expected_count, env.action_space.n)
                    self.assertEqual(resolve_action_set(action_set).name, env.mario_rl_action_set)
                    self.assertEqual(expected_count, env.mario_rl_action_count)
                finally:
                    env.close()

    def test_config_object_can_create_unpreprocessed_base_env_alias(self):
        config = MarioEnvConfig(
            env_id="SuperMarioBros-v0",
            render_mode="rgb_array",
            seed=77,
            action_set="right_only",
            preprocess=False,
            record_statistics=False,
        )
        env = make_env(config=config)

        try:
            obs, info = env.reset()
            self.assertEqual((240, 256, 3), obs.shape)
            self.assertEqual(5, env.action_space.n)
            self.assertEqual(1, info["world"])
            self.assertEqual(1, info["stage"])
        finally:
            env.close()

    def test_seeded_rollout_is_repeatable_on_same_platform(self):
        first = _first_rollout(seed=123)
        second = _first_rollout(seed=123)
        self.assertEqual(first, second)

    def test_stage_task_metadata_replaces_removed_random_stage_env(self):
        env_ids = available_env_ids()
        self.assertIn("SuperMarioBros-1-1-v0", env_ids)
        self.assertIn("SuperMarioBros1-1-v0", available_env_ids(include_aliases=True))
        self.assertIn("SuperMarioBros2USA-v0", env_ids)
        self.assertIn("SuperMarioBros3-1-1-v0", env_ids)
        self.assertNotIn("SuperMarioBrosRandomStages-v0", env_ids)

        first = choose_stage_env_id(seed=321)
        second = choose_stage_env_id(seed=321)
        self.assertEqual(first, second)
        self.assertIn(first, available_env_ids(game_family="smb1", single_stage=True))

        task = task_for_env_id(first)
        self.assertTrue(task.single_stage)
        self.assertEqual(first, task.env_id)

    def test_legacy_separator_free_stage_alias_still_creates_env(self):
        obs_shape, world, stage, _ = _first_rollout(
            seed=321,
            env_id="SuperMarioBros1-1-v0",
            actions=(),
        )
        self.assertEqual((2, 16, 16), obs_shape)
        self.assertEqual(1, world)
        self.assertEqual(1, stage)

    def test_factory_creates_representative_9x_game_family_envs(self):
        for env_id in (
            "SuperMarioBros-1-1-v0",
            "SuperMarioBros2-1-1-v0",
            "SuperMarioBros2USA-1-1-v0",
            "SuperMarioBros3-1-1-v0",
        ):
            with self.subTest(env_id=env_id):
                env = make_env(
                    env_id,
                    render_mode="rgb_array",
                    seed=123,
                    action_set="right_only",
                    preprocess=False,
                    record_statistics=False,
                )
                try:
                    obs, info = env.reset(seed=123)
                    self.assertEqual((240, 256, 3), obs.shape)
                    self.assertIn("task_id", info)
                    self.assertEqual(5, len(env.step(0)))
                finally:
                    env.close()

    def test_optional_video_recording_writes_gymnasium_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            env = make_env(
                "SuperMarioBros-1-1-v0",
                render_mode="rgb_array",
                seed=123,
                action_set="right_only",
                frame_skip=1,
                frame_stack=None,
                image_size=(16, 16),
                video_dir=tmpdir,
                video_episode_trigger=lambda episode_id: episode_id == 0,
                video_length=2,
            )
            try:
                env.reset(seed=123)
                env.step(0)
            finally:
                env.close()

            artifacts = [path.name for path in Path(tmpdir).iterdir()]
            self.assertTrue(any(name.endswith(".mp4") for name in artifacts), artifacts)

    def test_modern_package_does_not_import_legacy_gym_modules(self):
        import mario_rl.envs as envs

        package_root = Path(envs.__file__).resolve().parent
        source_text = "\n".join(
            path.read_text()
            for path in package_root.glob("*.py")
            if path.name != "__init__.py"
        )

        self.assertNotIn("import gym\n", source_text)
        self.assertNotIn("BinarySpaceToDiscreteSpaceEnv", source_text)
        self.assertNotIn("gym.wrappers.Monitor", source_text)
        self.assertEqual(np.uint8, MarioEnvConfig().dtype)
