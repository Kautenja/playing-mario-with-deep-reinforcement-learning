"""Mario Gymnasium environment factory contract tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from mario_rl.envs import MarioEnvConfig, make_env


def _first_rollout(seed=123, env_id="SuperMarioBros1-1-v0", actions=(0, 1, 0)):
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

    def test_factory_creates_preprocessed_gymnasium_env(self):
        env = make_env(
            "SuperMarioBros1-1-v0",
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

    def test_random_stage_env_preserves_seeded_stage_selection(self):
        first = _first_rollout(
            seed=321,
            env_id="SuperMarioBrosRandomStages-v0",
            actions=(),
        )
        second = _first_rollout(
            seed=321,
            env_id="SuperMarioBrosRandomStages-v0",
            actions=(),
        )
        self.assertEqual(first[1:3], second[1:3])

    def test_optional_video_recording_writes_gymnasium_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            env = make_env(
                "SuperMarioBros1-1-v0",
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
